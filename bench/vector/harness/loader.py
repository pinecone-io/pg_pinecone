"""Loads a dataset + its filter columns into the bench Postgres table."""
from __future__ import annotations

import numpy as np
import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector

from . import datasets


def filter_column_name(selectivity: float, mode: str) -> str:
    return f"filt_{mode}_{int(round(selectivity * 1000)):04d}"


def filter_eq_column_name(selectivity: float, mode: str) -> str:
    """The SAME qualifying mask as `filter_column_name`'s boolean column,
    mirrored into an int2 (1 = qualifying, 0 = not) companion column.

    pg_pinecone's AM only supports equality on an int2/int4/int8/text/varchar
    filter column (`am.rs`'s `ambuild`: a bare `boolean` column has no pg_pinecone
    opclass at all and is rejected at DDL time) -- every pgvector system in
    this harness queries the boolean column directly (`WHERE {bool_col}`),
    but a `pg_pinecone_native` system's `CREATE INDEX ... USING pinecone (embedding
    op, filter_col)` must declare an equality-filterable column instead, so
    it queries this one (`WHERE {eq_col} = 1`) against identical mask
    values -- same realized selectivity, same ground truth, just a type
    pg_pinecone's validation accepts.
    """
    return f"{filter_column_name(selectivity, mode)}_eq"


def filter_key(spec: dict) -> str:
    """Unique label of one filter cell: the pushed column alone, or
    `<pushed>+<residual>` when the spec carries a `residual` (a second column
    the index cannot push -- its qual stays an executor Filter). Without a
    residual this is exactly `filter_column_name`, so older results/caches keep
    their keys."""
    key = filter_column_name(spec["selectivity"], spec["mode"])
    r = spec.get("residual")
    if r:
        key += "+" + filter_column_name(r["selectivity"], r["mode"])
    return key


def column_specs(filter_specs: list[dict]) -> list[dict]:
    """The distinct {selectivity, mode} column specs behind `filter_specs`
    (pushed filters AND their residuals), in first-seen order -- what
    `create_and_load` must create columns for."""
    seen, out = set(), []
    for f in filter_specs:
        for s in (f, f.get("residual")):
            if s:
                name = filter_column_name(s["selectivity"], s["mode"])
                if name not in seen:
                    seen.add(name)
                    out.append({"selectivity": s["selectivity"], "mode": s["mode"]})
    return out


def create_and_load(conn: psycopg.Connection, table: str, vectors: np.ndarray,
                     cluster_ids: np.ndarray, filter_specs: list[dict], seed: int,
                     n_centers: int, tenant_ids: np.ndarray | None = None,
                     storage_type: str = "vector", column_storage: str | None = None):
    """(Re)create `table` and bulk-load vectors + cluster_id + one boolean column
    (for pgvector systems) + one int2 equality-mirror column (for pg_pinecone_native,
    see `filter_eq_column_name`) per filter spec ({selectivity, mode}).
    `column_storage` (e.g. "plain") is set on `embedding` before the COPY; None keeps the type's default.
    Returns {bool_column_name: realized_selectivity}.
    """
    register_vector(conn)
    if storage_type not in ("vector", "halfvec"):
        raise ValueError(f"unknown storage_type {storage_type!r}")
    dim = vectors.shape[1]

    cols_meta = [(filter_column_name(f["selectivity"], f["mode"]),
                  filter_eq_column_name(f["selectivity"], f["mode"]), f)
                 for f in filter_specs]
    col_defs = ", ".join(f"{bc} boolean, {ec} int2" for bc, ec, _ in cols_meta)

    masks = {bc: datasets.assign_filter_values(cluster_ids, n_centers, f["selectivity"], f["mode"], seed)
              for bc, _ec, f in cols_meta}

    col_names = ["id", "embedding", "cluster_id"]
    for bc, ec, _ in cols_meta:
        col_names += [bc, ec]
    if tenant_ids is not None:
        col_names.append("tenant_id")
        col_defs += (", " if col_defs else "") + "tenant_id int NOT NULL"
    n = vectors.shape[0]
    copy_sql = f"COPY {table} ({', '.join(col_names)}) FROM STDIN WITH (FORMAT BINARY)"
    # CREATE TABLE + COPY share one explicit transaction (disk-safety fix: under
    # wal_level=minimal, Postgres skips WAL for a bulk load into a relation
    # created earlier in the SAME transaction -- with conn in autocommit mode,
    # DROP/CREATE/COPY as separate implicit transactions defeated that
    # optimization, doubling peak disk usage during load (observed: 3.9GB table
    # + 3.3GB of WAL for it on a 1M x 768d load, on a machine with ~17GB free).
    # Load time and disk footprint aren't measured/reported metrics -- this
    # doesn't touch anything the benchmark actually compares.
    with conn.transaction():
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            f"CREATE TABLE {table} (id bigint PRIMARY KEY, embedding {storage_type}({dim}), "
            f"cluster_id int{',' if col_defs else ''} {col_defs})"
        )
        if column_storage is not None:
            conn.execute(f"ALTER TABLE {table} ALTER COLUMN embedding SET STORAGE {column_storage}")
        with conn.cursor().copy(copy_sql) as copy:
            types = ["int8", storage_type, "int4"]
            for _ in cols_meta:
                types += ["bool", "int2"]
            if tenant_ids is not None:
                types.append("int4")
            copy.set_types(types)
            for i in range(n):
                vector = HalfVector(vectors[i]) if storage_type == "halfvec" else vectors[i]
                row = [i, vector, int(cluster_ids[i])]
                for bc, _ec, _f in cols_meta:
                    row += [bool(masks[bc][i]), int(masks[bc][i])]
                if tenant_ids is not None:
                    row.append(int(tenant_ids[i]))
                copy.write_row(row)

    conn.execute(f"ANALYZE {table}")
    realized = {bc: float(np.mean(mask)) for bc, mask in masks.items()}
    return realized
