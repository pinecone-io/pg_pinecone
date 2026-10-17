"""Sustained update-stream (churn) workload: re-embed + delete/reinsert per
cycle, measuring recall@10 drift against freshly-recomputed ground truth and
post-VACUUM recovery.

Ground truth is recomputed every cycle over the table's *current* live rows:
the interesting signal is whether the index's approximate answers keep pace
with a moving dataset, not whether it still agrees with a stale answer.

Churn realism: re-embeds and reinserts are drawn
from the SAME generative model as the base dataset (persisted centers +
per-cluster sigma), not a random walk off the row's current position or a
fresh ad hoc N(0,1) singleton unrelated to any real cluster. Reinserted rows
also get their filter columns populated using the same assignment rule as
the base load, instead of left NULL (which would silently make them
invisible to every filtered-recall check).
"""
from __future__ import annotations

import numpy as np
from pgvector.psycopg import register_vector

from . import datasets as ds_mod
from . import groundtruth as gt_mod
from . import loader as loader_mod
from . import workloads as wl_mod
from .systems import System, distance_operator
from .util import timed


def fetch_all_vectors(conn, table: str, column: str):
    register_vector(conn)
    rows = conn.execute(f"SELECT id, {column} FROM {table} ORDER BY id").fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    vectors = np.stack([r[1].to_numpy().astype(np.float32) for r in rows])
    return ids, vectors


def churn_cycle(conn, table: str, column: str, dim: int, centers: np.ndarray,
                 cluster_sigmas: np.ndarray, next_id: int, reembed_pct: float, delete_pct: float,
                 rng: np.random.Generator, n_centers: int, filter_specs: list[dict],
                 base_cluster_ids: np.ndarray, seed: int):
    """One cycle of in-place re-embeds + delete/reinsert churn. Returns updated next_id."""
    rows = conn.execute(f"SELECT id, cluster_id FROM {table} ORDER BY id").fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    row_cluster_ids = np.array([r[1] for r in rows], dtype=np.int64)
    n_live = len(ids)

    # Re-embed: draw a FRESH sample from the row's own cluster's generative
    # distribution (center + that cluster's sigma) -- models re-embedding
    # (e.g. a document edited, a new model version) as "this item is still
    # the same topic, freshly encoded," not a random walk off wherever the
    # row currently sits (which drifts monotonically away from its center
    # every cycle and isn't what re-embedding means).
    n_reembed = int(round(reembed_pct * n_live))
    reembed_idx = rng.choice(n_live, size=n_reembed, replace=False)
    reembed_ids = ids[reembed_idx]
    reembed_clusters = row_cluster_ids[reembed_idx]
    sigma = cluster_sigmas[reembed_clusters]
    new_embed_vecs = (centers[reembed_clusters]
                       + rng.normal(0, 1, size=(n_reembed, dim)).astype(np.float32) * sigma[:, None])
    for rid, vec in zip(reembed_ids, new_embed_vecs):
        conn.execute(f"UPDATE {table} SET {column} = %s WHERE id = %s",
                      (vec.astype(np.float32), int(rid)))

    # Delete + reinsert: remove a random slice, insert the same count of
    # brand-new rows drawn fresh from the generative model (fresh cluster_id,
    # center + sigma for that cluster) -- not an ad hoc N(0,1) point unrelated
    # to any real cluster.
    remaining_mask = np.ones(n_live, dtype=bool)
    remaining_mask[reembed_idx] = False
    remaining_ids = ids[remaining_mask]
    n_churn = int(round(delete_pct * n_live))
    del_ids = rng.choice(remaining_ids, size=min(n_churn, len(remaining_ids)), replace=False)
    conn.execute(f"DELETE FROM {table} WHERE id = ANY(%s)", (del_ids.tolist(),))

    n_new = len(del_ids)
    new_cluster_ids = rng.integers(0, n_centers, size=n_new)
    new_sigma = cluster_sigmas[new_cluster_ids]
    new_vecs = (centers[new_cluster_ids]
                 + rng.normal(0, 1, size=(n_new, dim)).astype(np.float32) * new_sigma[:, None])

    # `filter_cols`/`eq_cols` mirror the SAME values (bool vs int2 companion
    # -- see `loader.filter_eq_column_name`) so pg_pinecone_native churn cycles (its
    # only path into these columns: local mode has no incremental aminsert,
    # so new rows only ever reach the table -- and later a rebuilt index --
    # via this COPY, never via a live index update) stay populated exactly
    # like the base load.
    filter_cols = {}
    eq_cols = {}
    for spec in filter_specs:
        col = loader_mod.filter_column_name(spec["selectivity"], spec["mode"])
        eq_col = loader_mod.filter_eq_column_name(spec["selectivity"], spec["mode"])
        values = ds_mod.assign_new_row_filter_values(
            new_cluster_ids, n_centers, spec["selectivity"], spec["mode"], seed, rng,
            base_cluster_ids=base_cluster_ids)
        filter_cols[col] = values
        eq_cols[eq_col] = values

    col_names = ["id", column, "cluster_id"] + list(filter_cols.keys()) + list(eq_cols.keys())
    types = ["int8", "vector", "int4"] + ["bool"] * len(filter_cols) + ["int2"] * len(eq_cols)
    with conn.cursor().copy(f"COPY {table} ({', '.join(col_names)}) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(types)
        for i in range(n_new):
            row = [next_id + i, new_vecs[i], int(new_cluster_ids[i])] + \
                  [bool(filter_cols[c][i]) for c in filter_cols] + \
                  [int(eq_cols[c][i]) for c in eq_cols]
            copy.write_row(row)
    next_id += n_new

    conn.execute(f"ANALYZE {table}")
    return next_id, dict(n_live_before=n_live, n_reembedded=n_reembed, n_deleted=n_new,
                          n_reinserted=n_new)


def measure_recall(conn, table, column, metric, system: System, sweep_value, queries,
                    ground_truth, k=10, where_clause=""):
    """`ground_truth` is the (ids, dists) pair from brute_force_topk -- recall
    is tie-aware (see groundtruth.py's module docstring)."""
    truth_ids, truth_dists = ground_truth
    op = distance_operator(metric)
    system.set_session_params(conn, sweep_value, None)  # once per connection per point
    recalls = []
    for q, truth_row, dist_row in zip(queries, truth_ids, truth_dists):
        rows = conn.execute(
            f"SELECT id FROM {table} {where_clause} ORDER BY {column} {op} %s LIMIT %s", (q, k)
        ).fetchall()
        ids = [r[0] for r in rows]
        recalls.append(gt_mod.recall_at_k(ids, truth_row, k, truth_dists_row=dist_row))
    return float(np.nanmean(recalls))


def run_churn(pg, table, column, metric, system: System, dataset, n_centers, cycles: int,
              reembed_pct: float, delete_pct: float, sweep_value, seed: int,
              filter_specs: list[dict] | None = None, vacuum_after_each_cycle: bool = True,
              k: int = 10, expected_index: str | None = None):
    """Runs `cycles` churn cycles, returning a list of per-cycle result dicts.

    `expected_index`, if given, is checked via EXPLAIN before cycle 0 AND
    again after every cycle's ANALYZE (the
    original check only ran once, before the first cycle -- but ANALYZE can
    change the planner's mind mid-run just as easily as the initial build
    can, and a churn cycle that silently degrades to a seqscan would make its
    recall numbers meaningless as a statement about the INDEX's drift. Each
    cycle's result records its own `plan_shape`/`index_used`, and a mismatch
    raises immediately -- same fail-loud posture as the original one-time
    check, just applied every cycle instead of only the first).
    """
    filter_specs = filter_specs or []
    rng = np.random.default_rng(seed)
    results = []
    op = distance_operator(metric)
    with pg.connect() as conn:
        register_vector(conn)
        if expected_index is not None:
            plan_shape, index_used, _ = wl_mod.capture_plan(
                pg.dsn(), table, column, op, system, sweep_value, None, "", dataset.queries[0], k)
            if plan_shape != f"index_scan:{expected_index}":
                raise RuntimeError(
                    f"churn index isolation check failed: expected plan to use index "
                    f"{expected_index!r}, got plan_shape={plan_shape!r} index_used={index_used!r}")

        next_id = int(conn.execute(f"SELECT max(id) + 1 FROM {table}").fetchone()[0])
        for cycle in range(cycles):
            with timed() as cycle_t:
                next_id, churn_info = churn_cycle(
                    conn, table, column, dataset.vectors.shape[1], dataset.centers,
                    dataset.cluster_sigmas, next_id, reembed_pct, delete_pct, rng, n_centers,
                    filter_specs, dataset.cluster_ids, seed)

            plan_shape = index_used = None
            if expected_index is not None:
                plan_shape, index_used, _ = wl_mod.capture_plan(
                    pg.dsn(), table, column, op, system, sweep_value, None, "", dataset.queries[0], k)

            ids, vectors = fetch_all_vectors(conn, table, column)
            truth = gt_mod.brute_force_topk(
                dataset.queries, vectors, k=100, metric=metric, row_ids=ids,
                cache_key=None,  # data mutates every cycle; caching would be stale by construction
            )
            recall_pre_vacuum = measure_recall(conn, table, column, metric, system, sweep_value,
                                                dataset.queries, truth, k=k)

            recall_post_vacuum = None
            vacuum_seconds = None
            if vacuum_after_each_cycle:
                with timed() as vac_t:
                    conn.execute(f"VACUUM {table}")
                vacuum_seconds = vac_t["seconds"]
                recall_post_vacuum = measure_recall(conn, table, column, metric, system,
                                                     sweep_value, dataset.queries, truth, k=k)

            results.append(dict(
                cycle=cycle, cycle_seconds=cycle_t["seconds"], vacuum_seconds=vacuum_seconds,
                recall_at_k_pre_vacuum=recall_pre_vacuum,
                recall_at_k_post_vacuum=recall_post_vacuum,
                plan_shape=plan_shape, index_used=index_used, k=k, **churn_info,
            ))

            if expected_index is not None and plan_shape != f"index_scan:{expected_index}":
                raise RuntimeError(
                    f"churn cycle {cycle} lost index isolation after ANALYZE: expected plan to "
                    f"use index {expected_index!r}, got plan_shape={plan_shape!r} "
                    f"index_used={index_used!r}. Partial per-cycle results: {results}")
    return results


def run_churn_pg_pinecone_native(pg, table, column, metric, system: System, dataset, n_centers,
                           cycles: int, reembed_pct: float, delete_pct: float, seed: int,
                           sweep_value, filter_specs: list[dict] | None = None, k: int = 10,
                           expected_index: str | None = None):
    """pg_pinecone_native's churn protocol -- REINDEX-shaped, NOT `run_churn`'s
    in-place live-index churn. Documented, honest comparability difference
    (a protocol constraint), not a like-for-like substitute:

    Local-mode pg_pinecone's `aminsert` is a hard-stub `ERROR` (incremental insert
    is not implemented yet) -- so ANY `INSERT`, or any `UPDATE` that touches
    the indexed vector column (always a non-HOT update, since the vector
    column IS indexed), raises immediately while this system's pg_pinecone index
    exists on the table. `run_churn`'s `churn_cycle` does both (re-embed via
    `UPDATE`, delete+reinsert via `COPY`), so it cannot run as-is against a
    live pg_pinecone_native index. The only way churn's mutated rows ever reach the
    index at all is a full rebuild. Per cycle:

      1. (index present) `VACUUM` -- reclaims already-dead TIDs
         (`ambulkdelete`'s O(churn) list-page sweep). Reported for the
         two-column comparison table even though it's
         logically redundant with the rebuild that immediately follows --
         it's real, separately-measurable work an honest accounting counts.
      2. `DROP INDEX` (measured; expected to be cheap).
      3. Mutate the table exactly like pgvector's churn (`churn_cycle`):
         re-embed `reembed_pct`, delete+reinsert `delete_pct` -- safe now,
         since no pg_pinecone index exists to reject the writes.
      4. `CREATE INDEX` fresh (`workloads.build_index`) -- THE number
         ("REINDEX time IS the number, count it fully"):
         `maintenance_seconds_total` sums all three steps, no discount.
      5. Measure recall against the fresh index. There is no "pre-vacuum
         recall" analog here -- no live index existed during the mutation to
         have drifted -- so recall reflects a normal fresh build every
         cycle, by construction (consistent with the empirical finding
         that pgvector churn showed zero recall drift, just structural here
         rather than measured).
    """
    filter_specs = filter_specs or []
    rng = np.random.default_rng(seed)
    results = []
    op = distance_operator(metric)
    idx_name = expected_index or system.index_name(table)
    with pg.connect() as conn:
        register_vector(conn)
        next_id = int(conn.execute(f"SELECT max(id) + 1 FROM {table}").fetchone()[0])
        for cycle in range(cycles):
            with timed() as vac_t:
                conn.execute(f"VACUUM {table}")
            vacuum_seconds = vac_t["seconds"]

            with timed() as drop_t:
                conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
            drop_seconds = drop_t["seconds"]

            with timed() as cycle_t:
                next_id, churn_info = churn_cycle(
                    conn, table, column, dataset.vectors.shape[1], dataset.centers,
                    dataset.cluster_sigmas, next_id, reembed_pct, delete_pct, rng, n_centers,
                    filter_specs, dataset.cluster_ids, seed)

            rebuild = wl_mod.build_index(pg, table, column, metric, system, dataset.vectors.shape[0])

            plan_shape, index_used, _ = wl_mod.capture_plan(
                pg.dsn(), table, column, op, system, sweep_value, None, "", dataset.queries[0], k)
            if plan_shape != f"index_scan:{idx_name}":
                raise RuntimeError(
                    f"pg_pinecone_native churn cycle {cycle}: rebuilt index isn't being used -- "
                    f"expected plan to use index {idx_name!r}, got plan_shape={plan_shape!r} "
                    f"index_used={index_used!r}")

            ids, vectors = fetch_all_vectors(conn, table, column)
            truth = gt_mod.brute_force_topk(
                dataset.queries, vectors, k=100, metric=metric, row_ids=ids,
                cache_key=None,  # data mutates every cycle; caching would be stale by construction
            )
            recall = measure_recall(conn, table, column, metric, system, sweep_value,
                                     dataset.queries, truth, k=k)

            maintenance_seconds_total = vacuum_seconds + drop_seconds + rebuild["build_seconds"]
            results.append(dict(
                cycle=cycle, vacuum_seconds=vacuum_seconds, drop_index_seconds=drop_seconds,
                cycle_seconds=cycle_t["seconds"], rebuild_seconds=rebuild["build_seconds"],
                maintenance_seconds_total=maintenance_seconds_total,
                index_size_bytes=rebuild["index_size_bytes"],
                recall_at_k_post_rebuild=recall,
                # No live-index recall exists mid-cycle (module docstring) --
                # these two keys are kept so downstream report code that
                # expects `run_churn`'s shape (`recall_at_k_pre_vacuum`/
                # `recall_at_k_post_vacuum`) degrades to "n/a" instead of a
                # KeyError, with the honest-comparability reason attached.
                recall_at_k_pre_vacuum=None, recall_at_k_post_vacuum=None,
                reindex_shaped=True,
                plan_shape=plan_shape, index_used=index_used, k=k, **churn_info,
            ))
    return results
