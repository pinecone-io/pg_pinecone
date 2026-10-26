# syntax=docker/dockerfile:1
ARG PG_MAJOR=17

FROM pgvector/pgvector:pg${PG_MAJOR} AS build
ARG PG_MAJOR
ARG PGRX_VERSION=0.19.2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang libclang-dev pkg-config curl ca-certificates git \
      postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH=/root/.cargo/bin:${PATH}
RUN cargo install cargo-pgrx --version ${PGRX_VERSION} --locked

COPY . /src
WORKDIR /src/pg_pinecone
ARG TARGETARCH
# amd64 at the generic x86-64 baseline is ~2.6x slower on the native engine's
# Hamming scan (no POPCNT); x86-64-v3 matches the release artifacts. arm64 baseline has CNT.
RUN cargo pgrx init --pg${PG_MAJOR} "$(which pg_config)" \
    && if [ "$TARGETARCH" = amd64 ]; then export RUSTFLAGS="-C target-cpu=x86-64-v3"; fi \
    && cargo pgrx install --release --pg-config "$(which pg_config)"

FROM pgvector/pgvector:pg${PG_MAJOR}
ARG PG_MAJOR
COPY --from=build /usr/lib/postgresql/${PG_MAJOR}/lib/pg_pinecone.so \
                  /usr/lib/postgresql/${PG_MAJOR}/lib/
COPY --from=build /usr/share/postgresql/${PG_MAJOR}/extension/pg_pinecone* \
                  /usr/share/postgresql/${PG_MAJOR}/extension/
