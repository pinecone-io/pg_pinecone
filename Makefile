# PGXN build for pg_pinecone.
#
# `pgxn install` runs `make all`, then `make install` (under sudo when the
# Postgres library directory is not writable). `all` compiles with cargo-pgrx
# from pg_pinecone/, so Cargo reads pg_pinecone/.cargo/config.toml.
# PGRX_PG_CONFIG_PATH points that build at PG_CONFIG, so it does not need
# `cargo pgrx init`.
# `install` only copies the packaged files, so it does not need Rust.
#
#   make install-pgrx     # once: cargo-pgrx matching pg_pinecone/Cargo.toml
#   make                  # package for the pg_config on PATH
#   make install          # copy that package into that Postgres
#   make uninstall
#   make pgxn-zip         # pg_pinecone-<version>.zip for manager.pgxn.org
#
# PostgreSQL 17 and 18. Override the target with PG_CONFIG=/path/to/pg_config.

PG_CONFIG ?= pg_config
DISTNAME = pg_pinecone

# $(shell) rather than != so this works with the GNU make 3.81 shipped on macOS.
DISTVERSION := $(shell sed -n 's/^version = "\([^"]*\)".*/\1/p' pg_pinecone/Cargo.toml)
PGRX_VERSION := $(shell sed -n 's/^pgrx = "=\([^"]*\)".*/\1/p' pg_pinecone/Cargo.toml)
PG_VERSION_LINE := $(shell $(PG_CONFIG) --version 2>/dev/null || true)
PG_MAJOR := $(shell printf '%s\n' "$(PG_VERSION_LINE)" | sed -n -E 's/^PostgreSQL ([0-9]+).*/\1/p')
STAGE = $(CURDIR)/target/pgxn/pg$(PG_MAJOR)

.PHONY: all package install uninstall install-pgrx pgxn-zip

all: package

package:
	set -eu; \
	case "$(PG_MAJOR)" in \
	  17|18) ;; \
	  *) printf '%s\n' "pg_pinecone supports PostgreSQL 17 and 18; $(PG_CONFIG) reports: '$(PG_VERSION_LINE)'" >&2; exit 1 ;; \
	esac; \
	command -v cargo >/dev/null 2>&1 || { printf '%s\n' "cargo is required to build pg_pinecone" >&2; exit 1; }; \
	cargo pgrx --version >/dev/null 2>&1 || { printf '%s\n' "cargo-pgrx $(PGRX_VERSION) is required; run 'make install-pgrx'" >&2; exit 1; }; \
	rm -rf "$(STAGE)"; \
	cd pg_pinecone && PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" cargo --locked pgrx package --pg-config "$(PG_CONFIG)" --out-dir "$(STAGE)"

install:
	set -eu; \
	stage="$(STAGE)"; \
	if [ ! -d "$$stage" ]; then $(MAKE) package PG_CONFIG="$(PG_CONFIG)"; fi; \
	pkglibdir="$$($(PG_CONFIG) --pkglibdir)"; \
	extdir="$$($(PG_CONFIG) --sharedir)/extension"; \
	libs="$$(find "$$stage" -type f \( -name pg_pinecone.so -o -name pg_pinecone.dylib -o -name pg_pinecone.dll \))"; \
	nlibs="$$(printf '%s\n' "$$libs" | grep -c . || true)"; \
	[ "$$nlibs" -eq 1 ] || { printf '%s\n' "expected one pg_pinecone library in $$stage, found $$nlibs" >&2; exit 1; }; \
	control="$$(find "$$stage" -type f -name pg_pinecone.control)"; \
	ncontrol="$$(printf '%s\n' "$$control" | grep -c . || true)"; \
	[ "$$ncontrol" -eq 1 ] || { printf '%s\n' "expected one pg_pinecone.control in $$stage, found $$ncontrol" >&2; exit 1; }; \
	nsql="$$(find "$$stage" -type f -name 'pg_pinecone--*.sql' | grep -c . || true)"; \
	[ "$$nsql" -ge 1 ] || { printf '%s\n' "no pg_pinecone SQL scripts in $$stage" >&2; exit 1; }; \
	mkdir -p "$$pkglibdir" "$$extdir"; \
	install -m 755 "$$libs" "$$pkglibdir/"; \
	install -m 644 "$$control" "$$extdir/"; \
	find "$$stage" -type f -name 'pg_pinecone--*.sql' -exec install -m 644 {} "$$extdir/" \;

uninstall:
	set -eu; \
	pkglibdir="$$($(PG_CONFIG) --pkglibdir)"; \
	extdir="$$($(PG_CONFIG) --sharedir)/extension"; \
	rm -f "$$pkglibdir/pg_pinecone.so" "$$pkglibdir/pg_pinecone.dylib" "$$pkglibdir/pg_pinecone.dll"; \
	rm -f "$$extdir/pg_pinecone.control"; \
	rm -f "$$extdir"/pg_pinecone--*.sql

install-pgrx:
	test -n "$(PGRX_VERSION)"
	cargo install cargo-pgrx --version "$(PGRX_VERSION)" --locked

# Upload this zip once per version. PGXN rejects a repeated version.
META.json: META.json.in pg_pinecone/Cargo.toml
	test -n "$(DISTVERSION)"
	sed 's/@CARGO_VERSION@/$(DISTVERSION)/g' META.json.in > META.json
	! grep -q '@CARGO_VERSION@' META.json

# git archive reads HEAD. META.json's version is taken from the working tree,
# so that tree's Cargo.toml has to match HEAD or the zip version will not match
# the sources inside it. PGXN rejects a second upload of the same version.
pgxn-zip: META.json
	@git cat-file -e HEAD:Makefile >/dev/null 2>&1 && \
	 git cat-file -e HEAD:META.json.in >/dev/null 2>&1 && \
	 git cat-file -e HEAD:.gitattributes >/dev/null 2>&1 && \
	 git diff --quiet HEAD -- Makefile META.json.in .gitattributes pg_pinecone/Cargo.toml || { \
	  printf '%s\n' "commit Makefile, META.json.in, .gitattributes, and pg_pinecone/Cargo.toml, then re-run make pgxn-zip" >&2; \
	  exit 1; \
	}
	rm -f $(DISTNAME)-$(DISTVERSION).zip
	git archive --format=zip \
	  --prefix=$(DISTNAME)-$(DISTVERSION)/ --add-file=META.json \
	  --prefix=$(DISTNAME)-$(DISTVERSION)/ \
	  -o $(DISTNAME)-$(DISTVERSION).zip HEAD
