#!/bin/zsh
# Build pg_pinecone against Postgres.app's PostgreSQL 18 and, when SIGN=1,
# wrap it in a notarized arm64 installer.
#
#   SIGN=0 packaging/postgresapp/build.zsh
#   SIGN=1 INSTALL_ROOT=/path/to/prefix packaging/postgresapp/build.zsh
#
# SIGN=0 compiles and checks linkage. SIGN=1 code-signs, builds the .pkg, and
# notarizes. Pass INSTALL_ROOT to package a tree already built (the release
# job does this so signing does not compile again). The install root is the
# prefix that contains lib/ and share/. Its path is written to
# target/postgresapp-install-root.txt.
set -euo pipefail

REPO_ROOT=${0:A:h:h:h}
PG_MAJOR=18
PG_CONFIG=${PG_CONFIG:-/Applications/Postgres.app/Contents/Versions/$PG_MAJOR/bin/pg_config}
SIGN=${SIGN:-0}

check_linked_libs() {
  local dylib=$1 id line dep i
  local -a lines
  id=$(otool -D "$dylib" | sed -n '2p')
  id="${id#"${id%%[![:space:]]*}"}"
  id="${id%"${id##*[![:space:]]}"}"
  [[ -n $id ]] || { echo "otool -D produced no install id for $dylib" >&2; exit 1; }
  lines=("${(@f)$(otool -L "$dylib")}")
  for (( i = 2; i <= ${#lines}; i++ )); do
    line=$lines[$i]
    dep=${line%% \(*}
    dep="${dep#"${dep%%[![:space:]]*}"}"
    dep="${dep%"${dep##*[![:space:]]}"}"
    [[ -z $dep ]] && continue
    [[ $dep == "$id" ]] && continue
    case $dep in
      /usr/lib/*|/System/*) ;;
      *)
        echo "non-system dependency in $dylib: $dep" >&2
        exit 1
        ;;
    esac
  done
}

compile() {
  local sdk
  command -v cargo >/dev/null || { echo "cargo is not on PATH" >&2; exit 1; }
  cargo pgrx --version >/dev/null 2>&1 || { echo "cargo-pgrx is not installed" >&2; exit 1; }
  [[ -x $PG_CONFIG ]] || { echo "Postgres.app pg_config not found at $PG_CONFIG" >&2; exit 1; }
  sdk=$(xcrun --show-sdk-path)
  [[ -n $sdk ]] || { echo "xcrun --show-sdk-path returned nothing" >&2; exit 1; }
  # init rewrites $PGRX_HOME/config.toml; keep it off ~/.pgrx so a local run
  # doesn't repoint `cargo pgrx test pg18` at Postgres.app.
  export PGRX_HOME=$REPO_ROOT/target/postgresapp-pgrx-home
  mkdir -p "$PGRX_HOME"
  cargo pgrx init --pg18="$PG_CONFIG"
  (
    cd "$REPO_ROOT/pg_pinecone"
    # RUSTFLAGS replaces pg_pinecone/.cargo/config.toml, so the macOS link
    # flag has to be repeated here. The sysroot flag is what a Postgres.app
    # build needs for bindgen to see the SDK headers.
    RUSTFLAGS='-Clink-arg=-Wl,-undefined,dynamic_lookup' \
      BINDGEN_EXTRA_CLANG_ARGS="-isysroot $sdk" \
      cargo pgrx package --pg-config "$PG_CONFIG"
  )
}

discover_install_root() {
  # pgrx nests each build under the pg_config's own pkglibdir, so a pg18 build
  # against another Postgres can sit next to this one. Take this path exactly.
  local dylib=$REPO_ROOT/target/release/pg_pinecone-pg${PG_MAJOR}$("$PG_CONFIG" --pkglibdir)/pg_pinecone.dylib
  [[ -f $dylib ]] || { echo "no pg_pinecone.dylib at $dylib" >&2; exit 1; }
  INSTALL_ROOT=${dylib:h:h:h}
  [[ -f $INSTALL_ROOT/share/postgresql/extension/pg_pinecone.control ]] \
    || { echo "pg_pinecone.control missing under $INSTALL_ROOT" >&2; exit 1; }
}

# Link check runs only on a fresh build. The sign job passes INSTALL_ROOT
# for a tree this function already accepted.
if [[ -z ${INSTALL_ROOT:-} ]]; then
  compile
  discover_install_root
  libs=("$INSTALL_ROOT"/lib/postgresql/*.dylib(N))
  (( ${#libs} )) || { echo "no dylibs under $INSTALL_ROOT/lib/postgresql" >&2; exit 1; }
  for lib in $libs; do
    check_linked_libs "$lib"
  done
  echo "linkage ok: ${libs[*]}" >&2
else
  [[ -f $INSTALL_ROOT/lib/postgresql/pg_pinecone.dylib ]] \
    || { echo "INSTALL_ROOT has no pg_pinecone.dylib: $INSTALL_ROOT" >&2; exit 1; }
fi

mkdir -p "$REPO_ROOT/target"
printf '%s\n' "$INSTALL_ROOT" > "$REPO_ROOT/target/postgresapp-install-root.txt"

if [[ $SIGN != 1 ]]; then
  exit 0
fi

: "${APPLE_API_KEY_ID:?APPLE_API_KEY_ID is required when SIGN=1}"
: "${APPLE_API_ISSUER_ID:?APPLE_API_ISSUER_ID is required when SIGN=1}"
: "${APPLE_API_KEY_P8:?APPLE_API_KEY_P8 is required when SIGN=1}"

EXTENSION_VERSION=$(grep -m1 '^version' "$REPO_ROOT/pg_pinecone/Cargo.toml" | cut -d'"' -f2)
[[ -n $EXTENSION_VERSION ]] || { echo "could not read extension version" >&2; exit 1; }

libs=("$INSTALL_ROOT"/lib/postgresql/*.dylib(N))
(( ${#libs} )) || { echo "no dylibs to sign under $INSTALL_ROOT/lib/postgresql" >&2; exit 1; }
for lib in $libs; do
  # --force matches ParadeDB. A linker signature can be replaced without it;
  # --force also covers a dylib that was signed for real on a re-run.
  codesign --force --sign "Developer ID Application" --timestamp "$lib"
done

BUILD=$(mktemp -d)
mkdir -p "$BUILD/Resources" "$BUILD/Scripts"
for file in "$REPO_ROOT"/packaging/postgresapp/Resources/*.html "$REPO_ROOT"/packaging/postgresapp/distribution.xml; do
  base=${file:t}
  if [[ $file == *.html ]]; then
    out="$BUILD/Resources/$base"
  else
    out="$BUILD/$base"
  fi
  sed -e "s|@EXTENSION_VERSION@|${EXTENSION_VERSION}|g" \
      -e "s|@PG_MAJOR_VERSION@|${PG_MAJOR}|g" \
      -e "s|@EXTENSION_NAME@|pg_pinecone|g" \
      "$file" > "$out"
done
cp "$REPO_ROOT/packaging/postgresapp/Scripts/preinstall" "$BUILD/Scripts/preinstall"
chmod +x "$BUILD/Scripts/preinstall"
cp "$REPO_ROOT/LICENSE" "$BUILD/Resources/license.txt"

FINAL="pg_pinecone-pg${PG_MAJOR}-${EXTENSION_VERSION}.pkg"
(
  cd "$BUILD"
  pkgbuild \
    --root "$INSTALL_ROOT" \
    --install-location "/Library/Application Support/Postgres/Extensions/${PG_MAJOR}/pg_pinecone" \
    --identifier "com.postgresapp.extension.${PG_MAJOR}.pg_pinecone" \
    --version "$EXTENSION_VERSION" \
    --scripts Scripts \
    --sign "Developer ID Installer" \
    "pg_pinecone-${PG_MAJOR}.pkg"
  productbuild \
    --distribution distribution.xml \
    --resources Resources \
    --package-path . \
    --sign "Developer ID Installer" \
    "$FINAL"
)
mv "$BUILD/$FINAL" "$REPO_ROOT/target/$FINAL"
rm -rf "$BUILD"

# Accept either the PEM itself or a base64 wrapping of it. GitHub secrets
# often store the .p8 base64-encoded, which is what ParadeDB's workflow decodes.
keyfile=$(mktemp)
trap 'rm -f "$keyfile"' EXIT
if [[ $APPLE_API_KEY_P8 == -----BEGIN* ]]; then
  printf '%s\n' "$APPLE_API_KEY_P8" > "$keyfile"
else
  printf '%s' "$APPLE_API_KEY_P8" | base64 --decode > "$keyfile"
fi
xcrun notarytool submit "$REPO_ROOT/target/$FINAL" \
  --key "$keyfile" \
  --key-id "$APPLE_API_KEY_ID" \
  --issuer "$APPLE_API_ISSUER_ID" \
  --wait
xcrun stapler staple "$REPO_ROOT/target/$FINAL"
echo "pkg: $REPO_ROOT/target/$FINAL" >&2
