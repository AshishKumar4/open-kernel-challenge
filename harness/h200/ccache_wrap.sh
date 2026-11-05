#!/bin/sh
# Installed as /usr/local/bin/ccache (ahead of the real one in PATH), the "ccache" torch puts before nvcc and c++: ccache
# with paths taken relative to the extension's build dir (ninja's cwd), so the same extension built in any build folder
# (/root/build/<trial>/ext/<name>) hits the same cache entry
self=$(readlink -f "$0")
for d in $(echo "$PATH" | tr ':' ' '); do
  c=$d/ccache
  [ -x "$c" ] && [ "$(readlink -f "$c")" != "$self" ] && CCACHE_BASEDIR="$PWD" CCACHE_NOHASHDIR=1 exec "$c" "$@"
done
exec "$@"
