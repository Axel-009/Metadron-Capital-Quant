#!/usr/bin/env bash
# Re-sync vendored repos to their latest upstream commit (or a given ref).
# Usage: scripts/update_vendor.sh [Kronos|Lean|OpenJarvis ...]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A URL=(
  [Kronos]=https://github.com/shiyu-coder/Kronos.git
  [Lean]=https://github.com/QuantConnect/Lean.git
  [OpenJarvis]=https://github.com/open-jarvis/OpenJarvis.git
)
declare -A EXCLUDE=(
  [Kronos]=""
  [Lean]="./Data"
  [OpenJarvis]="./desktop/src-tauri/binaries"
)

repos=("$@"); [ ${#repos[@]} -eq 0 ] && repos=(Kronos Lean OpenJarvis)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for r in "${repos[@]}"; do
  echo "== $r"
  git clone --depth 1 -q "${URL[$r]}" "$tmp/$r"
  sha=$(git -C "$tmp/$r" rev-parse HEAD); date=$(git -C "$tmp/$r" log -1 --format=%cs)
  args=(--exclude=.git); [ -n "${EXCLUDE[$r]}" ] && args+=(--exclude="${EXCLUDE[$r]}")
  rm -rf "vendor/$r"; mkdir -p "vendor/$r"
  (cd "$tmp/$r" && tar "${args[@]}" -cf - .) | tar -xf - -C "vendor/$r"
  big=$(find "vendor/$r" -type f -size +50M | head -1 || true)
  [ -n "$big" ] && echo "WARNING: file >50MB: $big (GitHub limit 100MB)"
  python3 - "$r" "$sha" "$date" <<'PY'
import re, sys
r, sha, date = sys.argv[1:]
p = "vendor/VENDOR.md"; s = open(p).read()
s = re.sub(rf"(\| {r} \| [^|]+\| )[0-9a-f]{{7,40}}( \| )[0-9-]+", rf"\g<1>{sha}\g<2>{date}", s)
open(p, "w").write(s)
PY
  echo "   -> $sha ($date)"
done
echo "Done. Run: python3 -m pytest tests -q"
