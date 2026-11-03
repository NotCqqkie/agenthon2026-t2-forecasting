#!/usr/bin/env bash
# Run one Track 2 public unit through an image under the platform's container settings.
# usage: scripts/run_unit.sh IMAGE UNIT_DIR OUT_DIR
set -euo pipefail
img=$1; u=$2; o=$3
st=$(mktemp -d); chmod 755 "$st"; mkdir -p "$st/panels" "$o"; chmod 777 "$o"
cp "$u"/*.toml "$u"/*.json "$u"/*.md "$st/" 2>/dev/null || true
cp -r "$u/text" "$st/" 2>/dev/null || true
cp "$u"/*.parquet "$st/panels/"
asof=$(python3 -c "import tomllib,sys;print(tomllib.load(open(sys.argv[1],'rb'))['provenance']['data_cutoff'])" "$st/card.toml")
docker run --rm --read-only --user 65534:65534 --cap-drop=ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m --pids-limit 256 --ulimit nofile=1024:1024 --ulimit nproc=256:256 \
  --cpus 2 --memory 4g --memory-swap 4g --network=none -v "$st":/input:ro -v "$(cd "$o" && pwd)":/output \
  -e QFBENCH_SEED=0 -e QFBENCH_NETWORK=none "$img" forecast --panels /input/panels/ --text /input/text/ --asof "$asof" \
  --out /output/forecast.parquet >/dev/null
rm -rf "$st"
