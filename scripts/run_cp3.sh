#!/usr/bin/env bash
# Checkpoint 3 evaluation, end to end, from the repository root:
#
#   ANTHROPIC_API_KEY=... scripts/run_cp3.sh            # everything
#   ANTHROPIC_API_KEY=... scripts/run_cp3.sh --smoke    # 2 manifest rows, to check the setup (~$0.05)
#
# One LLM run is enough for both comparisons: with --anticheat, summary.json reports the
# fooled rate of the verdict model alone (fooled_by_category, Assumption 1) and what was
# released once the anti-cheating layer had its say (released_by_category).
set -euo pipefail

cd "$(dirname "$0")/../backend"
DATA=../data
OUT=../results
SMOKE=()
if [[ "${1:-}" == "--smoke" ]]; then SMOKE=(--limit 2); OUT=$(mktemp -d); fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is not set (add it to the cloud environment, then start a new session)" >&2
  exit 1
fi

echo "== 1. GPS out of every photo, then the manifest"
python -m bord.ingest --data "$DATA" --strip-only
python -m bord.build_manifest --data "$DATA"

echo "== 2. Anti-cheating signals without an LLM; thresholds chosen on p001-p015, scored on the rest"
python -m bord.anticheat_eval --manifest "$DATA/manifest.csv" --out "$OUT" --calibrate-through p015

echo "== 3. Claude verifier with the anti-cheating stage, under the real budget (6.5 s call, 8 s deadline)"
python -m bord.evaluate --manifest "$DATA/manifest.csv" --out "$OUT" --provider anthropic --anticheat ${SMOKE[@]+"${SMOKE[@]}"}

if [[ -n "${SMOKE[*]:-}" ]]; then
  echo "smoke run done; rerun without --smoke for the full evaluation"
  exit 0
fi

# A call cut off at 6.5 s has no verdict, so accuracy cannot be scored from it. If more than
# 1 in 10 calls timed out, measure accuracy again with a relaxed timeout (as at Checkpoint 2).
LATEST=$(ls -d "$OUT"/2026*_anticheat | tail -1)
TIMEOUTS=$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['assumptions']['2']['llm_timeouts'] / max(1, s['n_submissions']))" "$LATEST/summary.json")
if python -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0.1 else 1)" "$TIMEOUTS"; then
  echo "== 3b. $TIMEOUTS of calls timed out: relaxed run (60 s call, 70 s deadline) for accuracy"
  python -m bord.evaluate --manifest "$DATA/manifest.csv" --out "$OUT" --provider anthropic --anticheat \
    --llm-timeout 60 --deadline 70
fi

echo "== 4. Comparison with the Checkpoint 2 runs"
python -m bord.compare_runs "$OUT"/2026*/ > ../docs/results_cp3.md
cat ../docs/results_cp3.md
