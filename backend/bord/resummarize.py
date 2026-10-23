"""Rewrite a run's summary.json from its saved predictions, without calling the model.

Use it after a change to how summaries are computed (Checkpoint 3: statuses in the
checkpoint template's words), so older runs read the same way as new ones::

    python -m bord.resummarize ../results/20261012T012236Z_qwen3-vl-8b-instruct

The ``run`` block (model, host, warm-up, settings) is kept as it was; everything else is
recomputed from ``predictions.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import RT_SERVER_DEADLINE_S, summarize


def resummarize(run_dir: Path) -> dict:
    records = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    summary_path = run_dir / "summary.json"
    run_meta = json.loads(summary_path.read_text(encoding="utf-8")).get("run", {}) if summary_path.exists() else {}
    deadline = run_meta.get("server_deadline_s", RT_SERVER_DEADLINE_S)
    summary = {"run": run_meta, **summarize(records, server_deadline_s=deadline)}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute summary.json from predictions.jsonl.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        a = resummarize(run_dir)["assumptions"]
        print(f"{run_dir.name}: A1 {a['1']['status']}, A2 {a['2']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
