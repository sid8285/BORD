"""Print a Markdown table comparing evaluation runs (for the checkpoint report).

Usage (from backend/)::

    python -m bord.compare_runs ../results/*/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _pct(rate, ci=None) -> str:
    if rate is None:
        return "n/a"
    text = f"{rate:.0%}"
    return f"{text} [{ci[0]:.0%}, {ci[1]:.0%}]" if ci else text


def _s(x, fmt="{:.2f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def _llm_p95(a2: dict, stages: dict) -> str:
    if a2.get("p95_llm_latency_at_least_s") is not None:  # timeouts: only a lower bound is known
        return f">= {a2['p95_llm_latency_at_least_s']:.2f}"
    if "p95_llm_latency_s" in a2:
        return _s(a2["p95_llm_latency_s"])
    return _s(stages.get("llm_s", {}).get("p95"))


def row(summary: dict, run_dir: Path) -> str:
    a = summary["assumptions"]
    rt = summary.get("realtime", {})
    stages = summary.get("stage_latency_s", {})
    released = summary.get("adversarial_released", {})
    return "| " + " | ".join([
        summary["run"].get("model", "?"),
        f"`{run_dir.name}`",
        f"{summary['n_honest']}/{summary['n_adversarial']}",
        _pct(summary["honest_accuracy"], summary.get("honest_accuracy_ci95")),
        _pct(summary["fooled_rate"], summary.get("fooled_rate_ci95")),
        _pct(released.get("rate"), released.get("ci95")),
        _llm_p95(a["2"], stages),
        _s(summary["latency_s"].get("p95")),
        f"{rt.get('deadline_misses', 'n/a')}",
        _s(summary["cost_usd"]["mean_per_call"], "${:.4f}"),
        a["1"]["status"],
        a["2"]["status"],
    ]) + " |"


HEADER = ("| Model | Run | Honest/adv. answered | Honest accuracy [95% CI] | Fooled rate [95% CI] "
          "| Adv. released | LLM p95 (s) | Server p95 (s) | Deadline misses | Mean cost/call "
          "| A1 | A2 |\n|" + "---|" * 12)


def main(argv: list[str] | None = None) -> int:
    dirs = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    print(HEADER)
    for d in dirs:
        f = d / "summary.json"
        if f.is_file():
            print(row(json.loads(f.read_text(encoding="utf-8")), d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
