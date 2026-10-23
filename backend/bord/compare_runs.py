"""Print Markdown tables comparing evaluation runs (for the checkpoint report).

Usage (from backend/)::

    python -m bord.compare_runs ../results/*/

The first table has one row per run (accuracy, fooled rate, latency, cost, assumption
statuses, image size and whether the anti-cheating stage was on). The second table has
one row per run and one column per attack category: how often a wager would have been
released for that kind of cheating attempt (k/n). Put the Checkpoint 2 baseline run and
a Checkpoint 3 ``--anticheat`` run on the same manifest side by side to read the effect
of the new layer directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .dataset import ADVERSARIAL_CATEGORIES


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
    run = summary["run"]
    return "| " + " | ".join([
        run.get("model", "?"),
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
        str(run.get("max_image_side", "n/a")),
        "on" if run.get("anticheat") else "off",
    ]) + " |"


def category_row(summary: dict, run_dir: Path) -> str:
    by_cat = summary.get("released_by_category") or {}
    cells = []
    for cat in ADVERSARIAL_CATEGORIES:
        p = by_cat.get(cat) or {}
        cells.append(f"{p['k']}/{p['n']}" if p.get("n") else "n/a")
    honest = summary.get("honest_completed_released") or {}
    honest_cell = f"{honest['k']}/{honest['n']}" if honest.get("n") else "n/a"
    return "| " + " | ".join([f"`{run_dir.name}`", *cells, honest_cell]) + " |"


HEADER = ("| Model | Run | Honest/adv. answered | Honest accuracy [95% CI] | Fooled rate [95% CI] "
          "| Adv. released | LLM p95 (s) | Server p95 (s) | Deadline misses | Mean cost/call "
          "| A1 | A2 | Image px | Anti-cheat |\n|" + "---|" * 14)

CATEGORY_HEADER = ("| Run | " + " | ".join(f"{c} released" for c in ADVERSARIAL_CATEGORIES)
                   + " | Honest completed released |\n|" + "---|" * (len(ADVERSARIAL_CATEGORIES) + 2))


def main(argv: list[str] | None = None) -> int:
    dirs = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    summaries = []
    for d in dirs:
        f = d / "summary.json"
        if f.is_file():
            summaries.append((d, json.loads(f.read_text(encoding="utf-8"))))
    print(HEADER)
    for d, s in summaries:
        print(row(s, d))
    with_categories = [(d, s) for d, s in summaries if s.get("released_by_category")]
    if with_categories:
        print()
        print(CATEGORY_HEADER)
        for d, s in with_categories:
            print(category_row(s, d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
