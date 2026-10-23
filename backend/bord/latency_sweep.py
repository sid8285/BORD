"""Re-measure LLM latency across model size, image size and hosting (Checkpoint 3, Assumption 2).

At Checkpoint 2 Qwen3-VL 8B on the Apple M4 Pro needed 10.42 s at p95 for one
two-photo call, over the 6.5 s LLM stage budget. The levers are fewer image
tokens (smaller photos), a smaller model, and faster hardware (a hosted API or
a GPU server). This runs the same manifest once per configuration with a relaxed
timeout, so every call completes and the latency is a measurement rather than a
cap, then prints the comparison table and which configurations fit the stage.

A configuration is ``model@max_image_side`` or ``model@max_image_side@provider``::

    python -m bord.latency_sweep --manifest ../data/manifest.csv --out ../results \\
        qwen3-vl:8b-instruct@1568 qwen3-vl:8b-instruct@768 \\
        qwen3-vl:4b-instruct@1568 qwen3-vl:4b-instruct@768 \\
        qwen3-vl-plus@768@dashscope          # needs DASHSCOPE_API_KEY

Local models must be pulled first (``ollama pull qwen3-vl:4b-instruct``). Add
``--anticheat`` to include the new stage in the timings, ``--limit N`` for a quick pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import compare_runs, evaluate
from .realtime import Budget

DEFAULT_CONFIGS = (
    "qwen3-vl:8b-instruct@1568",
    "qwen3-vl:8b-instruct@768",
    "qwen3-vl:4b-instruct@1568",
    "qwen3-vl:4b-instruct@768",
)


def parse_config(spec: str) -> dict:
    parts = spec.split("@")
    if len(parts) not in (2, 3) or not parts[1].isdigit():
        raise ValueError(f"configuration {spec!r}: expected model@max_image_side[@provider]")
    return {"model": parts[0], "side": int(parts[1]), "provider": parts[2] if len(parts) == 3 else "ollama"}


def evaluate_args(cfg: dict, manifest: Path, out: Path, timeout_s: float, anticheat: bool, limit: int | None) -> list[str]:
    argv = ["--manifest", str(manifest), "--out", str(out), "--provider", cfg["provider"],
            "--model", cfg["model"], "--max-image-side", str(cfg["side"]),
            # Relaxed timeout and deadline: every call completes, so latency is measured, not capped.
            "--llm-timeout", str(timeout_s), "--deadline", str(timeout_s + 10.0)]
    if anticheat:
        argv.append("--anticheat")
    if limit:
        argv += ["--limit", str(limit)]
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-measure verdict latency across model, image size and host.")
    parser.add_argument("configs", nargs="*", help="model@max_image_side[@provider]; default: 8B and 4B at 1568 and 768 px")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    parser.add_argument("--timeout", default=60.0, type=float, help="relaxed per-call timeout, seconds")
    parser.add_argument("--anticheat", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    configs = [parse_config(c) for c in (args.configs or DEFAULT_CONFIGS)]
    stage_budget = Budget().llm_s
    run_dirs: list[Path] = []
    fits: list[tuple[str, float | None]] = []
    for cfg in configs:
        before = set(args.out.iterdir()) if args.out.is_dir() else set()
        label = f"{cfg['model']} @ {cfg['side']} px ({cfg['provider']})"
        print(f"=== {label}", file=sys.stderr)
        code = evaluate.main(evaluate_args(cfg, args.manifest, args.out, args.timeout, args.anticheat, args.limit))
        if code != 0:
            return code
        new = sorted(set(args.out.iterdir()) - before)
        for d in new:
            run_dirs.append(d)
            summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            fits.append((label, summary["stage_latency_s"].get("llm_s", {}).get("p95")))

    print()
    compare_runs.main([str(d) for d in run_dirs])
    print(f"\nLLM stage budget: {stage_budget:.1f} s at p95")
    for label, p95 in fits:
        verdict = "n/a" if p95 is None else ("fits" if p95 <= stage_budget else "too slow")
        print(f"- {label}: p95 {'n/a' if p95 is None else f'{p95:.2f} s'} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
