| Model | Run | Honest/adv. answered | Honest accuracy [95% CI] | Fooled rate [95% CI] | Adv. released | LLM p95 (s) | Server p95 (s) | Deadline misses | Mean cost/call | A1 | A2 | Image px | Anti-cheat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-vl:8b-instruct | `20261012T011713Z_qwen3-vl-8b-instruct` | 0/0 | n/a | n/a | 0% [0%, 11%] | >= 6.50 | 6.70 | 0 | n/a | Untested | Untested | n/a | off |
| qwen3-vl:8b-instruct | `20261012T012236Z_qwen3-vl-8b-instruct` | 15/31 | 100% [80%, 100%] | 23% [11%, 40%] | 23% [11%, 40%] | 10.42 | 10.59 | 0 | $0.0000 | Weakened | Broken | n/a | off |

| Run | reused released | web_sourced released | ai_edited released | wrong_scene released | staged_partial released | Honest completed released |
|---|---|---|---|---|---|---|
| `20261012T011713Z_qwen3-vl-8b-instruct` | 0/11 | n/a | 0/5 | 0/15 | n/a | 0/12 |
| `20261012T012236Z_qwen3-vl-8b-instruct` | 2/11 | n/a | 5/5 | 0/15 | n/a | 12/12 |
