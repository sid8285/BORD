| Model | Run | Honest/adv. answered | Honest accuracy [95% CI] | Fooled rate [95% CI] | Adv. released | LLM p95 (s) | Server p95 (s) | Deadline misses | Mean cost/call | A1 | A2 | Image px | Anti-cheat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-vl:8b-instruct | `20261012T011713Z_qwen3-vl-8b-instruct` | 0/0 | n/a | n/a | 0% [0%, 11%] | >= 6.50 | 6.70 | 0 | n/a | Untested | Untested | n/a | off |
| qwen3-vl:8b-instruct | `20261012T012236Z_qwen3-vl-8b-instruct` | 15/31 | 100% [80%, 100%] | 23% [11%, 40%] | 23% [11%, 40%] | 10.42 | 10.59 | 0 | $0.0000 | Weakened | Broken | n/a | off |
| claude-opus-5-5 | `20261026T022525Z_claude-opus-5-5_anticheat` | 26/78 | 100% [87%, 100%] | 9% [4%, 17%] | 4% [1%, 11%] | 4.28 | 4.49 | 0 | $0.0223 | Broken | Broken | n/a | on |

| Run | reused released | web_sourced released | ai_edited released | wrong_scene released | staged_partial released | resubmitted released | recaptured released | Honest completed released |
|---|---|---|---|---|---|---|---|---|
| `20261012T011713Z_qwen3-vl-8b-instruct` | 0/11 | n/a | 0/5 | 0/15 | n/a | n/a | n/a | 0/12 |
| `20261012T012236Z_qwen3-vl-8b-instruct` | 2/11 | n/a | 5/5 | 0/15 | n/a | n/a | n/a | 12/12 |
| `20261026T022525Z_claude-opus-5-5_anticheat` | 0/16 | 0/26 | 3/5 | 0/26 | n/a | 0/3 | 0/3 | 16/19 |
