| Model | Run | Honest/adv. answered | Honest accuracy [95% CI] | Fooled rate [95% CI] | Adv. released | LLM p95 (s) | Server p95 (s) | Deadline misses | Mean cost/call | A1 | A2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-vl:8b-instruct | `20261012T011713Z_qwen3-vl-8b-instruct` | 0/0 | n/a | n/a | 0% [0%, 11%] | >= 6.50 | 6.70 | 0 | n/a | untested | untested |
| qwen3-vl:8b-instruct | `20261012T012236Z_qwen3-vl-8b-instruct` | 15/31 | 100% [80%, 100%] | 23% [11%, 40%] | 23% [11%, 40%] | 10.42 | 10.59 | 0 | $0.0000 | broken | broken |
