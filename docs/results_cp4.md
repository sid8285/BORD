# Checkpoint 4 results

Every number below is read from a committed `results/` folder. Rates carry 95% Wilson intervals; a category with fewer than 10 attempts is given as counts only.

## Staged-partial attacks (Assumption 7)

20 attacks (3 from Checkpoint 3, 17 new: a before photo, a half-done after framed to look finished, and the honest finished after) and their 20 honest pairs, `data/manifest_staged_cp4.csv`. Claude Opus 5.5 with the anti-cheating stage under the real 6.5 s LLM budget.

| Run | Staged released | Fooled (verdict "completed") | Honest completed not released | LLM p95 | Timeouts | Cost/call |
|---|---|---|---|---|---|---|
| Default prompt, `20261109T002513Z_claude-opus-5-5_anticheat` | 6/20 = 30% [15%, 52%] | 8/20 | 1/20 (1 hold) | 3.79 s | 0 | $0.0231 |
| Per-region prompt, `20261109T002754Z_claude-opus-5-5_anticheat_regions` | 3/20 = 15% [5%, 36%] | 4/20 | 6/20 = 30% [15%, 52%] (4 timeouts, 1 hold, 1 forfeit) | 6.58 s | 4 of 40 | $0.0280 |

The per-region prompt halves the staged-partial releases, but it writes a region list before its verdict, so it is about 2.8 s slower and 4 of 40 calls hit the 6.5 s timeout, each of which holds an honest user. Assumption 7: the catch target (at least 50%) is met at 85%; the false-hold target (at most 10%) is missed, with the whole interval above it.

Pixel check (`bord.regions`), `regions_20261108T224230Z`: caught 13/20 = 65% [43%, 82%], flagged 20/36 honest completed pairs = 56% [40%, 70%], 0.16 s at p95. No single feature separates the two groups (best single threshold, on the clutter ratio: 20/20 caught with 23/36 honest flagged). It stays a diagnostic signal.

## Multi-view and time-spread pilots (Assumption 6)

5 honest, completed tasks of each type, `data/tasks.jsonl`.

| Run | Multi-view held | Time-spread held |
|---|---|---|
| Checks only, `tasks_20261109T003143Z_none` | 1/5 | 3/5 |
| With Claude Opus 5.5, `tasks_20261109T003152Z_claude-opus-5-5` | 1/5 | 4/5 (1 of them forfeited) |

Why each honest task was held: mv05, one angle flagged by the scene check; ts04 and ts05, a check-in taken before its window opened (the pilot followed a stopwatch, while the app shows which check-in is due now); ts02, a close-up check-in scored 0.36 on scene similarity (threshold 0.45); ts03, Claude forfeited a coding session because the before and after photos show no visible change. A session task has no state change to see, so time-spread needs its own verdict question (was the person there, doing the activity, at each check-in) rather than the before/after one.

## Stripe test mode (Assumption 5)

`stripe_20261108T202720Z`: two holds authorized with manual capture; one cancelled, one captured, and a capture of the cancelled hold refused by Stripe. `capture_before` is 7.0 days after authorization. Calls: authorize 1.13 and 1.02 s (at task creation, off the verdict path), cancel 0.27 s, capture 0.72 s, over the 0.5 s decision-and-Stripe stage.

## Photos

`photo_check_20261109T003413Z`: all 157 photos in `data/` (pairs, attacks, web images and the task pilots), none with a GPS location.
