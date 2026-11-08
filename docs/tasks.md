# Tasks, wagers and deadlines (Checkpoint 4)

A task carries a wager held through Stripe test mode, a deadline on the server clock and a type.

| Type | What the user does | How it is verified |
|---|---|---|
| `single_scene` | one before and one after photo | the Checkpoint 3 pipeline |
| `multi_view` | the server assigns 2 to 4 distinct angles; each angle gets its own before and after photo | one pipeline run per angle, side by side; `combine_decisions` releases only if all release, forfeits only if all forfeit, otherwise holds; the after photos must differ from each other |
| `time_spread` | a session of `session_s` seconds with `n_checkins` random check-ins; each is photographed inside a 120 s window | every check-in is stamped, in its window, shows the session's scene and is not a copy of another; then the before/after pipeline |

Money: `POST /api/tasks` authorizes first (`capture_method=manual`); release cancels the hold, forfeit captures it and records a simulated charity payout, hold leaves the authorization in place. Set `STRIPE_SECRET_KEY=sk_test_...` for Stripe test mode; without it `SimulatedHolds` runs the same flow with no network. Live keys are refused. The ledger is appended to `data/ledger.jsonl` (`BORD_LEDGER`).

Deadlines: a submission that starts after the deadline is refused and the wager is forfeited; `POST /api/tasks/sweep` (also run on every task read) forfeits tasks nobody submitted. Deadlines are capped at 7 days, the length of a card authorization.

Check-in times are never sent to the browser ahead of time; the task view only says which check-in is due now.
