# Dataset spec (Checkpoint 1)

## 1. Self-collected honest set
- Target: **30 before/after pairs**, collected by Sid over the semester (15 by Checkpoint 2, 30 by Checkpoint 5).
- Captured through the BORD browser flow once it exists; earlier pairs are phone photos, labeled as such.
- Each pair gets a label (`completed` / `not_completed`), task type, task description, and capture timestamps.

## 2. Task types
| Type | Example | Evidence |
|---|---|---|
| Single-scene state change | Clean my room | One before photo and one after photo of the same scene |
| Multi-view | Organize the garage | Several prompted angles, before and after |
| Time-spread | Study for 2 hours | Check-in photos at random prompts during the session |

## 3. Adversarial set
Each honest pair is the base for cheating attempts. Each attempt is labeled with its category:
- **Reused:** an earlier photo of the user's own clean room, submitted again.
- **Web-sourced:** a stock or web image of a clean room.
- **AI-edited:** the real "before" photo edited by an image model to look clean.
- **Wrong scene:** an after photo of a different, already-clean room.
- **Staged partial:** only the photographed area is cleaned.

## 4. Public set
- [Messy vs Clean Room (Kaggle)](https://www.kaggle.com/datasets/cdawn1/messy-vs-clean-room): unpaired messy/clean room images, used as web-sourced adversarial material and for sanity checks.

## 5. Metrics recorded per submission
Verdict, confidence score, individual signal scores, latency (browser and server), API cost.
