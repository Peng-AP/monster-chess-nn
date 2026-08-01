# Artifact transfer: CPU box → GPU box, via GitHub

**Written 2026-08-01 on the GPU box.** The repo's *code* is synced through
GitHub, but the gitignored artifacts are not, and every experiment queued for
the GPU box is blocked on them. This file is the directive for the **old CPU
box** (`C:\Users\AaronPeng\Desktop\monster-chess-nn`): ship the artifacts on a
throwaway branch, then this file's second half runs back on the GPU box.

Delete this file (git rm, one-line commit) once the transfer is verified —
root holds only active work.

---

## What to ship (and what not to)

| Include | Why |
|---|---|
| `models/fresh_start_v17/` | the incumbent — no gate runs without it |
| `models/rejected/` (all 15) | `v18_ramp` is the sparring partner / 2nd gate opponent; the rest are the evidence base |
| `data/raw/combined_v17/` | the incumbent training corpus |
| `data/raw/combined_v17_dup1/` | concluded control, small, keeps §4.1 reproducible |
| `data/raw/human_games/white_2026_07/`, `black_2026_07/`, `curriculum_2026_07/` | **irreplaceable** owner games, incl. `game_00013.correction.md` |
| `data/raw/ps_monster/` | 829 PlayStrategy games, 43,939 records |
| `data/playstrategy/` | census + bundle + manifest — regenerable but a re-crawl is an hour of rate-limited API traffic |

**Do NOT ship `data/processed/`** — 13.4 GB, every dataset regenerable from
raw in ~20 min. Rebuild on the GPU box instead.

## On the CPU box

```bash
cd /c/Users/AaronPeng/Desktop/monster-chess-nn
git fetch origin && git status          # expect clean, main == origin/main at 6586e7a or later

# guard: GitHub rejects files >100MB. Expect zero hits; if any, STOP and report.
find models/fresh_start_v17 models/rejected data/raw/combined_v17 \
     data/raw/combined_v17_dup1 data/raw/ps_monster data/playstrategy \
     data/raw/human_games/white_2026_07 data/raw/human_games/black_2026_07 \
     data/raw/human_games/curriculum_2026_07 -type f -size +95M

git switch -c transfer/artifacts-2026-08

# two commits so each push stays under GitHub's ~2GB pack limit
git add -f models/fresh_start_v17 models/rejected
git commit -m "transfer: models"        # owner identity, no co-author trailer
git push -u origin transfer/artifacts-2026-08

git add -f data/raw/combined_v17 data/raw/combined_v17_dup1 \
           data/raw/ps_monster data/playstrategy \
           data/raw/human_games/white_2026_07 \
           data/raw/human_games/black_2026_07 \
           data/raw/human_games/curriculum_2026_07
git commit -m "transfer: raw data"
git push

git switch main                          # leave main untouched
```

If a push is rejected for size anyway, split the offending commit by
subdirectory and push after each commit — each push packs only what's new.

## Back on the GPU box

```bash
git fetch origin transfer/artifacts-2026-08

# writes the files into the worktree; they are gitignored on main, so git
# status stays clean and nothing lands on main. Byte-identity is guaranteed
# by git's content addressing — no separate checksum manifest needed.
git restore --source=origin/transfer/artifacts-2026-08 --worktree -- \
    models/fresh_start_v17 models/rejected \
    data/raw/combined_v17 data/raw/combined_v17_dup1 data/raw/ps_monster \
    data/playstrategy data/raw/human_games
```

Verify against HANDOFF §5.3/§6 before declaring done:
- `data/raw/ps_monster`: 829 games, 43,939 records
- `models/fresh_start_v17` loads and plays a move; `models/rejected/fresh_start_v18_ramp` present
- `data/raw/human_games/white_2026_07` has 15 games incl. `game_00013.correction.md`

Then clean up and rebuild:

```bash
git push origin --delete transfer/artifacts-2026-08
git branch -D transfer/artifacts-2026-08 2>/dev/null; git fetch --prune
# rebuild the processed dataset (~20 min): see README for the data_processor
# invocation that produced data/processed/combined_v17_r50h60 (ramp labels,
# 15ch, seed 42)
```

Note: the transfer blobs stay in GitHub's object store until its GC runs even
after the branch is deleted. Harmless; the repo's clone size returns to normal
for fresh clones once GC collects them.
