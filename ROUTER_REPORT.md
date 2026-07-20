
# ROUTER VALIDATION — started 2026-07-19 23:53:58

[07-19 23:53:58] START router_vs_v17
[07-20 00:59:57] SCORE router_vs_v17 (66.0 min): {'overall': 0.325, 'white': 0.35, 'black': 0.3}
[07-20 00:59:57] START router_vs_ramp
[07-20 01:38:12] SCORE router_vs_ramp (38.3 min): {'overall': 0.525, 'white': 0.8, 'black': 0.25}
[07-20 01:38:12] START router_anchor
[07-20 01:59:07] SCORE router_anchor (20.9 min): {'overall': 0.7, 'white': 0.9, 'black': 0.5}
[07-20 01:59:07] SUMMARY: vs_v17={'overall': 0.325, 'white': 0.35, 'black': 0.3} vs_ramp={'overall': 0.525, 'white': 0.8, 'black': 0.25} anchor={'overall': 0.7, 'white': 0.9, 'black': 0.5}
[07-20 01:59:07] VERDICT: FAIL — composite does not beat both parents; do not spend owner board time on it
[07-20 01:59:07] ALL STEPS COMPLETE

> Segment below = re-validation of the SIDE-AWARE spec (black_model=late, commit 2455aa1). Segment above = rejected phase-both-sides design.


# ROUTER VALIDATION — started 2026-07-20 04:06:09

[07-20 04:06:09] START router_vs_v17
[07-20 05:34:29] SCORE router_vs_v17 (88.3 min): {'overall': 0.575, 'white': 0.35, 'black': 0.8}
[07-20 05:34:29] START router_vs_ramp
[07-20 06:18:25] SCORE router_vs_ramp (43.9 min): {'overall': 0.5, 'white': 0.8, 'black': 0.2}
[07-20 06:18:25] START router_anchor
[07-20 06:39:26] SCORE router_anchor (21.0 min): {'overall': 0.75, 'white': 0.9, 'black': 0.6}
[07-20 06:39:26] SUMMARY: vs_v17={'overall': 0.575, 'white': 0.35, 'black': 0.8} vs_ramp={'overall': 0.5, 'white': 0.8, 'black': 0.2} anchor={'overall': 0.75, 'white': 0.9, 'black': 0.6}
[07-20 06:39:26] VERDICT: PASS — router inherits both strengths; ready for owner sessions (dropdown: router/router_v17_ramp)
[07-20 06:39:26] ALL STEPS COMPLETE
