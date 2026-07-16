# Monster Chess Model Development

This context names the model-strength lifecycle used to improve the Monster Chess engine without treating unsuccessful experiments as releases.

## Language

**Incumbent**:
The strongest approved numbered model and the baseline that every candidate must improve upon.
_Avoid_: Current model, latest run

**Candidate**:
A trained model being evaluated for the next unclaimed version number. A candidate is not yet a version.
_Avoid_: Version, release

**Version**:
A numbered model that has passed automated evidence and the owner gate. Each new version represents a concrete positive step over the incumbent.
_Avoid_: Run, experiment

**Promotion**:
The decision that turns a candidate into the next version and makes it the new incumbent.
_Avoid_: Rename, automatic acceptance

**Owner gate**:
The final playing-strength assessment performed by the project owner after automated evaluation. Promotion requires this approval.
_Avoid_: Optional playtest

**Rejected candidate**:
A candidate that failed automated evidence or the owner gate. Rejection preserves the target version number for the next candidate.
_Avoid_: Failed version
