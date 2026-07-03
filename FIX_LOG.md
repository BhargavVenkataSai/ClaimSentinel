# Fix Log — Evaluation Leakage

## What was wrong

`code/main.py`'s `infer_row()` had a shortcut: for any row whose `image_paths`
contained the string `"sample"`, it skipped the model entirely and returned the
gold-label row from `dataset/sample_claims.csv` (matched by `user_id`). A second,
now-unreachable block also hardcoded the exact output for four specific sample
cases (`user_002`/`case_002`, `user_008`/`case_008`, `user_033`/`case_019`,
`user_034`/`case_020`).

Because `code/evaluation/main.py` evaluates the "final strategy" by running the
pipeline on `dataset/sample_claims.csv`, this meant the model's real predictions
never appeared in the evaluation — every metric showed a perfect `1.000`, which
should always be a red flag rather than something to be proud of.

`code/README.md` separately and incorrectly claimed "It does not hardcode
per-test-row outputs."

A second, milder issue: the CLIP reference-image library used for similarity
retrieval was built from `sample_claims.csv`'s supporting images, and during
evaluation each sample row could match against its own image in that library —
a mild form of evaluating on the training signal.

## What did NOT need fixing

`output.csv`, the actual submitted predictions for `dataset/claims.csv`, was
never affected. `claims.csv` paths never contain the string `"sample"`, so the
leak path was never triggered for the graded test set. Every prediction in
`output.csv` came from the real CLIP+BLIP pipeline.

## What changed

1. Removed the `"sample" in image_paths` shortcut in `infer_row`.
2. Removed the four hardcoded per-case override blocks.
3. Added leave-one-out exclusion: `predict_from_references` now takes
   `exclude_paths` and `build_reference_library`/`ReferenceImage` track each
   reference's source path, so a row never matches against its own image when
   the reference library is scored on the sample set.
4. Corrected `code/README.md`'s claim and added a note explaining the fix.
5. Removed `__pycache__` from the repo.
6. Regenerated `evaluation_report.md` with honest metrics, with a note that the
   numbers shown were produced via the local fallback feature extractor (this
   environment didn't have the project's CLIP/BLIP weights cached) — re-run
   `python code/evaluation/main.py` with those weights available for the true
   numbers, which should be noticeably better than what's shown here.

## Why this matters for a resume project

A judge, recruiter, or interviewer who reads the code will eventually find a
block that says "if this is a sample row, return the answer key." That's the
kind of thing that turns "I built a multimodal evidence-review pipeline using
CLIP and BLIP" into a credibility problem in a five-minute code walkthrough.
The fixed version is slower to brag about — the real numbers aren't 1.000 — but
they're numbers you can actually defend, which is the entire point of the
evaluation section.
