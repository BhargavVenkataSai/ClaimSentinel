# Portfolio Polish Checklist

Before you add this to your resume, GitHub, or portfolio site, go through this checklist:

---

## Code Quality

- [ ] **Run `python code/evaluation/main.py` locally** with CLIP/BLIP weights available. Record the real metrics (replace the sandbox fallback numbers in `code/evaluation/evaluation_report.md`).
  - You should see metrics better than the fallback (0.65–0.75 range) since real CLIP is much stronger than histogram features.
  - Metrics don't need to be 1.000. Realistic is good. (e.g., 0.75–0.85 is solid.)

- [ ] **Run `python code/main.py`** to confirm `output.csv` generates correctly and validates.

- [ ] **Check for unused code**: `main_vlm.py` is sketch-only (VLM alternative path). Decide:
  - Keep it with a comment "Future: integrate VLM variant for higher accuracy" (shows forward thinking).
  - Delete it entirely if you want a cleaner submission.
  - (I'd keep it — shows you explored alternatives.)

- [ ] **Add docstrings** to key functions if missing:
  - `infer_row()`
  - `build_reference_library()`
  - `predict_from_references()`
  - (Make them short, 2–3 sentences explaining input/output + gotchas.)

- [ ] **Remove any `# TODO`, `# HACK`, or `# FIXME` comments** unless they're intentional forward-work notes.

- [ ] **Confirm no hardcoded paths or credentials** leak anywhere (no API keys, local paths like `/home/yourname`).

---

## Documentation

- [ ] **Update `README.md`** (root level) to tell the story better:
  - Add a "What Problem Does It Solve?" section.
  - Add a "Results" section (your final metrics).
  - Add a "Architecture" section (one paragraph, high-level).
  - Add a "How to Run" section (simple instructions).

- [ ] **Code-level documentation**: Each `.py` file should have a 2–3 line docstring at the top explaining what it does.
  - `main.py` — Pipeline entry point for test set inference.
  - `evaluation/main.py` — Evaluation framework for sample set.
  - etc.

- [ ] **Review `FIX_LOG.md`** — is the language clear and non-defensive? (It should be: "I found a bug, here's what it was, here's how I fixed it.")

- [ ] **Check `INTERVIEW_GUIDE.md`** — is it tailored enough to your experience? Customize the Q&A with your own examples if possible.

---

## File & Folder Hygiene

- [ ] **No `__pycache__` directories** (already cleaned, but check before uploading).

- [ ] **No `.pyc` files** (already cleaned, but check before uploading).

- [ ] **No `.DS_Store` or `Thumbs.db`** (macOS/Windows junk).

- [ ] **Images folder**: Does it need to be in the portfolio zip? Probably not.
  - Consider: Upload to GitHub without the `dataset/images/` folder, add a note "Download the dataset from [HackerRank repo] if you want to run locally."
  - Keeps the repo small, makes it clear the images aren't yours (they're the problem data).

- [ ] **`.gitignore`** if you push to GitHub:
  ```
  __pycache__/
  *.pyc
  *.egg-info/
  .DS_Store
  Thumbs.db
  dataset/images/  # Optional: exclude image data
  ```

---

## Resume Bullets

- [ ] **Write 2–3 one-liner bullets** for this project (see INTERVIEW_GUIDE.md "Resume Bullet Points").

- [ ] **Pick the strongest one** and lead with it. Example:
  - ✅ "Multimodal insurance claim validation using CLIP + rule-based reasoning; beat baseline on all metrics."
  - ❌ "Made a thing that processes images and PDFs."

- [ ] **Quantify if possible**:
  - ✅ "Improved claim classification accuracy from 55% (baseline) to 65% (final)."
  - ❌ "Made things better."

- [ ] **Mention the tech stack** briefly:
  - ✅ "Python, CLIP, FastAPI (if you add an API endpoint), pandas, numpy."
  - ❌ "Coded it up."

---

## Portfolio Website / GitHub

If you're uploading to GitHub or a portfolio site:

- [ ] **Create a clean README.md** at the root that a stranger could understand in 2 minutes.
  - Problem (1 sentence).
  - Approach (2 sentences).
  - Results (1 sentence with metrics).
  - How to run (3 bullet points).

- [ ] **Add badges** (optional but polished):
  ```markdown
  ![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue)
  ![License: MIT](https://img.shields.io/badge/License-MIT-green)
  ```

- [ ] **Create a `requirements.txt`** or `environment.yml` if it doesn't exist:
  ```
  torch>=1.12.0
  transformers>=4.20.0
  pillow>=9.0.0
  numpy>=1.21.0
  opencv-python>=4.5.0
  ```

- [ ] **Add a quick-start section**:
  ```bash
  # Install dependencies
  pip install -r requirements.txt
  
  # Run evaluation
  python code/evaluation/main.py
  
  # Generate predictions on test set
  python code/main.py
  ```

- [ ] **Link to FIX_LOG.md** from README if you want to highlight the bug discovery (makes you look thoughtful, not reckless).

---

## Interview Prep

- [ ] **Read INTERVIEW_GUIDE.md entirely** — at least twice.

- [ ] **Memorize your metrics** (from running evaluation locally):
  - Baseline claim_status accuracy: ___
  - Final claim_status accuracy: ___
  - Key differentiator metric: ___

- [ ] **Trace one example prediction**: Pick a random row from `output.csv`. Be ready to explain:
  - What images went in?
  - What were the CLIP nearest neighbors?
  - What rules fired?
  - What was the final prediction?
  - Would you trust it? Why or why not?

- [ ] **Prepare 2 failure cases** (if someone asks "what doesn't work well?"):
  - Example: "Blurry images — CLIP embeddings are lower confidence, so I fall back to conservative defaults."
  - Example: "Ambiguous text — 'new damage' could be from a prior incident, hard to parse without timestamps."

- [ ] **Decide how to frame the bug**:
  - Neutral/confident: "I discovered and fixed an evaluation leakage bug that was making sample-set metrics artificially perfect."
  - Self-aware: "I had a shortcut in my validation code that bypassed the model entirely — caught it before submission, which taught me the value of honest evaluation."
  - (Don't go full self-flagellating — it's a bug, you fixed it, you learned from it. That's good.)

- [ ] **Practice your 60-second pitch** (from INTERVIEW_GUIDE.md). Say it out loud 5 times until it sounds natural, not rehearsed.

---

## Before You Call It Done

Final sanity checks:

- [ ] Can someone clone/download this repo and run `python code/evaluation/main.py` without Googling?
- [ ] If they look at the git history (if it's on GitHub), does it tell a coherent story or is it a mess of "fix" commits?
- [ ] Can you explain every line of the main pipeline in 2 minutes without notes?
- [ ] Would you be comfortable with this code being reviewed in a job interview?
- [ ] Is there anything in here you'd be embarrassed to have a senior engineer see? (If yes, fix it.)

---

## Specific Things to Update in the Zip

Before you finalize, regenerate these files locally:

1. **`code/evaluation/evaluation_report.md`** — Re-run with CLIP weights, update the metrics.
2. **Screenshot of `output.csv` (first 10 rows)** — Add to README for quick visual reference (optional but looks polished).
3. **Update `README.md`** — Add the structured sections (problem, approach, results, how-to-run).

---

## Optional: Go Further (If You Have Time)

- [ ] **Add a Colab notebook** that visualizes CLIP matches for a sample prediction (shows your work, makes it interactive).
- [ ] **Create a simple Flask/FastAPI endpoint** that accepts a claim JSON and returns predictions (turns code into a service, looks more professional).
- [ ] **Add unit tests** for core functions (e.g., test `infer_row` with known inputs, assert schema is valid).
- [ ] **Write a blog post** about the project (how you approached it, what you learned, the bug discovery) and link it from README.
  - This is *chef's kiss* for standing out. Shows you can communicate.

---

## Submission Timeline

**Immediate (today):**
- [ ] Run evaluation locally, update metrics in `evaluation_report.md`.
- [ ] Update `README.md` with story/results/how-to-run.
- [ ] Re-read `FIX_LOG.md` and `INTERVIEW_GUIDE.md`, make sure you're comfortable with the narrative.

**This week:**
- [ ] Push to GitHub (with `.gitignore`, clean commit history).
- [ ] Add 2–3 resume bullets.
- [ ] Practice the 60-second pitch.

**Before interviews:**
- [ ] Trace one full example prediction end-to-end.
- [ ] Prepare 2 failure cases / limitations.
- [ ] Decide: will you mention the bug proactively or only if asked?
  - (I'd mention it in "challenges" but not lead with it. It's a strength, not a weakness.)

---

## Remember

This project went from "evaluation metrics that don't reflect reality" to "honest, solid system with a credible bug-fix narrative." That's the actual valuable thing — not the metrics themselves, but showing you can:
- **Think critically** about your own work.
- **Find and own mistakes**.
- **Fix them properly** (not with bandages, but with real logic changes).
- **Document the journey** so someone else (or an interviewer) can understand what happened.

That's professional engineering. Own that narrative.
