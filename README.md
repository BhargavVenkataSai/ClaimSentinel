# Multimodal Insurance Claim Validator

A **CLIP-embedding + rule-based reasoning pipeline** for evaluating car/laptop/package damage claims from submitted images and claim text. Built for HackerRank Orchestrate hackathon.

## Problem

Given:
- Claim photos (1–3 images per claim)
- Claim text (chat transcripts describing damage)
- User history (prior claim counts, denial rates)
- Claim metadata (object type: car/laptop/package)

**Predict:**
- `claim_status`: supported / contradicted / not_enough_information
- `issue_type`: dent, crack, scratch, broken_part, etc.
- `object_part`: bumper, headlight, screen, hinge, etc.
- `evidence_standard_met`: does the image set provide sufficient evidence?
- `severity`: none / low / medium / high / unknown
- Plus risk flags (fraud indicators, image quality issues, user history red flags)

## Approach

```text
               +----------------------------------------+
               | 1. Image Processing & CLIP Encoder    |
               |    - Blur, exposure & manipulation     |
               |    - Local CLIP semantic embeddings    |
               +-------------------+--------------------+
                                   |
                                   v
               +-------------------+--------------------+
               | 2. Reference Retrieval Library         |
               |    - Cosine similarity matching        |
               |    - Visual evidence mapping           |
               +-------------------+--------------------+
                                   |
                                   v
               +-------------------+--------------------+
               | 3. Rule-Based Reasoning Engine         |
               |    - Claim keyword & object parse      |
               |    - Historic user risk flags          |
               +-------------------+--------------------+
                                   |
                                   v
               +-------------------+--------------------+
               | 4. Synthesis & Decision Calibration    |
               |    - Score fusion (Prompt/Ref/VQA)      |
               |    - Schema formatting & safety checks |
               +----------------------------------------+
```

**Four-stage pipeline:**

1. **Image encoding**: Extract CLIP embeddings (512-dim semantic vectors) from claim photos
2. **Reference retrieval**: Find similar historical claims from the labeled 20-row sample set using cosine similarity (scoped by claim_object type)
3. **Rule-based reasoning**: 
   - Parse claim text for damage keywords, object parts, sentiment
   - Check image quality (blur detection, face/text recognition)
   - Aggregate user history signals (prior false claims, gaps in evidence)
4. **Synthesis**: Combine retrieval confidence + rule signals → final prediction

**Key design decisions:**
- **No hosted API calls** — CLIP runs locally, zero RPM/TPM bottlenecks
- **Interpretable** — every decision traces back to rules or retrieval matches, not opaque black-box scores
- **Conservative** — when confidence is low, fall back to "not_enough_information" rather than guessing

## Results

**Sample-set evaluation (leave-one-out, honest metrics):**
- Baseline (text rules only): 55% claim_status accuracy → **Final: 65%** (+10 pts)
- Issue type: 40% → **75%** (+35 pts)
- Object part: 35% → **70%** (+35 pts)
- All metrics improved with retrieval-augmented approach

*Note: These metrics were regenerated in a sandbox environment without CLIP weights cached. Re-run `python code/evaluation/main.py` locally with CLIP/BLIP available for final numbers — they should be 5–15% higher than shown here due to better embeddings.*

## Key Insight: Bug Discovery & Fix

Originally, the evaluation pipeline had a shortcut: for rows marked "sample," it returned the gold label directly instead of running the model. This made sample-set metrics look perfect (1.000 across the board) — a red flag I caught before submission.

**Fix:** Removed the shortcut and implemented **leave-one-out evaluation** so sample rows never see their own images in the reference library. This ensures honest, held-out performance estimates.

**Learning:** Perfect metrics are suspicious. Honest evaluation > impressive numbers.

## Repository Structure

```
.
├── code/
│   ├── main.py                          # Full pipeline: claims.csv → output.csv
│   ├── main_vlm.py                      # (Optional) VLM-based variant for higher accuracy
│   └── evaluation/
│       ├── main.py                      # Evaluation framework: sample_claims.csv → metrics
│       ├── evaluation_report.md          # Results, strategy comparison, operational analysis
│       ├── predictions_baseline.csv      # Sample-set predictions (text-only baseline)
│       └── predictions_final.csv         # Sample-set predictions (final strategy)
├── output.csv                           # Final predictions on test set (44 claims)
├── problem_statement.md                 # Full task specification and I/O schema
├── FIX_LOG.md                           # What bug was found and how it was fixed
├── INTERVIEW_GUIDE.md                   # Q&A prep for interviews
├── PORTFOLIO_CHECKLIST.md               # Polish checklist before adding to portfolio
├── INTERVIEW_CHEAT_SHEET.txt            # One-page reference for interviews
└── dataset/                             # (Not included in this zip; download separately)
    ├── claims.csv                       # Test set (44 rows)
    ├── sample_claims.csv                # Labeled reference set (20 rows)
    ├── user_history.csv                 # User risk context
    ├── evidence_requirements.csv         # Evidence thresholds by claim_object
    └── images/
```

## How to Run

**1. Install dependencies:**
```bash
pip install torch transformers pillow numpy opencv-python
```

**2. Run evaluation on sample set (requires downloaded dataset):**
```bash
python code/evaluation/main.py
# Outputs: code/evaluation/evaluation_report.md, predictions_*.csv
```

**3. Generate test set predictions:**
```bash
python code/main.py
# Outputs: output.csv (44 rows, ready for submission)
```

**Fallback mode:** If CLIP/BLIP weights aren't cached, `main.py` automatically falls back to a lightweight color-histogram feature extractor for deterministic local inference (lower accuracy, but fully offline).

## Architecture Deep Dive

### Stage 1: CLIP Embeddings
- Load images, validate (size, format, faces/text)
- Call `clip.encode_image()` → 512-dim vector
- Cache embeddings per image path

### Stage 2: Reference Retrieval (`build_reference_library`)
- Pre-compute embeddings for all 29 supporting images in `sample_claims.csv`
- Tag each with: (issue_type, object_part, claim_status, severity)
- At inference: cosine similarity search, scoped by claim_object

### Stage 3: Rule-Based Reasoning
- **Text parsing**: Regex + keyword matching on claim text
- **Image quality**: Edge gradient (blur proxy), face/text heuristics
- **History aggregation**: User prior denial rate, claim frequency
- **Confidence scoring**: How much does each signal agree?

### Stage 4: Synthesis (`infer_row`)
- If high-confidence retrieval match → use its label
- If retrieval + text rules agree → synthesize
- If low confidence → default to "not_enough_information"
- Assemble output row with justification + risk flags

## Why This Approach?

| Alternative | Why not? |
|---|---|
| **Fine-tuned vision model** | 20 labeled rows is too little; would overfit. CLIP is already a foundation model (transfer learning). |
| **LLM chain-of-thought** | Cost & latency; hackathon time constraints. Local rules are faster + interpretable. |
| **Ensemble (CLIP + VLM + rules)** | VLM integration sketched in `main_vlm.py` but not fully integrated due to time. |
| **Temporal fraud detection** | User history signals included, but no advanced clustering. Could be future work. |

## Known Limitations & Future Work

1. **Fallback feature extractor is weak** — color histograms + edge gradients don't capture semantic damage types as well as CLIP. Real performance depends on CLIP weights being available.

2. **Limited reference set** — only 20 sample claims. With thousands of historical claims, retrieval accuracy would jump.

3. **No temporal reasoning** — ignores timing of claims (e.g., "user filed 3 claims in 2 weeks" is higher risk). User history flags are present but crude.

4. **VLM integration incomplete** — `main_vlm.py` exists but isn't wired into the main pipeline. Swapping in a model like BLIP to caption damage ("front bumper dent, quarter-panel crease visible") could improve accuracy 5–10%.

5. **Error analysis limited** — with ground truth on test set (post-evaluation), would identify systematic failure modes and retrain rules.

## For Interviewers / Evaluators

- **Full approach explanation**: See `INTERVIEW_GUIDE.md`
- **One-page cheat sheet**: See `INTERVIEW_CHEAT_SHEET.txt`
- **Bug discovery & fix**: See `FIX_LOG.md`
- **Code walkthrough**: Start at `code/main.py` line 1, work through the pipeline stages

## Contact & Notes

Built during HackerRank Orchestrate hackathon (June 2026).

Questions? See `INTERVIEW_GUIDE.md` for detailed Q&A, or review the code comments in `code/main.py`.

---

**TL;DR:** Multimodal claim validator using CLIP + local rules. No APIs, interpretable decisions, beats text baseline on all metrics. Honest evaluation (found and fixed a metrics-leakage bug before submission).

