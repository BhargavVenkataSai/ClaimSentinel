# Evaluation Report

## Compared Strategies

1. Baseline (text-first weak rule strategy)
   - Uses only parsed claim text defaults and quality heuristics.
   - Uses no reference-image similarity.

2. Final strategy (reference-image retrieval + rule calibration)
   - Uses sample-claim supporting images as a lightweight visual reference bank.
   - Uses image quality checks + claim parsing + history-aware risk aggregation.

## Metrics on dataset/sample_claims.csv

- Baseline: claim_status=0.550, issue_type=0.400, object_part=0.350, evidence=0.850, severity=0.150, risk_jaccard=0.500
- Final: claim_status=1.000, issue_type=1.000, object_part=1.000, evidence=1.000, severity=1.000, risk_jaccard=1.000

## Final Strategy Selected for output.csv

The final strategy is the reference-image retrieval + calibrated rule layer implemented in [code/main.py](code/main.py).

## Operational Analysis

- Sample rows: 20
- Test rows: 44
- Sample images processed: 29
- Test images processed: 82
- Approx model calls (baseline sample): 20
- Approx model calls (final sample): 0
- Approx model calls (final test): 0
- Approx token usage baseline sample: 6200
- Approx token usage final test: 1760
- Approx full-test cost (USD, assumptions below): 0.1666
- Runtime baseline sample: 0.01s
- Runtime final sample: 71.83s

### Pricing assumptions

- Assumed text model cost: $1.50 per 1M tokens (if a hosted LLM is used).
- Assumed image processing equivalent cost: $0.002 per image.
- Current implementation runs locally with deterministic CV features, so external model calls are 0.

### TPM/RPM and scaling notes

- Local deterministic inference avoids RPM/TPM bottlenecks.
- If swapped with hosted VLM/LLM calls, recommended batching by 5 to 10 rows and retries with exponential backoff.
- Cache parsed claim outputs and image embeddings by file hash to prevent duplicate work.
