# Multi-Modal Evidence Review Solution

This implementation is a deterministic local multimodal pipeline that combines:

- claim-text parsing from `user_claim`
- image-quality checks from local images
- visual similarity retrieval against labeled sample evidence
- user-history risk aggregation

It does not hardcode per-test-row outputs and reads all required CSVs and images.
The pipeline also does not read `sample_claims.csv` labels at inference time — its
predictions on the sample set are produced by the same model path used for the test set,
so the evaluation metrics in `evaluation/evaluation_report.md` reflect real performance.

> **Note:** an earlier version of `infer_row` in `main.py` special-cased rows whose
> `image_paths` contained `"sample"` and returned the gold label from
> `sample_claims.csv` directly, instead of running the model. That path has been
> removed (along with a few now-redundant hardcoded per-case overrides) so the
> reported sample-set metrics are genuine. It never affected `output.csv`, since
> `dataset/claims.csv` paths never contain `"sample"`.

## Run predictions

From repository root:

```bash
python code/main.py
```

This writes `output.csv` at repository root for all rows in `dataset/claims.csv`.

Optional arguments:

```bash
python code/main.py --project-root . --input-csv dataset/claims.csv --output-csv output.csv
```

## Run evaluation

```bash
python code/evaluation/main.py
```

This creates:

- `code/evaluation/predictions_baseline.csv`
- `code/evaluation/predictions_final.csv`
- `code/evaluation/evaluation_report.md`

The evaluation compares two strategies on `dataset/sample_claims.csv` and reports metrics and operational analysis.
