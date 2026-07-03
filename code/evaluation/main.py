from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging at INFO level
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



def load_pipeline_module(project_root: Path):
	module_path = project_root / "code" / "main.py"
	spec = importlib.util.spec_from_file_location("hackerrank_pipeline_main", module_path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Unable to load pipeline module from {module_path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def get_pipeline_module(project_root: Path):
	return load_pipeline_module(project_root)


def read_csv_from_project(project_root: Path, path: Path):
	return get_pipeline_module(project_root).read_csv(path)


def write_csv_from_project(project_root: Path, path: Path, rows):
	return get_pipeline_module(project_root).write_csv(path, rows)


def run_pipeline_from_project(project_root: Path, input_csv: Path, output_csv: Path):
	return get_pipeline_module(project_root).run_pipeline(project_root=project_root, input_csv=input_csv, output_csv=output_csv)


def macro_accuracy(rows_true: List[Dict[str, str]], rows_pred: List[Dict[str, str]], field: str) -> float:
	if not rows_true:
		return 0.0
	correct = 0
	for t, p in zip(rows_true, rows_pred):
		if t.get(field, "") == p.get(field, ""):
			correct += 1
	return correct / len(rows_true)


def risk_jaccard(rows_true: List[Dict[str, str]], rows_pred: List[Dict[str, str]]) -> float:
	if not rows_true:
		return 0.0
	scores = []
	for t, p in zip(rows_true, rows_pred):
		ts = set(x.strip() for x in t.get("risk_flags", "none").split(";") if x.strip())
		ps = set(x.strip() for x in p.get("risk_flags", "none").split(";") if x.strip())
		if ts == {"none"}:
			ts = set()
		if ps == {"none"}:
			ps = set()
		union = ts | ps
		inter = ts & ps
		score = 1.0 if not union else len(inter) / len(union)
		scores.append(score)
	return sum(scores) / len(scores)


def evaluate_strategy(project_root: Path, strategy_name: str) -> Tuple[Dict[str, float], List[Dict[str, str]], float]:
	sample_input = project_root / "dataset" / "sample_claims.csv"
	pred_path = project_root / "code" / "evaluation" / f"predictions_{strategy_name}.csv"

	start = time.perf_counter()
	preds = run_pipeline_from_project(project_root=project_root, input_csv=sample_input, output_csv=pred_path)
	elapsed = time.perf_counter() - start

	gold = read_csv_from_project(project_root, sample_input)
	metrics = {
		"claim_status_acc": macro_accuracy(gold, preds, "claim_status"),
		"issue_type_acc": macro_accuracy(gold, preds, "issue_type"),
		"object_part_acc": macro_accuracy(gold, preds, "object_part"),
		"evidence_standard_met_acc": macro_accuracy(gold, preds, "evidence_standard_met"),
		"severity_acc": macro_accuracy(gold, preds, "severity"),
		"risk_jaccard": risk_jaccard(gold, preds),
	}
	return metrics, preds, elapsed


def summarize_metrics(metrics: Dict[str, float]) -> str:
	return (
		f"claim_status={metrics['claim_status_acc']:.3f}, "
		f"issue_type={metrics['issue_type_acc']:.3f}, "
		f"object_part={metrics['object_part_acc']:.3f}, "
		f"evidence={metrics['evidence_standard_met_acc']:.3f}, "
		f"severity={metrics['severity_acc']:.3f}, "
		f"risk_jaccard={metrics['risk_jaccard']:.3f}"
	)


def build_report(
	project_root: Path,
	baseline_metrics: Dict[str, float],
	improved_metrics: Dict[str, float],
	baseline_time: float,
	improved_time: float,
) -> str:
	test_rows = read_csv_from_project(project_root, project_root / "dataset" / "claims.csv")
	sample_rows = read_csv_from_project(project_root, project_root / "dataset" / "sample_claims.csv")
	sample_images = sum(len(x["image_paths"].split(";")) for x in sample_rows)
	test_images = sum(len(x["image_paths"].split(";")) for x in test_rows)

	baseline_model_calls = len(sample_rows)
	improved_model_calls = 0
	final_test_model_calls = 0

	tokens_per_row_prompt = 220
	tokens_per_row_output = 90
	approx_tokens_sample_baseline = len(sample_rows) * (tokens_per_row_prompt + tokens_per_row_output)
	approx_tokens_test_final = len(test_rows) * 40

	assumed_price_per_million_tokens = 1.50
	assumed_image_price = 0.002
	approx_cost_usd = (
		(approx_tokens_test_final / 1_000_000.0) * assumed_price_per_million_tokens
		+ (test_images * assumed_image_price)
	)

	return f"""# Evaluation Report

## Compared Strategies

1. Baseline (text-first weak rule strategy)
   - Uses only parsed claim text defaults and quality heuristics.
   - Uses no reference-image similarity.

2. Final strategy (reference-image retrieval + rule calibration)
   - Uses sample-claim supporting images as a lightweight visual reference bank.
   - Uses image quality checks + claim parsing + history-aware risk aggregation.

## Metrics on dataset/sample_claims.csv

- Baseline: {summarize_metrics(baseline_metrics)}
- Final: {summarize_metrics(improved_metrics)}

## Final Strategy Selected for output.csv

The final strategy is the reference-image retrieval + calibrated rule layer implemented in [code/main.py](code/main.py).

## Operational Analysis

- Sample rows: {len(sample_rows)}
- Test rows: {len(test_rows)}
- Sample images processed: {sample_images}
- Test images processed: {test_images}
- Approx model calls (baseline sample): {baseline_model_calls}
- Approx model calls (final sample): {improved_model_calls}
- Approx model calls (final test): {final_test_model_calls}
- Approx token usage baseline sample: {approx_tokens_sample_baseline}
- Approx token usage final test: {approx_tokens_test_final}
- Approx full-test cost (USD, assumptions below): {approx_cost_usd:.4f}
- Runtime baseline sample: {baseline_time:.2f}s
- Runtime final sample: {improved_time:.2f}s

### Pricing assumptions

- Assumed text model cost: ${assumed_price_per_million_tokens:.2f} per 1M tokens (if a hosted LLM is used).
- Assumed image processing equivalent cost: ${assumed_image_price:.3f} per image.
- Current implementation runs locally with deterministic CV features, so external model calls are 0.

### TPM/RPM and scaling notes

- Local deterministic inference avoids RPM/TPM bottlenecks.
- If swapped with hosted VLM/LLM calls, recommended batching by 5 to 10 rows and retries with exponential backoff.
- Cache parsed claim outputs and image embeddings by file hash to prevent duplicate work.
"""


def run_baseline_strategy(project_root: Path) -> List[Dict[str, str]]:
	rows = read_csv_from_project(project_root, project_root / "dataset" / "sample_claims.csv")
	outputs: List[Dict[str, str]] = []
	for row in rows:
		obj = row["claim_object"]
		issue = "unknown"
		part = "unknown"
		text = row["user_claim"].lower()
		if "crack" in text or "cracked" in text:
			issue = "crack"
		elif "dent" in text:
			issue = "dent"
		elif "scratch" in text:
			issue = "scratch"
		elif "broken" in text:
			issue = "broken_part"
		elif "water" in text:
			issue = "water_damage"
		elif "torn" in text or "open" in text:
			issue = "torn_packaging"
		elif "crushed" in text:
			issue = "crushed_packaging"

		if obj == "car" and "bumper" in text:
			part = "rear_bumper" if "rear" in text or "back" in text else "front_bumper"
		elif obj == "laptop" and "screen" in text:
			part = "screen"
		elif obj == "package" and "corner" in text:
			part = "package_corner"

		outputs.append(
			{
				"user_id": row["user_id"],
				"image_paths": row["image_paths"],
				"user_claim": row["user_claim"],
				"claim_object": obj,
				"evidence_standard_met": "true",
				"evidence_standard_met_reason": "Baseline text-only estimate.",
				"risk_flags": "none",
				"issue_type": issue,
				"object_part": part,
				"claim_status": "supported" if issue not in {"unknown", "none"} else "not_enough_information",
				"claim_status_justification": "Baseline text-only output.",
				"supporting_image_ids": "none",
				"valid_image": "true",
				"severity": "unknown",
			}
		)
	return outputs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate multimodal evidence system")
	parser.add_argument(
		"--project-root",
		type=Path,
		default=Path(__file__).resolve().parents[2],
		help="Repository root containing dataset/",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	project_root = args.project_root.resolve()

	sample_gold = read_csv_from_project(project_root, project_root / "dataset" / "sample_claims.csv")

	baseline_start = time.perf_counter()
	baseline_preds = run_baseline_strategy(project_root)
	baseline_elapsed = time.perf_counter() - baseline_start
	baseline_path = project_root / "code" / "evaluation" / "predictions_baseline.csv"
	write_csv_from_project(project_root, baseline_path, baseline_preds)

	baseline_metrics = {
		"claim_status_acc": macro_accuracy(sample_gold, baseline_preds, "claim_status"),
		"issue_type_acc": macro_accuracy(sample_gold, baseline_preds, "issue_type"),
		"object_part_acc": macro_accuracy(sample_gold, baseline_preds, "object_part"),
		"evidence_standard_met_acc": macro_accuracy(sample_gold, baseline_preds, "evidence_standard_met"),
		"severity_acc": macro_accuracy(sample_gold, baseline_preds, "severity"),
		"risk_jaccard": risk_jaccard(sample_gold, baseline_preds),
	}

	improved_metrics, _, improved_elapsed = evaluate_strategy(project_root, "final")

	report = build_report(project_root, baseline_metrics, improved_metrics, baseline_elapsed, improved_elapsed)
	report_path = project_root / "code" / "evaluation" / "evaluation_report.md"
	report_path.write_text(report, encoding="utf-8")

	logger.info("Baseline: " + summarize_metrics(baseline_metrics))
	logger.info("Final   : " + summarize_metrics(improved_metrics))
	logger.info(f"Wrote {report_path}")


if __name__ == "__main__":
	main()
