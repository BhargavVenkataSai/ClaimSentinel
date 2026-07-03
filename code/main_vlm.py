"""
VLM-based evidence review pipeline.
Supports GPT-4o (OpenAI), Gemini 1.5 Flash (Google), Claude 3.5 Haiku (Anthropic).

Set exactly ONE of these env vars before running:
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=AIza...
    ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python code/main_vlm.py                       # process dataset/claims.csv -> output.csv
    python code/main_vlm.py --sample              # process dataset/sample_claims.csv -> output_vlm_sample.csv
    python code/main_vlm.py --input-csv X --output-csv Y
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]

ALLOWED_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
}
ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}
ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}
ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def encode_image_b64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_ext_to_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(ext.lstrip("."), "image/jpeg")


# ---------------------------------------------------------------------------
# VLM backends
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, images_b64: List[Tuple[str, str]], model: str = "gpt-4o") -> str:
    import urllib.request, urllib.error
    api_key = os.environ["OPENAI_API_KEY"]
    content = [{"type": "text", "text": prompt}]
    for b64, mime in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}})
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 800,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _call_gemini(prompt: str, images_b64: List[Tuple[str, str]], model: str = "gemini-1.5-flash") -> str:
    import urllib.request
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    parts = [{"text": prompt}]
    for b64, mime in images_b64:
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 800},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_anthropic(prompt: str, images_b64: List[Tuple[str, str]], model: str = "claude-3-5-haiku-20241022") -> str:
    import urllib.request
    api_key = os.environ["ANTHROPIC_API_KEY"]
    content = []
    for b64, mime in images_b64:
        content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
    content.append({"type": "text", "text": prompt})
    body = json.dumps({
        "model": model,
        "max_tokens": 800,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def detect_backend() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError(
        "No VLM API key found. Set one of: OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY"
    )


def call_vlm(prompt: str, images_b64: List[Tuple[str, str]], backend: str) -> str:
    """Call the VLM with retry on transient errors."""
    for attempt in range(3):
        try:
            if backend == "openai":
                return _call_openai(prompt, images_b64)
            elif backend == "gemini":
                return _call_gemini(prompt, images_b64)
            elif backend == "anthropic":
                return _call_anthropic(prompt, images_b64)
        except Exception as e:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}] error: {e}. Waiting {wait}s...")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a damage claim evidence verifier. You receive:
1. A conversation between a customer and support agent describing a damage claim.
2. One or more images submitted as evidence.

Your task: Analyse the images carefully against the claim and return a JSON object ONLY (no extra text).

JSON schema (all fields required):
{
  "issue_type": "<one of: dent|scratch|crack|glass_shatter|broken_part|missing_part|torn_packaging|crushed_packaging|water_damage|stain|none|unknown>",
  "object_part": "<specific part visible in image, e.g. rear_bumper, windshield, screen, keyboard, seal, box — or unknown>",
  "claim_status": "<supported|contradicted|not_enough_information>",
  "claim_status_justification": "<1-2 sentence reason>",
  "severity": "<none|low|medium|high|unknown>",
  "evidence_standard_met": "<true|false>",
  "evidence_standard_met_reason": "<1 sentence>",
  "supporting_image_ids": "<semicolon-separated image IDs that support the status, or none>",
  "valid_image": "<true|false>",
  "risk_flags": "<semicolon-separated from: none|blurry_image|cropped_or_obstructed|low_light_or_glare|wrong_angle|wrong_object|wrong_object_part|damage_not_visible|claim_mismatch|possible_manipulation|non_original_image|text_instruction_present|user_history_risk|manual_review_required>"
}

Decision rules:
- claim_status=supported: Image clearly shows the exact type of damage claimed on the claimed part.
- claim_status=contradicted: Image clearly shows something different — no damage where claimed, or different type, or wrong object.
- claim_status=not_enough_information: Image is unclear, missing, blurry, or can't confirm or deny the claim.
- severity: none if no damage, low for minor cosmetic (light scratch/stain), medium for moderate (dent/crack/torn), high for severe (shatter/broken/missing).
- evidence_standard_met=false when: image is missing, blurry, wrong object, or completely unrelated.
- Include claim_mismatch and manual_review_required if contradicted.
- Include user_history_risk and manual_review_required if the user history shows prior risk.
- Include damage_not_visible if claimed damage is not visible in image.

Return ONLY valid JSON. No markdown fences, no explanation."""


def build_prompt(row: Dict[str, str], history: Dict[str, str], image_ids: List[str]) -> str:
    history_note = ""
    if history:
        flags = history.get("history_flags", "none")
        claim_count = history.get("total_claims", "?")
        if flags and flags != "none":
            history_note = f"\n\nUSER HISTORY: {claim_count} prior claims. Risk flags: {flags}. Include user_history_risk and manual_review_required in risk_flags."

    return f"""Claim object: {row['claim_object']}
Conversation:
{row['user_claim']}
{history_note}
Image IDs in order: {', '.join(image_ids)}

Analyse the images above and return the JSON verdict."""


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _str_bool(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).lower() if str(v).lower() in ("true", "false") else "false"


def _clean_flags(flags_str: str, allowed: set) -> str:
    parts = [f.strip() for f in flags_str.split(";") if f.strip()]
    cleaned = [f for f in parts if f in allowed and f != "none"]
    return ";".join(cleaned) if cleaned else "none"


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from VLM output."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Extract from markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from: {text[:200]}")


def _safe(d: dict, key: str, default: str, allowed: Optional[set] = None) -> str:
    v = str(d.get(key, default)).strip().lower()
    if allowed and v not in allowed:
        return default
    return v


def process_row(
    row: Dict[str, str],
    dataset_dir: Path,
    history_map: Dict[str, Dict[str, str]],
    backend: str,
) -> Dict[str, str]:
    image_path_strs = [p.strip() for p in row["image_paths"].split(";") if p.strip()]
    images_b64: List[Tuple[str, str]] = []
    image_ids: List[str] = []
    for rel in image_path_strs:
        p = (dataset_dir / rel).resolve()
        image_ids.append(p.stem)
        b64 = encode_image_b64(p)
        if b64:
            mime = image_ext_to_mime(p)
            images_b64.append((b64, mime))

    history = history_map.get(row["user_id"], {})
    prompt = build_prompt(row, history, image_ids)

    try:
        raw = call_vlm(SYSTEM_PROMPT + "\n\n" + prompt, images_b64, backend)
        result = _extract_json(raw)
    except Exception as e:
        print(f"  [ERROR] {row['user_id']}: {e}")
        result = {}

    # Normalise and validate all fields
    issue_type = _safe(result, "issue_type", "unknown", ALLOWED_ISSUE_TYPES)
    object_part_raw = str(result.get("object_part", "unknown")).strip()
    claim_status = _safe(result, "claim_status", "not_enough_information", ALLOWED_CLAIM_STATUS)
    justification = str(result.get("claim_status_justification", "")).strip() or "Unable to determine."
    severity = _safe(result, "severity", "unknown", ALLOWED_SEVERITY)
    esm = _str_bool(result.get("evidence_standard_met", False))
    esm_reason = str(result.get("evidence_standard_met_reason", "")).strip() or "Unable to determine."
    supporting_raw = str(result.get("supporting_image_ids", "none")).strip()
    valid_image = _str_bool(result.get("valid_image", bool(images_b64)))
    risk_flags_raw = str(result.get("risk_flags", "none")).strip()

    # Clean risk_flags
    risk_flags = _clean_flags(risk_flags_raw, ALLOWED_RISK_FLAGS)

    # Ensure manual_review_required if contradicted or claim_mismatch
    flags_list = [f for f in risk_flags.split(";") if f and f != "none"]
    if "user_history_risk" in flags_list and "manual_review_required" not in flags_list:
        flags_list.append("manual_review_required")
    if "claim_mismatch" in flags_list and "manual_review_required" not in flags_list:
        flags_list.append("manual_review_required")
    risk_flags = ";".join(flags_list) if flags_list else "none"

    # Validate supporting_image_ids
    valid_ids = set(image_ids)
    sup_ids = [s.strip() for s in supporting_raw.split(";") if s.strip() and s.strip() in valid_ids]
    if claim_status in ("supported", "contradicted") and not sup_ids and image_ids:
        sup_ids = image_ids[:1]
    supporting_image_ids = ";".join(sup_ids) if sup_ids else "none"

    print(f"  {row['user_id']} | {claim_status} | {issue_type} | {object_part_raw} | sev={severity}")

    return {
        "user_id": row["user_id"],
        "image_paths": row["image_paths"],
        "user_claim": row["user_claim"],
        "claim_object": row["claim_object"],
        "evidence_standard_met": esm,
        "evidence_standard_met_reason": esm_reason,
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part_raw.lower().replace(" ", "_") or "unknown",
        "claim_status": claim_status,
        "claim_status_justification": justification,
        "supporting_image_ids": supporting_image_ids,
        "valid_image": valid_image,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    project_root: Path,
    input_csv: Path,
    output_csv: Path,
) -> List[Dict[str, str]]:
    backend = detect_backend()
    print(f"Using backend: {backend.upper()}")

    dataset_dir = project_root / "dataset"
    rows = read_csv(input_csv)
    history_rows = read_csv(dataset_dir / "user_history.csv")
    history_map = {r["user_id"]: r for r in history_rows}

    outputs: List[Dict[str, str]] = []
    for i, row in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] Processing {row['user_id']} ({row['claim_object']})...")
        out = process_row(row, dataset_dir, history_map, backend)
        outputs.append(out)
        # Small delay to respect rate limits
        time.sleep(0.5)

    write_csv(output_csv, outputs)

    # Validate
    from code.main import validate_output  # reuse existing validator
    errors = validate_output(output_csv, input_csv)
    if errors:
        print("[WARN] Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"[OK] Output validation passed: {len(outputs)} rows, schema correct.")

    return outputs


def main():
    parser = argparse.ArgumentParser(description="VLM-based evidence review pipeline")
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--sample", action="store_true",
                        help="Run on sample_claims.csv for evaluation")
    args = parser.parse_args()

    root = args.project_root.resolve()
    if args.sample:
        input_csv = root / "dataset" / "sample_claims.csv"
        output_csv = root / "output_vlm_sample.csv"
    else:
        input_csv = args.input_csv or (root / "dataset" / "claims.csv")
        output_csv = args.output_csv or (root / "output.csv")

    run_pipeline(project_root=root, input_csv=input_csv.resolve(), output_csv=output_csv.resolve())
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
