import json
import logging
from pathlib import Path
from main import infer_row, build_reference_library, read_csv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_demo():
    # Setup paths relative to repository root
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "dataset"
    
    # 1. Load reference samples
    sample_claims_path = dataset_dir / "sample_claims.csv"
    logger.info(f"Loading reference claims from: {sample_claims_path}")
    sample_rows = read_csv(sample_claims_path)
    references = build_reference_library(dataset_dir, sample_rows)
    logger.info(f"Built reference library with {len(references)} images.")
    
    # 2. Hardcoded claim payload
    # Let's take a sample scenario where user_claim describes a dent on the rear bumper
    sample_claim = {
        "user_id": "user_200",
        "image_paths": "sample/car_dent_rear_bumper.jpg",
        "claim_object": "car",
        "user_claim": "The rear bumper of my car got dented after someone backed into it.",
    }
    
    # Simple history map setup
    history_map = {
        "user_200": {
            "history_summary": "Clean history, no prior claims flagged.",
            "history_flags": "none",
        }
    }
    
    logger.info(f"Running inference on claim:\n{json.dumps(sample_claim, indent=2)}")
    
    # 3. Perform inference
    try:
        result = infer_row(
            row=sample_claim,
            base_dir=dataset_dir,
            references=references,
            history_map=history_map
        )
        logger.info("Inference completed successfully.")
        
        # Pretty print result JSON
        print("\n--- INFERENCE RESULT JSON ---")
        print(json.dumps(result, indent=2))
        print("-----------------------------\n")
    except Exception as e:
        logger.exception(f"Failed to run inference: {e}")

if __name__ == "__main__":
    run_demo()
