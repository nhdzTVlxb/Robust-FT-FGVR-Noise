"""Central project configuration: paths and dataset constants.

All entry scripts read their CLI defaults from here, so relocating data or
outputs only requires editing this file.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# data
TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "test"
NUM_CLASSES = 500

# working artifacts (feature cache, checkpoints, experiment logs)
DEFAULT_WORK_DIR = BASE_DIR / "outputs"

# submission files
SUBMISSIONS_DIR = BASE_DIR / "submissions"
DEFAULT_OUTPUT_CSV = SUBMISSIONS_DIR / "pred_results.csv"
DEFAULT_OUTPUT_ZIP = SUBMISSIONS_DIR / "pred_results.zip"
DEFAULT_LORA_OUTPUT_CSV = SUBMISSIONS_DIR / "pred_results_lora.csv"
