"""
Configuration for MRI autoencoder data loading.
"""


# ============================================================
# Dataset
# ============================================================

DATA_ROOT = "data"

TRAIN_DIR = f"{DATA_ROOT}/train"
VAL_DIR = f"{DATA_ROOT}/val"
TEST_DIR = f"{DATA_ROOT}/test"


# ============================================================
# Volume preprocessing
# ============================================================

CROP_SIZE = (
    128,
    128,
    128,
)

NORMALIZE = True

RANDOM_CROP = True

AUGMENT = True


# ============================================================
# DataLoader
# ============================================================

BATCH_SIZE = 1

NUM_WORKERS = 4

PIN_MEMORY = True

PERSISTENT_WORKERS = True

SHUFFLE_TRAIN = True

SHUFFLE_VAL = False

SHUFFLE_TEST = False


# ============================================================
# Reproducibility
# ============================================================

SEED = 42
