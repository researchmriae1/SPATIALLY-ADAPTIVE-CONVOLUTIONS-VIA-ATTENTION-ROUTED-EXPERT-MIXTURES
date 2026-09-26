from torch.utils.data import DataLoader

from configs import data
from src.data.dataset import MRIDataset


train_dataset = MRIDataset(
    root_dir=data.TRAIN_DIR,
    crop_size=data.CROP_SIZE,
    normalize=data.NORMALIZE,
    augment=data.AUGMENT,
    random_crop=data.RANDOM_CROP,
)

val_dataset = MRIDataset(
    root_dir=data.VAL_DIR,
    crop_size=data.CROP_SIZE,
    normalize=data.NORMALIZE,
    augment=False,
    random_crop=False,
)

test_dataset = MRIDataset(
    root_dir=data.TEST_DIR,
    crop_size=data.CROP_SIZE,
    normalize=data.NORMALIZE,
    augment=False,
    random_crop=False,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=data.BATCH_SIZE,
    shuffle=data.SHUFFLE_TRAIN,
    num_workers=data.NUM_WORKERS,
    pin_memory=data.PIN_MEMORY,
    persistent_workers=data.PERSISTENT_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=data.BATCH_SIZE,
    shuffle=data.SHUFFLE_VAL,
    num_workers=data.NUM_WORKERS,
    pin_memory=data.PIN_MEMORY,
    persistent_workers=data.PERSISTENT_WORKERS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=data.BATCH_SIZE,
    shuffle=data.SHUFFLE_TEST,
    num_workers=data.NUM_WORKERS,
    pin_memory=data.PIN_MEMORY,
    persistent_workers=data.PERSISTENT_WORKERS,
)
