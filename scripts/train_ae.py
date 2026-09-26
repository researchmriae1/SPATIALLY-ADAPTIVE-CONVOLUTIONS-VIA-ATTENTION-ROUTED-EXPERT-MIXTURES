import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import ae, data
from data import MRIDataset
from models import AutoEncoder
from training import train_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders():
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

    loader_kwargs = {
        "batch_size": data.BATCH_SIZE,
        "num_workers": data.NUM_WORKERS,
        "pin_memory": data.PIN_MEMORY,
        "persistent_workers": (
            data.PERSISTENT_WORKERS
            and data.NUM_WORKERS > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=data.SHUFFLE_TRAIN,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=data.SHUFFLE_VAL,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=data.SHUFFLE_TEST,
        **loader_kwargs,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def create_model(device):
    model = AutoEncoder(
        encoder_experts=ae.ENCODER_EXPERTS,
        decoder_experts=ae.DECODER_EXPERTS,
    ).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {num_params:,}")

    return model


def main():

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    set_seed(ae.SEED)

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = create_model(device)

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ae.LEARNING_RATE,
        weight_decay=ae.WEIGHT_DECAY,
    )

    # ------------------------------------------------------------
    # Checkpoint path
    # ------------------------------------------------------------

    os.makedirs(
        ae.CHECKPOINT_DIR,
        exist_ok=True,
    )

    save_path = os.path.join(
        ae.CHECKPOINT_DIR,
        ae.BEST_CHECKPOINT_NAME,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=ae.NUM_EPOCHS,
        save_path=save_path,
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    main()
