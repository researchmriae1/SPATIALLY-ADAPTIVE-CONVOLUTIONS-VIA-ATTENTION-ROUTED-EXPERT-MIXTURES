import os

import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset


def center_crop_3d(
    vol,
    crop_size,
):
    """
    Center-crop a 3D volume.

    Args:
        vol: NumPy array of shape [D, H, W].
        crop_size: Target size (D, H, W).

    Returns:
        Cropped volume of shape [D, H, W].
    """

    D, H, W = vol.shape
    cd, ch, cw = crop_size

    if D < cd or H < ch or W < cw:
        raise ValueError(
            f"Volume {vol.shape} is smaller than "
            f"crop size {crop_size}."
        )

    d0 = (D - cd) // 2
    h0 = (H - ch) // 2
    w0 = (W - cw) // 2

    return vol[
        d0:d0 + cd,
        h0:h0 + ch,
        w0:w0 + cw,
    ]


def random_crop_3d(
    vol,
    crop_size,
):
    """
    Randomly crop a 3D volume.

    Args:
        vol: NumPy array of shape [D, H, W].
        crop_size: Target size (D, H, W).

    Returns:
        Cropped volume of shape [D, H, W].
    """

    D, H, W = vol.shape
    cd, ch, cw = crop_size

    if D < cd or H < ch or W < cw:
        raise ValueError(
            f"Volume {vol.shape} is smaller than "
            f"crop size {crop_size}."
        )

    d = np.random.randint(
        0,
        D - cd + 1,
    )

    h = np.random.randint(
        0,
        H - ch + 1,
    )

    w = np.random.randint(
        0,
        W - cw + 1,
    )

    return vol[
        d:d + cd,
        h:h + ch,
        w:w + cw,
    ]


class MRIDataset(Dataset):
    """
    MRI dataset for autoencoder training.

    Each sample is returned as:

        [C, D, H, W]

    with C = 1.
    """

    def __init__(
        self,
        root_dir=None,
        files=None,
        crop_size=(128, 128, 128),
        normalize=True,
        augment=True,
        random_crop=True,
    ):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.normalize = normalize
        self.augment = augment
        self.random_crop = random_crop

        # --------------------------------------------------
        # Resolve files
        # --------------------------------------------------

        if files is not None:

            self.files = sorted(files)

        elif root_dir is not None:

            self.files = sorted(
                [
                    os.path.join(
                        root_dir,
                        filename,
                    )
                    for filename in os.listdir(root_dir)
                    if filename.endswith(".nii")
                    or filename.endswith(".nii.gz")
                ]
            )

        else:
            raise ValueError(
                "Either root_dir or files must be provided."
            )

        if len(self.files) == 0:
            raise RuntimeError(
                "No NIfTI files found."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(
        self,
        idx,
    ):
        filepath = self.files[idx]

        # --------------------------------------------------
        # Load NIfTI
        # --------------------------------------------------

        vol = nib.load(
            filepath
        ).get_fdata().astype(
            np.float32
        )

        # Handle possible 4D NIfTI files
        if vol.ndim == 4:
            vol = vol[..., 0]

        # --------------------------------------------------
        # Convert [H, W, D] -> [D, H, W]
        # --------------------------------------------------

        vol = np.transpose(
            vol,
            (2, 0, 1),
        )

        # --------------------------------------------------
        # Percentile-based normalization
        # --------------------------------------------------

        if self.normalize:

            vmin, vmax = np.percentile(
                vol,
                (1, 99),
            )

            vol = np.clip(
                vol,
                vmin,
                vmax,
            )

            vol = (
                vol - vmin
            ) / (
                vmax - vmin + 1e-8
            )

        # --------------------------------------------------
        # Crop
        # --------------------------------------------------

        if self.random_crop:

            vol = random_crop_3d(
                vol,
                self.crop_size,
            )

        else:

            vol = center_crop_3d(
                vol,
                self.crop_size,
            )

        # --------------------------------------------------
        # Spatial augmentation
        # --------------------------------------------------

        if self.augment:

            if np.random.rand() < 0.5:
                vol = vol[:, :, ::-1]

            if np.random.rand() < 0.5:
                vol = vol[:, ::-1, :]

        # --------------------------------------------------
        # Convert to tensor
        # --------------------------------------------------

        hr = torch.from_numpy(
            vol.copy()
        ).float()

        # [D, H, W] -> [1, D, H, W]
        hr = hr.unsqueeze(0)

        return hr
