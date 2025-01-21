import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def tiff_to_safetensors_patches(
        *,
        file: Path,
        patch_directory: Path,
        train_fraction: float,
        patch_size: int,
        overlap: int = 0,
        pbar_enabled: bool = True,
):
    image = Image.open(file)

    image_width, image_height  = image.size

    num_patches_x = image_width // (patch_size - overlap)
    num_patches_y = image_height // (patch_size - overlap)

    _x = (image_width - patch_size) // ((image_width + patch_size - 1) // patch_size - 1)
    _y = (image_height - patch_size) // ((image_height + patch_size - 1) // patch_size - 1)

    pbar = tqdm(
        range(0, num_patches_x * num_patches_y),
        desc="patches",
        disable=not pbar_enabled,
    )

    with logging_redirect_tqdm():
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                patch_x = x * (_x - overlap)
                patch_y = y * (_y - overlap)

                patch = image.crop((patch_x, patch_y, patch_x + patch_size, patch_y + patch_size))

                patch_np = np.array(patch, dtype=np.float32)
                patch_tensor = torch.tensor(patch_np).unsqueeze(0)

                split = "train" if torch.rand(1) < train_fraction else "test"
                file_id = uuid.uuid4()

                metadata = {
                    'filename': str(file),
                    'slice_number': str(-1),
                    'x': str(patch_x),
                    'y': str(patch_y),
                    'patch_size': str(patch_size),
                    'overlap': str(overlap),
                }
                save_file(
                    {'image': patch_tensor},
                    Path(patch_directory) / split / f"{file_id}.safetensors",
                    metadata=metadata,
                )
                pbar.update(1)
    pbar.close()
