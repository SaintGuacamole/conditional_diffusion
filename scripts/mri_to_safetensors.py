
import logging
import uuid
from pathlib import Path

import fire
import torch
import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize

from diffusion.datasets import dicom_to_safetensors_mri


def main(
    mri_root_dir: str,
    target_dir: str,
    split: str
):
    tiff_directory = Path(mri_root_dir)
    target_dir = Path(target_dir)

    (target_dir / split).mkdir(parents=True, exist_ok=True)

    transforms = Resize(size=(256, 256), interpolation=InterpolationMode.BILINEAR)

    images = list(Path(tiff_directory).glob("**/*.dcm"))
    logging.info(f"Found {len(images)} dicom images.")

    for file_name in tqdm.tqdm(images):

        file_id = uuid.uuid4()

        dicom_to_safetensors_mri(
            file_path=file_name,
            out_filename=target_dir / split / f"{file_id}",
            transforms=transforms,
        )


if __name__ == "__main__":
    import lovely_tensors
    lovely_tensors.monkey_patch()

    torch.cuda.manual_seed(0)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fire.Fire(main)