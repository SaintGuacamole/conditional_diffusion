import logging
import uuid
from pathlib import Path

import fire
import torch
import tqdm

from diffusion.datasets import dicom_to_safetensors


def main(
    kits_root_dir: str,
    target_dir: str,
    train_fraction: float = 0.8
):
    dicom_directory = Path(kits_root_dir)
    target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "train").mkdir(parents=True, exist_ok=True)
    (target_dir / "test").mkdir(parents=True, exist_ok=True)

    images = list(Path(dicom_directory).glob("**/*.dcm"))
    logging.info(f"Found {len(images)} dicom images.")

    for file_name in tqdm.tqdm(images):

        if "Segmentation" in str(file_name):
            continue
        else:
            file_id = uuid.uuid4()

            split = "train" if torch.rand(1) < train_fraction else "test"
            dicom_to_safetensors(
                file_path=file_name,
                out_filename=target_dir / split / f"{file_id}"
            )


if __name__ == "__main__":
    import lovely_tensors
    lovely_tensors.monkey_patch()

    torch.manual_seed(0)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fire.Fire(main)