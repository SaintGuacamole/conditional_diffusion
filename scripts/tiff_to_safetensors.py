import logging
from pathlib import Path

import fire
from PIL import Image

from diffusion.datasets import tiff_to_safetensors_patches

Image.MAX_IMAGE_PIXELS = None

def main(
        tiff_directory: str,
        patch_directory: str,
        patch_size: int,
        overlap: int,
):
    tiff_directory = Path(tiff_directory)
    patch_directory = Path(patch_directory)

    patch_directory.mkdir(parents=True, exist_ok=True)
    (patch_directory / "train").mkdir(parents=True, exist_ok=True)
    (patch_directory / "test").mkdir(parents=True, exist_ok=True)

    tiffs = list(Path(tiff_directory).glob("**/*.tif*"))
    logging.info(f"Found {len(tiffs)} tiffs.")

    for i, tiff in enumerate(tiffs):

        logging.info(f"Processing tiff {i+1} {tiff}")

        tiff_to_safetensors_patches(
            file=tiff,
            patch_directory=patch_directory,
            train_fraction=0.8,
            patch_size=patch_size,
            overlap=overlap,
            pbar_enabled=True,
        )


if __name__ == "__main__":
    import lovely_tensors
    lovely_tensors.monkey_patch()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fire.Fire(main)