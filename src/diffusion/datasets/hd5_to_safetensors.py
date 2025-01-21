from pathlib import Path
from typing import Union

import numpy as np
import torch
from safetensors.torch import save_file


def hd5_3d_to_safetensors_2d(
    path: Union[Path, str],
    out_directory: Union[Path, str],
):
    import h5py
    path = Path(path).with_suffix(".hdf5")

    f = h5py.File(path, 'r')

    images = torch.tensor(np.array(f.get("data")), dtype=torch.float32)
    print(images.shape)
    f.close()

    nr_images = images.size(0)

    if nr_images > 999:
        raise ValueError("Only expecting up to 999 images from same file, code filename change neccessary.")

    for i in range(nr_images):
        save_file(
            {'image': images[i]},
            (Path(out_directory) / f"{path.stem}_{i:03}").with_suffix(".safetensors")
        )
        break
