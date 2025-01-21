from pathlib import Path
from typing import Literal, Callable, Union, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):

    split: Literal["train", "test", "val"]
    collate_fn: Callable[[dict[str, Tensor]],dict[str, Tensor]]

    def __init__(
            self,
            *,
            split,
            collate_fn,
    ):
        super(Dataset, self).__init__()
        self.split=split
        self.collate_fn = collate_fn

    def to_dataloader(
            self,
            batch_size: int,
            shuffle: Optional[bool] = None,
            num_workers: int = 0,
            collate_fn: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
            pin_memory: bool = False,
            drop_last: bool = True,
            prefetch_factor: Union[int, None] = None,
            persistent_workers: bool = False
    ):
        return get_dataloader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )

def get_dataloader(
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: Optional[bool] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        prefetch_factor: Union[int, None] = None,
        persistent_workers: bool = False
) -> DataLoader:

    shuffle = shuffle or dataset.split == "train"
    collate_fn = collate_fn or dataset.collate_fn

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        generator=torch.Generator().manual_seed(49)
    )


DatasetClass = Literal["ImageDataset"]

def get_dataloader_pair(
    dataset_class: DatasetClass,
    *,
    dataset_dir: Path,
    device: Literal["cpu", "cuda", "mps"],
    batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    image_size: Optional[int] = None,
    rotation: Tuple[float, float] = (-90, 90),
    circle_mask: bool = True,
    mean: float = 0.5,
    std: float = 0.5
) -> Tuple[DataLoader, DataLoader]:

    match dataset_class:
        case "ImageDataset":
            assert image_size is not None
            assert batch_size is not None

            train_dataset = SafetensorsImageDataset(
                directory=dataset_dir,
                split="train",
                device=device,
                image_size=image_size,
                rotation=rotation,
                circle_mask=circle_mask,
                mean=mean,
                std=std,
            )

            train_loader = train_dataset.to_dataloader(batch_size=batch_size, shuffle=True)

            test_dataset = SafetensorsImageDataset(
                directory=Path(dataset_dir),
                split="test",
                device=device,
                image_size=image_size,
                rotation=rotation,
                circle_mask=circle_mask,
                mean=mean,
                std=std,
            )
            test_batch_size = test_batch_size or batch_size

            test_loader = test_dataset.to_dataloader(batch_size=test_batch_size, shuffle=False)

            return train_loader, test_loader

        case _:
            raise NotImplementedError

from .dicom_to_safetensors import *
from .hd5_to_safetensors import *
from .safetensors_image_dataset import *
from .tiff_to_safetensor import *