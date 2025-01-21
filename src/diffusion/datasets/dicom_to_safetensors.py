import logging
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import safetensors.torch
import torch


def dicom_to_safetensors(
    file_path: Union[Path, str],
    out_filename: Union[Path, str],
    clip_hu: Optional[Tuple[float, float]] = (-1000, 3000)
):
    import pydicom
    file_path = Path(file_path).with_suffix(".dcm")

    dicom = pydicom.dcmread(file_path)

    rescale_slope = dicom.RescaleSlope
    rescale_intercept = dicom.RescaleIntercept

    pixel_array = torch.from_numpy(dicom.pixel_array.astype(np.float32))

    pixel_array = pixel_array * rescale_slope + rescale_intercept

    if clip_hu is not None:
        pixel_array = pixel_array.clip(min=clip_hu[0], max=clip_hu[1])

    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())

    if pixel_array.ndim == 2:
        if pixel_array.shape != (512, 512):
            print(pixel_array.shape)
            return

        metadata = {
            'instance_number': str(getattr(dicom, 'InstanceNumber', -1)),
            'patient_id': str(getattr(dicom, 'PatientID', 'unknown')),
            'study_id': str(getattr(dicom, 'StudyID', 'unknown')),
            'series_id': str(getattr(dicom, 'SeriesInstanceUID', 'unknown')),
            'filename': str(file_path),
            'slice_number': str(-1)
        }

        out_filename = Path(out_filename).with_suffix(".safetensors")

        safetensors.torch.save_file(
            {
                'image': pixel_array,
            },
            out_filename,
            metadata,
        )
    elif pixel_array.ndim == 3:
        if pixel_array.shape[1] != 512 or pixel_array.shape[2] != 512:
            print(pixel_array.shape)
            return

        for i in range(pixel_array.shape[0]):
            metadata = {
                'instance_number': str(getattr(dicom, 'InstanceNumber', -1)),
                'patient_id': str(getattr(dicom, 'PatientID', 'unknown')),
                'study_id': str(getattr(dicom, 'StudyID', 'unknown')),
                'series_id': str(getattr(dicom, 'SeriesInstanceUID', 'unknown')),
                'filename': str(file_path),
                'slice_number': str(f'{i:04d}')
            }
            out_filename = (Path(f"{out_filename}-{i}")).with_suffix(".safetensors")

            safetensors.torch.save_file(
                {
                    'image': pixel_array[i],
                },
                out_filename,
                metadata,
            )
    else:
        return


def dicom_to_safetensors_mri(
    file_path: Union[Path, str],
    out_filename: Union[Path, str],
    transforms
):
    import pydicom
    file_path = Path(file_path).with_suffix(".dcm")

    dicom = pydicom.dcmread(file_path)
    try:
        image_array = dicom.pixel_array

        # Convert to float for processing
        image_array = image_array.astype(float)
        image = torch.tensor(image_array)

        window_center = dicom.WindowCenter
        window_width = dicom.WindowWidth
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2

        normalized = image.clamp(window_min, window_max)
        normalized = (normalized - window_min) / (window_max - window_min)

        normalized = transforms(normalized.unsqueeze(0))
        metadata = {
            'instance_number': str(getattr(dicom, 'InstanceNumber', -1)),
            'patient_id': str(getattr(dicom, 'PatientID', 'unknown')),
            'study_id': str(getattr(dicom, 'StudyID', 'unknown')),
            'series_id': str(getattr(dicom, 'SeriesInstanceUID', 'unknown')),
            'filename': str(file_path),
            'slice_number': str(-1)
        }
        out_filename = Path(out_filename).with_suffix(".safetensors")

        safetensors.torch.save_file(
            {
                'image': normalized,
            },
            out_filename,
            metadata,
        )
    except Exception as e:
        logging.warning(e)


if __name__ == "__main__":

    import uuid
    from safetensors.torch import safe_open

    name = uuid.uuid4()
    dicom_to_safetensors(
        file_path="/mnt/b/kits_source_data/C4KC-KiTS/KiTS-00209/10-23-2003-NA-abdomenpelvisw-99291/300.000000-Segmentation-18083/1-1.dcm",
        out_filename=f"{name}",
    )

    with safe_open(f'{name}.safetensors', framework='pt', device="cpu") as f:

        for key in f.keys():
            print(key)

            for k, v in f.metadata().items():
                print(k, v)