import logging
from dataclasses import astuple
from functools import partial
from pathlib import Path
from typing import Union, Literal, List, Optional, Tuple, Callable, Any

import fire
import torch
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.utils import save_image

from conditional_diffusion import FBPNormalizationType, SamplingMethod, ConditioningType, get_input_dict
from diffusion import Normalization
from diffusion.datasets import get_dataloader_pair
from diffusion.experiment import FileLogger
from diffusion.experiment.masked_sinogram_mixin import MaskedSinogramMixin
from diffusion.functional import float_psnr, Metrics
from diffusion.functional.batch_slice_mask import subsample_mask
from diffusion.models import SinogramConditionedUnet, UNetModel
from diffusion.nn import CircleMask
from diffusion.nn.leap_projector_wrapper import SimpleProjector
from diffusion.nn.pipelines import VariablePipeline
from diffusion.nn.rotations import ExtractIntoRotations
from diffusion.nn.slice_mask import SliceRandomMask


class ActiveLearning(MaskedSinogramMixin):

    def __init__(
        self,
        *,
        dataset_dir: Union[str, Path],
        device: torch.device = "cuda",
        pretrained_model_path: Union[str, Path],
        nll_unet_path: Optional[Union[str, Path]] = None,
        mode: Optional[Literal["nll", "nll_sino", "diffusion", "equidistant"]] = "diffusion",
        image_size: int = 512,
        image_rotation: Union[float, Tuple[float, float]] = 90.0,
        conditioning_type: ConditioningType = ConditioningType.NONE,
        sinogram_n_angles: int = 384,
        fbp_normalization: FBPNormalizationType = FBPNormalizationType.MASK_AMOUNT,
        sinogram_normalization: Normalization = Normalization.MINUS_ONE_ONE_TO_ZERO_ONE,
        sinogram_mask_value: float = 0.,
        sampling_method: SamplingMethod = SamplingMethod.CG,
        sample_dir: Optional[Union[str, Path]] = None,
        n_inference_steps: int = 50,
        n_consistency_steps: int = 20,
        n_samples_per_iteration: int = 10,
    ):

        super().__init__()
        self.file_logger = None
        torch.manual_seed(0)

        match conditioning_type:
            case ConditioningType.NONE:
                self.model_cls = UNetModel
            case ConditioningType.FBP:
                self.model_cls = UNetModel
            case ConditioningType.STACK:
                self.model_cls = SinogramConditionedUnet
                self.extract_into_rotations = ExtractIntoRotations(
                    n_rotations=sinogram_n_angles,
                    size=image_size,
                    device=device,
                    circle_mask_value=sinogram_mask_value
                )
            case _:
                raise NotImplementedError(f'conditioning_type {conditioning_type} not implemented')
        self.conditioning_type = conditioning_type
        train_loader, test_loader = get_dataloader_pair(
            "ImageDataset",
            dataset_dir=Path(dataset_dir),
            device=device,
            batch_size=1,
            test_batch_size=1,
            image_size=image_size,
            rotation=(-image_rotation, image_rotation) if isinstance(image_rotation, float) else image_rotation,
        )
        self.test_loader = test_loader

        self.projector = SimpleProjector(
            device=device,
            nr_angles=sinogram_n_angles,
            image_size=image_size,
            batch_size=1,
        )
        self.circle_mask = CircleMask(
            size=image_size,
            device=device
        )

        self.slice_random_mask = SliceRandomMask(
            keep_min=0,
            keep_max=0,
            device=device,
            mask_value=sinogram_mask_value,
        )
        self.sinogram_normalization = sinogram_normalization
        self.sinogram_mask_value = sinogram_mask_value

        logging.info(f"Loading model checkpoint from {pretrained_model_path}, and compiling.")
        self.model = self.model_cls.from_pretrained(pretrained_model_name_or_path=f"{pretrained_model_path}/unet")
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        torch.set_float32_matmul_precision('high')
        self.model = torch.compile(self.model)

        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path=f"{pretrained_model_path}/noise_scheduler")

        self.pipeline = VariablePipeline(model=self.model, scheduler=self.noise_scheduler)
        self.device = device

        match fbp_normalization:
            case FBPNormalizationType.NONE:
                self.normalize_sparse_fbp = lambda x, **_: x
            case FBPNormalizationType.VANILLA:
                from diffusion.functional.sinogram_normalization import vanilla_normalize
                self.normalize_sparse_fbp = vanilla_normalize
            case FBPNormalizationType.MASK_AMOUNT:
                from diffusion.functional.sinogram_normalization import normalize_by_mask_amount
                self.normalize_sparse_fbp = normalize_by_mask_amount
            case _:
                raise ValueError(f"Normalization type not supported. Found {sinogram_normalization}")

        if sample_dir is not None:
            sample_dir = Path(sample_dir)
        self.sample_dir = sample_dir

        self.metrics = Metrics()

        self.subsample_mask = partial(subsample_mask, generator=torch.Generator(device=device).manual_seed(0), device=device)

        self.n_inference_steps = n_inference_steps
        self.n_consistency_steps = n_consistency_steps

        self.sampling_method = sampling_method

        if mode == "nll_unet":
            assert nll_unet_path is not None
            self.nll_model = UNetModel.from_pretrained(pretrained_model_name_or_path=f"{nll_unet_path}/unet")
        elif mode == "nll_sino":
            assert nll_unet_path is not None
            self.nll_model = UNetModel.from_pretrained(pretrained_model_name_or_path=f"{nll_unet_path}/unet")
        else:
            if nll_unet_path is not None:
                logging.warning(f"Passed nll_unet_path={nll_unet_path}. Will not be used as mode={mode} not in nll_unet or nll_sino")

        self.mode = mode
        self.n_samples_per_iteration = n_samples_per_iteration

    def find_farthest_point(self, mask):

        _mask = mask.clone()[0, 0, :, 0]

        _mask = _mask.float()

        indices = torch.nonzero(_mask).squeeze()

        positions = torch.arange(len(_mask), device=_mask.device)

        boundary_positions = torch.tensor([-1, len(_mask)], device=_mask.device)
        all_positions = torch.cat([boundary_positions, indices])

        distances = torch.abs(positions.unsqueeze(1) - all_positions.unsqueeze(0))

        min_distances = distances.min(dim=1)[0]

        farthest_idx = torch.argmax(min_distances)

        return farthest_idx

    def compute_uncertainty(
        self,
        samples: List[Tensor],
        mode: str = "sino-std"
    ):

        match mode:
            case "std":
                samples = torch.stack(samples, dim=0)
                std = torch.std(samples, dim=0)
                return self.projector(std)

            case "var":
                samples = torch.stack(samples, dim=0)
                var = torch.var(samples, dim=0)
                return self.projector(var)

            case "sino-std":
                for i, sample in enumerate(samples):
                    samples[i] = self.projector(sample)
                samples = torch.stack(samples, dim=0)
                return torch.std(samples, dim=0)

            case "sino-var":
                for i, sample in enumerate(samples):
                    samples[i] = self.projector(sample)
                samples = torch.stack(samples, dim=0)
                return torch.var(samples, dim=0)

            case _:
                raise NotImplementedError(f"Mode {mode} not supported. Please choose from ['std']")


    def __get_fbp(
        self,
        *,
        masked_sinogram: Tensor,
        mask_amount: Tensor,
    ):
        fbp = self.projector.fbp(masked_sinogram.sample)
        fbp = self.circle_mask(fbp.clamp(0., 1.0), mask_value=0.)
        return self.normalize_sparse_fbp(x=fbp, mask_amount=mask_amount).detach().clone()

    def __get_stack(
        self,
        *,
        masked_sinogram: Tensor,
    ):
        return self.extract_into_rotations(masked_sinogram)

    def __get_model_input(
        self,
        masked_sinogram: Tensor,
        mask_amount: Tensor,
    ) -> Callable:
        match self.conditioning_type:
            case ConditioningType.NONE:
                model_input = partial(get_input_dict, conditioning=None)
            case ConditioningType.FBP:
                fbp = self.projector.fbp(masked_sinogram)
                fbp = self.circle_mask(fbp.clamp(0., 1.0), mask_value=0.)
                fbp = self.normalize_sparse_fbp(x=fbp, mask_amount=mask_amount)

                model_input = partial(get_input_dict, conditioning=fbp.detach())
            case ConditioningType.STACK:
                extracted = self.extract_into_rotations(masked_sinogram)
                model_input = partial(get_input_dict, conditioning=extracted.detach())
            case _:
                raise NotImplementedError
        return model_input

    def __run_inference(
        self,
        image_shape,
        model_input: Callable,
        masked_sinogram: Tensor,
        mask: Tensor,
    ):
        match self.sampling_method:
            case SamplingMethod.VANILLA:
                outputs = self.pipeline.vanilla(
                    target_shape=image_shape,
                    device=self.device,
                    model_input=model_input,
                    n_inference_steps=self.n_inference_steps or 50,
                )
            case SamplingMethod.DPS:
                outputs = self.pipeline.dps(
                    target_shape=image_shape,
                    device=self.device,
                    model_input=model_input,
                    masked_sinogram=masked_sinogram,
                    mask=mask,
                    projector=self.projector,
                    n_inference_steps=self.n_inference_steps or 1000,
                )
            case SamplingMethod.CG:
                outputs = self.pipeline.cg(
                    target_shape=image_shape,
                    device=self.device,
                    model_input=model_input,
                    masked_sinogram=masked_sinogram,
                    mask=mask,
                    projector=self.projector,
                    n_inference_steps=self.n_inference_steps or 50,
                    n_consistency_steps=self.n_consistency_steps or 20,
                )
            case _:
                raise NotImplementedError(f'sampling_method {self.sampling_method} not implemented')
        outputs = outputs.clamp(-1., 1.).add(1.).mul(.5)
        outputs = self.circle_mask(outputs, mask_value=0.)
        return outputs

    def __log_metrics(
        self,
        outputs: Tensor,
        target: Tensor,
        index: int,
        n_angles: Union[str, int],
        sample: int
    ) -> None:
        metrics = self.metrics(outputs, target)
        metrics_dict = [
            {'image_nr': index, 'n_angles': n_angles, 'sample': sample, 'psnr': p.item(), 'ssim': s.item()}
            for i, p, s in zip(range(target.shape[0]), metrics.psnr, metrics.ssim)
        ]
        self.file_logger.log_batch(metrics_dict)

    def __get_next_angle(
        self,
        image_shape,
        model_input: Callable,
        masked_sinogram: Tensor,
        mask: Tensor,
        mask_amount: Tensor,
        index: int,
        step: int,
        target_image: Tensor,
        cg_pred: bool = True
    ) -> Union[int, Tensor]:
        match self.mode:
            case "diffusion":
                samples = []

                for i in range(self.n_samples_per_iteration):
                    outputs = self.__run_inference(
                        image_shape=image_shape,
                        model_input=model_input,
                        masked_sinogram=masked_sinogram,
                        mask=mask,
                    )
                    save_image(outputs, f"{self.sample_dir}/{index}-{step}-{i}.png")

                    self.__log_metrics(outputs, target_image, index, step, step)

                    samples.append(outputs)

                uncertainty = self.compute_uncertainty(samples)
                uncertainty = uncertainty * (1 - mask) + 0. * mask

                plt.imsave(
                    f"{self.sample_dir}/{index}-{step}-uncertainty.png",
                    uncertainty[0, 0].clone().cpu().numpy(),
                    vmin=0,
                    vmax=0.01
                )
                uncertainty_per_angle = uncertainty.mean(dim=-1)[0, 0]
                highest_uncertainty_angle = torch.argsort(uncertainty_per_angle, descending=True)[0]
                return highest_uncertainty_angle
            case "equidistant":
                outputs = self.__run_inference(
                    image_shape=image_shape,
                    model_input=model_input,
                    masked_sinogram=masked_sinogram,
                    mask=mask,
                )
                self.__log_metrics(outputs, target_image, index, step, step)
                save_image(outputs, f"{self.sample_dir}/{index}-{step}-{0}.png")
                return self.find_farthest_point(mask=mask)
            case "nll_sino":
                fbp = self.__get_fbp(masked_sinogram=masked_sinogram, mask_amount=mask_amount)
                outputs = self.nll_model(sample=fbp).sample
                if cg_pred:
                    _, log_var = outputs.chunk(2, dim=1)
                    outputs = self.__run_inference(
                        image_shape=image_shape,
                        model_input=model_input,
                        masked_sinogram=masked_sinogram,
                        mask=mask,
                    )
                else:
                    outputs, log_var = outputs.chunk(2, dim=1)
                var = torch.exp(log_var).clamp(1e-6, 1e3)
                var = self.projector(var)
                var = var * (1 - mask) + 0. * mask

                uncertainty_per_angle = var.mean(dim=-1)[0, 0]
                highest_uncertainty_angle = torch.argsort(uncertainty_per_angle, descending=True)[0]

                plt.imsave(
                    f"{self.sample_dir}/{index}-{step}-uncertainty.png",
                    var[0, 0].clone().cpu().numpy(),
                    vmin=0,
                    vmax=0.01
                )
                self.__log_metrics(outputs, target_image, index, step, step)
                save_image(outputs, f"{self.sample_dir}/{index}-{step}-{0}.png")

                return highest_uncertainty_angle
            case "nll":
                outputs_sino = self.nll_model(
                    sample=masked_sinogram,
                ).sample
                _, log_var = outputs_sino.chunk(2, dim=1)
                outputs = self.__run_inference(
                    image_shape=image_shape,
                    model_input=model_input,
                    masked_sinogram=masked_sinogram,
                    mask=mask,
                )
                var = torch.exp(log_var).clamp(1e-5, 1e3)
                var = var * (1 - mask) + 0. * mask

                uncertainty_per_angle = var.mean(dim=-1)[0, 0]
                highest_uncertainty_angle = torch.argsort(uncertainty_per_angle, descending=True)[0]

                plt.imsave(
                    f"{self.sample_dir}/{index}-{step}-uncertainty.png",
                    var[0, 0].clone().cpu().numpy(),
                    vmin=0,
                    vmax=0.01
                )
                self.__log_metrics(outputs, target_image, index, step, step)
                save_image(outputs, f"{self.sample_dir}/{index}-{step}-{0}.png")

                return highest_uncertainty_angle
            case _:
                raise NotImplementedError

    def run_batch(
        self,
        index: int,
        batch: Any,
        starting_n_angles: int = 12,
        target_n_angles: int = 60,
    ):
        n_iterations = target_n_angles - starting_n_angles
        with torch.no_grad():
            image = batch["image"].to(self.device)
            target_image = image.clone().add(1.).mul(.5)
            masked_sinogram, sinogram, mask, mask_amount = astuple(self.fixed_sparsity_sinogram(
                x=image,
                keep_n_angles=target_n_angles,
                return_dict=True,
            ))

        model_input = self.__get_model_input(
            masked_sinogram=masked_sinogram,
            mask_amount=mask_amount,
        )

        outputs = self.__run_inference(
            image_shape=image.shape,
            model_input=model_input,
            masked_sinogram=masked_sinogram,
            mask=mask,
        )

        self.__log_metrics(outputs, target_image, index, 'baseline', 0)

        save_image(outputs, f"{self.sample_dir}/{index}-baseline.png")

        mask = self.subsample_mask(mask, n_iterations)
        masked_sinogram, mask, mask_amount = astuple(
            self.slice_random_mask.apply_mask(sinogram.clone(), mask, return_dict=True)
        )

        model_input = self.__get_model_input(
            masked_sinogram=masked_sinogram,
            mask_amount=mask_amount,
        )

        for step in range(n_iterations):

            next_angle = self.__get_next_angle(
                image_shape=image.shape,
                model_input=model_input,
                masked_sinogram=masked_sinogram,
                mask=mask,
                mask_amount=mask_amount,
                index=index,
                step=step+starting_n_angles,
                target_image=target_image,
            )

            mask[:, :, next_angle, :] = 1.

            save_image(mask, f"{self.sample_dir}/{index}-{step+starting_n_angles}-mask.png")

            masked_sinogram, mask, mask_amount = astuple(self.slice_random_mask.apply_mask(sinogram.clone(), mask, return_dict=True))

            model_input = self.__get_model_input(
                masked_sinogram=masked_sinogram,
                mask_amount=mask_amount,
            )

        outputs = self.__run_inference(
            image_shape=image.shape,
            model_input=model_input,
            masked_sinogram=masked_sinogram,
            mask=mask,
        )

        save_image(outputs, f"{self.sample_dir}/{index}-final.png")

        self.__log_metrics(outputs, target_image, index, 'final', 0)

        return target_image, outputs, mask

    def run(
        self,
        n_samples: int = 10,
        starting_n_angles: int = 12,
        target_n_angles: int = 60,
    ):
        from torch.utils.data import Subset, DataLoader
        self.sample_dir.mkdir(exist_ok=False, parents=True)

        self.file_logger = FileLogger(
            filename=self.sample_dir / "metrics.csv",
            fieldnames=['image_nr', 'n_angles', 'sample', 'psnr', 'ssim'],
        )

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        indices = torch.randperm(len(self.test_loader))[:n_samples]
        dataset = Subset(self.test_loader.dataset, indices)

        eval_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            collate_fn=self.test_loader.collate_fn,
        )

        for i, batch in enumerate(eval_loader):

            self.run_batch(
                index=i,
                batch=batch,
                starting_n_angles=starting_n_angles,
                target_n_angles=target_n_angles,
            )

if __name__ == "__main__":
    """
    Usage example:
    
    HELP:
    python active_learning.py --help
    
    RUN diffusion:
        python active_learning.py --sample-dir="{{sample_dir}}" --dataset_dir="{{dataset_dir}}" --pretrained_model_path="{{model_checkpoint}}" --conditioning_type="{{"none"|"stack"|"fbp"}}" run
        
    RUN diffusion + UNet:
        python active_learning.py --sample-dir="{{sample_dir}}" --dataset_dir="{{dataset_dir}}" --pretrained_model_path="{{model_checkpoint}}" --conditioning_type="{{"none"|"stack"|"fbp"}}" --mode="{{"nll"|"nll_sino"}}" --nll_unet_path=""{{nll_checkpoint}}" run
    """
    import lovely_tensors

    lovely_tensors.monkey_patch()

    fire.Fire(ActiveLearning)