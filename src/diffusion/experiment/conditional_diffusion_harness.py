import json
import logging
import os
from dataclasses import astuple
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Union, Tuple, Optional, Any, List, Iterable, Literal

import torch
import torch.nn.functional as F
import torchvision
import wandb
from diffusers import get_scheduler, DDPMScheduler
from diffusers.configuration_utils import register_to_config
from torch import Tensor
from torch.optim import AdamW
from tqdm import tqdm
from transformers import SchedulerType

from diffusion import Normalization
from diffusion.datasets import get_dataloader_pair, Dataset
from diffusion.experiment.noise_mixin import NoiseMixin
from diffusion.experiment.base_harness import BaseExperimentHarness
from diffusion.experiment.file_logger import FileLogger
from diffusion.experiment.masked_sinogram_mixin import MaskedSinogramMixin
from diffusion.functional.metrics import Metrics
from diffusion.models import SinogramConditionedUnet, UNetModel
from diffusion.nn import UpBlockType, DownBlockType, CircleMask
from diffusion.nn.leap_projector_wrapper import SimpleProjector
from diffusion.nn.pipelines.variable_pipeline import VariablePipeline
from diffusion.nn.rotations import ExtractIntoRotations
from diffusion.nn.scheduler import GaussianDiffusion
from diffusion.nn.slice_mask import SliceRandomMask

os.environ["WANDB_SILENT"] = "true"

class ConditioningType(str, Enum):
    NONE = "none"
    FBP = "fbp"
    STACK = "stack"

class SamplingMethod(str, Enum):
    VANILLA = "vanilla"
    DPS = "dps"
    CG = "cg"

class FBPNormalizationType(str, Enum):
    NONE = "none"
    VANILLA = "vanilla"
    MASK_AMOUNT = "mask_amount"

def get_input_dict(
    sample: Tensor,
    embeddings: Tuple[Tensor, ...],
    conditioning: Optional[Tensor] = None
):
    if not isinstance(embeddings, Tuple):
        embeddings = (embeddings,)
    if conditioning is None:
        return {
            'sample': sample,
            'embeddings': embeddings,
        }
    else:
        return {
            'sample': sample,
            'embeddings': embeddings,
            'conditioning': conditioning,
        }

class ConditionalDiffusionHarness(BaseExperimentHarness, NoiseMixin, MaskedSinogramMixin):

    @register_to_config
    def __init__(
        self,
        *,
        dataset_dir: Union[str, Path],
        device: torch.device = "cuda",
        batch_size: int = 3,
        image_size: int = 512,
        image_rotation: Union[float, Tuple[float, float]] = 90.0,
        conditioning_type: ConditioningType = ConditioningType.NONE,
        output_dir: Union[str, Path] = None,
        unet_up_blocks: Tuple[UpBlockType, ...] = ("AttnUpBlock", "AttnUpBlock", "ResnetUpBlock", "ResnetUpBlock", "ResnetUpBlock", "ResnetUpBlock"),
        unet_down_blocks: Tuple[DownBlockType, ...] = ("ResnetDownBlock", "ResnetDownBlock", "ResnetDownBlock", "ResnetDownBlock", "AttnDownBlock", "AttnDownBlock"),
        unet_layer_sizes = (64, 64, 128, 128, 256, 256),
        unet_mid_block_num_layers: int = 2,
        sinogram_n_angles: int = 384,
        sinogram_encoder_output_n_channels: Optional[int] = 64,
        fbp_normalization: FBPNormalizationType = FBPNormalizationType.MASK_AMOUNT,
        sinogram_normalization: Normalization = Normalization.MINUS_ONE_ONE_TO_ZERO_ONE,
        sinogram_mask_value: float = 0.,
        project: str = None,
        run: str = None,
        epochs: int = 10,
        visualize_every_n_steps: int = 4000,
        validate_every_n_steps: int = 4000,
        save_checkpoint_every_n_steps: int = 4000,
        n_validation_steps: int = 100,
        ema_decay: float = 0.9999,
        mask_range: Tuple[float, Optional[float]] = (12, None),
        lr: float = 1e-4,
        lr_scheduler: SchedulerType = SchedulerType.CONSTANT,
        lr_scheduler_warmup_steps: int = 0,
        x0_loss_weight: float = 0.0,
        eval_angles: List[int] = None,
        sampling_method: SamplingMethod = SamplingMethod.VANILLA,
        variance_type: Literal["learned_range", "fixed_small"] = "fixed_small",
    ):

        # TODO: fix enum parsing issue with fire
        conditioning_type = ConditioningType(conditioning_type)
        fbp_normalization = FBPNormalizationType(fbp_normalization)
        sinogram_normalization = Normalization(sinogram_normalization)
        lr_scheduler = SchedulerType(lr_scheduler)
        sampling_method = SamplingMethod(sampling_method)

        self.input_prep = None
        train_loader, test_loader = get_dataloader_pair(
            "ImageDataset",
            dataset_dir=Path(dataset_dir),
            device=device,
            batch_size=batch_size,
            test_batch_size=batch_size,
            image_size=image_size,
            rotation=(-image_rotation, image_rotation) if isinstance(image_rotation, float) else image_rotation,
        )

        match conditioning_type:
            case ConditioningType.NONE:
                self.model = UNetModel(
                    sample_size=(image_size, image_size),
                    in_channels=1,
                    out_channels=1,
                    embeddings=("positional",),
                    up_block_types=unet_up_blocks,
                    down_block_types=unet_down_blocks,
                    block_out_channels=unet_layer_sizes,
                    mid_block_num_layers=unet_mid_block_num_layers,
                )
                self.model_cls = UNetModel
            case ConditioningType.FBP:
                self.model = UNetModel(
                    sample_size=(image_size, image_size),
                    in_channels=2,
                    out_channels=1,
                    embeddings=("positional",),
                    up_block_types=unet_up_blocks,
                    down_block_types=unet_down_blocks,
                    block_out_channels=unet_layer_sizes,
                    mid_block_num_layers=unet_mid_block_num_layers,
                )
                self.model_cls = UNetModel
            case ConditioningType.STACK:
                assert sinogram_encoder_output_n_channels is not None, f"sinogram_encoder_output_n_channels must be specified for STACK conditioning"
                self.model = SinogramConditionedUnet(
                    sample_size=(image_size, image_size),
                    in_channels=1,
                    out_channels=1,
                    embeddings=("positional",),
                    up_block_types=unet_up_blocks,
                    down_block_types=unet_down_blocks,
                    block_out_channels=unet_layer_sizes,
                    mid_block_num_layers=unet_mid_block_num_layers,
                    sinogram_n_angles=sinogram_n_angles,
                    sinogram_encoding_dim=sinogram_encoder_output_n_channels,
                )
                self.model_cls = SinogramConditionedUnet
                self.extract_into_rotations = ExtractIntoRotations(
                    n_rotations=sinogram_n_angles,
                    size=image_size,
                    device=device,
                    circle_mask_value=sinogram_mask_value
                )
            case _:
                raise NotImplementedError(f"Conditioning type {conditioning_type} not implemented.")
        self.model.to(device)
        self.device = device
        self.conditioning_type = conditioning_type

        self.set_prepare_inputs(conditioning_type=conditioning_type)

        super(ConditionalDiffusionHarness, self).__init__(
            project=project,
            output_dir=output_dir,
            epochs=epochs,
            run_name=run,
            starting_step=0,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            validate_every_n_steps=validate_every_n_steps,
            visualize_every_n_steps=visualize_every_n_steps,
            save_checkpoint_every_n_steps=save_checkpoint_every_n_steps,
            ema_model=self.model,
            ema_decay=ema_decay,
            limit_validation_steps=n_validation_steps
        )
        if mask_range[1] is None or mask_range[1] < 0:
            mask_range = (mask_range[0], sinogram_n_angles)

        self.projector = SimpleProjector(
            device=device,
            nr_angles=sinogram_n_angles,
            image_size=image_size,
            batch_size=batch_size,
        )
        self.slice_random_mask = SliceRandomMask(
            keep_min=mask_range[0],
            keep_max=mask_range[1],
            device=device,
            mask_value=sinogram_mask_value,
        )
        self.sinogram_normalization = sinogram_normalization
        self.sinogram_mask_value = sinogram_mask_value

        self.circle_mask = CircleMask(
            size=image_size,
            device=device
        )

        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=lr,
        )
        self.lr = lr

        self.lr_scheduler = get_scheduler(
            name=lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=lr_scheduler_warmup_steps,
            num_training_steps=self.total_steps - self.starting_step,
            num_cycles=1
        )
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_warmup_steps = lr_scheduler_warmup_steps
        self.x0_loss_weight = x0_loss_weight

        self.metrics = Metrics()
        self.sinogram_n_angles = sinogram_n_angles
        if eval_angles is None:
            eval_angles = [20, 30, 40, 50, 75, 100, self.sinogram_n_angles]
        self.eval_angles = eval_angles

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.noise_scheduler_cls = DDPMScheduler

        if variance_type == "learned_range":
            self.gaussian_diffusion = GaussianDiffusion(num_train_timesteps=1000)
        self.variance_type = variance_type

        self.sampling_method = sampling_method

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

    def save_model(
        self,
        path: Path,
    ):

        self.model.save_pretrained(save_directory=path / "unet")
        self.noise_scheduler.save_pretrained(save_directory=path / "noise_scheduler")

    def load_model(
            self,
            path: Path
    ):

        self.model = self.model_cls.from_pretrained(pretrained_model_name_or_path=path / "unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path=path / "noise_scheduler")

        self.model.to(self.device)

        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.lr,
        )

        self.lr_scheduler = get_scheduler(
            name=self.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_scheduler_warmup_steps,
            num_training_steps=self.total_steps - self.starting_step,
            num_cycles=1
        )

    def unconditional_training_input_prep(self, image: Tensor):

        noise_output = self.add_noise(image, device=self.device)

        sinogram_output = self.rand_masked_sinogram(
            x=image,
            return_dict=True
        )

        model_input = get_input_dict(
            sample=noise_output.noisy_x,
            embeddings=(noise_output.timesteps,)
        )

        return model_input, noise_output, sinogram_output

    def __get_fbp(
        self,
        *,
        masked_sinogram: Tensor,
        mask_amount: Tensor,
    ):
        fbp = self.projector.fbp(masked_sinogram.sample)
        fbp = self.circle_mask(fbp.clamp(0., 1.0), mask_value=0.)
        return self.normalize_sparse_fbp(x=fbp, mask_amount=mask_amount).detach().clone()

    def fbp_training_input_prep(self, image: Tensor):

        noise_output = self.add_noise(image, device=self.device)

        sinogram_output = self.rand_masked_sinogram(
            x=image,
            return_dict=True
        )
        fbp = self.__get_fbp(
            masked_sinogram=sinogram_output.sample,
            mask_amount=sinogram_output.mask_amount,
        )
        model_input = get_input_dict(
            sample=noise_output.noisy_x,
            embeddings=(noise_output.timesteps, ),
            conditioning=fbp.detach()
        )
        return model_input, noise_output, sinogram_output

    def __get_stack(
        self,
        *,
        masked_sinogram: Tensor,
    ):
        return self.extract_into_rotations(masked_sinogram)

    def stack_input_prep(self, image: Tensor):

        noise_output = self.add_noise(image, device=self.device)

        sinogram_output = self.rand_masked_sinogram(
            x=image,
            return_dict=True
        )

        extracted = self.__get_stack(masked_sinogram=sinogram_output.sample)

        model_input = get_input_dict(
            sample=noise_output.noisy_x,
            embeddings=(noise_output.timesteps, ),
            conditioning=extracted.detach()
        )
        return model_input, noise_output, sinogram_output

    def set_prepare_inputs(self, conditioning_type: ConditioningType):
        match conditioning_type:
            case ConditioningType.NONE:
                self.input_prep = self.unconditional_training_input_prep
                logging.info(f"Using unconditional diffusion model")
            case ConditioningType.FBP:
                self.input_prep = self.fbp_training_input_prep
                logging.info(f"Using FBP conditioned diffusion model")
            case ConditioningType.STACK:
                self.input_prep = self.stack_input_prep
                logging.info(f"Using stack conditioned diffusion model")
            case _:
                raise NotImplementedError(f'Unknown conditioning type: {conditioning_type}')

    def __compute_losses(
        self,
        *,
        model_output: Tensor,
        image: Optional[Tensor] = None,
        noise_output,
        sinogram_output,
    ):
        if self.variance_type == "learned_range":
            loss_dict = self.gaussian_diffusion.training_losses(
                model=lambda *_: model_output,
                x_start=image,
                t=noise_output.timesteps,
                noise=noise_output.noise,
                x_t=noise_output.noisy_x
            )
            loss = loss_dict['loss'].mean()
            vb_loss = loss_dict['vb'].mean()
        else:
            loss = F.mse_loss(model_output, noise_output.noise, reduction="mean")
            vb_loss = None

        if self.x0_loss_weight > 0:
            x0_hat = torch.zeros_like(model_output, device=self.device)

            for i in range(model_output.shape[0]):
                x0_hat[i] = self.noise_scheduler.step(
                    model_output[i].unsqueeze(0),
                    noise_output.timesteps[i],
                    noise_output.noisy_x[i]
                ).pred_original_sample.squeeze(0)

            x0_hat_sino = self.compute_sinogram(x0_hat)
            x0_hat_sino = (
                x0_hat_sino * sinogram_output.mask
                + (1 - sinogram_output.mask) * self.sinogram_mask_value
            )

            sinogram_loss = F.mse_loss(x0_hat_sino, sinogram_output.sample, reduction="mean")
            sinogram_loss = self.x0_loss_weight * sinogram_loss

            loss += sinogram_loss

            return loss, sinogram_loss, vb_loss
        else:
            return loss, None, vb_loss

    def train_step(self, step: int, batch: Any):

        image = batch["image"].to(self.device)
        with torch.no_grad():
            model_inputs, noise_output, sinogram_output = self.input_prep(image)

        model_output = self.model(**model_inputs).sample
        loss, sinogram_loss, vb_loss = self.__compute_losses(
            model_output=model_output,
            noise_output=noise_output,
            sinogram_output=sinogram_output
        )
        loss_dict = {
            'training_loss': loss.item(),
        }
        if sinogram_loss is not None:
            loss_dict['training_x0_hat_loss'] = sinogram_loss.item()
        if vb_loss is not None:
            loss_dict['training_vb_loss'] = vb_loss.item()

        wandb.log(loss_dict, step)

        return loss

    def validation_step(self, step: int, batch: Any):
        image = batch["image"].to(self.device)

        model_inputs, noise_output, sinogram_output = self.input_prep(image)

        model_output = self.model(**model_inputs).sample
        loss, sinogram_loss, vb_loss = self.__compute_losses(
            model_output=model_output,
            noise_output=noise_output,
            sinogram_output=sinogram_output
        )
        loss_dict = {}
        if sinogram_loss is not None:
            loss_dict['test_x0_hat_loss'] = sinogram_loss.item()
        if vb_loss is not None:
            loss_dict['test_vb_loss'] = vb_loss.item()

        wandb.log(loss_dict, step)
        return loss


    def __run_inference(
        self,
        image: Tensor,
        pipeline: VariablePipeline,
        angle: int,
        n_inference_steps: Optional[int] = None,
        n_consistency_steps: Optional[int] = None,
        cg_mask: Optional[Tensor] = None,
    ):
        masked_sinogram, sinogram, mask, mask_amount = astuple(self.fixed_sparsity_sinogram(
            x=image.clone(),
            sinogram=None,
            keep_n_angles=angle,
            return_dict=True
        ))

        conditioning = (
            self.__get_fbp(masked_sinogram=masked_sinogram,
                           mask_amount=mask_amount).detach() if self.conditioning_type == ConditioningType.FBP else (
                self.__get_stack(
                    masked_sinogram=masked_sinogram).detach() if self.conditioning_type == ConditioningType.STACK else None
            )
        )
        model_input = partial(get_input_dict, conditioning=conditioning)

        match self.sampling_method:
            case SamplingMethod.VANILLA:
                outputs = pipeline.vanilla(
                    target_shape=image.shape,
                    device=self.device,
                    model_input=model_input,
                    n_inference_steps=n_inference_steps or 50,
                )
            case SamplingMethod.DPS:
                outputs = pipeline.dps(
                    target_shape=image.shape,
                    device=self.device,
                    model_input=model_input,
                    masked_sinogram=masked_sinogram,
                    mask=mask,
                    projector=self.projector,
                    n_inference_steps=n_inference_steps or 1000,
                )
            case SamplingMethod.CG:
                outputs = pipeline.cg(
                    target_shape=image.shape,
                    device=self.device,
                    model_input=model_input,
                    masked_sinogram=masked_sinogram,
                    mask=mask,
                    projector=self.projector,
                    n_inference_steps=n_inference_steps or 50,
                    n_consistency_steps=n_consistency_steps or 20,
                    cg_mask=cg_mask,
                )
            case _:
                raise NotImplementedError(f"Unknown sampling method: {self.sampling_method}")

        outputs = outputs.clamp(-1., 1.).add(1.).mul(.5)
        outputs = self.circle_mask(outputs, mask_value=0.)

        return outputs

    def visualize(self, step: int, batch: Any):
        if self.variance_type == "fixed_small":
            scheduler = self.noise_scheduler_cls()
        else:
            scheduler = self.noise_scheduler_cls(variance_type="learned_range")
        pipeline = VariablePipeline(model=self.model, scheduler=scheduler)

        image = batch["image"].to(self.device)
        image_zero_one = image.add(1.).mul(.5)

        for angle in self.eval_angles:

            outputs = self.__run_inference(
                image=image,
                pipeline=pipeline,
                angle=angle,
            )

            metrics = self.metrics(outputs, image_zero_one)

            combined = torch.cat((outputs, image_zero_one, (outputs - image_zero_one).abs().mul(.5)), dim=3)

            images = [wandb.Image(combined[i], caption=f"{i}") for i in range(combined.shape[0])]

            wandb.log({
                f'samples_{angle}': images,
                f'psnr_{angle}': metrics.psnr.mean().item(),
                f'ssim_{angle}': metrics.ssim.mean().item(),
            }, step=step)

    def evaluate(
        self,
        *,
        sampling_method: SamplingMethod = SamplingMethod.VANILLA,
        n_batches: int = 10,
        dataset: Optional[Dataset] = None,
        angles: Optional[Iterable[int]] = None,
        sample_dir: Optional[Union[str, Path]] = None,
        save_samples: bool = True,
        step: Optional[int] = None,
        checkpoint: Optional[str] = None,
        n_consistency_steps: Optional[int] = None,
        n_inference_steps: Optional[int] = None,
        drop_last_n_cg_steps: int = 0,
    ):
        # TODO: fix
        sampling_method = SamplingMethod(sampling_method)

        assert step is not None or checkpoint is not None
        self.load_checkpoint(directory=checkpoint, step=step)

        from torch.utils.data import Subset, DataLoader
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        if dataset is None:
            indices = torch.randperm(len(self.test_loader))[:n_batches * self.config.batch_size]
            dataset = Subset(self.test_loader.dataset, indices)

        if angles is None:
            angles = self.eval_angles

        eval_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            collate_fn=self.test_loader.collate_fn,
        )

        self.ema_model.apply_shadow()

        logging.info(
            f"Running evaluation for angles {angles}, with sampling method {sampling_method}."
            f"Test set size: {len(eval_loader) * self.batch_size}."
        )

        progress_bar = tqdm(
            total=len(eval_loader) * len(angles),
            desc="Evaluate",
        )

        pipeline = VariablePipeline(model=self.model, scheduler=self.noise_scheduler)

        to_image = torchvision.transforms.ToPILImage(mode="L")

        if sample_dir is None:
            sample_dir = Path(self.output_dir) / "samples"

        sample_dir = Path(sample_dir)

        sample_dir.mkdir(parents=True, exist_ok=False)

        conf = self.config
        conf['eval_sampling_method'] = sampling_method
        conf['step'] = step
        conf['checkpoint'] = checkpoint
        conf['n_consistency_steps'] = n_consistency_steps
        conf['n_inference_steps'] = n_inference_steps
        conf['drop_last_n_cg_steps'] = drop_last_n_cg_steps

        with open(sample_dir / "config.json", "w") as f:
            json.dump(conf, f)

        file_logger = FileLogger(
            filename=sample_dir / "metrics.csv",
            fieldnames=['n_angles', 'sample', 'psnr', 'ssim'],
        )

        if drop_last_n_cg_steps > 0:
            cg_mask = torch.ones((n_inference_steps,), device=self.device)
            cg_mask[:-drop_last_n_cg_steps] = 0
        else:
            cg_mask = None

        for angle in angles:

            self.slice_random_mask.reset_rng()

            for index, batch in enumerate(eval_loader):

                image = batch["image"].to(self.device)

                outputs = self.__run_inference(
                    image=image,
                    pipeline=pipeline,
                    angle=angle,
                    n_inference_steps=n_inference_steps,
                    n_consistency_steps=n_consistency_steps,
                    cg_mask=cg_mask,
                )

                image_zero_one = image.add(1.).mul(.5)
                metrics = self.metrics(outputs, image_zero_one)

                metrics_dict = [
                    {'n_angles': angle, 'sample': index * image.shape[0] + i, 'psnr': p.item(), 'ssim': s.item()}
                    for i, p, s in zip(range(image.shape[0]), metrics.psnr, metrics.ssim)
                ]
                file_logger.log_batch(metrics_dict)

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        'psnr': metrics.psnr.mean().item(),
                        'ssim': metrics.ssim.mean().item(),
                    }
                )

                if save_samples:
                    for b in range(image.shape[0]):

                        pred_image = to_image(outputs[b])
                        pred_image.save(f"{str(sample_dir)}/pred-{angle}-{index * image.shape[0] + b}.png")

                        gt_image = to_image(image_zero_one[b])
                        gt_image.save(f"{str(sample_dir)}/gt-{angle}-{index * image.shape[0] + b}.png")

            file_logger.flush()
        file_logger.flush()
        progress_bar.close()
