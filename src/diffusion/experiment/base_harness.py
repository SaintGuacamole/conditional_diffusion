import itertools
import logging
import re
import shutil
import warnings
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Union, Optional

import torch
import wandb
from diffusers import ConfigMixin
from torch import Tensor, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.experiment.ema_harness import EMAModel

class BaseExperimentHarness(ConfigMixin):

    config_name = "config"
    train_loader: DataLoader
    test_loader: DataLoader
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    progress_bar: tqdm

    def __init__(
        self,
        *,
        project: str,
        output_dir: Union[str, Path],
        epochs: int,
        run_name: Optional[str] = None,
        starting_step: int = 0,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        test_batch: Any = None,
        validate_every_n_steps: int = 1000,
        visualize_every_n_steps: int = 1000,
        save_checkpoint_every_n_steps: int = 500,
        device: Literal["cuda", "cpu", "mps"] = "cpu",
        seed: int = 49,
        limit_validation_steps: Optional[int] = None,
        ema_decay: float = 0.99999,
        ema_model: Optional[Any] = None,
        shuffle_visualize: bool = False,
        dtype=torch.float16,
    ):
        torch.manual_seed(seed)
        self.project = project
        if run_name is None:
            run_name = str(datetime.now())
        self.run_name = run_name
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        self.epochs = epochs

        self.train_loader = train_loader
        self.train_iterator = iter(train_loader)

        if starting_step != 0:
            self.load_checkpoint(starting_step)

        self.test_loader = test_loader
        if test_batch is not None:
            self.test_batch = test_batch
        else:
            self.test_batch = next(iter(test_loader))
        if shuffle_visualize:
            self.test_iterator = iter(test_loader)
        self.shuffle_visualize = shuffle_visualize

        self.starting_step = starting_step
        self.total_steps = len(self.train_loader) * epochs
        self.validate_every_n_steps = validate_every_n_steps
        self.visualize_every_n_steps = visualize_every_n_steps
        self.save_checkpoint_every_n_steps = save_checkpoint_every_n_steps
        self.device = device
        self.wandb_id = None
        self.limit_validation_steps = limit_validation_steps

        self.ema_model = None
        if ema_model is not None:
            self.ema_model = EMAModel(model=ema_model, decay=ema_decay)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.dtype = dtype
        self.scaler = GradScaler(device=device)

        self.ssim = None
        self.fid = None

    def _get_next_batch(self):

        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            batch = next(self.train_iterator)

        return batch

    @abstractmethod
    def train_step(
        self,
        step: int,
        batch: Any
    ):
        raise NotImplementedError


    @abstractmethod
    def validation_step(
        self,
        step: int,
        batch,
    ) -> Tensor:
        raise NotImplementedError


    def _validate(
        self,
        step: int
    ):
        with torch.no_grad():
            progress_bar = tqdm(
                self.test_loader,
                desc="Validation",
                disable=True,
            )

            if self.ema_model is not None:
                self.ema_model.apply_shadow()

            test_loss = torch.tensor(0.0, device=self.device)
            count = 0
            for batch in progress_bar:
                test_loss += self.validation_step(step, batch)
                count+=1
                if self.limit_validation_steps is not None and count >= self.limit_validation_steps:
                    break
            test_loss /= count
            wandb.log({"test_loss": test_loss}, step=step)

            if self.ema_model is not None:
                self.ema_model.restore()

        torch.cuda.empty_cache()

    @abstractmethod
    def visualize(
        self,
        step: int,
        batch
    ):

        raise NotImplementedError


    def _visualize(
        self,
        step: int,
        batch,
        apply_ema: bool = True
    ):
        if self.ema_model is not None and apply_ema:
            self.ema_model.apply_shadow()

        self.visualize(step=step, batch=batch)

        if self.ema_model is not None and apply_ema:
            self.ema_model.restore()

        torch.cuda.empty_cache()


    def _do_train_step(
        self,
        step: int,
        batch: Any,
    ):
        logging.debug("_do_train_step")
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            loss = self.train_step(step, batch)
        self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        if self.ema_model is not None:
            self.ema_model.update(
                step=step
            )

        logs = {"step_loss": loss.detach().item()}
        self.progress_bar.set_postfix(**logs)


    def initialize_wandb(self):
        if self.wandb_id is not None:
            _ = wandb.init(
                dir=self.output_dir,
                project=self.project,
                name=self.run_name,
                id=self.wandb_id,
                resume="allow",
                allow_val_change=True,
            )
        else:
            run = wandb.init(
                dir=self.output_dir,
                project=self.project,
                config=self.config,
                name=self.run_name,
            )
            self.wandb_id = run.id
            with open(f"{self.output_dir}/run_id.txt", "w+") as f:
                f.write(self.wandb_id)

    @abstractmethod
    def save_model(
        self,
        path: Union[str, Path]
    ):
        raise NotImplementedError


    @abstractmethod
    def load_model(
        self,
        path: Union[str, Path]
    ):
        raise NotImplementedError


    def load_checkpoint(
        self,
        step: Optional[int],
        resume_wandb: bool = True,
        directory: Optional[Union[str, Path]] = None,
    ):
        if directory is None:
            path = Path(self.output_dir) / f"checkpoint-{step}"
        else:
            path = Path(directory)
            if step is None:
                match = re.search(r'checkpoint-(\d+)', str(path))
                if match:
                    step = int(match.group(1))
                else:
                    raise ValueError(f"{path} does not contain a step number, and no step is provided")

        logging.info(f"Loading checkpoint from {path}")

        self.load_model(path)
        optim_state_dict = torch.load(f"{path}/optimizer.pt", weights_only=True)
        self.optimizer.load_state_dict(optim_state_dict)
        del optim_state_dict

        scheduler_state_dict = torch.load(f"{path}/lr_scheduler.pt", weights_only=True)
        self.lr_scheduler.load_state_dict(scheduler_state_dict)
        del scheduler_state_dict

        rng_state = torch.load(f"{path}/rng.pt", weights_only=True)
        torch.set_rng_state(rng_state)
        del rng_state

        if self.device == "cuda" and (path / "cuda_rng.pt").exists():
            cuda_rng_state = torch.load(f"{path}/cuda_rng.pt", weights_only=True)
            torch.cuda.set_rng_state(cuda_rng_state)
            del cuda_rng_state
            logging.info(f"Loaded cuda rng state from {path}/cuda_rng.pt")

        if self.ema_model is not None:
            self.ema_model.load_checkpoint(f"{path}/ema.pt", self.model)
            logging.info(f"Loaded ema model from {path}/ema.pt")

        step = step + 1
        current_index = step % len(self.train_loader)
        self.train_iterator = itertools.islice(self.train_iterator, current_index, None)
        self.starting_step = step

        if resume_wandb:
            if (Path(self.output_dir) / "run_id.txt").exists():
                with open(f"{self.output_dir}/run_id.txt", "r") as f:
                    self.wandb_id = f.read()
            else:
                warnings.warn(f"Could not find wand id file at {self.output_dir}/run_id.txt. Starting new wand run")
        else:
            self.wandb_id = None

        logging.info(f"Loaded checkpoint from {path}")

    def _save_checkpoint(
            self,
            step: int,
            keep_last: int = 10
    ):
        path = Path(self.output_dir) / f"checkpoint-{step}"
        self.save_model(path)
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
        torch.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
        torch.save(torch.get_rng_state(), f"{path}/rng.pt")

        if self.device == "cuda":
            torch.save(torch.cuda.get_rng_state(), f"{path}/cuda_rng.pt")

        if self.ema_model is not None:
            self.ema_model.save_checkpoint(f"{path}/ema.pt")

        checkpoints = sorted(Path(self.output_dir).glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[-1]))
        if len(checkpoints) > keep_last:
            checkpoints_to_delete = checkpoints[:-keep_last]  # Keep the last 'keep_last' checkpoints
            for checkpoint in checkpoints_to_delete:
                shutil.rmtree(checkpoint)  # Remove the checkpoint directory
                print(f"Deleted {checkpoint}")

    def train(
        self,
        step: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ):
        if step is not None:
            assert checkpoint is None, f"Either provide a step or checkpoint argument, not both"
            self.load_checkpoint(step=step)
        if checkpoint is not None:
            assert step is None, f"Either provide a step or checkpoint argument, not both"
            self.load_checkpoint(directory=checkpoint)
        self.initialize_wandb()

        self.progress_bar = tqdm(
            initial=self.starting_step,
            total=self.total_steps,
            desc="Steps",
        )

        for step in range(self.starting_step, self.total_steps):

            batch = self._get_next_batch()
            self._do_train_step(step, batch)

            if step % self.save_checkpoint_every_n_steps == 0 and step != 0:
                self._save_checkpoint(step)

            if step % self.visualize_every_n_steps == 0 and step != 0:
                if self.shuffle_visualize:
                    try:
                        self.test_batch = next(self.test_iterator)
                    except StopIteration:
                        self.test_iterator = iter(self.test_loader)
                        self.test_batch = next(self.test_iterator)

                self._visualize(step, self.test_batch)

            if step % self.validate_every_n_steps == 0 and step != 0:
                self._validate(step)

            self.progress_bar.update(1)

        self.progress_bar.close()
        wandb.finish()
        logging.info("Finished training")

        if self.ema_model is not None:
            self.ema_model.apply_shadow()
        self._save_checkpoint(self.total_steps)
