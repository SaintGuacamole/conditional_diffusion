from copy import deepcopy
from pathlib import Path
from typing import Union, Optional

import torch
from torch import nn


class EMAModel:

    model: torch.nn.Module
    def __init__(
        self,
        model: nn.Module,
        decay=0.9999,
        warmup_steps: int = 25000,
        warmup_decay: float = 0.999,
        ramp: bool = True,
        use_buffers: bool = True
    ):
        self.model = model
        self.decay = decay
        self.use_buffers = use_buffers

        self.shadow = deepcopy(model)
        self.shadow.eval()

        for param in self.shadow.parameters():
            param.requires_grad_(False)

        if use_buffers:
            for buf in self.shadow.buffers():
                buf.requires_grad_(False)

        self.backup = None
        self.training_mode = model.training
        self.warmup_steps = warmup_steps
        self.warmup_decay = warmup_decay
        self.ramp = ramp


    def update(
        self,
        step: int,
    ):
        with torch.no_grad():
            if step < self.warmup_steps:
                if self.ramp:
                    decay = step / self.warmup_steps * (self.decay - self.warmup_decay) + self.warmup_decay
                else:
                    decay = self.warmup_decay
            else:
                decay = self.decay
            for shadow_param, param in zip(self.shadow.parameters(), self.model.parameters()):
                if param.requires_grad:
                    shadow_param.data.lerp_(param.data, 1 - decay)

            # Update buffers if needed (e.g., BatchNorm stats)
            if self.use_buffers:
                for shadow_buf, buf in zip(self.shadow.buffers(), self.model.buffers()):
                    shadow_buf.data.copy_(buf.data)

    def apply_shadow(self):

        self.backup = deepcopy(self.model)

        # Load shadow params and buffers into original model
        with torch.no_grad():
            for param, shadow_param in zip(self.model.parameters(), self.shadow.parameters()):
                if param.requires_grad:
                    param.data.copy_(shadow_param.data)

            if self.use_buffers:
                for buf, shadow_buf in zip(self.model.buffers(), self.shadow.buffers()):
                    buf.data.copy_(shadow_buf.data)
        self.model.eval()


    def restore(self):

        if self.backup is None:
            raise RuntimeError("No backup exists. Must call apply_shadow first.")

        with torch.no_grad():
            for param, backup_param in zip(self.model.parameters(), self.backup.parameters()):
                if param.requires_grad:
                    param.data.copy_(backup_param.data)

            if self.use_buffers:
                for buf, backup_buf in zip(self.model.buffers(), self.backup.buffers()):
                    buf.data.copy_(backup_buf.data)

        if self.training_mode:
            self.model.train()
        else:
            self.model.eval()

    def ema_forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


    def reinit(self, model):
        """Reinitialize EMA with a new model."""
        self.model = model
        self.shadow = deepcopy(model)
        self.shadow.eval()

        for param in self.shadow.parameters():
            param.requires_grad_(False)

        if self.use_buffers:
            for buf in self.shadow.buffers():
                buf.requires_grad_(False)

        self.backup: Optional[nn.Module] = None
        self.training_mode = model.training

    def save_checkpoint(self, path: Union[Path, str]):

        torch.save({
            'shadow_state_dict': self.shadow.state_dict(),
            'backup_state_dict': self.backup.state_dict() if (self.backup is not None) else None,
            'decay': self.decay,
            'use_buffers': self.use_buffers,
            'training_mode': self.training_mode if hasattr(self, 'training_mode') else True
        }, path)

    def load_checkpoint(self, path: Union[Path, str], model: nn.Module):

        self.model = model

        self.shadow = deepcopy(self.model)
        self.shadow.eval()
        state = torch.load(path, map_location='cpu', weights_only=True)

        # Load shadow model state
        self.shadow.load_state_dict(state['shadow_state_dict'])

        for param in self.shadow.parameters():
            param.requires_grad_(False)

        # Load backup if it exists
        if state['backup_state_dict'] is not None:
            if self.backup is None:
                self.backup = deepcopy(self.model)
            self.backup.load_state_dict(state['backup_state_dict'])
        else:
            self.backup = None

        self.decay = state['decay']
        self.use_buffers = state.get('use_buffers', True)  # Default to True for backward compatibility
        self.training_mode = state.get('training_mode', True)

        if self.use_buffers:
            for buf in self.shadow.buffers():
                buf.requires_grad_(False)
        del state

    @property
    def device(self):
        """Get the device of the shadow model."""
        return next(self.shadow.parameters()).device

    def to(self, device):
        """Move models to specified device."""
        self.shadow = self.shadow.to(device)
        if self.backup is not None:
            self.backup = self.backup.to(device)
        return self