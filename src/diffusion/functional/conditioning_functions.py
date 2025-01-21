from typing import Optional

import torch
from torch import Tensor
from torch.fft import fftshift, fft2

from diffusion.functional import complex_mse
from diffusion.nn.leap_projector_wrapper import SimpleProjector

def conjugate_gradient(
    x,
    target,
    mask: Tensor,
    device,
    projector: SimpleProjector,
    n_steps: int = 100,
    lam=1e5
):
    # CGAlgorithm algorithm
    # https://github.com/wustl-cig/DOLCE/blob/main/dataFidelities/CTClass.py
    z_k = x
    target = target * mask
    sparse_fbp = projector.fbp(target)

    x = torch.zeros_like(z_k, device=device)

    rhs = sparse_fbp + lam * z_k - projector.fbp(projector(x) * mask) + lam * x
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(r.conj() * r)
    while i <= n_steps and rTr > 1e-10:
        Ap = projector.fbp(projector(p) * mask) + lam * p
        alpha = rTr / torch.sum(p.conj() * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj() * r)
        beta = rTrNew / rTr
        p = r + beta * p
        i += 1
        rTr = rTrNew

def alternating_proximal_gradient_method(
    x: Tensor,
    target: Tensor,
    mask: Tensor,
    rho,
    device: torch.device,
    projector: SimpleProjector,
    n_steps: int = 30,
    grad_weight: float = 5e-6
):
    # APGM algorithm
    # https://github.com/wustl-cig/DOLCE/blob/main/dataFidelities/CTClass.py
    x = x.detach().clone()
    z = x.detach().clone()

    s = x
    t = torch.tensor([1.]).float().to(device)
    for _ in range(n_steps):
        projected = projector(s)
        diff = mask * (projected - target)
        grad = projector.fbp(diff)

        xnext = s - grad_weight * grad - rho * (s - z)

        tnext = 0.5 * (1 + torch.sqrt(1 + 4 * t * t))
        s = xnext + ((t - 1) / tnext) * (xnext - x)

        t = tnext
        x = xnext
    return x


def diffusion_posterior_sampling(
    *,
    x_t_minus_one: Tensor,
    measurement: Tensor,
    measurement_mask: Tensor,
    x_t: Tensor,
    x_0_hat: Tensor,
    projector: SimpleProjector,
    scale: float = 0.3
):
    x_0_sino = projector(x_0_hat)
    difference = measurement - x_0_sino
    difference = difference * measurement_mask

    norm = torch.linalg.norm(difference)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t)[0]
    x_t_minus_one = x_t_minus_one - norm_grad * scale

    return x_t_minus_one, norm


def mri_fast_sampling(
    x: Tensor,
    target: Tensor,
    lr: float,
    mask: Optional[Tensor] = None,
    n_steps: int = 20,
):
    x = x.detach().clone().requires_grad_(True)

    if mask is None:
        mask = torch.ones_like(target)

    target = target.detach().clone()

    optim = torch.optim.Adam([x], lr=lr)

    for i in range(n_steps):
        optim.zero_grad()

        k_space_hat = fftshift(fft2(x))

        loss = complex_mse(k_space_hat, target, reduction="none")

        loss = (loss * mask).mean()

        loss.backward()
        optim.step()

    torch.cuda.empty_cache()
    return x.detach().clone()


def fast_sampling(
    x: Tensor,
    target: Tensor,
    lr: float,
    projector: SimpleProjector,
    mask: Optional[Tensor] = None,
    n_steps: int = 20,
):
    x = x.detach().clone().requires_grad_(True)

    if mask is None:
        mask = torch.ones_like(target)

    target = target.detach().clone()

    optim = torch.optim.Adam([x], lr=lr)

    for i in range(n_steps):
        optim.zero_grad()

        x_sino = projector.forward(x)

        loss = torch.nn.functional.mse_loss(x_sino, target, reduction="none")

        loss = (loss * mask).mean()

        loss.backward()
        optim.step()

    torch.cuda.empty_cache()
    return x.detach().clone()
