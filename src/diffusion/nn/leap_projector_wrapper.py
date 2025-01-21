
import leaptorch
import torch
from torch import nn, Tensor

from diffusion.nn import NormalizationType
from diffusion.nn.circle_mask import CircleMask


class Projector(nn.Module):

    def __init__(
            self,
            *,
            batch_size: int,
            device: str,
            nr_angles: int,
            cols: int,
            rows: int = 1,
            pixel_size: int = 1,
            angular_range: int = 180,
            circle_mask: bool = False,
            normalization: NormalizationType = None,
            forward_project: bool = True,
            use_static: bool = True,
    ):
        super(Projector, self).__init__()

        if device == "cuda":
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)

        leap_projector = leaptorch.Projector(
            forward_project=forward_project,
            use_static=use_static,
            use_gpu=device != "cpu",
            gpu_device=device,
            batch_size=batch_size
        )

        leap_projector.leapct.set_parallelbeam(
            numAngles=nr_angles,
            numRows=rows,
            numCols=cols,
            pixelHeight=pixel_size,
            pixelWidth=pixel_size,
            centerRow=0.5 * (rows - 1),
            centerCol=0.5 * (cols - 1),
            phis=leap_projector.leapct.setAngleArray(
                numAngles=nr_angles,
                angularRange=angular_range
            )
        )

        leap_projector.leapct.set_default_volume()
        leap_projector.allocate_batch_data()

        self.leap_projector = leap_projector

        self.circle_mask = None
        if circle_mask:
            self.circle_mask = CircleMask(
                size=cols
            )

        match normalization:
            case NormalizationType.ZERO_TO_ONE:
                self.normalization = lambda x: x / cols
            case NormalizationType.MINUS_ONE_TO_ONE:
                self.normalization = lambda x: x / cols * 2. - 1.
            case _:
                self.normalization = lambda x: x

        self.cols = cols
        self.batch_size = batch_size

    def forward(self, x: Tensor) -> Tensor:
        if self.circle_mask is not None:
            x = self.circle_mask(x)
        x = self.leap_projector(x)
        x = self.normalization(x)
        return x

    def fbp(self, x: Tensor) -> Tensor:
        return self.leap_projector.fbp(x)

    def fbp_scaled(self, x: Tensor) -> Tensor:
        return self.leap_projector.fbp(x * self.cols)

    def fbp_channel_wise_scaled(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()

        x = (x.reshape(b * c, h, 1, w) * w)

        fbp = self.leap_projector.fbp(x)
        fbp = fbp[:b * c]

        fbp = fbp.reshape(b, c, w, w)

        return fbp


class SimpleProjector(leaptorch.BaseProjector):
    def __init__(
        self,
        *,
        device: str,
        nr_angles: int,
        image_size: int,
        batch_size: int = 1
    ):
        if device == "cuda":
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)

        super(SimpleProjector, self).__init__(
            use_static=True,
            use_gpu=device != "cpu",
            gpu_device=device,
            batch_size=batch_size
        )


        self.leapct.set_parallelbeam(
            numAngles=nr_angles,
            numRows=1,
            numCols=image_size,
            pixelHeight=1,
            pixelWidth=1,
            centerRow=0.,
            centerCol=0.5 * (image_size - 1),
            phis=self.leapct.setAngleArray(
                numAngles=nr_angles,
                angularRange=180
            )
        )

        self.leapct.set_default_volume()
        self.allocate_batch_data()
        self.image_size = image_size

    def forward(self, x):
        output = torch.zeros_like(self.proj_data)
        vol = torch.zeros_like(self.vol_data)
        if self.use_gpu:
            sino = leaptorch.ProjectorFunctionGPU.apply(x, output, vol, self.param_id_t)
        else:
            sino = leaptorch.ProjectorFunctionCPU.apply(x, output, vol, self.param_id_t)

        return sino.permute((0, 2, 1, 3)) / self.image_size


    def fbp(self, x) -> Tensor:
        xp = x.permute((0, 2, 1, 3)) * self.image_size
        output = torch.zeros_like(self.vol_data)
        for batch in range(xp.shape[0]):
            if self.use_gpu:
                self.leapct.FBP_gpu(xp[batch], output[batch])
            else:
                self.leapct.FBP_cpu(xp[batch], output[batch])

        return output

    def forward_grad(self, x):
        if self.use_gpu:
            sino = leaptorch.ProjectorFunctionGPU.apply(x, self.proj_data, self.vol_data, self.param_id_t)
        else:
            sino = leaptorch.ProjectorFunctionCPU.apply(x, self.proj_data, self.vol_data, self.param_id_t)

        return sino.permute((0, 2, 1, 3)) / self.image_size


    def fbp_grad(self, x):

        xp = x.permute((0, 2, 1, 3)) * self.image_size
        for batch in range(xp.shape[0]):
            if self.use_gpu:
                self.leapct.FBP_gpu(xp[batch], self.vol_data[batch])