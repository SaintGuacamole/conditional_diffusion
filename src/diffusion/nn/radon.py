
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RadonTransform(nn.Module):
    def __init__(self, image_size, angles=180, circle=True):

        super().__init__()

        self.image_size = image_size
        self.circle = circle
        self.angles = torch.linspace(0, 180, angles, dtype=torch.float32)

        self.proj_coords = self._get_projection_coordinates()

    def _get_projection_coordinates(self):

        if self.circle:
            size = self.image_size
        else:
            size = int(self.image_size * math.sqrt(2))

        x = torch.linspace(-size / 2, size / 2, size)
        y = torch.linspace(-size / 2, size / 2, size)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        coords = torch.stack([X.flatten(), Y.flatten()], dim=0)
        return coords

    def _rotate_points(self, points, angle):

        angle_rad = torch.tensor(angle * math.pi / 180)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        rotation_matrix = torch.tensor([[cos_a, -sin_a],
                                        [sin_a, cos_a]], device=points.device)

        return torch.mm(rotation_matrix, points)

    def forward(self, x):

        batch_size = x.shape[0]
        device = x.device

        if self.circle:

            center = self.image_size // 2
            Y, X = torch.meshgrid(torch.arange(self.image_size, device=device),
                                  torch.arange(self.image_size, device=device))
            dist_from_center = torch.sqrt((X - center) ** 2 + (Y - center) ** 2)
            mask = dist_from_center <= center
            x = x * mask.to(x.dtype)

        if self.circle:
            proj_size = self.image_size
        else:
            proj_size = int(self.image_size * math.sqrt(2))

        if not self.circle:
            pad_size = (proj_size - self.image_size) // 2
            x = F.pad(x, (pad_size,) * 4)

        sinogram = torch.zeros(batch_size, len(self.angles), proj_size,
                               device=device, dtype=x.dtype)

        coords = self.proj_coords.to(device)

        for i, angle in enumerate(self.angles):

            rotated_coords = self._rotate_points(coords, angle.item())

            rotated_coords = rotated_coords / (proj_size / 2)
            rotated_coords = rotated_coords.T.view(1, -1, 2)

            grid = rotated_coords.expand(batch_size, -1, -1)

            samples = F.grid_sample(x.view(batch_size, 1, proj_size, proj_size),
                                    grid.view(batch_size, proj_size, -1, 2),
                                    align_corners=True,
                                    mode='bilinear')

            proj = samples.sum(dim=2)
            sinogram[:, i] = proj.squeeze(1)

        return sinogram

def _get_fourier_filter(size, filter_name, device):

    n = torch.cat([
        torch.arange(1, size // 2 + 1, 2, device=device),
        torch.arange(size // 2 - 1, 0, -2, device=device)
    ])

    f = torch.zeros(size, device=device)
    f[0] = 0.25
    f[1::2] = -1 / (math.pi * n) ** 2

    fourier_filter = 2 * torch.real(torch.fft.fft(f))

    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        omega = math.pi * torch.fft.fftfreq(size, device=device)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega
    elif filter_name == "cosine":
        freq = torch.linspace(0, math.pi, size, endpoint=False, device=device)
        cosine_filter = torch.fft.fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        window = torch.hamming_window(size, device=device)
        fourier_filter *= torch.fft.fftshift(window)
    elif filter_name == "hann":
        window = torch.hann_window(size, device=device)
        fourier_filter *= torch.fft.fftshift(window)
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter.unsqueeze(1)


class InverseRadonTransform(nn.Module):
    def __init__(self,
                 image_size,
                 angles=180,
                 filter_name="shepp-logan",
                 circle=True,
                 interpolation='bilinear'
                 ):

        super().__init__()

        self.image_size = image_size
        self.circle = circle
        self.filter_name = filter_name
        self.interpolation = interpolation
        self.angles = torch.linspace(0, 180, angles, dtype=torch.float32)

        self.proj_size = image_size

        self._precompute_backprojection_params()

        self.fourier_filter = None

    def _precompute_backprojection_params(self):
        y, x = torch.meshgrid(
            torch.arange(self.image_size, dtype=torch.float32) - self.image_size // 2,
            torch.arange(self.image_size - 1, -1, -1, dtype=torch.float32) - self.image_size // 2,
            indexing='ij'
        )

        self.x = x
        self.y = y

        if self.circle:
            radius = self.image_size // 2
            self.circle_mask = (x.pow(2) + y.pow(2)) <= radius ** 2

    def _filter_sinogram(self, sinogram):

        device = sinogram.device

        if self.fourier_filter is None or self.fourier_filter.device != device:
            self.fourier_filter = _get_fourier_filter(
                self.proj_size, self.filter_name, device)

        pad_size = max(64, 2 ** (self.proj_size - 1).bit_length())
        pad_width = pad_size - self.proj_size

        padded_sinogram = F.pad(sinogram, (pad_width // 2, pad_width - pad_width // 2), mode='reflect')

        fourier = torch.fft.fft(padded_sinogram, dim=-1)

        filter_shaped = self.fourier_filter.view(1, 1, -1).expand_as(fourier)
        filtered_fourier = fourier * filter_shaped
        filtered = torch.real(torch.fft.ifft(filtered_fourier, dim=-1))

        if pad_width % 2 == 0:
            filtered = filtered[:, :, pad_width // 2:pad_width // 2 + self.proj_size]
        else:
            filtered = filtered[:, :, pad_width // 2:pad_width // 2 + self.proj_size]

        return filtered

    def forward(self, sinogram):

        batch_size = sinogram.shape[0]
        device = sinogram.device
        dtype = sinogram.dtype

        x = self.x.to(device)
        y = self.y.to(device)

        self.proj_size = sinogram.shape[2]

        filtered_sinogram = self._filter_sinogram(sinogram)

        output = torch.zeros(batch_size, 1, self.image_size, self.image_size,
                             device=device, dtype=dtype)

        for i, angle in enumerate(self.angles):
            angle_rad = angle * torch.tensor(math.pi) / 180

            t = x * torch.sin(angle_rad) + y * torch.cos(angle_rad)

            t = t + self.proj_size // 2

            t = (2 * t / (self.proj_size - 1)) - 1
            zeros = torch.zeros_like(t)

            grid = torch.stack([t, zeros], dim=-1).unsqueeze(0)
            grid = grid.expand(batch_size, -1, -1, -1)

            proj = filtered_sinogram[:, i:i + 1]

            proj = proj.unsqueeze(-1).transpose(-1, -2)

            backprojection = F.grid_sample(
                proj, grid,
                align_corners=True,
                mode=self.interpolation,
                padding_mode='zeros'
            )

            output += backprojection

        output = output * (math.pi / (2 * len(self.angles)))

        if self.circle:
            mask = self.circle_mask.to(device).unsqueeze(0).unsqueeze(0)
            output = output * mask.to(dtype)

        return output