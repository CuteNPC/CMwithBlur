from .torch_dct import dct_2d, idct_2d
import torch
import math

class DCTBlur(torch.nn.Module):

    def __init__(
        self, 
        sigma_blur_max, 
        img_dim, 
        min_scale = 0.001,
        logsnr_min=-10, 
        logsnr_max=10,
            ) -> None:
        super().__init__()
        self.sigma_blur_max = sigma_blur_max
        if self.sigma_blur_max is None:
            return
        self.img_dim = img_dim
        self.min_scale = min_scale
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        # 计算频率
        self.freq = torch.pi * torch.linspace(0, img_dim - 1, img_dim) / img_dim
        self.labda = self.freq[None, None, None, :]**2 + self.freq[None, None, :, None]**2
        self.limit_max = math.atan(math.exp(-0.5 * logsnr_max))
        self.limit_min = math.atan(math.exp(-0.5 * logsnr_min)) - self.limit_max
        self.register_buffer('fix_labda', self.labda)
        print(f"Enable DCTBlur, sigma_blur_max = ", sigma_blur_max)
        self.fix_labda.requires_grad_(False)

    def get_alpha(self, t):
        sigma_blur = self.sigma_blur_max * torch.sin(t * torch.pi / 2)**2
        dissipation_time = (sigma_blur**2 / 2)[..., None, None, None]
        freq_scaling = torch.exp(-self.fix_labda * dissipation_time) * (1 - self.min_scale) + self.min_scale
        return freq_scaling

    @torch.no_grad()
    def diffuse(self, x, t):
        x_freq = dct_2d(x, norm='ortho')
        alpha = self.get_alpha(t)
        z_t = idct_2d(alpha * x_freq, norm='ortho')
        return z_t
    
    def __call__(self, x, timesteps, _bins = 2):
        if self.sigma_blur_max is None:
            return x
        else:
            times = timesteps / (_bins - 1)
            if not isinstance(times, torch.Tensor):
                times = torch.full((x.shape[0],), times, device = x.device)
            return self.diffuse(x, times)
