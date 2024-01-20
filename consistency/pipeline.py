import math
from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from noise_utils.blur import DCTBlur


class ConsistencyPipeline(DiffusionPipeline):
    unet: UNet2DModel

    def __init__(
        self,
        unet: UNet2DModel,
        dct_blur: Optional[DCTBlur]
    ) -> None:
        super().__init__()
        self.register_modules(unet=unet)
        self.dct_blur = dct_blur

    @torch.no_grad()
    def __call__(
        self,
        num_sample: int = 1,
        steps: int = 3,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        labels: Optional[List[int]] = None,
        pow: Optional[float] = None,
        time_min: float = 0.008,
        time_max: float = 20.0,
        data_std: float = 0.5,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        img_size = self.unet.config.sample_size
        shape = (num_sample, 3, img_size, img_size)
        if labels is not None:
            labels = torch.tensor(labels, device = self.unet.device)

        model = self.unet

        time: float = time_max

        sample = randn_tensor(shape, generator=generator, device=self.unet.device) * time

        for step in range(steps):
            if step > 0:
                time = self.search_previous_time(time)
                sigma = math.sqrt(time**2 - time_min**2 + 1e-6)
                if self.dct_blur:
                    sample = self.dct_blur(sample, (time / time_max) ** pow)
                sample = sample + sigma * randn_tensor(sample.shape, device=self.unet.device, generator=generator)

            out = model(sample, torch.tensor([time], device=self.unet.device), labels).sample

            skip_coef = data_std**2 / ((time - time_min) ** 2 + data_std**2)
            out_coef = data_std * time / (time**2 + data_std**2) ** (0.5)

            sample = (sample * skip_coef + out * out_coef).clamp(-1.0, 1.0)

        sample = (sample / 2 + 0.5).clamp(0, 1)
        image = sample.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def search_previous_time(self, time, time_min: float = 0.002, time_max: float = 80.0):
        return (2 * time + time_min) / 3

    def cuda(self):
        self.to("cuda")
