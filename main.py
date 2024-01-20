import argparse
import math
import io
import os
import shutil
import torch
import contextlib

from diffusers import UNet2DModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from consistency import Consistency
from consistency.loss import PerceptualLoss
from consistency.pipeline import ConsistencyPipeline
from torchvision.datasets import CIFAR10

from pathlib import Path
from PIL import Image
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="A simple training script for consistency models."
    )
    parser.add_argument("--dataset-name",                           type=str,   default="cifar10")
    parser.add_argument("--resolution",                             type=int,   default=32, help="The resolution for input images.")
    parser.add_argument("--batch-size",                             type=int,   default=80, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num-workers",                            type=int,   default=4)
    parser.add_argument("--max-epochs",                             type=int,   default=280)
    parser.add_argument("--learning-rate",                          type=float, default=8e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--data-std",                               type=int,   default=0.5, help="Standard deviation of the dataset")
    parser.add_argument("--time-min",                               type=float, default=0.008)
    parser.add_argument("--time-max",                               type=float, default=20)
    parser.add_argument("--bins-min",                               type=float, default=2)
    parser.add_argument("--bins-max",                               type=float, default=150)
    parser.add_argument("--bins-rho",                               type=float, default=7)
    parser.add_argument("--initial-ema-decay",                      type=float, default=0.9)
    parser.add_argument("--sigma-blur-max",                         type=float, default=None)
    
    parser.add_argument("--save-samples-every-n-epochs",            type=int,   default=10)
    parser.add_argument("--num-samples",                            type=int,   default=16, help="The number of images to generate for evaluation.")
    parser.add_argument("--sample-steps",                           type=int,   default=10)
    parser.add_argument("--sample-seed",                            type=int,   default=42)
    parser.add_argument("--use-ema",                                            action="store_true")
    
    parser.add_argument("--wandb-project",                          type=str,   default="consistency")
    parser.add_argument("--log-every-n-steps",                      type=int,   default=100)
    
    parser.add_argument("--task-name",                              type=str,   default=None)
    parser.add_argument("--load-checkpoint-path",                   type=str,   default=None)
    parser.add_argument("--cond",                                   action="store_true")
    parser.add_argument("--eval",                                   action="store_true")
    parser.add_argument("--sample-with-blur",                       action="store_true")
    parser.add_argument("--sample-blur-pow",                        type=float,   default=1)

    args = parser.parse_args()
    return args


def main(args):
    
    # 下载可能失败，将下载到本地的文件拷贝到缓存中，直接在缓存中读取
    def copy_predownload_to_cache():
        cache_path = os.path.expanduser("~") + "/.cache/torch/hub/checkpoints"
        os.makedirs(cache_path, exist_ok = True)
        for file in ["squeezenet1_1-b8a52dc0.pth", "vgg16-397923af.pth", "weights-inception-2015-12-05-6726825d.pth"]:
            shutil.copyfile("./data/" + file, cache_path + "/" + file)
    copy_predownload_to_cache()

    # 数据集准备
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = CIFAR10("data", train=True, transform=augmentations, download=True)
    dataset_length = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    cond_classes = (10 if args.cond else None)
    
    # Simplified NCSN++ Architecture tailored for CIFAR-10
    # See https://huggingface.co/google/ncsnpp-ffhq-1024/blob/main/config.json
    unet = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        attention_head_dim=8,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=(
            "SkipDownBlock2D",
            "AttnSkipDownBlock2D",
            "SkipDownBlock2D",
            "SkipDownBlock2D",
        ),
        downsample_padding=1,
        act_fn="silu",
        center_input_sample=True,
        mid_block_scale_factor=math.sqrt(2),
        up_block_types=(
            "SkipUpBlock2D",
            "SkipUpBlock2D",
            "AttnSkipUpBlock2D",
            "SkipUpBlock2D",
        ),
        num_class_embeds=cond_classes
    )

    # Use both VGG and SqueezeNet as loss
    loss_fn = PerceptualLoss(net_type=("vgg", "squeeze"))

    args.checkpoint_dir = f"train_result/{args.task_name}/checkpoint"
    args.sample_path = f"train_result/{args.task_name}/sample"
    if not args.sample_with_blur:
        args.sample_blur_pow = None
    
    configs = {
        "model": unet,
        "loss_fn": loss_fn,
        "learning_rate": args.learning_rate,
        "data_std": args.data_std,
        "time_min": args.time_min,
        "time_max": args.time_max,
        "bins_min": args.bins_min,
        "bins_max": args.bins_max,
        "bins_rho": args.bins_rho,
        "initial_ema_decay": args.initial_ema_decay,
        "samples_path": args.sample_path,
        "save_samples_every_n_epochs": args.save_samples_every_n_epochs,
        "num_samples": args.num_samples,
        "sample_steps": args.sample_steps,
        "use_ema": args.use_ema,
        "sample_seed": args.sample_seed,
        "sigma_blur_max": args.sigma_blur_max,
        "cond_classes": cond_classes,
    }
    
    if not args.eval:

        consistency = Consistency(**configs)

        trainer = Trainer(
            accelerator="auto",
            logger=WandbLogger(project=args.wandb_project, log_model=True),
            callbacks=[
                ModelCheckpoint(
                    dirpath=args.checkpoint_dir,
                    save_top_k=10,
                    monitor="loss",
                    save_last=True
                )
            ],
            max_epochs=args.max_epochs,
            precision=32,
            log_every_n_steps=args.log_every_n_steps,
            gradient_clip_algorithm="norm",
            gradient_clip_val=1.0,
        )

        trainer.fit(consistency, dataloader)

    else:

        dataset.transform = None
        consistency = Consistency.load_from_checkpoint(
            checkpoint_path=args.load_checkpoint_path, **configs
        )
        pipeline = ConsistencyPipeline(
            unet=consistency.model_ema if consistency.use_ema else consistency.model,
            dct_blur = consistency.dct_blur if args.sample_blur_pow is not None else None,
            
            )
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        str_sample_blur_pow = f"{args.sample_blur_pow:.2f}" if args.sample_blur_pow else f"{args.sample_blur_pow}"
        eval_sample_path = f"train_result/{args.task_name}/eval_sample_{args.sample_steps}_{args.sample_seed}_{str_sample_blur_pow}"

        shutil.rmtree(eval_sample_path, ignore_errors=True)
        Path(eval_sample_path).mkdir(parents=True, exist_ok=True)
        batch_list = [args.batch_size] * (dataset_length // args.batch_size)
        if dataset_length % args.batch_size:
            batch_list += [dataset_length % args.batch_size]
        if cond_classes is not None:
            labels_list = [i % cond_classes for i in range(dataset_length)]
        generator = torch.Generator(device=DEVICE).manual_seed(args.sample_seed)
        id = 0
        for batch in tqdm.tqdm(batch_list):
            labels = None
            if cond_classes is not None:
                labels = labels_list[id : id+batch]
            image_list = pipeline(
                num_sample = batch,
                steps = args.sample_steps,
                generator = generator,
                labels = labels,
                pow = args.sample_blur_pow,
                ).images
            for image in image_list:
                image.save(f"{eval_sample_path}/{id:06}.jpeg")
                id += 1

if __name__ == "__main__":
    args = parse_args()
    main(args)
