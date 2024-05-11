# Consistency Model with Blurring Noise 😋

Final Project of Fundamentals of Generative Modeling at PKU 2023 fall.

## Environment requirements 🔨

To set the environment, just install the latest versions of

`pillow`
`torch`
`torchvision`
`pytorch-lightning`
`diffusers`
`torchmetrics`
`lpips`

## Training 🚂

The `main.py` is used for training the model.

Unconditional training consistency model without blurring noise

`python main.py --batch-size=80  --use-ema                           --task-name=blur_0_uncond`

Unconditional training consistency model with blurring noise

`python main.py --batch-size=80  --use-ema --sigma-blur-max=2        --task-name=blur_2_uncond`

Conditional training consistency model with blurring noise

`python main.py --batch-size=80  --use-ema --sigma-blur-max=2 --cond --task-name=blur_2_cond  `

Change the sigma of blurring noise

`python main.py --batch-size=80  --use-ema --sigma-blur-max=3        --task-name=blur_2_uncond`

## Generation 📷

The `main.py` can also be used to generate samples.

Unconditional one step sampling

`python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=2            --task-name=blur_2_uncond   --load-checkpoint-path   "ckpt/blur_2_uncond.ckpt"  --sample-seed=1898 --sample-steps=1                                         `

Unconditional multi-step sampling

`python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=2            --task-name=blur_2_uncond   --load-checkpoint-path   "ckpt/blur_2_uncond.ckpt"  --sample-seed=1898 --sample-steps=5  --sample-with-blur  --sample-blur-pow=1`

Conditional multi-step sampling

`python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=2   --cond   --task-name=blur_2_cond     --load-checkpoint-path   "ckpt/blur_2_cond.ckpt"    --sample-seed=1898 --sample-steps=5  --sample-with-blur  --sample-blur-pow=1`

## Evaluation 🔍

The `evaluation.py` is used for evaluating FID and IS of the generated images.

Evaluation the image folder "image_folder_dir"

`python evaluation.py image_folder_dir    `

## URL of the open-source code 👋

https://github.com/junhsss/consistency-models

https://github.com/AaltoML/generative-inverse-heat-dissipation
