CUDA_VISIBLE_DEVICES=0,1   python main.py --batch-size=80  --use-ema                           --task-name=blur_0_uncond
CUDA_VISIBLE_DEVICES=0,1   python main.py --batch-size=80  --use-ema --sigma-blur-max=1        --task-name=blur_1_uncond
CUDA_VISIBLE_DEVICES=0,1   python main.py --batch-size=80  --use-ema --sigma-blur-max=2        --task-name=blur_2_uncond
CUDA_VISIBLE_DEVICES=0,1   python main.py --batch-size=80  --use-ema --sigma-blur-max=3        --task-name=blur_3_uncond
CUDA_VISIBLE_DEVICES=0,1   python main.py --batch-size=80  --use-ema                    --cond --task-name=blur_0_cond  
CUDA_VISIBLE_DEVICES=0,1   python main.py --batch-size=80  --use-ema --sigma-blur-max=2 --cond --task-name=blur_2_cond  
