CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema                                 --task-name=blur_0_uncond   --load-checkpoint-path   "ckpt/blur_0_uncond.ckpt"  --sample-seed=1898 --sample-steps=1                                         
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema                                 --task-name=blur_0_uncond   --load-checkpoint-path   "ckpt/blur_0_uncond.ckpt"  --sample-seed=1898 --sample-steps=3                                         
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema                                 --task-name=blur_0_uncond   --load-checkpoint-path   "ckpt/blur_0_uncond.ckpt"  --sample-seed=1898 --sample-steps=5                                         

CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=1            --task-name=blur_1_uncond   --load-checkpoint-path   "ckpt/blur_1_uncond.ckpt"  --sample-seed=1898 --sample-steps=1                                         
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=1            --task-name=blur_1_uncond   --load-checkpoint-path   "ckpt/blur_1_uncond.ckpt"  --sample-seed=1898 --sample-steps=3  --sample-with-blur  --sample-blur-pow=1
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=1            --task-name=blur_1_uncond   --load-checkpoint-path   "ckpt/blur_1_uncond.ckpt"  --sample-seed=1898 --sample-steps=5  --sample-with-blur  --sample-blur-pow=1

CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=2            --task-name=blur_2_uncond   --load-checkpoint-path   "ckpt/blur_2_uncond.ckpt"  --sample-seed=1898 --sample-steps=1                                         
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=2            --task-name=blur_2_uncond   --load-checkpoint-path   "ckpt/blur_2_uncond.ckpt"  --sample-seed=1898 --sample-steps=3  --sample-with-blur  --sample-blur-pow=1
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=2            --task-name=blur_2_uncond   --load-checkpoint-path   "ckpt/blur_2_uncond.ckpt"  --sample-seed=1898 --sample-steps=5  --sample-with-blur  --sample-blur-pow=1

CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=3            --task-name=blur_3_uncond   --load-checkpoint-path   "ckpt/blur_3_uncond.ckpt"  --sample-seed=1898 --sample-steps=1                                         
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=3            --task-name=blur_3_uncond   --load-checkpoint-path   "ckpt/blur_3_uncond.ckpt"  --sample-seed=1898 --sample-steps=3  --sample-with-blur  --sample-blur-pow=1
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=3            --task-name=blur_3_uncond   --load-checkpoint-path   "ckpt/blur_3_uncond.ckpt"  --sample-seed=1898 --sample-steps=5  --sample-with-blur  --sample-blur-pow=1

CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema                        --cond   --task-name=blur_0_cond     --load-checkpoint-path   "ckpt/blur_0_cond.ckpt"    --sample-seed=1898 --sample-steps=1                                         
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema                        --cond   --task-name=blur_0_cond     --load-checkpoint-path   "ckpt/blur_0_cond.ckpt"    --sample-seed=1898 --sample-steps=3                                         
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema                        --cond   --task-name=blur_0_cond     --load-checkpoint-path   "ckpt/blur_0_cond.ckpt"    --sample-seed=1898 --sample-steps=5                                         

CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=2   --cond   --task-name=blur_2_cond     --load-checkpoint-path   "ckpt/blur_2_cond.ckpt"    --sample-seed=1898 --sample-steps=1                                         
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=2   --cond   --task-name=blur_2_cond     --load-checkpoint-path   "ckpt/blur_2_cond.ckpt"    --sample-seed=1898 --sample-steps=3  --sample-with-blur  --sample-blur-pow=1
CUDA_VISIBLE_DEVICES=0  python   main.py   --eval   --batch-size=1250   --use-ema   --sigma-blur-max=2   --cond   --task-name=blur_2_cond     --load-checkpoint-path   "ckpt/blur_2_cond.ckpt"    --sample-seed=1898 --sample-steps=5  --sample-with-blur  --sample-blur-pow=1
