from torch_fidelity.metrics import calculate_metrics
from cleanfid.fid import compute_fid
import sys, os, shutil
from torchvision.datasets import CIFAR10
from pathlib import Path
from PIL import Image
import tqdm

dataset = CIFAR10("data", train=True, download=True)
dataset_length = len(dataset)
ground_truth_path = f"data/cifar10img"
extract_cifar10 = True
if os.path.isdir(ground_truth_path):
    if len(os.listdir(ground_truth_path)) == dataset_length:
        extract_cifar10 = False
if extract_cifar10:
    shutil.rmtree(ground_truth_path, ignore_errors=True)
    Path(ground_truth_path).mkdir(parents=True, exist_ok=True)
    for i in tqdm.trange(dataset_length):
        image: Image.Image = dataset[i][0]
        image.save(f"{ground_truth_path}/{i:06}.jpeg")

fdir1 = sys.argv[1]
fdir2 = ground_truth_path

metrics_dict  = calculate_metrics(
            input1=fdir1,
            input2=fdir2,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
            batch_size=500
            )
fid = metrics_dict['frechet_inception_distance']
is_mean = metrics_dict['inception_score_mean']
is_std = metrics_dict['inception_score_std']
# fid2 = compute_fid(
    # fdir1=fdir1,
    # fdir2=fdir2,
    # mode="clean",
    # device="cuda",
    # batch_size=500
# )

print(fdir1)
print(f"FID = {fid}")
print(f"IS = {is_mean}")
# print(f"is_std = {is_std}")
# print(f"fid2 = {fid2}")