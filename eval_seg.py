import argparse
import os
import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch.utils import data
import numpy as np

from networks import *
from datasets import *
from utils import *
from train_test import *

parser = argparse.ArgumentParser()

parser.add_argument("--result_dir", type=str, help="path of results dir of test")
parser.add_argument("--split", type=float, help="cycle or dataset split")
parser.add_argument("--ckpt", type=int, help="best model checkpoint")

opt = parser.parse_args()

test_dataset = KvasirSegDataset('./sessile-main-Kvasir-SEG')

generator = GeneratorUNet(in_channels=3, out_channels=1)

test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
 )

path = f"{opt.result_dir}/saved_models/generator_{opt.split}_{opt.ckpt}.pth"

print(f"Evaluating {path}")

generator.load_state_dict(torch.load(path))

generator = generator.cuda()

test_mean_iou = calculate_mean_iou(generator, test_dataloader, True)

print(f"mean iou is {test_mean_iou}")