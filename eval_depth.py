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

test_dataset = DepthDataset("test.txt")

generator = GeneratorUNet(in_channels=3, out_channels=1)

test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
 )

generator.load_state_dict(torch.load(f"{opt.result_dir}/saved_models/generator_{opt.split}_{opt.ckpt}.pth"))

generator = generator.cuda()


rmse_list = []
mae_list = []

generator.eval()

for i, batch in enumerate(test_dataloader):
        imgs = batch["A"]
        gts = batch["B"]
        imgs = imgs.cuda()
        gts = gts.cuda()
        with torch.no_grad():
                preds, _ = generator(imgs)
        imgs = imgs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        gts = gts.detach().cpu().numpy()
        gts = inv_normalize(gts)
        preds = inv_normalize(preds)
        imgs = inv_normalize(imgs)
        rmse_list.append(cal_rmse(255*gts, 255 * preds))
        mae_list.append(cal_mae(255 * gts, 255 * preds))

print(f"Mean RMSE is : {np.mean(rmse_list)}")