import argparse
from sklearn.decomposition import PCA
import numpy as np
import pickle as pk
from torch.utils import data


from datasets import DepthDataset, KvasirSegDataset

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="depth", help="generate PCA for depth or segmentation task")
opt = parser.parse_args()

def pca_for_depth():
    Train_dataset = DepthDataset('train.txt')
    train_dataloader = data.DataLoader(Train_dataset, batch_size=6000, shuffle=True)
    batches = next(iter(train_dataloader))
    real_A = batches['A']
    B, C, H, W = real_A.shape
    flatten_batch = real_A.reshape(-1, C * H * W).numpy()
    depth_PCA = PCA(n_components=512)
    depth_PCA.fit(flatten_batch)
    flatten_batch=None
    real_A=None
    batches=None
    pk.dump(depth_PCA, open("pca.pkl","wb"))

def pca_for_seg():
    Train_dataset = KvasirSegDataset('./Kvasir-SEG')
    train_dataloader = data.DataLoader(Train_dataset, batch_size=1000, shuffle=True)
    batches = next(iter(train_dataloader))
    real_A = batches['A']
    B, C, H, W = real_A.shape
    flatten_batch = real_A.reshape(-1, C * H * W).numpy()
    depth_PCA = PCA(n_components=512)
    depth_PCA.fit(flatten_batch)
    flatten_batch=None
    real_A=None
    batches=None
    pk.dump(depth_PCA, open("pca.pkl","wb"))

if opt.task == "depth":
    pca_for_depth()
elif opt.task == "seg":
    pca_for_seg()
else:
    print("Incorrect arugment")
