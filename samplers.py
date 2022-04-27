import os
import torch
from tqdm import tqdm
import numpy as np
import pickle as pk
from torch.utils.data import DataLoader

from config import *
from kcenterGreedy import kCenterGreedy
from train_test import inv_normalize
from utils import *
from data.sampler import SubsetSequentialSampler



def get_kcg(models, labeled_data_size, unlabeled_loader, unlabeled_indices, budget, opt, save_features=False, out_path=None, splits=None):
    models['generator'].eval()
    if opt.cuda:
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for batches in tqdm(unlabeled_loader):
            inputs = batches['A']
            if opt.cuda:
                inputs = inputs.cuda()
            _, features_batch = models['generator'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        if save_features:
            os.makedirs(f'{out_path}/features', exist_ok=True)
            np.save(f'./{out_path}/features/feats_after{splits}.npy', feat)
        num_unlabeled_indices = len(unlabeled_indices)
        new_av_idx = np.arange(num_unlabeled_indices,(num_unlabeled_indices + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, budget)
        #other_idx = [x for x in range(num_unlabeled_indices) if x not in batch]
    return batch



def get_kcg_pca(labeled_data_size, unlabeled_loader, unlabeled_indices, budget):
    features = torch.tensor([])
    pca_model = pk.load(open("pca.pkl",'rb'))
    for batches in unlabeled_loader:
        inputs = batches['A']
        B, C, H, W = inputs.shape
        inputs = inputs.reshape(-1, C * H * W).numpy()
        features_batch = pca_model.transform(inputs)
        features = torch.cat((features, torch.from_numpy(features_batch)), 0)
    feat = features.detach().numpy()
    num_unlabeled_indices = len(unlabeled_indices)
    new_av_idx = np.arange(num_unlabeled_indices,(num_unlabeled_indices + labeled_data_size))
    sampling = kCenterGreedy(feat)  
    batch = sampling.select_batch_(new_av_idx, budget)
    return batch

def get_topk_bvsb(models, unlabeled_loader, budget, opt):
    models['generator'].eval()
    with torch.no_grad():
        unc_preds = []
        for batches in tqdm(unlabeled_loader):
            inputs = batches['A']

            if opt.cuda:
                inputs = inputs.cuda()

            preds,_ = models['generator'](inputs)
            preds = inv_normalize(preds)
            bvsb = torch.abs(2*preds - 1)
            uncertainty = 1 - bvsb
            avg_uncertainty = torch.mean(uncertainty)
            unc_preds.append(avg_uncertainty)
        unc_preds = torch.stack(unc_preds)

    scores, querry_indices = torch.topk(unc_preds, budget)

    return querry_indices, scores

def get_uncertain_and_diverse(train_dataset, unlabeled_indices, labeled_indices, models, num_uncertain, 
                              num_diverse, opt):
    # Get uncertain examples
    unlabeled_loader = DataLoader(train_dataset, batch_size=1, 
                                sampler=SubsetSequentialSampler(unlabeled_indices))
    unc_arg, _ = get_topk_bvsb(models, unlabeled_loader, num_uncertain, opt)

    # Get diverse samples
    div_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                    sampler=SubsetSequentialSampler(np.append(unlabeled_indices, labeled_indices)))
    if opt.method == "UncertainWithPCA":
        div_arg = get_kcg_pca(len(labeled_indices), div_loader, unlabeled_indices, num_diverse)
    else:
        div_arg = get_kcg(models, len(labeled_indices), div_loader, unlabeled_indices, num_diverse, opt)

    #Combine uncertain and diverse
    arg = np.union1d(unc_arg.detach().cpu().numpy(), div_arg)

    return arg