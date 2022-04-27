
import random
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from data.sampler import SubsetSequentialSampler
from config import *
from utils import *
from datasets import *
from samplers import *
from train_test import read_data
from query_model import VAE, Discriminator

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle, opt):
    
    vae = models['vae']
    discriminator = models['discriminator']
    vae.train()
    discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()
    
    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 2

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader, labels=False)
    unlabeled_data = read_data(unlabeled_dataloader, labels=False)

    train_iterations = int( (BUDGET*cycle+ INITIAL_BUDGET) * EPOCHV / opt.batch_size)

    for iter_count in range(train_iterations):
        labeled_imgs = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            #labels = labels.cuda()

        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    #labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)
            
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    #labels = labels.cuda()
        if (iter_count + 1) % 100 == 0:
            print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))

    
def query_samples(models, train_dataset, unlabeled_indices, labeled_indices, cycle, opt, split=None):

    if opt.method == 'VAAL':
        unlabeled_loader = DataLoader(train_dataset, batch_size=opt.batch_size, 
                                    sampler=SubsetSequentialSampler(unlabeled_indices), 
                                    pin_memory=True)
        labeled_loader = DataLoader(train_dataset, batch_size=opt.batch_size, 
                                    sampler=SubsetSequentialSampler(labeled_indices), 
                                    pin_memory=True)
        vae = VAE()
        discriminator = Discriminator(32)
        query_models      = {'vae': vae, 'discriminator': discriminator}
        
        optim_vae = torch.optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

        train_vaal(query_models,optimizers, labeled_loader, unlabeled_loader, cycle, opt)
        
        all_preds = []

        for batch in tqdm(unlabeled_loader):                       
            images = batch['A'].cuda()
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        _, arg = torch.topk(all_preds, BUDGET)
        indices_to_label = list(torch.tensor(unlabeled_indices)[arg].numpy())
    
    if opt.method == 'Uncertain':
        unlabeled_loader = DataLoader(train_dataset, batch_size=1, 
                                    sampler=SubsetSequentialSampler(unlabeled_indices))
        arg, _ = get_topk_bvsb(models, unlabeled_loader, BUDGET)
        indices_to_label = list(torch.tensor(unlabeled_indices)[arg].numpy())

    if opt.method == 'UncertainWithCoreset':
        indices_to_label = []

        while len(indices_to_label) < BUDGET:
            intmd_unlabeled = np.setdiff1d(unlabeled_indices, indices_to_label)
            intmd_labeled = np.append(labeled_indices, indices_to_label).astype('int')

            #Calculate number of uncertain and diverse examples to sample
            num_to_sample = BUDGET - len(indices_to_label)
            num_uncertain = int(opt.gamma * num_to_sample)
            num_diverse = num_to_sample - num_uncertain

            #Get uncertain and diverse indices
            arg = get_uncertain_and_diverse(train_dataset, intmd_unlabeled, intmd_labeled,
                                            models, num_uncertain, num_diverse, opt)
            indices = list(intmd_unlabeled[arg])
            indices_to_label += indices
    
    if opt.method == "UncertainWithPCA":
        indices_to_label = []

        while len(indices_to_label) < BUDGET:
            intmd_unlabeled = np.setdiff1d(unlabeled_indices, indices_to_label)
            intmd_labeled = np.append(labeled_indices, indices_to_label).astype('int')

            #Calculate number of uncertain and diverse examples to sample
            num_to_sample = BUDGET - len(indices_to_label)
            num_uncertain = int(opt.gamma * num_to_sample)
            num_diverse = num_to_sample - num_uncertain

            #Get uncertain and diverse indices
            arg = get_uncertain_and_diverse(train_dataset, intmd_unlabeled, intmd_labeled,
                                            models, num_uncertain, num_diverse, opt)
            indices = list(intmd_unlabeled[arg])
            indices_to_label += indices
        
    if opt.method == 'CoreSetPCA':
        unlabeled_loader = DataLoader(train_dataset, sampler=SubsetSequentialSampler(np.append(unlabeled_indices, labeled_indices)), 
                batch_size=opt.batch_size)
        arg = get_kcg_pca(INITIAL_BUDGET + BUDGET*cycle, unlabeled_loader, unlabeled_indices, BUDGET)
        indices_to_label = list(torch.tensor(unlabeled_indices)[arg].numpy())
        
    if opt.method == 'Random':
        random.shuffle(unlabeled_indices)
        arg = np.random.randint(len(unlabeled_indices), size=len(unlabeled_indices))
        indices_to_label = unlabeled_indices[arg][:BUDGET]
    
    if opt.method == 'CoreSet':
        unlabeled_loader = DataLoader(train_dataset, sampler=SubsetSequentialSampler(np.append(unlabeled_indices, labeled_indices)), 
                batch_size=opt.batch_size)
        arg = get_kcg(models, INITIAL_BUDGET + BUDGET*cycle, unlabeled_loader, unlabeled_indices, BUDGET, opt,
                        opt.save_feat, opt.output_path, split)

        indices_to_label = list(torch.tensor(unlabeled_indices)[arg].numpy())

    return indices_to_label
