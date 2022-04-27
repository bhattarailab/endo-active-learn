import time
from cv2 import mean
import numpy as np
import datetime
import sys
from numpy import random

import torch
from tqdm import tqdm
import wandb
from torchvision.utils import save_image
from utils import cal_rmse, cal_mae, MeanIOU
from config import *

def inv_normalize(img_tensor):
    return (img_tensor/2 + 0.5) 

def sample_images(batches_done, val_dataloader, Tensor, generator, out_path):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = imgs["A"].type(Tensor)
    real_B = imgs["B"].type(Tensor)
    fake_B, _ = generator(real_A)
    real_A = inv_normalize(real_A)
    real_B = inv_normalize(real_B)
    fake_B = inv_normalize(fake_B)
    img_sample = torch.cat((fake_B.data, real_B.data), -1)
    save_image(img_sample, "%s/images/%s_y.png" % (out_path, batches_done), normalize=True)
    save_image(real_A, "%s/images/%s_x.png" % (out_path, batches_done),  normalize=True)

def calculate_metrics(generator, data_loader, cuda):
    generator.eval()
    rmse_list = []
    mae_list = []
    rmse_list_cm = []
    mae_list_cm = []
    for i, batch in enumerate(data_loader):    
        imgs = batch["A"]
        gts = batch["B"]
        
        if cuda:
            imgs = imgs.cuda()
            gts = gts.cuda()
        
        with torch.no_grad():
            preds, _ = generator(imgs)
        imgs = imgs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        gts = gts.detach().cpu().numpy()
        gts = inv_normalize(gts)
        preds = inv_normalize(preds)
        rmse_list.append(cal_rmse(255 * gts, 255 * preds))
        mae_list.append(cal_mae(255 * gts, 255 * preds))
        rmse_list_cm.append(cal_rmse(20 * gts, 20 * preds))
        mae_list_cm.append(cal_mae(20 * gts, 20 * preds))

    return np.mean(rmse_list), np.mean(mae_list), np.mean(rmse_list_cm), np.mean(mae_list_cm)

def calculate_mean_iou(generator, data_loader, cuda):
    generator.eval()
    metric = MeanIOU()
    metric.reset()
    for batch in tqdm(data_loader):    
        imgs = batch["A"]
        masks = batch["B"]
        
        if cuda:
            imgs = imgs.cuda()
            masks = masks.cuda()
        
        with torch.no_grad():
            preds, _ = generator(imgs)
        masks = inv_normalize(masks.detach().cpu())
        preds = inv_normalize(preds.detach().cpu())
        metric.add(preds, masks)
        
    return metric.value()    

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for batch in dataloader:
                yield batch['A'], batch['B']
    else:
        while True:
            for batch in dataloader:
                yield batch['A']


def train(models, dataloaders, criteria, opt, split=1):

    generator = models['generator']
    discriminator = models['discriminator']
    train_loader =  dataloaders['train_loader']
    val_loader = dataloaders['val_loader']
    criterion_GAN = criteria['criterion_GAN']
    criterion_pixelwise = criteria['criterion_pixelwise']

    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    prev_time = time.time()
    runs = wandb.init(project="ActiveLearning", entity = "abyss", name=f"{opt.method}_{split}", reinit=True)
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(train_loader):
            #Training Model
            generator.train()
            discriminator.train()

            # Model inputs
            real_A = batch["A"].type(opt.Tensor)
            real_B = batch["B"].type(opt.Tensor)

            # Adversarial ground truths
            valid = opt.Tensor(np.ones((real_A.size(0), *opt.patch)))
            fake = opt.Tensor(np.zeros((real_A.size(0), *opt.patch)))

            if opt.cuda:
                real_A = real_A.cuda()
                real_B = real_B.cuda()
                valid = valid.cuda()
                fake = fake.cuda()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            fake_B, _ = generator(real_A)
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()

            optimizer_D.step()

            # ------------------
            #  Train Generators
            # ------------------
            mean_lossG = 0
            mean_loss_GAN = 0
            mean_loss_pixel = 0

            for _ in range(opt.num_gen_steps):

                optimizer_G.zero_grad()

                # GAN loss
                fake_B, _ = generator(real_A)
                pred_fake = discriminator(fake_B, real_A)
                loss_GAN = criterion_GAN(pred_fake, valid)
                # Pixel-wise loss
                loss_pixel = criterion_pixelwise(fake_B, real_B)

                # Total loss
                loss_G = loss_GAN + opt.lambda_pixel * loss_pixel


                loss_G.backward()

                optimizer_G.step()

                mean_lossG += loss_G.item() / opt.num_gen_steps
                mean_loss_GAN += loss_GAN.item() / opt.num_gen_steps
                mean_loss_pixel += loss_pixel.item() / opt.num_gen_steps




            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(train_loader) + i
            batches_left = opt.n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(train_loader),
                    loss_D.item(),
                    mean_lossG,
                    mean_loss_pixel,
                    mean_loss_GAN,
                    time_left,
                )
            )

            wandb.log({
                    "D loss": loss_D.item(),
                    "G loss":mean_lossG,
                    "pixel loss":mean_loss_pixel,
                    "adv loss":mean_loss_GAN
                })

                
        
        #Calculating Metrics

        #mean_rmse, mean_mae, mean_rmse_cm, mean_mae_cm = calculate_metrics(generator, val_loader, opt.cuda)
        mean_iou = calculate_mean_iou(generator, val_loader, opt.cuda)
        # wandb.log({
        #     "mean_rmse": mean_rmse,
        #     "mean_mae": mean_mae,
        #     "mean_rmse(cm)": mean_rmse_cm,
        #     "mean_mae(cm)": mean_mae_cm
        # })
        wandb.log({
            "val_mean_iou": mean_iou
        })


        #Saving image sample
        sample_images(batches_done, val_loader, opt.Tensor, generator, opt.output_path)

        # Save model checkpoints
        if epoch % opt.checkpoint_interval == 0:
            torch.save(generator.state_dict(), "%s/saved_models/generator_%s_%d.pth" % (opt.output_path, str(split), epoch))
            torch.save(discriminator.state_dict(), "%s/saved_models/discriminator_%s_%d.pth" % (opt.output_path, str(split), epoch))
            torch.save(optimizer_G.state_dict(), "%s/saved_models/generator_optimizer_%s.pth" % (opt.output_path, str(split)))
            torch.save(optimizer_D.state_dict(), "%s/saved_models/discriminator_Optimizer_%s.pth" % (opt.output_path, str(split)))




    


