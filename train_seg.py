import argparse
import os
import random
import numpy as np

import torch
from torch.utils import data


from networks import *
from datasets import *
#from pix2pix.train_test import train_adv
from utils import *
from config import *
from train_test import train, train_adv, calculate_mean_iou
from selection_methods import query_samples



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--train_dir", type=str, default="./Kvasir-SEG", help="Path to training dataset")
parser.add_argument("--test_dir", type=str, default='./sessile-Kvasir-SEG', help='Path to test dataset')
parser.add_argument("--output_path" , type=str, default="results", help="Output paths")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--lr_SSL", type=float, default=0.0002, help="adam: learning rate for SSL")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
#parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--in_channels", type=int, default=3, help="number of input image channels")
parser.add_argument("--out_channels", type=int, default=1, help="number of output channels")
parser.add_argument("--num_gen_steps", type=int, default=1, help="Number of times to train generator")
#parser.add_argument(
 #   "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
#)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--method", type=str, default='Full', help='Training Type')
parser.add_argument("--alpha", type=float, default=0.0, help="alpha value for uncertainty score weighting")
parser.add_argument("--gamma", type=float, default=0.0, help="sampling ratio")
parser.add_argument("--lambda_adv", type=float, default=1.0, help='lambda for adv loss')
parser.add_argument("--k", type=int, default=1000, help='Choose topk uncertain data')
parser.add_argument("--unc_sampling", action='store_true', help="Do uncertainty sampling")
parser.add_argument("--save_feat", action='store_true', help="Save bottleneck features")
parser.add_argument("--data_fraction", type=float, default=0.0, help="Train with Fraction of data sampled by AL sampler")
parser.add_argument("--frac_path", type=str, default="./Coreset", help="Folder to load indices from")
opt = parser.parse_args()
print(opt)

os.makedirs("%s/images" % opt.output_path, exist_ok=True)
os.makedirs("%s/saved_models" % opt.output_path, exist_ok=True)
os.makedirs("%s/metrics" % opt.output_path, exist_ok=True)
os.makedirs("%s/final_models" % opt.output_path, exist_ok=True)


opt.cuda = torch.cuda.is_available()

## Set Seed
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)

# Configure dataset

train_dataset = KvasirSegDataset(opt.train_dir)
test_dataset = KvasirSegDataset(opt.test_dir)




# Loss weight of L1 pixel-wise loss between translated image and real image
opt.lambda_pixel = 200

# Calculate output of image discriminator (PatchGAN)
opt.patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)



# Tensor type
opt.Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

# Setting up indices
all_indices = np.arange(len(train_dataset))
dev_indices = np.load('./validation_indices.npy')
train_indices = np.setdiff1d(all_indices, dev_indices)

fix_seed(2021)

if opt.method == "Full":
    initial_indices = list(train_indices)
    SPLITS = [1.0]

elif opt.method == 'Fraction':
    cycle = opt.data_fraction - 0.05
    initial_indices = np.load(os.path.join(opt.frac_path,f'current_indices_after{cycle}.npy'))
    SPLITS = [opt.data_fraction]

else:
    # random.seed(2021)
    initial_indices = random.sample(list(train_indices), INITIAL_BUDGET)

train_sampler = data.sampler.SubsetRandomSampler(initial_indices)
dev_sampler = data.sampler.SubsetRandomSampler(dev_indices)


# dataset with labels available
train_dataloader = data.DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=opt.batch_size, num_workers=opt.n_cpu)
val_dataloader = data.DataLoader(train_dataset, sampler=dev_sampler,
        batch_size=1, num_workers=1)
test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

dataloaders = {
    'train_loader': train_dataloader,
    'val_loader': val_dataloader,
    'test_loader': test_dataloader
}

current_indices = list(initial_indices)
# ----------
#  Training
# ----------

for cycle, split in enumerate(SPLITS):

    generator = GeneratorUNet(in_channels=opt.in_channels, out_channels=opt.out_channels)
    discriminator = Discriminator(in_channels=opt.in_channels + opt.out_channels)
    
    models = {
        'generator' : generator,
        'discriminator': discriminator,
    }

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    criteria = {
        'criterion_GAN': criterion_GAN,
        'criterion_pixelwise': criterion_pixelwise
    }


    if opt.epoch != 0:
        # Load pretrained models
        print(f"Loading model for split {str(split)} from epoch {opt.epoch} ")
        generator.load_state_dict(torch.load("%s/saved_models/generator_%s_%d.pth" % (opt.output_path, str(split), opt.epoch)))
        discriminator.load_state_dict(torch.load("%s/saved_models/discriminator_%s_%d.pth" % (opt.output_path, str(split), opt.epoch)))
    else:
        # Initialize weights
        print("Initializing weights")
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    unlabeled_indices = np.setdiff1d(list(train_indices), current_indices)
    random.shuffle(unlabeled_indices)




    # Train the model
    if opt.method == "AdversaryAL":
        train_adv(models, dataloaders, criteria, opt, split)
    else:
        train(models, dataloaders, criteria, opt, split)

    if cycle == len(SPLITS) - 1:
        print('Finished')
        break
    # Sample querries for labelling
    
    
    indices_to_label = query_samples(models, train_dataset, unlabeled_indices, current_indices, cycle, opt, split)
    current_indices = list(current_indices) + list(indices_to_label)
    sampler = data.sampler.SubsetRandomSampler(current_indices)
    train_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=opt.batch_size)
    dataloaders['train_loader'] = train_dataloader
    np.save(f'{opt.output_path}/current_indices_after{split}.npy', current_indices)


        
