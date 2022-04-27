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
from depth_config import *
from train_test import train
from selection_methods import query_samples



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="depthdata", help="name of the dataset")
parser.add_argument("--output_path" , type=str, default="results", help="Output paths")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--in_channels", type=int, default=3, help="number of input image channels")
parser.add_argument("--out_channels", type=int, default=1, help="number of output channels")
parser.add_argument("--num_gen_steps", type=int, default=1, help="Number of times to train generator")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--method", type=str, default='Full', help='Training Type')
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
train_dataset = DepthDataset("train.txt")
val_dataset = DepthDataset("valid.txt")




# Loss weight of L1 pixel-wise loss between translated image and real image
opt.lambda_pixel = 200

# Calculate output of image discriminator (PatchGAN)
opt.patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)



# Tensor type
opt.Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

# Setting up indices

all_indices = set(np.arange(NUM_IMAGES))

fix_seed(2021)

if opt.method == "Full":
    initial_indices = list(all_indices)
    SPLITS = [1.0]

elif opt.method == 'Fraction':
    cycle = opt.data_fraction - 0.05
    initial_indices = np.load(os.path.join(opt.frac_path,f'current_indices_after{cycle}.npy'))
    SPLITS = [opt.data_fraction]

else:
    # random.seed(2021)
    initial_indices = random.sample(list(all_indices), INITIAL_BUDGET)

sampler = data.sampler.SubsetRandomSampler(initial_indices)

# dataset with labels available
train_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
        batch_size=opt.batch_size, num_workers=opt.n_cpu)
val_dataloader = data.DataLoader(val_dataset, batch_size=1,shuffle=True, num_workers=1)

dataloaders = {
    'train_loader': train_dataloader,
    'val_loader': val_dataloader
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

    unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
    
    # Train the model
    train(models, dataloaders, criteria, opt, split)

    
    if cycle == len(SPLITS) - 1:
        print('Finished')
        break


    # Sample querries for labelling
    random.shuffle(unlabeled_indices)
    if COMPLETE:
        SUBSET = len(unlabeled_indices)
    subset = unlabeled_indices[:SUBSET]
    indices_to_label = query_samples(models, train_dataset, subset, current_indices, cycle, opt, split)
    current_indices = list(current_indices) + list(indices_to_label)
    sampler = data.sampler.SubsetRandomSampler(current_indices)
    train_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=opt.batch_size)
    dataloaders['train_loader'] = train_dataloader
    np.save(f'{opt.output_path}/current_indices_after{split}.npy', current_indices)


        
