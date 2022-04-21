from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class DepthDataset(Dataset):
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        with open(self.dataset_file) as file:
            self.data = [line.strip() for line in file]
        self.transform_x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5])
        ])
        self.transform_y = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5,
            std=0.5)
        ])
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
                idx = idx.tolist()
        x_filename, y_filename = self.data[idx].split(',')
        x_image = Image.open(x_filename).convert('RGB')
        y_image = Image.open(y_filename)

        return {"A": self.transform_x(x_image), "B": self.transform_y(y_image)}

class KvasirSegDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = 'images'
        self.mask_dir = 'masks'
        
        self.transform_x = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5])
        ])
        self.transform_y = transforms.Compose([
            transforms.Resize((256,256)),            
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5,
            std=0.5)
        ])
        self.image_files = [f for f in sorted(os.listdir(os.path.join(self.root_dir, self.image_dir))) 
                            if not f.startswith('.')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_fn = os.path.join(self.root_dir, self.image_dir, self.image_files[idx])
        mask_fn = os.path.join(self.root_dir, self.mask_dir, self.image_files[idx])
        
        image = self.transform_x(Image.open(image_fn).convert('RGB'))
        mask = self.transform_y(Image.open(mask_fn))
        mask = torch.unsqueeze(mask[0], dim=0)

        return {"A": image, "B": mask}
