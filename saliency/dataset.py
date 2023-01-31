import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
from glob import glob
import os.path

import torchvision.transforms as transforms

class SaliencyDataset(Dataset):
    input_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
        transforms.ToTensor(),
    ])

    output_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

    def __init__(self, image_dir, gt_dir):
        self.gt_dir      = gt_dir
        self.image_dir   = image_dir
        self.image_files = glob(os.path.join(self.image_dir, "*.jpg"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Ground truth images have the same filenames as the raw
        # images, but are PNG files
        gt_path  = os.path.join(
            self.gt_dir,
            os.path.splitext(os.path.basename(img_path))[0] + ".png"
        )

        image = read_image(img_path)
        gt    = read_image(gt_path)

        # Choose a random seed so the input and output transforms
        # are randomized in the same way
        _seed = np.random.randint(2**16)

        torch.manual_seed(_seed)
        image = self.input_transform(image)

        torch.manual_seed(_seed)
        gt = self.output_transform(gt)

        return image, gt
