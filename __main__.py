from math import inf
import torch
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch import nn
from torchvision.io import read_image
from torchvision import models
import torchvision.transforms as transforms
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from glob import glob
import os.path
from datetime import datetime, time
import numpy as np
import random
from time import time

import matplotlib.pyplot as plt

from tqdm import tqdm

class SaliencyDataset(Dataset):
    def __init__(self, image_dir, gt_dir, transform=None, target_transform=None):
        self.gt_dir           = gt_dir
        self.img_dir          = image_dir
        self.transform        = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(glob(os.path.join(self.img_dir, "*.jpg")))

    def __getitem__(self, idx):
        images   = glob(os.path.join(self.img_dir, "*.jpg"))
        img_path = images[idx]

        gt_path  = os.path.join(
            self.gt_dir,
            os.path.splitext(os.path.basename(img_path))[0] + ".png")

        image = read_image(img_path)
        gt    = read_image(gt_path)

        _seed = np.random.randint(2**16)

        torch.manual_seed(_seed)
        if self.transform:
            image = self.transform(image)

        torch.manual_seed(_seed)
        if self.target_transform:
            gt = self.target_transform(gt)

        return image, gt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(
            # Convolution
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 7, stride=1, padding=3),
            nn.ReLU(),
        )

        self.dropout1 = nn.Dropout()

        self.unpool_deconv5 = nn.MaxUnpool2d(2, 2)

        self.deconv5 = nn.Sequential(
            # Transpose convolution
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.unpool_deconv4 = nn.MaxUnpool2d(2, 2)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.unpool_deconv3 = nn.MaxUnpool2d(2, 2)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.unpool_deconv2 = nn.MaxUnpool2d(2, 2)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.unpool_deconv1 = nn.MaxUnpool2d(2, 2)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.dropout2 = nn.Dropout()

        self.score = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        x, indices1 = self.conv1(x)
        x, indices2 = self.conv2(x)
        x, indices3 = self.conv3(x)
        x, indices4 = self.conv4(x)
        x, indices5 = self.conv5(x)
        x           = self.conv6(x)
        x           = self.dropout1(x)
        x           = self.unpool_deconv5(x, indices5)
        x           = self.deconv5(x)
        x           = self.unpool_deconv4(x, indices4)
        x           = self.deconv4(x)
        x           = self.unpool_deconv3(x, indices3)
        x           = self.deconv3(x)
        x           = self.unpool_deconv2(x, indices2)
        x           = self.deconv2(x)
        x           = self.unpool_deconv1(x, indices1)
        x           = self.deconv1(x)
        x           = self.dropout2(x)
        x           = self.score(x)

        return x

data_transform = transforms.Compose([
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
    transforms.Grayscale(3),
    transforms.ToTensor(),
])

dataset = SaliencyDataset("/media/misc/dataset/saliency/img/resized", "/media/misc/dataset/saliency/gt/resized", data_transform, output_transform)
n_train = int(0.8 * len(dataset))
data_split = random_split(dataset, [n_train, len(dataset) - n_train])

train_loader = DataLoader(data_split[0], batch_size=4, shuffle=True)
test_loader  = DataLoader(data_split[1], batch_size=4, shuffle=True)

device = "cpu"
if torch.cuda.is_available():
     device = "cuda"

print(f"Using {device}")

model = Model().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = torch.nn.L1Loss()
scheduler1 = ReduceLROnPlateau(optimizer)
scheduler2 = ExponentialLR(optimizer, gamma=0.9)

def train_epoch(idx, writer):
    running_loss = 0
    a = 0
    with tqdm(train_loader, unit="batch") as tqepoch:
        for i, data in enumerate(tqepoch):
            images, gt = data
            images, gt = images.to(device), gt.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, gt)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            a = i + 1
            tqepoch.set_postfix(loss=running_loss / a)

    return running_loss / a

EPOCHS = 80

def train():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/saliency_trainer_{now}")

    best_vloss = inf
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")

        model.train(True)
        avg_loss = train_epoch(epoch, writer)
        model.train(False)

        running_vloss = 0.0
        a = 0
        for i, vdata in enumerate(test_loader):
            vimages, vgts = vdata
            vimages, vgts = vimages.to(device), vgts.to(device)
            voutput = model(vimages)
            vloss = loss_fn(voutput, vgts)
            running_vloss += vloss.item()
            a = i

        avg_vloss = running_vloss / (a + 1)
        print(f"Loss train {avg_loss}, valid {avg_vloss}")

        scheduler1.step(avg_vloss)
        scheduler2.step()

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/validation', avg_vloss, epoch)

        writer.flush()

        if avg_vloss < best_vloss:
            print(f"new PB Δ{best_vloss - avg_vloss}")
            best_vloss = avg_vloss
            model_path = f"models/model_{now}_{epoch}"
            torch.save(model.state_dict(), model_path)

train()

# model.load_state_dict(torch.load("model_20220827_211445_10"))
# model.eval()

# image, _ = next(iter(test_loader))

# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(image[0][0])

# image = image.to(device)
# pred = model(image)
# pred = pred.detach().cpu()
# print(pred.shape)
# axes[1].imshow(pred[0].permute((1,2,0)))
# plt.show()
