import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

import torchvision

from datetime import datetime
from math import inf

import matplotlib.pyplot as plt

from tqdm import tqdm
import time

from saliency.dataset import SaliencyDataset
from saliency.model   import SaliencyModel

dataset_train = SaliencyDataset("/media/misc/dataset/DUTS-TR-aug/DUTS-TR-Image/resized",
                               "/media/misc/dataset/DUTS-TR-aug/DUTS-TR-Mask/resized")

dataset_valid = SaliencyDataset("/media/misc/dataset/DUTS-TE-aug/DUTS-TE-Image/resized",
                                "/media/misc/dataset/DUTS-TE-aug/DUTS-TE-Mask/resized")

BATCH_SIZE = 4

train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=True)

device = "cpu"
if torch.cuda.is_available():
     device = "cuda"

print(f"Using {device}")

model = SaliencyModel().to(device)
#model = SaliencyModel()

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = torch.nn.BCEWithLogitsLoss()
#loss_fn   = torch.nn.L1Loss()
#scheduler1 = ReduceLROnPlateau(optimizer)
#scheduler2 = ExponentialLR(optimizer, gamma=0.85)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2)

def train_epoch(idx, writer):
    running_loss = 0
    a = 0
    with tqdm(train_loader, unit="batch", desc=f"Epoch {idx + 1}") as tqepoch:
        iters = len(tqepoch)
        for i, data in enumerate(tqepoch):
            images, gt = data
            images, gt = images.to(device), gt.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, gt)
            loss.backward(retain_graph=False)
            optimizer.step()

            scheduler.step(idx + i / iters)

            running_loss += loss.item()

            a = i + 1
            tqepoch.set_postfix(loss=running_loss / a)

    return running_loss / a

EPOCHS = 20

def train():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/saliency_trainer_{now}")

    best_vloss = inf
    for epoch in range(0, EPOCHS):
        model.train(True)
        avg_loss = train_epoch(epoch, writer)
        model.train(False)

        running_vloss = 0.0
        a = 0
        with tqdm(test_loader, unit="batch", desc="Validation") as loader:
             for i, vdata in enumerate(loader):
                  vimages, vgts = vdata
                  vimages, vgts = vimages.to(device), vgts.to(device)
                  voutput = model(vimages)
                  vloss = loss_fn(voutput, vgts)
                  running_vloss += vloss.item()
                  a = i

        avg_vloss = running_vloss / (a + 1)
        print(f"Loss train {avg_loss}, valid {avg_vloss}")

        #scheduler1.step(avg_vloss)
        # scheduler2.step()

        writer.add_scalars('Average loss', {
             'train': avg_loss,
             'validation': avg_vloss
        }, epoch)

        writer.add_scalar('Average loss/divergence',
                          abs(avg_loss - avg_vloss), epoch)

        writer.flush()

        if avg_vloss < best_vloss:
            print(f"new PB Δ{best_vloss - avg_vloss}")
            best_vloss = avg_vloss

        model_path = f"models/model_{avg_vloss:.4f}_{now}_{epoch}_"
        torch.save(model.state_dict(), model_path)

#model.load_state_dict(torch.load("models/model_0.1781_20220829_180858_19_"))
#model.eval()
#train()

model.load_state_dict(torch.load("models/model_0.1781_20220829_180858_19_"))
model.eval()

image, gt = next(iter(test_loader))

fig, axes = plt.subplots(BATCH_SIZE, 4)

image = image.to(device)
beg_time = time.time_ns()
pred = model(image)
end_time = time.time_ns()

axes[0][0].title.set_text("Source image")
axes[0][1].title.set_text("Ground truth")
axes[0][2].title.set_text("Predicted saliency map")
axes[0][3].title.set_text("Masked image")

for i in range(0, BATCH_SIZE):
     loss = loss_fn(pred[i].detach().cpu(), gt[i])

     axes[i][0].imshow(image[i].detach().cpu().permute((1,2,0)))
     axes[i][1].imshow(gt[i][0])

     prediction = pred[i]
     pred_image: torch.Tensor = torch.sigmoid(prediction)
     axes[i][2].imshow(pred_image.detach().cpu().permute((1,2,0)))
     axes[i][2].yaxis.set_label_position("right")
     axes[i][2].set_ylabel(f"(Loss: {loss:.4f})", rotation=0, ha="left")

     axes[i][3].imshow(pred_image.detach().cpu().permute((1,2,0)) * image[i].detach().cpu().permute((1,2,0)))

     with open(f"pred_{i}.png", "wb") as out:
         torchvision.utils.save_image(pred_image, out)

time_taken_ms = (end_time - beg_time) / 1e6
time_per_image = time_taken_ms / BATCH_SIZE
print(f"Batch took {time_taken_ms}ms -- {time_per_image}ms avg per image")

plt.show()
