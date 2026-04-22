import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sklearn.metrics import precision_score, recall_score

# ==============================
# CONFIG (CPU SAFE)
# ==============================
IMAGE_SIZE = 96
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = torch.device("cpu")

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

os.makedirs("outputs_seg", exist_ok=True)
os.makedirs("outputs_seg/plots", exist_ok=True)

# ==============================
# DATASET
# ==============================
class LesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, brightness=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.brightness = brightness

        self.img_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

        # ✅ IMPORTANT FIX: nearest interpolation
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

        print("Building valid image-mask pairs...")
        self.pairs = []

        for root, _, files in os.walk(image_dir):
            for file in files:
                if not file.lower().endswith(".jpg"):
                    continue

                img_path = os.path.join(root, file)
                base_id = os.path.splitext(file)[0]

                mask_path = os.path.join(mask_dir, base_id + "_segmentation.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(mask_dir, base_id + ".png")

                if os.path.exists(mask_path):
                    self.pairs.append((img_path, mask_path))

        print(f"✅ Found {len(self.pairs)} valid pairs")

        if len(self.pairs) == 0:
            raise RuntimeError("No matching masks found.")

        # ==============================
        # SAMPLE WEIGHTS (imbalance)
        # ==============================
        print("Computing lesion-area weights...")
        weights = []

        for _, mask_path in tqdm(self.pairs):
            mask = Image.open(mask_path).convert("L")
            lesion_area = (np.array(mask) > 0).sum()
            weights.append(1.0 / (lesion_area + 1))

        self.sample_weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.brightness:
            image = transforms.functional.adjust_brightness(image, 1.5)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask

# ==============================
# U-NET
# ==============================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bridge = self.bridge(self.pool2(d2))

        u2 = self.up2(bridge)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return torch.sigmoid(self.out(u1))

# ==============================
# LOSSES
# ==============================
bce_loss = nn.BCELoss()

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)

# ==============================
# METRICS
# ==============================
def compute_metrics(pred, target):
    pred_bin = (pred > 0.5).cpu().numpy().astype(int).flatten()
    target_bin = target.cpu().numpy().astype(int).flatten()

    intersection = (pred_bin & target_bin).sum()
    union = (pred_bin | target_bin).sum()

    dice = (2 * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-6)
    iou = intersection / (union + 1e-6)
    precision = precision_score(target_bin, pred_bin, zero_division=0)
    recall = recall_score(target_bin, pred_bin, zero_division=0)

    return dice, iou, precision, recall

# ==============================
# TRAIN FUNCTION
# ==============================
def train_model(brightness=False):
    print(f"\nTraining (brightness={brightness})")

    dataset = LesionDataset(IMAGE_DIR, MASK_DIR, brightness)

    # train/val split
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_weights = dataset.sample_weights[train_idx]

    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            sampler=SubsetRandomSampler(val_idx))

    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = combined_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # ✅ checkpoint
        torch.save(
            model.state_dict(),
            f"outputs_seg/unet_epoch_{epoch+1}_{'bright' if brightness else 'normal'}.pth"
        )

    return model, val_loader

# ==============================
# EVALUATE
# ==============================
def evaluate(model, loader):
    model.eval()
    dices, ious, precisions, recalls = [], [], [], []

    with torch.no_grad():
        for imgs, masks in loader:
            preds = model(imgs.to(DEVICE)).cpu()
            for p, m in zip(preds, masks):
                d, i, pr, re = compute_metrics(p, m)
                dices.append(d)
                ious.append(i)
                precisions.append(pr)
                recalls.append(re)

    return {
        "Dice": np.mean(dices),
        "IoU": np.mean(ious),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
    }

# ==============================
# MAIN
# ==============================
model_normal, loader_normal = train_model(False)
metrics_normal = evaluate(model_normal, loader_normal)

model_bright, loader_bright = train_model(True)
metrics_bright = evaluate(model_bright, loader_bright)

comparison = pd.DataFrame(
    [metrics_normal, metrics_bright],
    index=["Normal", "Brightness"]
)
comparison.to_csv("outputs_seg/metric_comparison.csv")

print("\n✅ Week 2 Segmentation Completed!")