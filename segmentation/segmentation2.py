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
# CONFIG
# ==============================
IMAGE_SIZE = 96
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = torch.device("cpu")

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

os.makedirs("outputs_seg", exist_ok=True)

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

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

        print("Building image-mask pairs...")
        self.pairs = []

        for root, _, files in os.walk(image_dir):
            for file in files:
                if not file.lower().endswith(".jpg"):
                    continue

                img_path = os.path.join(root, file)
                base = os.path.splitext(file)[0]

                mask_path = os.path.join(mask_dir, base + "_segmentation.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(mask_dir, base + ".png")

                if os.path.exists(mask_path):
                    self.pairs.append((img_path, mask_path))

        print(f"✅ Found {len(self.pairs)} pairs")

        # imbalance weights
        weights = []
        print("Computing lesion-area weights...")
        for _, m in tqdm(self.pairs):
            mask = Image.open(m).convert("L")
            area = (np.array(mask) > 0).sum()
            weights.append(1.0 / (area + 1))

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
# BLOCKS
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

# ==============================
# ATTENTION BLOCK
# ==============================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Conv2d(F_int, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi

# ==============================
# U-NET
# ==============================
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
# ATTENTION U-NET
# ==============================
class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(128, 256)

        self.att2 = AttentionBlock(128, 128, 64)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.att1 = AttentionBlock(64, 64, 32)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bridge = self.bridge(self.pool2(d2))

        u2 = self.up2(bridge)
        d2_att = self.att2(u2, d2)
        u2 = torch.cat([u2, d2_att], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        d1_att = self.att1(u1, d1)
        u1 = torch.cat([u1, d1_att], dim=1)
        u1 = self.conv1(u1)

        return torch.sigmoid(self.out(u1))

# ==============================
# LOSS
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
def train_model(model_class, tag):
    dataset = LesionDataset(IMAGE_DIR, MASK_DIR)

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

    model = model_class().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nTraining {tag}...")

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

        print(f"{tag} Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(),
                   f"outputs_seg/{tag}_epoch_{epoch+1}.pth")

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
model_unet, loader_unet = train_model(UNet, "unet")
metrics_unet = evaluate(model_unet, loader_unet)

model_att, loader_att = train_model(AttentionUNet, "attention_unet")
metrics_att = evaluate(model_att, loader_att)

comparison = pd.DataFrame(
    [metrics_unet, metrics_att],
    index=["UNet", "Attention_UNet"]
)

comparison.to_csv("outputs_seg/segmentation_model_comparison.csv")

print("\n✅ Week 2 Multi-Model Segmentation Completed!")