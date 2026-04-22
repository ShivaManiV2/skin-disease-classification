import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn as nn

# ==============================
# CONFIG
# ==============================
IMAGE_SIZE = 96
DEVICE = torch.device("cpu")

IMAGE_PATH = "data/images/ISIC_0024306.jpg"   # 🔴 change
MASK_PATH = "data/masks/ISIC_0024306.png"  # 🔴 change

UNET_PATH = "outputs_seg/unet_epoch_5.pth"
ATT_PATH = "outputs_seg/attention_unet_epoch_5.pth"

# ==============================
# TRANSFORMS
# ==============================
img_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor()
])

# ==============================
# MODEL BLOCKS (same as training)
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
# LOAD IMAGE & MASK
# ==============================
image = Image.open(IMAGE_PATH).convert("RGB")
mask_gt = Image.open(MASK_PATH).convert("L")

img_tensor = img_transform(image).unsqueeze(0).to(DEVICE)
mask_gt_tensor = mask_transform(mask_gt)[0].numpy()

# ==============================
# LOAD MODELS
# ==============================
unet = UNet().to(DEVICE)
unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
unet.eval()

att_unet = AttentionUNet().to(DEVICE)
att_unet.load_state_dict(torch.load(ATT_PATH, map_location=DEVICE))
att_unet.eval()

# ==============================
# PREDICT
# ==============================
with torch.no_grad():
    pred_unet = unet(img_tensor)[0][0].cpu().numpy()
    pred_att = att_unet(img_tensor)[0][0].cpu().numpy()

pred_unet_bin = (pred_unet > 0.5).astype(np.uint8)
pred_att_bin = (pred_att > 0.5).astype(np.uint8)

# ==============================
# VISUALIZE
# ==============================
plt.figure(figsize=(12,6))

plt.subplot(1,4,1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(mask_gt_tensor, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(pred_unet_bin, cmap="gray")
plt.title("UNet Prediction")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(pred_att_bin, cmap="gray")
plt.title("Attention UNet Prediction")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs_seg/model_mask_comparison.png", dpi=300)
plt.show()

print("✅ Saved: outputs_seg/model_mask_comparison.png")