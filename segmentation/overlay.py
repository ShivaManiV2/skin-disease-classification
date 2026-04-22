import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn as nn

# ==============================
# CONFIG (EDIT IMAGE ID ONLY)
# ==============================
IMAGE_SIZE = 96
DEVICE = torch.device("cpu")

IMAGE_PATH = "data/images/ISIC_0024306.jpg"  # 🔴 change if needed
ATT_MODEL_PATH = "outputs_seg/attention_unet_epoch_5.pth"

# ==============================
# SAFE MASK FINDER
# ==============================
def find_mask(image_id):
    p1 = f"data/masks/{image_id}_segmentation.png"
    p2 = f"data/masks/{image_id}.png"

    if os.path.exists(p1):
        return p1
    elif os.path.exists(p2):
        return p2
    else:
        raise FileNotFoundError(f"Mask not found for {image_id}")

# ==============================
# TRANSFORMS
# ==============================
img_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=InterpolationMode.NEAREST
    ),
    transforms.ToTensor()
])

# ==============================
# MODEL BLOCKS
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
print("Loading image...")

image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))

image_id = os.path.basename(IMAGE_PATH).replace(".jpg", "")
mask_path = find_mask(image_id)

print(f"Using mask: {mask_path}")

mask_gt = Image.open(mask_path).convert("L")
mask_gt_tensor = mask_transform(mask_gt)[0].numpy()
gt_bin = (mask_gt_tensor > 0.5).astype(np.uint8)

# ==============================
# LOAD MODEL
# ==============================
print("Loading Attention U-Net...")

model = AttentionUNet().to(DEVICE)
model.load_state_dict(torch.load(ATT_MODEL_PATH, map_location=DEVICE))
model.eval()

# ==============================
# PREDICT
# ==============================
img_tensor = img_transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(img_tensor)[0][0].cpu().numpy()

pred_bin = (pred > 0.5).astype(np.uint8)

# ==============================
# OVERLAY FUNCTION
# ==============================
def overlay_mask(image_np, mask_np, color, alpha=0.4):
    overlay = image_np.copy()
    color_mask = np.zeros_like(image_np)
    color_mask[mask_np == 1] = color

    blended = image_np * (1 - alpha) + color_mask * alpha
    blended = blended.astype(np.uint8)

    overlay[mask_np == 1] = blended[mask_np == 1]
    return overlay

# create overlays
overlay_gt = overlay_mask(image_np, gt_bin, (0, 255, 0))   # green
overlay_pred = overlay_mask(image_np, pred_bin, (255, 0, 0))  # red

# ==============================
# PLOT
# ==============================
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.imshow(image_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(overlay_gt)
plt.title("Ground Truth (Green)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay_pred)
plt.title("Attention U-Net (Red)")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs_seg/overlay_attention_comparison.png", dpi=300)
plt.show()

print("✅ Saved: outputs_seg/overlay_attention_comparison.png")