import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# FOLDERS
# ==============================
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)

# ==============================
# CONFIG
# ==============================
IMAGE_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 15
DEVICE = torch.device("cpu")

DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# ==============================
# LOAD DATA
# ==============================
print("Loading metadata...")
df = pd.read_csv(METADATA_PATH)

df['label'] = df['dx'].astype('category').cat.codes.astype(int)
num_classes = df['label'].nunique()

print("\nClass Distribution:")
print(df['dx'].value_counts())

# ==============================
# SPLIT
# ==============================
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

# ==============================
# CLASS WEIGHTS
# ==============================
class_counts = train_df['label'].value_counts().sort_index()
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes
class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)

# ==============================
# IMAGE PATH
# ==============================
def get_image_path(image_id):
    path = os.path.join(DATA_DIR, "images", image_id + ".jpg")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path

# ==============================
# DATASET
# ==============================
class HAMDataset(Dataset):
    def __init__(self, dataframe, transform=None, brightness_factor=1.0):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.brightness_factor = brightness_factor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = get_image_path(row['image_id'])

        image = Image.open(img_path).convert("RGB")

        if self.brightness_factor != 1.0:
            image = transforms.functional.adjust_brightness(
                image, self.brightness_factor
            )

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label

# ==============================
# TRANSFORMS
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# LOADERS
# ==============================
train_loader = DataLoader(
    HAMDataset(train_df, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    HAMDataset(test_df, test_transform),
    batch_size=BATCH_SIZE
)

# ==============================
# EVALUATION
# ==============================
def evaluate_model(model, loader, name):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(f"\n{name} Classification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"outputs/{name.lower().replace(' ', '_')}_cm.png")
    plt.close()

    return np.mean(np.array(all_preds) == np.array(all_labels))

# ==============================
# TRAIN FUNCTION
# ==============================
def train_model(model, model_name):
    print(f"\n===== Training {model_name} =====")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"\n[{model_name}] Epoch {epoch+1}/{EPOCHS}")
        print(f"Loss: {running_loss/len(train_loader):.4f}, Train Acc: {correct/total:.4f}")

    return model

# ==============================
# 🔥 MODEL 1 — MobileNetV2
# ==============================
print("\nLoading MobileNetV2...")
mobilenet = models.mobilenet_v2(
    weights=MobileNet_V2_Weights.IMAGENET1K_V1
)

for param in mobilenet.features.parameters():
    param.requires_grad = False

for param in mobilenet.features[-4:].parameters():
    param.requires_grad = True

mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)
mobilenet = mobilenet.to(DEVICE)

mobilenet = train_model(mobilenet, "MobileNetV2")
acc_m = evaluate_model(mobilenet, test_loader, "MobileNetV2 Test")

# ==============================
# 🔥 MODEL 2 — ResNet18
# ==============================
print("\nLoading ResNet18...")
resnet = models.resnet18(
    weights=ResNet18_Weights.IMAGENET1K_V1
)

for param in resnet.parameters():
    param.requires_grad = False

for param in resnet.layer4.parameters():
    param.requires_grad = True

resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = resnet.to(DEVICE)

resnet = train_model(resnet, "ResNet18")
acc_r = evaluate_model(resnet, test_loader, "ResNet18 Test")

# ==============================
# COMPARISON
# ==============================
print("\n==============================")
print("FINAL COMPARISON")
print("==============================")
print(f"MobileNetV2 Accuracy: {acc_m:.4f}")
print(f"ResNet18 Accuracy: {acc_r:.4f}")

print("\n✅ Week 1 Classification Completed!")