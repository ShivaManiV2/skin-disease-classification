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
from torchvision.models import MobileNet_V2_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)

# ==============================
# CONFIG (CPU FRIENDLY)
# ==============================
IMAGE_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 15
DEVICE = torch.device("cpu")

DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# ==============================
# LOAD METADATA
# ==============================
print("Loading metadata...")
df = pd.read_csv(METADATA_PATH)

df['label'] = df['dx'].astype('category').cat.codes.astype(int)
num_classes = df['label'].nunique()

# ==============================
# DATASET EXPLORATION
# ==============================
print("\nClass Distribution:")
print(df['dx'].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(x='dx', data=df)
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.close()

# ==============================
# TRAIN VAL TEST SPLIT
# ==============================
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

print(f"Train: {len(train_df)}")
print(f"Val: {len(val_df)}")
print(f"Test: {len(test_df)}")

# ==============================
# CLASS WEIGHTS (KEEP)
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
#  TRAIN TRANSFORMS (AUGMENTED)
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# 🔥 TEST TRANSFORMS (NO AUG)
# ==============================
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# DATA LOADERS (NO SAMPLER)
# ==============================
train_loader = DataLoader(
    HAMDataset(train_df, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    HAMDataset(val_df, test_transform),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    HAMDataset(test_df, test_transform),
    batch_size=BATCH_SIZE
)

# ==============================
# TRANSFER LEARNING MODEL
# ==============================
print("\nLoading MobileNet (Transfer Learning)...")

weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)

# 🔥 freeze most layers
for param in model.features.parameters():
    param.requires_grad = False

# 🔥 unfreeze top blocks (important)
for param in model.features[-4:].parameters():
    param.requires_grad = True

model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=3e-4)

print("Transfer learning model ready.")

# ==============================
# TRAINING LOOP
# ==============================
print("\nTraining started...")
history = []

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

    train_acc = correct / total
    epoch_loss = running_loss / len(train_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}")

    torch.save(
        model.state_dict(),
        f"outputs/checkpoints/model_epoch_{epoch+1}.pth"
    )

    history.append({
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "train_accuracy": train_acc
    })

pd.DataFrame(history).to_csv("outputs/training_history.csv", index=False)

# ==============================
# EVALUATION
# ==============================
def evaluate_model(loader, name="Test"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(f"\n{name} Classification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    return all_labels, all_preds

# ==============================
# TEST
# ==============================
true_labels, predictions = evaluate_model(test_loader, "Test")

print("\n✅ Week 1 pipeline completed!")