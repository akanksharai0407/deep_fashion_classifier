# -*- coding: utf-8 -*-
"""DeepLearningF.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KaWelAV-PhGmTDXc5f4MvtcMnYhoR-vH

# Importing the necessary libraries
"""

!pip install kagglehub seaborn
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,classification_report
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision import models

"""# Data overview"""

# Download dataset
import kagglehub
path = kagglehub.dataset_download("vikashrajluhaniwal/fashion-images")

# Download CSV
data = pd.read_csv(path + '/data/fashion.csv')
print("Column names", data.columns.tolist())
print(data.head())

"""Find the unique values of each columns and also count of each value in categorical columns to understand is the data balanced or not

"""

## Image filtering and distribution visualization

def make_path(row):
    return os.path.join(path, 'data', row['Category'], row['Gender'],
                      'Images', 'images_with_product_ids', row['Image'])

data['full_path']= data.apply(make_path, axis=1)
data = data[data['full_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Images found: {len(data)}")

print("Number of unique values:")
print(f"- Category: {data['Category'].nunique()}")
print(f"- SubCategory: {data['SubCategory'].nunique()}")
print(f"- ProductType: {data['ProductType'].nunique()}")

# Output of the distribution by category

def display_value_counts(column_name):
    counts = data[column_name].value_counts().reset_index()
    counts.columns= [column_name, 'Quantity']
    print(f"\nDistribution by '{column_name}':")
    display(counts)

# Output
display_value_counts('Category')

print(data['ProductType'].unique())

"""# Data preprocessing

For further training of models, we chose the Product type category.
"""

# Calculating the quantity by category

category_counts= data['ProductType'].value_counts()

# Filtering categories that have at least 50 images
filtered_categories = category_counts[category_counts >=50]
desired = filtered_categories.index.tolist()

# Filtering
filtered_df = data[data['ProductType'].isin(desired)].reset_index(drop=True)
print(f" Filtered {len(desired)} category:")
print(filtered_df['ProductType'].value_counts())

"""Here we can see visualized examples of each ProductType"""

classes = sorted(data['ProductType'].unique())

# Pick 10 random indices
num_samples = 10
random_indices= random.sample(range(len(data)), num_samples)
fig, axes = plt.subplots(2,5, figsize=(12,5))
for i, idx in enumerate(random_indices):
    row= data.iloc[idx]
    img= Image.open(row['full_path']).convert('RGB')
    ax= axes[i//5, i%5]
    ax.imshow(img)
    ax.set_title(row['ProductType'])
    ax.axis('off')
plt.tight_layout()
plt.show()

"""Encoding of class labels"""

encoder = LabelEncoder()
filtered_df['label'] = encoder.fit_transform(filtered_df['ProductType'])
num_classes = len(encoder.classes_)
print("Classes:", encoder.classes_)

"""Splitting data into train/val/test"""

train_val_df, test_df = train_test_split(filtered_df, test_size=0.10, stratify=filtered_df['label'], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2222, stratify=train_val_df['label'], random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

"""# Model setup and data preparation

Setting up hyperparameters and basic transformation of images to feed into the model
"""

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10

# Data Preparation
tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

"""Creating Dataset and DataLoader"""

class FashionDataset(Dataset):
     def __init__(self, df, transforms=None):
         self.df = df.reset_index(drop=True)
         self.transforms= transforms

     def __len__(self):
         return len(self.df)

     def __getitem__(self, idx):
         row = self.df.iloc[idx]
         img = Image.open(row['full_path']).convert('RGB')
         if self.transforms:
            img = self.transforms(img)
         return img, row['label']

train_loader = DataLoader(FashionDataset(train_df, transforms=tfms), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FashionDataset(val_df, transforms=tfms), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(FashionDataset(test_df, transforms=tfms), batch_size=BATCH_SIZE, shuffle=False)

"""# CNN model definition and training"""

class CNN(nn.Module):
     def __init__(self,num_classes):
         super(CNN, self).__init__()
         self.conv1= nn.Conv2d(3, 32, 3, padding=1)
         self.conv2= nn.Conv2d(32, 64, 3, padding=1)
         self.dropout1 = nn.Dropout(0.25)
         self.dropout2 = nn.Dropout(0.5)
         self.fc1 = nn.Linear(64*64*64, 128)
         self.fc2 = nn.Linear(128,num_classes)

     def forward(self, x):
         x = F.relu(self.conv1(x))
         x = F.relu(self.conv2(x))
         x = F.max_pool2d(x, 2)
         x = self.dropout1(x)
         x = torch.flatten(x, 1)
         x = F.relu(self.fc1(x))
         x = self.dropout2(x)
         x = self.fc2(x)
         return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

history = {"Loss": [], "Val Loss": [], "Accuracy": [], "Val Accuracy": []}
pbar = tqdm(range(EPOCHS))

for _ in pbar:
    model.train()
    total, correct, running_loss=0,0,0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs= model(imgs)
        loss= criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0,0,0
    with torch.no_grad():
         for imgs, labels in val_loader:
             imgs, labels = imgs.to(device), labels.to(device)
             outputs = model(imgs)

             val_loss += criterion(outputs, labels).item()*imgs.size(0)
             val_correct += (outputs.argmax(1)== labels).sum().item()
             val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    #Logging
    history["Loss"].append(train_loss)
    history["Val Loss"].append(val_loss)
    history["Accuracy"].append(train_acc)
    history["Val Accuracy"].append(val_acc)

    pbar.set_postfix_str(
        f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, "
        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
    )

    print(f"Epoch {_+1}/{EPOCHS} - "
          f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

"""**Model evaluation**"""

# Accuracy and Loss curves
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), history["Accuracy"], label='Train')
plt.plot(range(1, EPOCHS+1), history["Val Accuracy"], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(range(1,EPOCHS+1), history["Loss"], label='Train')
plt.plot(range(1,EPOCHS+1), history["Val Loss"], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

"""**Quantitative results:**"""

# Testing model
big_test_loader = DataLoader(FashionDataset(test_df, transforms=tfms), batch_size=len(test_df), shuffle=False)

testImgs, testLabels = next(iter(big_test_loader))
testImgs = testImgs.to(device)
testLabels = testLabels.to(device)

# Forward pass
model.eval()
with torch.no_grad():
     out = model(testImgs)
     preds = out.argmax(1)

# Accuracy
acc = accuracy_score(testLabels.cpu(), preds.cpu()) * 100
print(f"Test Accuracy: {acc:.4f}%")

# Confusion Matrix
cm = confusion_matrix(testLabels.cpu(), preds.cpu())

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="hot",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()

#  Classification Report : Precision, Recall, F1-score
print("Classification Report for CNN model:")
print(classification_report(testLabels.cpu(), preds.cpu(), target_names=encoder.classes_))

# Pick one random image from test set
sample_row = test_df.sample(1).iloc[0]
img_path = sample_row['full_path']
true_label = sample_row['label']

# Load and preprocess image
img = Image.open(img_path).convert("RGB")
img_tensor = tfms(img).unsqueeze(0).to(device)

# Forward pass through CNN
model.eval()
with torch.no_grad():
     outputs = model(img_tensor)
     probs = torch.softmax(outputs, dim=1).squeeze()

# Get top-5 predictions
topk_indices = torch.topk(probs, 5).indices.tolist()

print("Top-5 predicted classes (CNN):")
for idx in topk_indices:
    prob = probs[idx].item()
    print(f"[{idx}] {encoder.classes_[idx]:20} ({prob * 100:.2f}%)")

# Show image with true label
plt.imshow(img)
plt.title(f"True: {true_label}")
plt.axis("off")
plt.show()

"""# ResNet model definition and training"""

from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)
model_resnet = model_resnet.to(device)

optimizer = optim.Adam(model_resnet.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

#Training
model_resnet.train()

history = {
    "train_loss": [],
    "train_acc": []
}
for epoch in range(EPOCHS):
    epoch_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

    for i, (x, y) in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model_resnet(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct/total)

        # Validation
    model_resnet.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
         for x_val, y_val in val_loader:
             x_val, y_val = x_val.to(device), y_val.to(device)
             val_outputs = model_resnet(x_val)
             val_loss += criterion(val_outputs, y_val).item()*x_val.size(0)
             val_preds = val_outputs.argmax(dim=1)
             val_correct += (val_preds == y_val).sum().item()
             val_total += y_val.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/total:.4f}, Accuracy: {correct/total:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    history["train_loss"].append(epoch_loss / total)
    history["train_acc"].append(correct / total)
    history.setdefault("val_loss", []).append(val_loss)
    history.setdefault("val_acc", []).append(val_acc)

    model_resnet.train()

# Accuracy curve
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet Train Accuracy")
plt.legend()
plt.show()

# Loss curve
plt.plot(history["train_loss"], label="Train Loss", color="orange")
plt.plot(history["val_loss"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet Train Loss")
plt.legend()
plt.show()

# Evaluation on test set
model_resnet.eval()
y_true, y_pred = [],[]
test_loss, correct = 0.0,0

with torch.no_grad():
     pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluation")

     for i, (x, y) in pbar:
        x, y = x.to(device), y.to(device)
        outputs = model_resnet(x)
        loss = criterion(outputs, y)
        test_loss += loss.item() * x.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Metrics
test_loss /= len(test_loader.dataset)
test_acc = correct / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Classification Report
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="hot",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (ResNet Test Set)")
plt.tight_layout()
plt.show()

"""# ViT model definition and training"""

VIT_BATCH_SIZE = 64

#dataloader
vit_train_loader = DataLoader(FashionDataset(train_df, transforms=tfms), batch_size=VIT_BATCH_SIZE, shuffle=True)
vit_val_loader = DataLoader(FashionDataset(val_df, transforms=tfms), batch_size=VIT_BATCH_SIZE, shuffle=False)
vit_test_loader = DataLoader(FashionDataset(test_df, transforms=tfms), batch_size=VIT_BATCH_SIZE, shuffle=False)

# ViT
class Patch(nn.Module):
    def __init__(self, patch_size, dim, channels=3):
        super().__init__()

        self.dim=dim
        patches = (128 // patch_size) **2
        self.patchify = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        x = self.patchify(x)
        batch_size, channels, _, _= x.shape
        x =x.reshape(batch_size, channels, -1).transpose(-1, -2)

        cls= self.cls_token.expand(batch_size, 1, self.dim)
        x= torch.cat([cls,x], dim=1)
        x= x+self.pos_embedding
        return x

#  FeedForward Layer
class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.feedforward(x)

class SelfAttn(nn.Module):
    def __init__(self, scaling, in_dim, out_dim):
        super().__init__()

        self.scaling = scaling

        self.qkv = nn.Linear(in_dim, out_dim)
    def forward(self,x):
        x = x.reshape(1, *x.shape).expand(3, *x.shape)
        Q, K, V = self.qkv(x)

        return torch.softmax((Q@K.permute(0,2,1))*self.scaling, dim=-1)@V

class MultiHeadSelfAttn(nn.Module):
    def __init__(self, in_dim, heads):
        super().__init__()
        out_dim = in_dim//heads
        scaling = out_dim**-0.5

        self.heads = nn.ModuleList([SelfAttn(scaling, in_dim, out_dim) for _ in range(heads)])
        self.linear = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self,x):
        out_heads = [head(x) for head in self.heads]
        x= torch.concatenate(out_heads, dim=-1)
        x= self.linear(x)

        return x

#  ViT Block
class ViTBlock(nn.Module):
     def __init__(self, dim, heads):
         super().__init__()
         self.ln_before = nn.LayerNorm(dim)
         self.msa = MultiHeadSelfAttn(in_dim=dim, heads=heads)
         self.ln_after = nn.LayerNorm(dim)
         self.feedforward = FeedForward(dim, dim)

     def forward(self, x):
         out_ln_before = self.ln_before(x)
         out_msa = self.msa(out_ln_before)

         x = out_msa+x

         out_ln_after = self.ln_after(x)
         out_ffd = self.feedforward(out_ln_after)

         return out_ffd + x

class ViTBase(nn.Module):
    def __init__(self, dim, patch_size, depth, heads, channels=3):
        super().__init__()
        self.patch = Patch(patch_size, dim, channels)
        self.vit_blocks = nn.ModuleList([ViTBlock(dim=dim, heads=heads) for _ in range(depth)])

    def forward(self,x):
        x= self.patch(x)
        for block in self.vit_blocks:
            x = block(x)
        return x

class ViTClassifier(nn.Module):
     def __init__(self, vit_blocks=4, hidden_dim=128, out_dim=12, heads=4, patch_size=16):
         super().__init__()
         self.vit = ViTBase(dim=hidden_dim, patch_size=patch_size, depth=vit_blocks, heads=heads)
         self.feedforward = FeedForward(hidden_dim, out_dim)

     def forward(self,x):
         x = self.vit(x)
         cls = x[:, 0]
         x= self.feedforward(cls)
         x= torch.softmax(x, dim=-1)
         return x

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 30
log_frequency = max(1,len(train_loader) // 10)

vit = ViTClassifier(vit_blocks=4, hidden_dim=128, out_dim=12, heads=4, patch_size=16)
vit = vit.to(device)
vit.train()

opt = torch.optim.Adam(vit.parameters(), lr=3e-5)

def val(epoch, data, model, device=None):
    if device is None:
       device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    hist_loss, hist_acc =0,0

    for idx, (samples, labels) in enumerate(data):
        samples = samples.to(device)
        labels = labels.to(device)

        preds = model(samples)
        loss = nn.functional.cross_entropy(preds, labels)
        max_preds = torch.argmax(preds, dim=1)
        acc= (max_preds == labels).sum() / len(labels)

        hist_loss += loss.item()
        hist_acc += acc.item()

    hist_loss /= idx
    hist_acc /= idx

    print(f"[Val {epoch}] loss: {hist_loss:.4f}, acc: {hist_acc:.4f}")
    return hist_loss, hist_acc

history = {
    "train": {"acc": [0 for _ in range(epochs)], "loss": [0 for _ in range(epochs)]},
    "val":   {"acc": [0 for _ in range(epochs)], "loss": [0 for _ in range(epochs)]},
}

for epoch in range(epochs):
    vit.train()
    for idx, (samples, labels) in enumerate(train_loader):
        opt.zero_grad()
        samples = samples.to(device).type(torch.float32)
        labels = labels.to(device)

        preds = vit(samples)
        loss = nn.functional.cross_entropy(preds, labels)
        loss.backward()
        opt.step()
        max_preds = torch.argmax(preds, dim=1)
        acc = (max_preds == labels).sum() / len(labels)

        history["train"]["loss"][epoch] += loss.item()
        history["train"]["acc"][epoch]  += acc.item()


    history["train"]["loss"][epoch] /= len(train_loader)
    history["train"]["acc"][epoch]  /= len(train_loader)

    print(f"[{epoch+1}/{epochs}] avg loss: {history['train']['loss'][epoch]:.5f}, acc: {history['train']['acc'][epoch]:.5f}")
    history["val"]["loss"][epoch], history["val"]["acc"][epoch] = val(epoch+1, val_loader, vit)

vit.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = vit(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"\n Test Accuracy: {test_acc * 100:.2f}%")

plt.figure(figsize=(15, 5))
plt.suptitle("ViT - FashionClassifier")

# loss
plt.subplot(1, 2, 1)
plt.title("Loss")
plt.plot(history["train"]["loss"], label="train")
plt.plot(history["val"]["loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# accuracy
plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.plot(history["train"]["acc"], label="train")
plt.plot(history["val"]["acc"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.figtext(0.5, 0.01, f"Test Accuracy: {test_acc * 100:.2f}%", ha="center", fontsize=12, color="green")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""2 experiment with adjusted the hyperparameters: reduced learning rate to 3e-5, used Dropout , increased layer to 6"""

!pip install vit-pytorch
from vit_pytorch import ViT

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout))

    def forward(self,x):
        return self.ff(x)

class ViTClassifier(nn.Module):
    def __init__(self, vit_blocks=6, hidden_dim=128, out_dim=12, heads=4, patch_size=16):
        super().__init__()
        self.vit= ViT(image_size=128, patch_size=patch_size, num_classes=hidden_dim,
                      dim=hidden_dim, depth=vit_blocks, heads=heads, mlp_dim=hidden_dim)
        self.feedforward= FeedForward(hidden_dim, out_dim)

    def forward(self, x):
        cls = self.vit(x)
        return self.feedforward(cls)

from torch.optim import Adam

def val(epoch, data, model, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    val_loss, val_acc = 0, 0

    for samples, labels in data:
        samples, labels = samples.to(device), labels.to(device)
        preds = model(samples)
        loss = loss_fn(preds, labels)
        acc = (preds.argmax(1) == labels).sum().item() / len(labels)
        val_loss += loss.item()
        val_acc += acc

    return val_loss / len(data), val_acc / len(data)

vit = ViTClassifier().to(device)
optimizer = Adam(vit.parameters(), lr=3e-5)
EPOCHS = 30

history = {"train": {"acc": [], "loss": []}, "val": {"acc": [], "loss": []}}

for epoch in range(EPOCHS):
    vit.train()
    total_loss, total_acc = 0, 0

    for samples, labels in train_loader:
        samples, labels = samples.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = vit(samples)
        loss = nn.CrossEntropyLoss()(preds, labels)
        loss.backward()
        optimizer.step()

        acc = (preds.argmax(1) == labels).sum().item() / len(labels)
        total_loss += loss.item()
        total_acc += acc

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    val_loss, val_acc = val(epoch+1, val_loader, vit, device)

    history["train"]["loss"].append(avg_loss)
    history["train"]["acc"].append(avg_acc)
    history["val"]["loss"].append(val_loss)
    history["val"]["acc"].append(val_acc)

    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

vit.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vit(images)
        predicted = outputs.argmax(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# Accuracy
plt.figure(figsize=(10, 4))
plt.plot(history["train"]["acc"], label="Train Accuracy")
plt.plot(history["val"]["acc"], label="Val Accuracy")
plt.title("ViT Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.figure(figsize=(10, 4))
plt.plot(history["train"]["loss"], label="Train Loss")
plt.plot(history["val"]["loss"], label="Val Loss")
plt.title("ViT Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

vit.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vit(images)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report (ViT Test Set):")
print(classification_report(y_true, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("ViT Confusion Matrix on Test Set")
plt.tight_layout()
plt.show()

""" Implementing top-k image predictions"""

#
sample_row = filtered_df.sample(1).iloc[0]
img_path = sample_row['full_path']
true_label = sample_row['ProductType']

# Load and preprocess
img = Image.open(img_path).convert("RGB")
img_tensor = tfms(img).unsqueeze(0).to(device)

# Predict top-3 classes
model = vit
model.eval()
with torch.no_grad():
    outputs = model(img_tensor).squeeze(0)

k = 5
topk_indices = torch.topk(outputs, k=k).indices.tolist()

# Show result
print("Top-5 predicted classes ViT:")
for idx in topk_indices:
    prob = torch.softmax(outputs, -1)[idx].item()
    print(f"[{idx}] {encoder.classes_[idx]:<20} ({prob * 100:.2f}%)")

# Show image with prediction
plt.imshow(img)
plt.title(f"True: {true_label}")
plt.axis("off")
plt.show()

"""# CLIP using"""

!pip install git+https://github.com/openai/CLIP.git

import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)
text_labels = ["white top", " red dress", "trousers", "heels", "a photo of sport shoes "]
text_tokens = clip.tokenize(text_labels).to(device)

# Downloading random image from filtered_df
sample_row = filtered_df.sample(1).iloc[0]
img_path = sample_row['full_path']
true_label = sample_row['ProductType']
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

#  CLIP
with torch.no_grad():
    image_features = F.normalize(model_clip.encode_image(image), dim=-1)
    text_features = F.normalize(model_clip.encode_text(text_tokens), dim=-1)
    similarity = image_features @ text_features.T

topk = similarity[0].topk(3)
print("\Top-3 CLIP predictions:")
for idx, score in zip(topk.indices, topk.values):
    print(f"{text_labels[idx]:<30} ({score.item()*100:.2f}%)")

plt.imshow(Image.open(img_path))
plt.title(f"True: {true_label}")
plt.axis("off")
plt.show()