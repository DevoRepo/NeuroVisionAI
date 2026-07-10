import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

!pip install -q kagglehub timm

import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import timm

import matplotlib.pyplot as plt

import kagglehub

import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)

import os

print("Dataset Path:", path)
print()

for item in os.listdir(path):
    print(item)

train_dir = os.path.join(path, "Training")
test_dir = os.path.join(path, "Testing")

print("Training:", train_dir)
print("Testing :", test_dir)

print("\nTraining Classes:")
print(os.listdir(train_dir))

print("\nTesting Classes:")
print(os.listdir(test_dir))

for cls in sorted(os.listdir(train_dir)):
    folder = os.path.join(train_dir, cls)
    print(f"{cls:12s} : {len(os.listdir(folder))} images")

print("\nTesting Set\n")

for cls in sorted(os.listdir(test_dir)):
    folder = os.path.join(test_dir, cls)
    print(f"{cls:12s} : {len(os.listdir(folder))} images")

print(sorted(os.listdir(train_dir)))

import matplotlib.pyplot as plt
from PIL import Image
import os
import random

classes = sorted(os.listdir(train_dir))

plt.figure(figsize=(12, 12))

for i, cls in enumerate(classes):

    folder = os.path.join(train_dir, cls)

    img_name = random.choice(os.listdir(folder))

    img_path = os.path.join(folder, img_name)

    img = Image.open(img_path)

    plt.subplot(2, 2, i + 1)

    plt.imshow(img, cmap="gray")

    plt.title(cls)

    plt.axis("off")

plt.show()

from collections import Counter

sizes = Counter()

for cls in classes:
    folder = os.path.join(train_dir, cls)

    for img_name in os.listdir(folder):

        img = Image.open(os.path.join(folder, img_name))

        sizes[img.size] += 1

print("Number of unique image sizes:", len(sizes))

print("\nMost common image sizes:\n")

for size, count in sizes.most_common(10):
    print(size, ":", count)

from google.colab import drive
drive.mount('/content/drive')

import os

PROJECT_DIR = "/content/drive/MyDrive/NeuroVision_AI"

folders = [
    PROJECT_DIR,
    f"{PROJECT_DIR}/models",
    f"{PROJECT_DIR}/checkpoints",
    f"{PROJECT_DIR}/outputs",
    f"{PROJECT_DIR}/plots",
    f"{PROJECT_DIR}/reports",
    f"{PROJECT_DIR}/logs",
    f"{PROJECT_DIR}/exports",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project folders created successfully!")

from torchvision import transforms

train_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((224,224)),

    transforms.RandomRotation(10),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05,0.05),
        scale=(0.95,1.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std =[0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std =[0.229,0.224,0.225]
    )
])

from torchvision.datasets import ImageFolder

train_dataset = ImageFolder(
    root=train_dir,
    transform=train_transform
)

test_dataset = ImageFolder(
    root=test_dir,
    transform=test_transform
)

print("Classes:", train_dataset.classes)
print("Class to Index:", train_dataset.class_to_idx)

print("Training Images :", len(train_dataset))
print("Testing Images  :", len(test_dataset))

from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

images, labels = next(iter(train_loader))

print("Image Batch Shape :", images.shape)
print("Label Shape       :", labels.shape)

import matplotlib.pyplot as plt
import torchvision

images, labels = next(iter(train_loader))

grid = torchvision.utils.make_grid(images[:8], nrow=4)

grid = grid.permute(1, 2, 0)

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

grid = grid.numpy()
grid = std * grid + mean
grid = grid.clip(0, 1)

plt.figure(figsize=(10, 5))
plt.imshow(grid)
plt.axis("off")
plt.show()

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)

print(model)

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")

for param in model.parameters():
    param.requires_grad = False

print(model.classifier)

import torch.nn as nn

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(
        in_features=1280,
        out_features=4
    )
)

for param in model.classifier.parameters():
    param.requires_grad = True

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable Parameters: {trainable_params:,}")

model = model.to(device)

print(model.classifier)

import torch.nn as nn

criterion = nn.CrossEntropyLoss()

import torch.optim as optim

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32

BEST_MODEL_PATH = (
    f"{PROJECT_DIR}/models/best_model.pth"
)

print("Loss Function :", criterion)
print("Optimizer     :", optimizer)
print("Scheduler     :", scheduler)
print("Epochs        :", NUM_EPOCHS)

import time
import copy

best_accuracy = 0.0

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 40)

    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    start_time = time.time()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        running_correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = 100 * running_correct / total

    train_losses.append(epoch_loss)

    train_accuracies.append(epoch_acc)

    scheduler.step()

    elapsed = time.time() - start_time

    print(f"Loss     : {epoch_loss:.4f}")
    print(f"Accuracy : {epoch_acc:.2f}%")
    print(f"Time     : {elapsed:.1f} sec")

model.eval()

running_loss = 0.0
running_correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        running_correct += (predicted == labels).sum().item()

val_loss = running_loss / len(test_loader)

val_acc = 100 * running_correct / total

val_losses.append(val_loss)

val_accuracies.append(val_acc)

print(f"Validation Loss     : {val_loss:.4f}")
print(f"Validation Accuracy : {val_acc:.2f}%")

if val_acc > best_accuracy:

    best_accuracy = val_acc

    torch.save(
        model.state_dict(),
        BEST_MODEL_PATH
    )

    print("✅ Best model saved!")

print("\nTraining Complete!")

print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

print(model.features)

model.load_state_dict(torch.load(BEST_MODEL_PATH))

for param in model.parameters():
    param.requires_grad = False

# Unfreeze last MBConv block
for param in model.features[6].parameters():
    param.requires_grad = True

# Unfreeze final MBConv block
for param in model.features[7].parameters():
    param.requires_grad = True

# Unfreeze final 1x1 convolution
for param in model.features[8].parameters():
    param.requires_grad = True

# Keep classifier trainable
for param in model.classifier.parameters():
    param.requires_grad = True

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable Parameters : {trainable_params:,}")

import torch.optim as optim

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

#Fine-tuning Epochs
NUM_EPOCHS = 10


FINE_TUNED_MODEL_PATH = (
        f"{PROJECT_DIR}/models/best_model_finetuned.pth"
    )

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 40)

    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    start_time = time.time()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        running_correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = 100 * running_correct / total

    train_losses.append(epoch_loss)

    train_accuracies.append(epoch_acc)

    #scheduler.step()

    elapsed = time.time() - start_time

    print(f"Loss     : {epoch_loss:.4f}")
    print(f"Accuracy : {epoch_acc:.2f}%")
    print(f"Time     : {elapsed:.1f} sec")
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            running_correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(test_loader)

    val_acc = 100 * running_correct / total

    val_losses.append(val_loss)

    val_accuracies.append(val_acc)

    print(f"Validation Loss     : {val_loss:.4f}")
    print(f"Validation Accuracy : {val_acc:.2f}%")

    scheduler.step(val_acc)
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.7f}")

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(
            model.state_dict(),
            FINE_TUNED_MODEL_PATH
        )

        print(f"✅ New Best Model Saved! ({best_accuracy:.2f}%)")

import os

print(os.path.exists(FINE_TUNED_MODEL_PATH))

print(FINE_TUNED_MODEL_PATH)

size = os.path.getsize(FINE_TUNED_MODEL_PATH)

print(f"{size/1024/1024:.2f} MB")

import os

for root, dirs, files in os.walk(PROJECT_DIR):
    print(root)

    for file in files:
        print("   ", file)

from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import torch.nn as nn
import torchvision

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

PROJECT_DIR = "/content/drive/MyDrive/NeuroVision_AI"

MODEL_PATH = (
    f"{PROJECT_DIR}/models/best_model_finetuned.pth"
)

print(MODEL_PATH)

model = efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280,4)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model Loaded Successfully!")

print(model.classifier)

import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)

import os

print("Dataset Path:", path)
print()

for item in os.listdir(path):
    print(item)

from torchvision import transforms
from torchvision.datasets import ImageFolder

test_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std =[0.229,0.224,0.225]
    )
])

TEST_DIR = "/kaggle/input/brain-tumor-mri-dataset/Testing"

test_dataset = ImageFolder(
    TEST_DIR,
    transform=test_transform
)

from torch.utils.data import DataLoader

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

y_true = []
y_pred = []

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())

        y_pred.extend(predictions.cpu().numpy())

print(len(y_true))
print(len(y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

print(cm)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels = test_dataset.classes

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

plt.figure(figsize=(8,8))

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Brain Tumor Classification Confusion Matrix")

plt.show()

from sklearn.metrics import classification_report

print(
    classification_report(
        y_true,
        y_pred,
        target_names=test_dataset.classes,
        digits=4
    )
)

!pip install -q grad-cam

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2
import numpy as np

wrong_image = None
wrong_label = None
wrong_prediction = None

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        preds = outputs.argmax(1)

        for i in range(len(labels)):

            if preds[i] != labels[i]:

                wrong_image = images[i]

                wrong_label = labels[i].item()

                wrong_prediction = preds[i].item()

                break

        if wrong_image is not None:
            break

print("True Class :", test_dataset.classes[wrong_label])
print("Predicted  :", test_dataset.classes[wrong_prediction])

target_layers = [model.features[8]]

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import HiResCAM

cam = HiResCAM(
    model=model,
    target_layers=target_layers
)

input_tensor = wrong_image.unsqueeze(0)

targets = [ClassifierOutputTarget(wrong_prediction)]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]

import matplotlib.pyplot as plt

rgb_img = wrong_image.permute(1,2,0).cpu().numpy()

mean = np.array([0.485,0.456,0.406])
std  = np.array([0.229,0.224,0.225])

rgb_img = std * rgb_img + mean
rgb_img = np.clip(rgb_img,0,1)

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(rgb_img)
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(visualization)
plt.title("Grad-CAM")
plt.axis("off")

plt.show()

import os

GRADCAM_DIR = f"{PROJECT_DIR}/gradcam"

folders = [
    "glioma_correct",
    "glioma_wrong",
    "meningioma_correct",
    "pituitary_correct"
]

for folder in folders:
    os.makedirs(os.path.join(GRADCAM_DIR, folder), exist_ok=True)

print("Folders Ready!")

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt
import numpy as np
import os

cam = GradCAM(
    model=model,
    target_layers=[model.features[8]]
)

targets_needed = {
    "glioma_correct":10,
    "glioma_wrong":10,
    "meningioma_correct":10,
    "pituitary_correct":10
}

saved = {
    "glioma_correct":0,
    "glioma_wrong":0,
    "meningioma_correct":0,
    "pituitary_correct":0
}

classes = test_dataset.classes

model.eval()

for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        probs = torch.softmax(outputs.detach(), dim=1)

        for i in range(len(images)):

            true_name = classes[labels[i]]

            pred_name = classes[preds[i]]

            category = None

            if true_name=="glioma" and pred_name=="glioma":
                category="glioma_correct"

            elif true_name=="glioma" and pred_name!="glioma":
                category="glioma_wrong"

            elif true_name=="meningioma" and pred_name=="meningioma":
                category="meningioma_correct"

            elif true_name=="pituitary" and pred_name=="pituitary":
                category="pituitary_correct"

            if category is None:
                continue

            if saved[category] >= targets_needed[category]:
                continue

            img = images[i].unsqueeze(0)

            grayscale_cam = cam(
                input_tensor=img,
                targets=[ClassifierOutputTarget(preds[i].item())]
            )[0]

            rgb = images[i].permute(1,2,0).cpu().numpy()

            mean=np.array([0.485,0.456,0.406])
            std=np.array([0.229,0.224,0.225])

            rgb=std*rgb+mean
            rgb=np.clip(rgb,0,1)

            overlay = show_cam_on_image(
                rgb,
                grayscale_cam,
                use_rgb=True
            )

            fig,ax=plt.subplots(1,3,figsize=(12,4))

            ax[0].imshow(rgb)
            ax[0].set_title("Original")

            ax[1].imshow(grayscale_cam,cmap="jet")
            ax[1].set_title("Heatmap")

            ax[2].imshow(overlay)
            ax[2].set_title(
                f"T:{true_name}\nP:{pred_name}\n{probs[i].max().item()*100:.1f}%"
            )

            for a in ax:
                a.axis("off")

            save_path=os.path.join(
                GRADCAM_DIR,
                category,
                f"{category}_{saved[category]+1:02d}.png"
            )

            plt.savefig(save_path,bbox_inches="tight")
            plt.close()

            saved[category]+=1

        if all(saved[k]>=targets_needed[k] for k in targets_needed):
            break

print(saved)
print("Done!")

