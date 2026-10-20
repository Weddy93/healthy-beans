import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import TransferLearningModel


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "classification",
    "training"
)

VAL_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "classification",
    "validation"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "finetuned_resnet18.pth"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")


model = TransferLearningModel(
    num_classes=len(train_dataset.classes)
)

model = model.to(device)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=0.0001
)


num_epochs = 15

best_val_accuracy = 0.0


for epoch in range(num_epochs):

    print()
    print(
        f"Epoch {epoch + 1}/{num_epochs}"
    )

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    train_progress = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, labels in train_progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = running_loss / len(train_dataset)

    train_accuracy = (
        correct / total
    ) * 100


    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    val_progress = tqdm(
        val_loader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for images, labels in val_progress:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item() * images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    val_loss = val_loss / len(val_dataset)

    val_accuracy = (
        val_correct / val_total
    ) * 100


    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"Best model saved! "
            f"Validation Accuracy: "
            f"{val_accuracy:.2f}%"
        )


print()
print("==============================================")
print("Fine-tuning complete!")
print("==============================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Model saved to: {MODEL_PATH}"
)