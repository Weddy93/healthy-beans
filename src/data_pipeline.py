from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset" / "classification"

TRAIN_DIR = DATASET_ROOT / "training"
VALIDATION_DIR = DATASET_ROOT / "validation"
TEST_DIR = DATASET_ROOT / "test"

# 2. IMAGE NORMALIZATION

NORMALIZATION = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 3. TRAINING TRANSFORMS

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    NORMALIZATION
])
# 4. VALIDATION AND TEST TRANSFORMS

evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    NORMALIZATION
])

# 5. CREATE DATASETS

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    VALIDATION_DIR,
    transform=evaluation_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=evaluation_transform
)

# 6. DATASET INFORMATION

print("Classes:")
print(train_dataset.classes)

print()

print("Class mapping:")
print(train_dataset.class_to_idx)

print()

print("Dataset sizes:")
print(f"Training:   {len(train_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test:       {len(test_dataset)}")

# 7. BATCH SIZE

BATCH_SIZE = 32

# 8. DATA LOADERS

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
# 9. VERIFY ONE TRAINING BATCH

images, labels = next(iter(train_loader))

print()

print("First training batch:")
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")

print()

print("Label values in batch:")
print(labels)

print()

print("Image value range:")
print(f"Minimum: {images.min().item():.4f}")
print(f"Maximum: {images.max().item():.4f}")

assert images.shape[0] <= BATCH_SIZE
assert images.shape[1:] == (3, 224, 224)

assert labels.shape[0] == images.shape[0]

assert torch.all((labels >= 0) & (labels <= 2))

print()
print("[OK] Training DataLoader created.")
print("[OK] Validation DataLoader created.")
print("[OK] Test DataLoader created.")
print("[OK] Training data is shuffled.")
print("[OK] Validation data is not shuffled.")
print("[OK] Test data is not shuffled.")
print("[OK] Images have shape [batch, 3, 224, 224].")
print("[OK] Labels contain valid class indices.")
print()
print("PHASE 5 DATA PIPELINE VERIFICATION COMPLETE")