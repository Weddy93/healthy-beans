from pathlib import Path

import matplotlib.pyplot as plt

from torchvision import datasets, transforms
# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "classification"
# 2. ImageNet normalization

NORMALIZATION = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
# 3. Training transformations

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Data augmentation
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

    # Convert image to PyTorch tensor
    transforms.ToTensor(),

    # Normalize for pretrained models
    NORMALIZATION
])
# 4. Validation/test transformations

evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZATION
])
# 5. Load datasets

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "training",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    DATASET_DIR / "validation",
    transform=evaluation_transform
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=evaluation_transform
)
# 6. Display dataset information

print("\n" + "=" * 60)
print("HEALTHY BEANS - PHASE 4")
print("DATA AUGMENTATION VERIFICATION")
print("=" * 60)

print("\nClasses:")
print(train_dataset.classes)

print("\nDataset sizes:")
print(f"Training:   {len(train_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test:       {len(test_dataset)}")

# 7. Verify a training image

image, label = train_dataset[0]

print("\nTraining sample:")
print(f"Tensor shape: {image.shape}")
print(f"Label index:  {label}")

# 8. Verify validation image

validation_image, validation_label = validation_dataset[0]

print("\nValidation sample:")
print(f"Tensor shape: {validation_image.shape}")
print(f"Label index:  {validation_label}")

# 9. Verify test image

test_image, test_label = test_dataset[0]

print("\nTest sample:")
print(f"Tensor shape: {test_image.shape}")
print(f"Label index:  {test_label}")

# 10. Final checks

assert image.shape == (3, 224, 224)
assert validation_image.shape == (3, 224, 224)
assert test_image.shape == (3, 224, 224)

assert train_dataset.classes == [
    "als",
    "bean_rust",
    "healthy"
]

print("\n[OK] Training augmentation pipeline created.")
print("[OK] Horizontal flipping enabled.")
print("[OK] Small rotation enabled.")
print("[OK] Small translation enabled.")
print("[OK] Mild brightness/contrast variation enabled.")

print("\n[OK] Validation data has NO augmentation.")
print("[OK] Test data has NO augmentation.")

print("[OK] Images remain 224x224.")
print("[OK] ImageNet normalization is applied.")
print("[OK] Original dataset files were not modified.")

print("\nPHASE 4 AUGMENTATION VERIFICATION COMPLETE")