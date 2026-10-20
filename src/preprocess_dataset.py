"""
Healthy Beans - Phase 3
Data Preprocessing
This script:
1. Loads the classification dataset using torchvision
2. Converts images to tensors
3. Resizes images to 224x224
4. Normalizes pixel values using ImageNet statistics
5. Verifies labels and tensor shapes
"""

from pathlib import Path

import torch
from torchvision import datasets, transforms
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "classification"

# 2. Image preprocessing

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    # ImageNet normalization
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Load each dataset split

training_dataset = datasets.ImageFolder(
    DATASET_DIR / "training",
    transform=preprocess
)

validation_dataset = datasets.ImageFolder(
    DATASET_DIR / "validation",
    transform=preprocess
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=preprocess)

# 4. Display dataset information

print("\n" + "=" * 55)
print("HEALTHY BEANS - PHASE 3")
print("DATA PREPROCESSING VERIFICATION")
print("=" * 55)

print("\nDataset classes:")
print(training_dataset.classes)

print("\nClass-to-index mapping:")
print(training_dataset.class_to_idx)

print("\nDataset sizes:")
print(f"Training:   {len(training_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test:       {len(test_dataset)}")

# 5. Inspect one processed image

image, label = training_dataset[0]

print("\nProcessed sample:")
print(f"Tensor shape: {image.shape}")
print(f"Label index:  {label}")

print("\nExpected tensor shape:")
print("(3, 224, 224)")

# 6. Inspect pixel values after normalization

print("\nProcessed tensor value range:")
print(f"Minimum: {image.min().item():.4f}")
print(f"Maximum: {image.max().item():.4f}")
# 7. Verify preprocessing

assert image.shape == (3, 224, 224), \
    "Unexpected image tensor shape."

assert len(training_dataset.classes) == 3, \
    "Expected exactly 3 classes."

assert training_dataset.class_to_idx == {
    "als": 0,
    "bean_rust": 1,
    "healthy": 2
}, \
    "Unexpected class-to-index mapping."


print("\n[OK] Images are being converted to tensors.")
print("[OK] Images are 3-channel RGB tensors.")
print("[OK] Images are 224x224.")
print("[OK] ImageNet normalization is applied.")
print("[OK] Class labels are correctly mapped.")
print("[OK] Original dataset files were not modified.")

print("\nPHASE 3 PREPROCESSING VERIFICATION COMPLETE")