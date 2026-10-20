import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import TransferLearningModel


# Find the main project folder
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# Set the location of the test dataset
TEST_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "classification",
    "test"
)


# Load the fine-tuned ResNet18 model
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "finetuned_resnet18.pth"
)


# Use GPU if available, otherwise use CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# Prepare the test images
# No augmentation is used because the test set must remain untouched
test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load the test dataset
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


# Create batches for testing
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# Display test dataset information
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {test_dataset.classes}")
print(f"Class mapping: {test_dataset.class_to_idx}")


# Create the fine-tuned ResNet18 model
model = TransferLearningModel(
    num_classes=len(test_dataset.classes)
)


# Load the saved fine-tuned weights
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


# Move the model to the selected device
model = model.to(device)


# Put the model into evaluation mode
model.eval()


print()
print("Fine-tuned ResNet18 loaded successfully.")


# Store predictions and actual labels
all_predictions = []
all_labels = []


# Turn off gradient calculations during testing
with torch.no_grad():

    # Process each test batch
    for images, labels in test_loader:

        images = images.to(device)

        # Make predictions
        outputs = model(images)

        # Select the class with the highest prediction score
        _, predictions = torch.max(
            outputs,
            1
        )

        # Store predictions and actual labels
        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# Calculate the number of correct predictions
correct = sum(
    prediction == label
    for prediction, label in zip(
        all_predictions,
        all_labels
    )
)


# Calculate test accuracy
accuracy = (
    correct / len(all_labels)
) * 100


print()
print("==============================================")
print("FINE-TUNED RESNET18 TEST RESULTS")
print("==============================================")


print(
    f"\nTest Accuracy: {accuracy:.2f}%"
)


# Display precision, recall and F1-score
print("\nClassification Report:")
print("----------------------------------------------")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        digits=4
    )
)


# Display the confusion matrix
print("Confusion Matrix:")
print("----------------------------------------------")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)


print()
print("==============================================")
print("Fine-tuned ResNet18 evaluation complete!")
print("==============================================")