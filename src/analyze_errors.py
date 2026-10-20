
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
from PIL import Image

from model import BaselineCNN


# Find the main project folder
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# Location of the test dataset
TEST_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "classification",
    "test"
)


# Location of the best Scheduled CNN
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "scheduled_cnn.pth"
)


# Folder where misclassified images will be copied
ERROR_DIR = os.path.join(
    PROJECT_DIR,
    "error_analysis"
)


# Use GPU if available, otherwise use CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# Use the same preprocessing as the test evaluation
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


# Create the test data loader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# Display dataset information
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {test_dataset.classes}")
print(f"Class mapping: {test_dataset.class_to_idx}")


# Create the Scheduled CNN
model = BaselineCNN(
    num_classes=len(test_dataset.classes)
)


# Load the trained Scheduled CNN
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
print("Scheduled CNN loaded successfully.")


# Create folders for each type of classification error
for actual_class in test_dataset.classes:

    for predicted_class in test_dataset.classes:

        if actual_class != predicted_class:

            folder_name = (
                f"{actual_class}_to_{predicted_class}"
            )

            folder_path = os.path.join(
                ERROR_DIR,
                folder_name
            )

            os.makedirs(
                folder_path,
                exist_ok=True
            )


# Keep track of the number of errors in each category
error_counts = defaultdict(int)


# Keep track of correctly classified images
correct_count = 0

# Keep track of all images
total_count = 0


# Run the model on the test dataset
with torch.no_grad():

    image_index = 0

    for images, labels in test_loader:

        # Move images and labels to the selected device
        images = images.to(device)
        labels = labels.to(device)


        # Get model predictions
        outputs = model(images)


        # Select the class with the highest prediction score
        _, predictions = torch.max(
            outputs,
            1
        )


        # Process every image in the current batch
        for i in range(len(images)):

            actual_index = labels[i].item()
            predicted_index = predictions[i].item()

            actual_class = test_dataset.classes[
                actual_index
            ]

            predicted_class = test_dataset.classes[
                predicted_index
            ]


            # Get the original image path
            original_path, _ = test_dataset.samples[
                image_index
            ]


            total_count += 1


            # Check whether the prediction was correct
            if actual_index == predicted_index:

                correct_count += 1

            else:

                # Create the error category
                error_type = (
                    f"{actual_class}_to_{predicted_class}"
                )

                error_counts[error_type] += 1


                # Create the destination folder
                destination_folder = os.path.join(
                    ERROR_DIR,
                    error_type
                )


                # Create a unique filename
                filename = os.path.basename(
                    original_path
                )

                destination_path = os.path.join(
                    destination_folder,
                    filename
                )


                # Copy the original image
                image = Image.open(
                    original_path
                )

                image.save(
                    destination_path
                )


            image_index += 1


# Calculate accuracy
accuracy = (
    correct_count / total_count
) * 100


print()
print("==============================================")
print("SCHEDULED CNN ERROR ANALYSIS")
print("==============================================")


print(
    f"\nTotal test images: {total_count}"
)

print(
    f"Correct predictions: {correct_count}"
)

print(
    f"Incorrect predictions: "
    f"{total_count - correct_count}"
)

print(
    f"Test Accuracy: {accuracy:.2f}%"
)


# Display the number of errors for each category
print()
print("Misclassification Summary:")
print("----------------------------------------------")


for actual_class in test_dataset.classes:

    for predicted_class in test_dataset.classes:

        if actual_class != predicted_class:

            error_type = (
                f"{actual_class}_to_{predicted_class}"
            )

            count = error_counts[error_type]

            print(
                f"{error_type}: {count}"
            )


print()
print("==============================================")
print("Error analysis complete!")
print("==============================================")

print(
    f"\nMisclassified images saved to:"
)

print(ERROR_DIR)