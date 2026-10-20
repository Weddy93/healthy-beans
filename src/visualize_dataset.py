from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------
# 1. Project and dataset paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "classification" / "training"

# The actual class folder names in your dataset
CLASSES = ["als", "bean_rust", "healthy"]

# Number of random images to display per class
SAMPLES_PER_CLASS = 6

# Make results reproducible when you run the script again
RANDOM_SEED = 42

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------
# 2. Find images
# ---------------------------------------------------------
def get_images(class_dir):
    """Return all supported image files inside a class folder."""
    return [
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    random.seed(RANDOM_SEED)

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found:")
        print(DATASET_DIR)
        return

    # -----------------------------------------------------
    # 3. Create the visual grid
    # -----------------------------------------------------
    fig, axes = plt.subplots(
        len(CLASSES),
        SAMPLES_PER_CLASS,
        figsize=(15, 9)
    )

    # Make axes 2D even if settings change later
    if len(CLASSES) == 1:
        axes = [axes]

    print("\nHealthy Beans - Visual Dataset Inspection")
    print("=" * 50)

    for row, class_name in enumerate(CLASSES):
        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"[ERROR] Class folder not found: {class_dir}")
            continue

        images = get_images(class_dir)

        if len(images) < SAMPLES_PER_CLASS:
            selected = images
        else:
            selected = random.sample(images, SAMPLES_PER_CLASS)

        print(f"{class_name}: displaying {len(selected)} sample(s)")

        for col in range(SAMPLES_PER_CLASS):
            ax = axes[row][col]
            ax.axis("off")

            if col >= len(selected):
                continue

            image_path = selected[col]

            try:
                with Image.open(image_path) as image:
                    # Convert to RGB for consistent display
                    image = image.convert("RGB")
                    ax.imshow(image)

                ax.set_title(
                    image_path.name,
                    fontsize=8
                )

            except Exception as error:
                ax.text(
                    0.5,
                    0.5,
                    "Could not load image",
                    ha="center",
                    va="center"
                )
                print(f"[WARNING] Could not read {image_path}: {error}")

        # Class name on the left side of each row
        axes[row][0].set_ylabel(
            class_name.replace("_", " ").title(),
            fontsize=12,
            fontweight="bold",
            rotation=90,
            labelpad=15
        )

    fig.suptitle(
        "Healthy Beans - Training Dataset Visual Inspection",
        fontsize=16,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.01,
        "Random samples from each class | Original dataset files were not modified",
        ha="center",
        fontsize=9
    )

    plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])

    # -----------------------------------------------------
    # 4. Display the grid locally
    # -----------------------------------------------------
    plt.show()

    print("\nVisual inspection checklist:")
    print("  1. Are the leaves clearly visible?")
    print("  2. Are disease symptoms visible and consistent with the class?")
    print("  3. Are there unrelated objects or non-leaf images?")
    print("  4. Is image quality generally good?")
    print("  5. Are there extreme lighting, blur, or unusual backgrounds?")
    print("  6. Do ALS and Bean Rust look sufficiently distinguishable?")
    print("  7. Do the Healthy images actually appear healthy?")


if __name__ == "__main__":
    main()