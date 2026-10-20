"""
Dataset Audit

This script audits the classification dataset without modifying it.

Checks:
1. Dataset structure
2. Number of images per split
3. Number of images per class
4. Image dimensions
5. Image formats
6. Image color modes
7. Corrupted/unreadable images
8. Duplicate images
9. Potential duplicates across dataset splits
"""

from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import hashlib             
# CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "classification"

SPLITS = ["training", "validation", "test"]

EXPECTED_CLASSES = ["als", "bean_rust", "healthy"]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}
# HELPER FUNCTIONS

def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash of a file.

    Files with the same hash are byte-for-byte identical.
    """

    sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception:
        return None


def get_image_files(folder):
    """
    Return all supported image files inside a folder.
    """

    return [
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
# DATASET STRUCTURE

def check_dataset_structure():
    print("\n" + "=" * 70)
    print("1. DATASET STRUCTURE")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory does not exist:")
        print(f"        {DATASET_DIR}")
        return False

    print(f"[OK] Dataset directory found:")
    print(f"     {DATASET_DIR}")

    structure_ok = True

    for split in SPLITS:

        split_path = DATASET_DIR / split

        if not split_path.exists():
            print(f"[FAIL] Missing split: {split}")
            structure_ok = False
            continue

        print(f"[OK] Split found: {split}")

        for class_name in EXPECTED_CLASSES:

            class_path = split_path / class_name

            if class_path.exists():
                print(f"     [OK] {class_name}")
            else:
                print(f"     [FAIL] Missing class: {class_name}")
                structure_ok = False

    return structure_ok

# IMAGE COUNTS

def count_images():
    print("\n" + "=" * 70)
    print("2. IMAGE COUNTS")
    print("=" * 70)

    total_images = 0

    for split in SPLITS:

        split_path = DATASET_DIR / split

        print(f"\n{split.upper()}")

        split_total = 0

        for class_name in EXPECTED_CLASSES:

            class_path = split_path / class_name

            if not class_path.exists():
                print(f"  {class_name:<15}: MISSING")
                continue

            images = get_image_files(class_path)

            count = len(images)
            split_total += count

            print(f"  {class_name:<15}: {count}")

        print(f"  {'TOTAL':<15}: {split_total}")

        total_images += split_total

    print("\n" + "-" * 70)
    print(f"TOTAL DATASET IMAGES: {total_images}")
# IMAGE PROPERTIES

def inspect_images():
    print("\n" + "=" * 70)
    print("3. IMAGE PROPERTIES")
    print("=" * 70)

    dimensions = Counter()
    formats = Counter()
    modes = Counter()

    corrupted_images = []

    total_images = 0

    for split in SPLITS:

        split_path = DATASET_DIR / split

        if not split_path.exists():
            continue

        for class_name in EXPECTED_CLASSES:

            class_path = split_path / class_name

            if not class_path.exists():
                continue

            for image_path in get_image_files(class_path):

                total_images += 1

                try:
                    with Image.open(image_path) as image:

                        # Verify that the image can actually be read
                        image.verify()

                    # Reopen after verify because verify() invalidates
                    # the image object.
                    with Image.open(image_path) as image:

                        dimensions[image.size] += 1
                        formats[image.format] += 1
                        modes[image.mode] += 1

                except Exception:
                    corrupted_images.append(
                        str(image_path.relative_to(PROJECT_ROOT))
                    )

    print("\nIMAGE DIMENSIONS")

    for dimension, count in dimensions.most_common():
        print(f"  {dimension}: {count}")

    print("\nIMAGE FORMATS")

    for image_format, count in formats.most_common():
        print(f"  {image_format}: {count}")

    print("\nCOLOR MODES")

    for mode, count in modes.most_common():
        print(f"  {mode}: {count}")

    print("\nCORRUPTED / UNREADABLE IMAGES")

    if corrupted_images:
        print(f"  Found: {len(corrupted_images)}")

        for image in corrupted_images:
            print(f"  - {image}")

    else:
        print("  None found")

# CLASS BALANCE

def analyze_class_balance():
    print("\n" + "=" * 70)
    print("4. CLASS BALANCE")
    print("=" * 70)

    class_totals = Counter()

    for split in SPLITS:

        split_path = DATASET_DIR / split

        if not split_path.exists():
            continue

        for class_name in EXPECTED_CLASSES:

            class_path = split_path / class_name

            if not class_path.exists():
                continue

            count = len(get_image_files(class_path))

            class_totals[class_name] += count

    total = sum(class_totals.values())

    if total == 0:
        print("[ERROR] No images found.")
        return

    print("\nOverall class distribution:")

    for class_name, count in class_totals.items():

        percentage = (count / total) * 100

        print(
            f"  {class_name:<15}: "
            f"{count:<6} "
            f"({percentage:.2f}%)"
        )

    print(f"\nTotal images: {total}")

# DUPLICATE DETECTION

def find_duplicates():
    print("\n" + "=" * 70)
    print("5. DUPLICATE IMAGE CHECK")
    print("=" * 70)

    hashes = defaultdict(list)

    for split in SPLITS:

        split_path = DATASET_DIR / split

        if not split_path.exists():
            continue

        for class_name in EXPECTED_CLASSES:

            class_path = split_path / class_name

            if not class_path.exists():
                continue

            for image_path in get_image_files(class_path):

                file_hash = calculate_file_hash(image_path)

                if file_hash:
                    hashes[file_hash].append(
                        image_path.relative_to(PROJECT_ROOT)
                    )

    duplicate_groups = []

    for file_hash, files in hashes.items():

        if len(files) > 1:
            duplicate_groups.append(files)

    if not duplicate_groups:

        print("[OK] No duplicate images found.")

    else:

        print(
            f"[WARNING] Found "
            f"{len(duplicate_groups)} duplicate groups."
        )

        for index, group in enumerate(duplicate_groups, start=1):

            print(f"\nDuplicate group {index}:")

            for file in group:
                print(f"  - {file}")

# DATA LEAKAGE CHECK

def check_split_leakage():
    print("\n" + "=" * 70)
    print("6. TRAIN / VALIDATION / TEST LEAKAGE CHECK")
    print("=" * 70)

    split_hashes = {}

    for split in SPLITS:

        split_hashes[split] = {}

        split_path = DATASET_DIR / split

        if not split_path.exists():
            continue

        for class_name in EXPECTED_CLASSES:

            class_path = split_path / class_name

            if not class_path.exists():
                continue

            for image_path in get_image_files(class_path):

                file_hash = calculate_file_hash(image_path)

                if file_hash:
                    split_hashes[split][file_hash] = image_path

    leakage_found = False

    for i in range(len(SPLITS)):

        for j in range(i + 1, len(SPLITS)):

            split_a = SPLITS[i]
            split_b = SPLITS[j]

            hashes_a = set(split_hashes.get(split_a, {}))
            hashes_b = set(split_hashes.get(split_b, {}))

            overlap = hashes_a.intersection(hashes_b)

            if overlap:

                leakage_found = True

                print(
                    f"[WARNING] {len(overlap)} identical image(s) "
                    f"found in {split_a} and {split_b}."
                )

            else:

                print(
                    f"[OK] No identical images between "
                    f"{split_a} and {split_b}."
                )

    if not leakage_found:
        print("\n[OK] No obvious data leakage detected.")

# MAIN

def main():

    print("=" * 70)
    print("              HEALTHY BEANS")
    print("              DATASET AUDIT")
    print("=" * 70)

    print(f"\nDataset location:")
    print(DATASET_DIR)

    structure_ok = check_dataset_structure()

    if not structure_ok:

        print("\n[ERROR] Dataset structure needs attention.")
        print("Please fix the structure before continuing.")

        return

    count_images()

    inspect_images()

    analyze_class_balance()

    find_duplicates()

    check_split_leakage()

    print("\n" + "=" * 70)
    print("DATASET AUDIT COMPLETE")
    print("=" * 70)

    print(
        "\nNo files were modified, moved, renamed, "
        "or deleted by this script."
    )


if __name__ == "__main__":
    main()