# Healthy Beans 🌱

A machine learning project for classifying French bean leaf images into three categories: **Angular Leaf Spot (ALS)**, **Bean Rust**, and **Healthy**. The project uses PyTorch and a convolutional neural network (CNN) to explore image classification, evaluate model performance, and improve predictions through controlled experiments.

> **Project status:** In progress. This README will be updated as the project develops.

## Current Results

The strongest model so far is the scheduled CNN.

| Model | Test Accuracy | Macro F1-score |
|---|---:|---:|
| Baseline CNN | 70.05% | 70.48% |
| CNN with data augmentation | 84.02% | 84.20% |
| Regularized CNN | 85.75% | 85.97% |
| **Scheduled CNN (current best)** | **89.41%** | **89.47%** |
| Frozen ResNet18 transfer learning | 76.80% | 77.02% |
| Fine-tuned ResNet18 | 74.25% | 74.76% |

These are results from the current dataset and experiment setup, not a guarantee of performance on new images. The project is still being improved.

## Classes

The classifier currently predicts one of these classes:

- `als` — Angular Leaf Spot
- `bean_rust` — Bean Rust
- `healthy` — Healthy bean leaf

## Dataset

The dataset is arranged in separate folders for training, validation, and testing. Each split contains one folder per class.

```text
dataset/
└── classification/
    ├── training/
    │   ├── als/
    │   ├── bean_rust/
    │   └── healthy/
    ├── validation/
    │   ├── als/
    │   ├── bean_rust/
    │   └── healthy/
    └── test/
        ├── als/
        ├── bean_rust/
        └── healthy/
```

Current dataset counts:

| Split | Images |
|---|---:|
| Training | 5,100 |
| Validation | 1,093 |
| Test | 1,095 |
| **Total** | **7,288** |

The model uses 224 × 224 image inputs and ImageNet normalization. Training augmentation has included horizontal flipping, rotations, and brightness/contrast changes. Validation and test images are not augmented.

**Dataset note:** Before pushing to GitHub, check the dataset's license and sharing permissions. Image datasets can be large, so the image files are generally better stored outside Git unless the dataset license and repository size make inclusion appropriate.

## Project Structure

```text
healthy-beans/
├── dataset/
│   └── classification/
│       ├── training/
│       ├── validation/
│       └── test/
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── analyze_errors.py
├── models/
│   ├── baseline_cnn.pth
│   ├── augmented_cnn.pth
│   ├── regularized_cnn.pth
│   ├── scheduled_cnn.pth
│   ├── transfer_learning.pth
│   └── finetuned_resnet18.pth
├── error_analysis/
├── requirements.txt
├── .gitignore
└── README.md
```

This is the current working layout. Files and checkpoints may change as development continues.

## Technology Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- scikit-learn
- Matplotlib
- tqdm

Check `requirements.txt` for the project's installed dependencies.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/healthy-beans.git
cd healthy-beans
```

Replace `YOUR-USERNAME` with your GitHub username and adjust the repository name if you choose a different one.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the project's Python directly through `.venv\Scripts\python.exe`, or adjust your local PowerShell execution policy as appropriate.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a PyTorch and Torchvision combination compatible with your Python version and hardware. This project has been developed and tested locally on CPU; training can take several minutes per epoch.

### 4. Add the dataset

Place the dataset into the directory structure shown above. Ensure that the class-folder names match exactly:

- `als`
- `bean_rust`
- `healthy`

Do not put the same images into multiple splits, as this can cause data leakage and make evaluation results misleading.

## Running the Project

Run commands from the repository root with the virtual environment activated.

### Train a model

```powershell
python src/train.py
```

The training script's configuration determines which model is trained and where the checkpoint is saved. Review `src/train.py` before running it, especially if you want to preserve an existing checkpoint.

### Evaluate a model

```powershell
python src/evaluate.py
```

Check `src/evaluate.py` to confirm which checkpoint it loads before running evaluation.

### Analyze misclassified images

```powershell
python src/analyze_errors.py
```

The error-analysis script evaluates the scheduled CNN on the test set and copies misclassified images into `error_analysis/` folders grouped by actual and predicted class. This helps identify patterns in the model's mistakes.

## Current Findings

- The scheduled CNN currently performs best among the experiments recorded here.
- Healthy leaves are classified more reliably than the two disease classes.
- The most common confusion is **Bean Rust predicted as ALS**, followed by **ALS predicted as Bean Rust**.
- The current error-analysis results show 116 incorrect predictions out of 1,095 test images.
- The next planned experiment is to investigate targeted data augmentation for the visually similar disease classes. The existing best checkpoint will be retained as a benchmark.

## Development Roadmap

- [x] Audit and organize the dataset
- [x] Verify image preprocessing
- [x] Train a baseline CNN
- [x] Test data augmentation
- [x] Add regularization and early stopping
- [x] Experiment with learning-rate scheduling
- [x] Evaluate CNN performance
- [x] Experiment with ResNet18 transfer learning
- [x] Analyze CNN classification errors
- [ ] Test targeted augmentation for ALS and Bean Rust
- [ ] Compare new experiments against the current benchmark
- [ ] Explore confidence thresholds or out-of-distribution image handling
- [ ] Consider adding more bean-leaf disease classes
- [ ] Build an inference interface or API
- [ ] Document deployment and usage

## Important Notes

- Model accuracy depends on the dataset, image quality, and similarity between disease symptoms.
- This is an ongoing learning and development project; predictions should not be treated as a substitute for expert agricultural diagnosis.
- Keep a record of each experiment's settings and test results so comparisons remain meaningful.
- Do not evaluate model changes repeatedly against the test set during tuning. Use the validation set for model selection and reserve the test set for final comparisons.
- If the model is retrained, confirm that the class-to-index mapping remains consistent with the saved checkpoint.

## Contributing and Updates

This is a personal project that will be updated as new experiments and improvements are completed. The README's results, roadmap, and project structure will be revised over time.

## License

A license has not yet been selected. Add a `LICENSE` file before explicitly allowing others to reuse or distribute this project. Also verify the license of any dataset and pretrained weights you use.
