Below is the complete content for the README file in Markdown format. You can copy the text into a file named `README.md` in your repository. This file contains all the important details including project overview, key decisions, directory structure, setup, execution instructions, model architecture, file descriptions, and future directions.

```markdown
# ACADA: Adversarial Contrastive Domain Adaptation for Object Detection

## Overview

ACADA is a domain-adaptive object detection model that concurrently combines adversarial domain adaptation and contrastive learning to align feature distributions between a labeled source domain and multiple unlabeled target domains. The core innovation lies in processing global features for adversarial alignment and local features for instance-level contrastive (InfoNCE) loss concurrently in a single forward pass through a shared ResNet50 backbone.

This repository contains all necessary scripts and modules for data preprocessing, model training, evaluation, and a full workflow orchestration.

---

## Table of Contents

- [Overview](#overview)
- [Key Decisions & Clarifications](#key-decisions--clarifications)
- [Directory Structure](#directory-structure)
- [Setup and Dependencies](#setup-and-dependencies)
- [Execution Instructions](#execution-instructions)
  - [Preprocessing (Target Synthesis)](#preprocessing-target-synthesis)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Full Workflow via Shell Script](#full-workflow-via-shell-script)
- [Model Architecture Overview](#model-architecture-overview)
- [File Descriptions](#file-descriptions)
- [Command-Line Arguments](#command-line-arguments)
- [Future Directions](#future-directions)
- [License](#license)
- [Contact](#contact)

---

## Key Decisions & Clarifications

1. **Dataset Selection:**
   - **Source:** We use the Pascal VOC 2012 dataset (downloadable from Kaggle) for its high-quality annotations and standard train/validation splits.
   - **Target:** Three synthesized target domains—foggy, low-light, and artistic—simulate diverse domain shifts.

2. **Target Synthesis:**
   - The `target_synthesis.py` script applies specific effects (e.g., Gaussian blur and contrast reduction for foggy; brightness reduction for low-light; increased saturation for artistic) to source images.
   - The images are split into training and validation sets and stored in organized directories.

3. **Model Architecture:**
   - **Shared Backbone:** A pretrained ResNet50 extracts features.
   - **Global Branch (Adversarial):** Global average pooling is applied followed by a Gradient Reversal Layer (GRL) with dynamic lambda scheduling and a shallow domain classifier (one hidden FC layer with batch normalization, ReLU, and dropout).
   - **Local Branch (Contrastive):** A projection head (1x1 convolution, ReLU, adaptive pooling) produces compact embeddings for InfoNCE loss.
   - The design allows for future integration of a full detection head.

4. **Training Strategy:**
   - Data loaders are created for the source and each target domain. A round-robin sampling method ensures balanced exposure to each target domain.
   - Losses include:
     - **Supervised Detection Loss:** A placeholder loss on the source domain.
     - **Domain Adversarial Loss:** Cross-entropy loss on global features.
     - **Contrastive (InfoNCE) Loss:** For local embeddings.
   - Dynamic GRL lambda scheduling (using a logistic function) is used, and detailed diagnostics (including gradient norm logging) are performed.

5. **Evaluation:**
   - Evaluation computes:
     - Supervised detection loss on the source validation set.
     - Domain classification accuracy for both source and each target domain.
     - InfoNCE loss on target domains by pairing with source.
   - Sample visualizations and detailed logs are saved for reference.

6. **Command-Line Flexibility:**
   - All scripts accept command-line arguments with default values, ensuring flexibility and reproducibility.

---

## Directory Structure

```
project/
├── data/
│   ├── source/
│   │   ├── all_images/      # (Optional) Raw Pascal VOC images
│   │   ├── train/           # Source training images (after splitting)
│   │   └── val/             # Source validation images (after splitting)
│   └── target/
│       ├── foggy/
│       │   ├── train/       # Synthesized foggy training images
│       │   └── val/         # Synthesized foggy validation images
│       ├── lowlight/
│       │   ├── train/       # Synthesized low-light training images
│       │   └── val/         # Synthesized low-light validation images
│       └── artistic/
│           ├── train/       # Synthesized artistic training images
│           └── val/         # Synthesized artistic validation images
├── preprocessing/
│   └── target_synthesis.py  # Script for synthesizing target domain images
├── checkpoints/             # Model checkpoints saved during training
├── logs/                    # Training logs (TensorBoard logs)
├── logs_evaluation/         # Evaluation logs (TensorBoard logs for evaluation)
├── evaluation_visuals/      # Saved sample visualizations from evaluation
├── docs/                    # Documentation, presentation slides, thesis drafts, etc.
├── dataset.py               # Data loading and preprocessing module
├── model.py                 # ACADA model architecture with dual branches
├── train.py                 # Training pipeline
├── evaluation.py            # Evaluation script for the trained model
├── run_all.sh               # Shell script to run the complete workflow
├── requirements.txt         # List of required Python packages
└── README.md                # This file
```

---

## Setup and Dependencies

1. **Python Version:**  
   Recommended: Python 3.7 or higher.

2. **Install Dependencies:**  
   Run:
   ```bash
   pip install -r requirements.txt
   ```
   This file includes packages such as `torch`, `torchvision`, `tensorboard`, and `Pillow`.

3. **(Optional) Docker/Conda:**  
   For reproducibility, you can containerize the project using Docker or create a Conda environment.

---

## Execution Instructions

### Preprocessing (Target Synthesis)

- **Purpose:**  
  Generate target domain images (foggy, low-light, artistic) from the source images.
- **Command:**
  ```bash
  python preprocessing/target_synthesis.py \
    --source_train data/source/train \
    --source_val data/source/val \
    --output_foggy_train data/target/foggy/train \
    --output_foggy_val data/target/foggy/val \
    --output_lowlight_train data/target/lowlight/train \
    --output_lowlight_val data/target/lowlight/val \
    --output_artistic_train data/target/artistic/train \
    --output_artistic_val data/target/artistic/val \
    --fog_intensity 0.5 \
    --lowlight_intensity 0.5 \
    --artistic_intensity 0.5 \
    --split_ratio 0.8
  ```

### Training

- **Purpose:**  
  Train the ACADA model using source and synthesized target data.
- **Command:**
  ```bash
  python train.py \
    --batch_size 8 \
    --num_epochs 20 \
    --lr 0.001 \
    --lambda_adv 1.0 \
    --lambda_con 0.1 \
    --lambda_det 1.0 \
    --temperature 0.07 \
    --source_train data/source/train \
    --source_val data/source/val \
    --target_foggy data/target/foggy \
    --target_lowlight data/target/lowlight \
    --target_artistic data/target/artistic \
    --checkpoint_dir checkpoints \
    --log_dir logs
  ```

### Evaluation

- **Purpose:**  
  Evaluate the model on source and target validation sets.
- **Command:**
  ```bash
  python evaluation.py \
    --checkpoint_path checkpoints/best_model.pth \
    --batch_size 8 \
    --source_val data/source/val \
    --target_foggy_val data/target/foggy/val \
    --target_lowlight_val data/target/lowlight/val \
    --target_artistic_val data/target/artistic/val \
    --log_dir logs_evaluation
  ```

### Full Workflow via Shell Script

- **Purpose:**  
  Orchestrate the entire workflow in one go.
- **Command:**
  ```bash
  ./run_all.sh
  ```
  Ensure the script is executable:
  ```bash
  chmod +x run_all.sh
  ```

---

## Model Architecture Overview

The ACADA model consists of:

- **Shared Backbone:**  
  A pretrained ResNet50 extracts features from input images.

- **Global Branch (Adversarial):**  
  - Global average pooling produces a 2048-dimensional feature vector.
  - A Gradient Reversal Layer (GRL) with dynamic lambda scheduling reverses gradients.
  - A shallow domain classifier (one hidden FC layer with batch normalization, ReLU, and dropout) predicts domain labels.

- **Local Branch (Contrastive):**  
  - A projection head (1x1 convolution, ReLU, adaptive average pooling) generates 256-dimensional embeddings.
  - These embeddings are used with an InfoNCE loss for instance-level alignment.

---

## File Descriptions

- **dataset.py:**  
  Loads images from specified directories, supports source (with optional annotations) and target domains, and returns additional metadata (file names).

- **preprocessing/target_synthesis.py:**  
  Applies domain shift effects (foggy, low-light, artistic) to source images and saves the synthesized images into organized train/val directories.

- **model.py:**  
  Defines the ACADA model with a shared ResNet50 backbone, a global branch (with GRL and a domain classifier), and a local branch (with a projection head). Implements dynamic GRL lambda scheduling.

- **train.py:**  
  Implements the training pipeline, including data loading (with round-robin sampling for target domains), loss computations (supervised detection, domain adversarial, and contrastive InfoNCE), logging of detailed diagnostics (including gradient norms), and checkpoint saving.

- **evaluation.py:**  
  Evaluates the trained model on source and target validation sets, computing detection loss (placeholder), domain classification accuracy, and InfoNCE loss. Saves visualizations and logs metrics.

- **run_all.sh:**  
  A shell script to execute the entire workflow: target synthesis, training, and evaluation.

- **requirements.txt:**  
  Lists all required Python packages.

---

## Command-Line Arguments

Each script accepts command-line arguments for flexibility:

- **target_synthesis.py:**  
  - `--source_train`, `--source_val`
  - `--output_foggy_train`, `--output_foggy_val`, etc.
  - `--fog_intensity`, `--lowlight_intensity`, `--artistic_intensity`
  - `--split_ratio`

- **train.py:**  
  - Hyperparameters: `--batch_size`, `--num_epochs`, `--lr`, `--lambda_adv`, `--lambda_con`, `--lambda_det`, `--temperature`
  - Data directories: `--source_train`, `--source_val`, `--target_foggy`, `--target_lowlight`, `--target_artistic`
  - `--checkpoint_dir`, `--log_dir`

- **evaluation.py:**  
  - `--checkpoint_path`
  - Validation directories: `--source_val`, `--target_foggy_val`, `--target_lowlight_val`, `--target_artistic_val`
  - `--batch_size`, `--log_dir`

- **run_all.sh:**  
  - Uses the above defaults; modify as needed.

---

## Future Directions

- **Full Detection Head Integration:**  
  Extend the model to include an RPN and ROI head for complete object detection.

- **Advanced Hyperparameter Tuning:**  
  Experiment with dynamic loss weighting, improved contrastive pair sampling, and more sophisticated GRL lambda schedules.

- **Ablation Studies:**  
  Evaluate the impact of each component (adversarial vs. contrastive) through controlled experiments.

- **Enhanced Logging:**  
  Consider integrating additional tools (e.g., Weights & Biases) for in-depth experiment tracking.

---

## Contact

For questions or issues regarding this project, please contact [Balaji S] at [223ec6216@nitrkl.ac.in].
```

---

