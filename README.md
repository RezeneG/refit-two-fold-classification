# Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## 📋 Overview

This repository contains the official implementation of the paper: **"Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection"** (CSO7013 Final Assessment).

Zero-inflated data—where the target variable contains an abundance of zero values—poses challenges for supervised learning. This project investigates whether a two-fold classification approach (separating activity detection from appliance identification) improves performance on appliance detection tasks compared to conventional end-to-end models.

### 🔬 Research Question

> *Does a two-fold classification approach achieve higher classification performance on zero‑inflated appliance data than end‑to‑end classifiers?*

### 📊 Key Results (after fixing preprocessing)
| Model | Macro F1 |
|-------|----------|
| Random Forest | 0.48 ± 0.14 |
| XGBoost (end-to-end) | 0.56 ± 0.12 |
| **Two-fold XGBoost** | **0.81 ± 0.08** |

- **Statistical significance**: McNemar’s test p < 0.001  
- **Primary bottleneck**: Activity detection (Stage 1 AUPRC = 0.89), not appliance identification (Stage 2 F1 = 0.93).

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- 16 GB RAM (recommended)
- ~2 hours runtime on a standard laptop

### Installation

```bash
# Clone the repository
git clone https://github.com/RezeneG/refit-two-fold-classification.git
cd refit-two-fold-classification

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

Data Preparation

Obtain the REFIT dataset

Go to https://repository.lboro.ac.uk/articles/dataset/REFIT_Electrical_Load_Measurement/2070091

Register (free) and download all CSV files (Household_1.csv … Household_20.csv).

Place the files

Move all downloaded CSV files into data/raw/.

Run preprocessing

bash
python data/preprocess.py
This creates features.csv, targets.csv, and household_ids.csv in data/processed/.

Train Models
Train the baseline and two‑fold models in order:

bash
python src/train_baseline.py --model random_forest
python src/train_baseline.py --model xgboost
python src/train_two_fold.py
Model files will be saved in the models/ directory.

Evaluate

bash
python src/evaluate.py

Results are saved to:

results/tables/main_results.csv – overall comparison

results/tables/per_class_performance.csv – per‑appliance metrics

results/figures/confusion_matrix.png – confusion matrix figure

📁 Repository Structure

text
refit-two-fold-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── download_data.py      # Download instructions
│   ├── preprocess.py          # Preprocessing pipeline
│   ├── raw/                    # Place downloaded CSV files here (not in git)
│   └── processed/               # Generated features/targets (not in git)
│
├── src/
│   ├── train_baseline.py       # Random Forest & XGBoost training
│   ├── train_two_fold.py        # Two‑fold XGBoost training
│   ├── evaluate.py              # Evaluation and figure generation
│   └── utils.py                  # Helper functions
│
├── config/
│   └── hyperparameters.yaml      # All hyperparameter settings
│
├── seeds/
│   └── seed_42.txt               # Fixed random seed documentation
│
├── models/                        # Created during training (not in git)
└── results/                       # Created during evaluation (not in git)

🔧 Reproducibility Guarantees
Fixed random seed: 42 for all stochastic processes (data splitting, model initialisation, etc.).

Temporal split: 80% training (early period), 20% testing (later period) per household – no leakage.

Imputation: Missing values (NaN) are handled with SimpleImputer (mean strategy) inside each training script.

Hardware tested: Intel i5‑8250U, 16 GB RAM, Windows 10/11, macOS 14, Ubuntu 22.04.

📝 Dataset Information

Name: REFIT Electrical Load Measurement

Source: Loughborough University, UK

License: CC BY 4.0

Period: 2013–2015

Households: 20

Original resolution: 8 seconds (downsampled to 1 minute)

Appliances: 9 categories (kettle, washing machine, dishwasher, fridge, freezer, microwave, television, monitor, lighting)

Class distribution after preprocessing: ~87% inactive, ~13% active (varies by appliance)

Ethical Considerations
Data collected with informed consent, anonymised, and not redistributable.

The code respects the original license and provides only download instructions.

📄 License

Code: MIT License

Dataset: CC BY 4.0 (as specified by REFIT)

📚 Citation

If you use this code or findings in your research, please cite:

bibtex
@article{rezene2026benchmarking,
  title={Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection},
  author={Rezene, Ghebrehiwot.},
  journal={CSO7013 Machine_Learning_Final_Assessment},
  year={2026},
  note={Code available: https://github.com/RezeneG/refit-two-fold-classification}
}
🙏 Acknowledgements

REFIT project team for making the dataset publicly available.

Loughborough University for data collection and curation.

Last updated: February 2026

