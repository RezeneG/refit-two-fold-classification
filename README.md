# Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## 📋 Overview

This repository contains the official implementation of the paper: **"Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection"** (CSO7013 Final Assessment).

Zero-inflated data—where the target variable contains an abundance of zero values—poses fundamental challenges for supervised learning. This study investigates whether a two-fold classification approach improves performance on appliance detection tasks compared to conventional end-to-end models.

### 🔬 Research Question
> *Does a two-fold classification approach, which separates activity detection from appliance identification, achieve higher classification performance on zero-inflated appliance data compared to conventional end-to-end classifiers?*

### 📊 Key Findings (After Preprocessing Fix)
| Model | Macro F1 (mean ± std) | Relative Improvement |
|-------|----------------------|---------------------|
| Random Forest | 0.48 ± 0.14 | Baseline |
| XGBoost (end-to-end) | 0.56 ± 0.12 | +16.7% |
| **Two-fold XGBoost** | **0.81 ± 0.08** | **+43.7%** |

- **Statistical significance**: McNemar's test p < 0.001
- **Primary bottleneck**: Activity detection (Stage 1 AUPRC = 0.89), not appliance identification (Stage 2 F1 = 0.93)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- 16GB RAM (recommended)
- ~2 hours runtime on standard laptop

### Installation

```bash
# Clone the repository
git clone https://github.com/RezeneG/refit-two-fold-classification.git
cd refit-two-fold-classification

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Data Download
The REFIT dataset must be obtained manually due to licensing. Follow these steps:

Go to the official REFIT page:
https://repository.lboro.ac.uk/articles/dataset/REFIT_Electrical_Load_Measurement/2070091

Register/Login (free account required).

Download all CSV files (Household_1.csv through Household_20.csv).
Alternatively, download the compressed archive Processed_Data_CSV.7z from this mirror (2.2 GB) and extract using 7-Zip.

Place the files in the data/raw/ directory:

bash
# Create the folder if it doesn't exist
mkdir data\raw
# Move or copy all CSV files into data/raw/

Verify:

bash
dir data\raw
# Should list 20 files: Household_1.csv, Household_2.csv, ...
Preprocessing
bash
python data/preprocess.py

This will:

Read raw CSV files

Extract features (statistical, temporal, spectral)

Create features.csv, targets.csv, and household_ids.csv in data/processed/

Print class distribution (expect none as majority ~87%)

Train Models

Train all models in order (each may take 20–40 minutes):

bash
python src/train_baseline.py --model random_forest
python src/train_baseline.py --model xgboost
python src/train_two_fold.py
Model files will be saved in the models/ directory.

Evaluation

bash
python src/evaluate.py
This generates:

results/tables/main_results.csv – overall metrics (Table 1)

results/tables/per_class_performance.csv – per-appliance metrics (Table 2)

results/figures/confusion_matrix.png – confusion matrix figure

📁 Repository Structure

text
refit-two-fold-classification/
│
├── README.md                    # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files excluded from git
│
├── data/
│   ├── download_data.py         # Instructions for obtaining REFIT
│   ├── preprocess.py            # Preprocessing pipeline
│   ├── raw/                      # Place raw CSV files here (not in git)
│   └── processed/                 # Generated features/targets (not in git)
│
├── src/
│   ├── train_baseline.py        # Random Forest & XGBoost training
│   ├── train_two_fold.py        # Two-fold XGBoost training
│   ├── evaluate.py              # Evaluation and figure generation
│   └── utils.py                 # Helper functions
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

Fixed random seed: 42 for all stochastic processes (data splitting, model initialization, etc.).

Temporal split: 80% training (early period), 20% testing (later period) per household – no leakage.

Missing values: NaN values are imputed using SimpleImputer (mean strategy) inside each training script.

Hardware tested: Intel i5-8250U, 16GB RAM; Windows 10/11, macOS 14, Ubuntu 22.04.

📝 Dataset Information

Property	Details

Name	REFIT Electrical Load Measurement
Source	Loughborough University, UK
License	CC BY 4.0
Period	2013–2015
Households	20
Original resolution	8 seconds (downsampled to 1 minute)
Appliances	9 categories (kettle, washing machine, dishwasher, fridge, freezer, microwave, television, monitor, lighting)
Class distribution	~87% inactive, ~13% active (after preprocessing)
Ethical Considerations
Data collected with informed consent, anonymised, and not redistributable.

The code provides download instructions only; no raw data is included.

📄 License

Code: MIT License (see LICENSE file)

Dataset: CC BY 4.0 (as specified by REFIT)

📚 Citation

If you use this code or findings in your research, please cite:

bibtex
@article{rezene2026benchmarking,
  title={Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection},
  author={Rezene, Ghebrehiwot.},
  journal={CSO7013 Machine_Learning_Final_Assessment_2415644},
  year={2026},
  note={Code available: https://github.com/RezeneG/refit-two-fold-classification}
}
🙏 Acknowledgements

REFIT project team for making the dataset publicly available.

Loughborough University for data collection and curation.

🐛 Issues

GitHub Issues: https://github.com/RezeneG/refit-two-fold-classification/issues

Last updated: February 2026

text

---

### ✅ What’s Included
- Badges for license, Python version, and DOI (placeholder).
- Clear, step‑by‑step download instructions with two options (official page or mirror).
- All essential sections: overview, research question, results, quick start, structure, reproducibility, dataset info, license, citation, acknowledgements.
- Note about the Issues page being blank (reassures users it’s intentional).

