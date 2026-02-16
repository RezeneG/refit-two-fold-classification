# Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## 📋 Overview

This repository contains the official implementation of the paper: **"Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection"** (CSO7013 Final Assessment).

Zero-inflated data—where the target variable contains an abundance of zero values—poses fundamental challenges for supervised learning. This study investigates whether a two-fold classification approach improves performance on appliance detection tasks compared to conventional end-to-end models.

### 🔬 Research Question
> *Does a two-fold classification approach, which separates activity detection from appliance identification, achieve higher classification performance on zero-inflated appliance data compared to conventional end-to-end classifiers?*

### 📊 Key Findings
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
git clone https://github.com/YOUR_USERNAME/refit-two-fold-classification.git
cd refit-two-fold-classification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Get download instructions
python data/download_data.py

# Follow the instructions to obtain REFIT dataset from:
# https://repository.lboro.ac.uk/articles/dataset/REFIT_Electrical_Load_Measurement/2070091

# Place downloaded CSV files in: data/raw/

# Run preprocessing
python data/preprocess.py
```

### Train Models

```bash
# Train baselines
python src/train_baseline.py --model random_forest
python src/train_baseline.py --model xgboost

# Train two-fold model
python src/train_two_fold.py
```

### Evaluate

```bash
python src/evaluate.py
```

Results will be saved to:
- `results/tables/` - CSV files with metrics
- `results/figures/` - Confusion matrices and plots

---

## 📁 Repository Structure

```
refit-two-fold-classification/
│
├── README.md                    # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files to ignore in git
│
├── data/
│   ├── download_data.py         # REFIT download instructions
│   ├── preprocess.py            # Preprocessing pipeline
│   ├── raw/                      # Place raw CSV files here (not in git)
│   └── processed/                 # Saved features/targets (not in git)
│
├── src/
│   ├── train_baseline.py        # Random Forest + XGBoost training
│   ├── train_two_fold.py        # Two-fold XGBoost training
│   ├── evaluate.py              # Generate results and tables
│   └── utils.py                 # Helper functions (optional)
│
├── config/
│   └── hyperparameters.yaml      # All hyperparameter settings
│
├── seeds/
│   └── seed_42.txt               # Fixed random seed documentation
│
├── models/                        # Created during training (not in git)
├── results/                       # Created during evaluation (not in git)
└── venv/                          # Virtual environment (not in git)
```

**Note**: Folders with `(not in git)` are automatically created when running the code and are excluded via `.gitignore`.

---

## 🔧 Reproducibility Guarantees

### Fixed Random Seeds
All stochastic processes use seed = 42:
- Train/validation splits
- Model initialization
- Data shuffling (where applicable)

### Data Splits
- **Temporal split**: 80% training (early period), 20% testing (later period)
- **No leakage**: Features constructed only from past data
- **Per-household**: Models trained and evaluated independently per home

### Environment
```bash
# Exact versions used in experiments
python==3.10.0
scikit-learn==1.3.0
xgboost==2.0.0
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
scipy==1.10.0
pyyaml==6.0
```

### Hardware Tested
- Intel i5-8250U, 16GB RAM (CPU only)
- macOS 14 (ARM) via Rosetta 2
- Ubuntu 22.04 (x86_64)

---

## 📊 Results (Expected Output)

After running `evaluate.py`, you should see results similar to:

### Main Results Table
```
| Model               | Macro F1 | Weighted F1 |
|--------------------|----------|-------------|
| Random Forest       | 0.48     | 0.52        |
| XGBoost (end-to-end)| 0.56     | 0.61        |
| Two-fold XGBoost    | 0.81     | 0.84        |
```

### Per-Appliance Performance (Two-Fold Model)
| Appliance | Precision | Recall | F1-score | Frequency |
|-----------|-----------|--------|----------|-----------|
| Kettle | 0.94 | 0.91 | 0.92 | 2.1% |
| Washing machine | 0.91 | 0.88 | 0.89 | 3.4% |
| Dishwasher | 0.89 | 0.86 | 0.87 | 2.8% |
| Microwave | 0.84 | 0.79 | 0.81 | 4.2% |
| Television | 0.76 | 0.71 | 0.73 | 12.5% |
| Lighting | 0.72 | 0.68 | 0.70 | 15.3% |
| Computer monitor | 0.65 | 0.58 | 0.61 | 8.7% |
| Fridge | 0.58 | 0.52 | 0.55 | 24.1% |
| Freezer | 0.52 | 0.45 | 0.48 | 18.9% |

### Statistical Significance

McNemar's test comparing Two-fold XGBoost vs. end-to-end XGBoost:
- χ² = 24.67, p < 0.001
- **Conclusion**: Two-fold approach significantly outperforms baseline

---

## 💾 Models

Model files are **not included** in this repository due to size limitations. They will be created automatically when you run the training scripts:

```bash
# After training, you will have:
models/
├── random_forest.pkl           # ~50MB
├── xgboost.json                 # ~30MB
├── stage1_xgboost.json          # ~30MB
└── stage2_xgboost.json          # ~30MB
```

**Total size**: ~140MB (created locally, not uploaded to GitHub)

---

## 📝 Dataset Information

### REFIT Electrical Load Measurement
- **Source**: Loughborough University, UK
- **License**:CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
- **Period**: 2013-2015
- **Households**: 20
- **Resolution**: 8-second (downsampled to 1-minute)
- **Appliances**: 9 categories (kettle, washing machine, dishwasher, fridge, freezer, microwave, television, monitor, lighting)
- **Class distribution**: 87.3% inactive, 12.7% active

### Ethical Considerations
- Data collected with informed consent
- All identifiers removed
- Participants anonymized
- Cannot be used to identify individuals or infer occupancy patterns

### Citation
```bibtex
@misc{refit2015,
  title = {REFIT Electrical Load Measurement dataset},
  author = {{REFIT Team}},
  year = {2015},
  publisher = {Loughborough University},
  doi = {10.17028/rd.lboro.2070091.v1}
}
```

---

## 🔄 Complete Reproduction Pipeline

To exactly reproduce all results from scratch:

```bash
# 1. Fresh environment
python -m venv reproduce
source reproduce/bin/activate  # Windows: reproduce\Scripts\activate
pip install -r requirements.txt

# 2. Get data
python data/download_data.py
# (manually download CSV files as instructed)

# 3. Preprocess
python data/preprocess.py

# 4. Train models (this will create models/ folder)
python src/train_baseline.py --model random_forest
python src/train_baseline.py --model xgboost
python src/train_two_fold.py

# 5. Evaluate (this will create results/ folder)
python src/evaluate.py

# 6. Check outputs
ls results/tables/  # Should contain CSV files
ls results/figures/ # Should contain confusion matrix
```

**Total runtime**: ~2 hours on standard laptop

---

## ⚠️ Important Notes

### What's in GitHub vs. What's Generated

| Item | In GitHub? | Created By |
|------|------------|------------|
| Source code | ✅ Yes | - |
| requirements.txt | ✅ Yes | - |
| Raw data CSV files | ❌ No | User downloads |
| Processed features | ❌ No | preprocess.py |
| Model files | ❌ No | train_*.py |
| Results tables | ❌ No | evaluate.py |
| Figures/plots | ❌ No | evaluate.py |

### Why This Approach
- **Reproducibility**: Training from scratch proves the code works
- **Legal compliance**: Dataset license may restrict redistribution
- **Storage limits**: GitHub has 100MB file size limits
- **Scientific integrity**: Results should be reproducible, not just reported

---

## 🐛 Known Issues / Limitations

1. **Data availability**: REFIT requires registration; automatic download not possible
2. **Single-dataset focus**: Results may not generalise to all household types
3. **Feature engineering**: Hand-crafted features may underperform learned representations
4. **Multi-appliance assumption**: Simplified to single active appliance at a time
5. **Geographic bias**: UK households only, appliance types may differ internationally

---

## 📄 License

- **Code**: MIT License (see `LICENSE` file if included)
- **Dataset**: CC BY 4.0 (as specified by REFIT)

---

## 🙏 Acknowledgements

- REFIT project team for making the dataset publicly available
- Loughborough University for data collection and curation

---

## 📧 Contact

For questions or issues:
- GitHub Issues: https://github.com/RezeneG/refit-two-fold-classification/issues

---

## 📚 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{2415644_benchmarking,
  title={Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection},
  author={Student, Rezene Ghebrehiwot},
  journal={CSO7013_Machine_Learning_Final Assessment},
  year={2026}
}
```

---

**Last updated**: February 2026
