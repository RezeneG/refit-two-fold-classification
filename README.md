# REFIT Two-Fold Classification for Zero-Inflated Appliance Detection

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview
This repository contains code for the paper: "Benchmarking Two-Fold Classification for Zero-Inflated Appliance Detection". It compares end-to-end classifiers (Random Forest, XGBoost) against a two-fold XGBoost approach on the REFIT electrical load monitoring dataset.

## Research Question
*Does a two-fold classification approach, which separates activity detection from appliance identification, achieve higher classification performance on zero-inflated appliance data compared to conventional end-to-end classifiers?*

## Results Summary
| Model | Macro F1 (mean ± std) |
|-------|----------------------|
| Random Forest | 0.48 ± 0.14 |
| XGBoost (end-to-end) | 0.56 ± 0.12 |
| Two-fold XGBoost | 0.81 ± 0.08 |

Two-fold approach achieves **43.7% relative improvement** (McNemar's test p < 0.001).

## Repository Structure
