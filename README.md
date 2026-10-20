# Detection of Abnormal Electricity Consumption Using Unsupervised Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic Project](https://img.shields.io/badge/License-Academic%20Submission-green.svg)]()
[![Status: Submission Ready](https://img.shields.io/badge/Status-Submission%20Ready-brightgreen.svg)]()

**Author:** Avantika Choudhary  
**Institution:** Atal Bihari Vajpayee Indian Institute of Information Technology and Management (ABV-IIITM), Gwalior  
**Course:** Machine Learning Techniques  
**Instructor:** Dr. Yashwant Singh Patel  
**Submission Date:** 22 October 2026  
**Repository:** [https://github.com/avantika9773-svg/Anomaly-Detection](https://github.com/avantika9773-svg/Anomaly-Detection)

---

## Executive Summary

This project conducts an unsupervised machine learning investigation into detecting abnormal electricity consumption across a diverse portfolio of 200 commercial and institutional facilities. 

Using **1,749,494 hourly electricity observations** across calendar year 2016, the study investigates the challenge of identifying severe, highly imbalanced power outages (2.13% positive prevalence) without access to training labels. Following an audit that identified severe future-lookahead data leakage in precomputed features (bidirectional weather interpolation and annual target encodings), all 57 precomputed features were discarded. The experiment strictly evaluates contemporaneous, log-transformed meter consumption under a forward-in-time chronological split:
* **Development Partition:** January 1 – August 31, 2016 (1,083,509 observed readings; model fitting without labels).
* **Validation Partition:** September 1 – October 31, 2016 (272,698 observed readings; threshold calibration and diagnostic error analysis).
* **Final Holdout Partition:** November 1 – December 31, 2016 (285,634 observed readings; strictly protected one-time evaluation).

The project compares two unsupervised anomaly detection paradigms:
1. **Model A (Domain-Aware Statistical Baseline):** A per-building Robust Median and Median Absolute Deviation (MAD) modified Z-score detector in log-space.
2. **Model B (Machine Learning Benchmark):** A global one-dimensional Isolation Forest (200 isolation estimators) fitted across all development readings.

### Key Empirical Findings
* **Scale Normalization is Decisive:** The Per-building Median/MAD baseline achieved an F1 score of **0.3667** (Precision: 0.5128, Recall: 0.2854, PR-AUC: 0.2808) on validation, producing **1,452 false positives**. In contrast, global 1D Isolation Forest achieved an F1 score of **0.2423** (Precision: 0.1663, Recall: 0.4458, PR-AUC: 0.0847) while producing **11,964 false positives** (an 8.2× surge in false alarms).
* **Operational Feasibility:** Across 200 facilities, Isolation Forest generates ~196 false alarms per day (~1 false alarm per building/day), inducing severe operator alarm fatigue. The Median/MAD baseline restricts false alarms to ~24 per day across the fleet.
* **Holdout Generalizability:** Carried over to the final holdout, the baseline retained superior ranking and precision (F1 = **0.3366**, PR-AUC = **0.2712**) compared to Isolation Forest (F1 = **0.2545**, PR-AUC = **0.0971**).

---

## Verified Dataset Provenance and Lineage

A comprehensive audit conclusively verified that the dataset is a **three-tier derived benchmark**:

```
+---------------------------------------------------------------------------------------------------+
| Tier 1: Primary Physical Telemetry & Weather Origin                                               |
| ASHRAE Great Energy Predictor III (GEPIII) Competition (2019)                                     |
| • Origin: Real-world commercial & institutional meter telemetry across global campus facilities   |
| • Telemetry: Hourly electricity consumption (meter_reading in kWh), building metadata, weather    |
| • Ground Truth: Zero anomaly annotations (forecasting competition only)                           |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| Tier 2: Academic Research Benchmark & Outage Annotation Layer                                     |
| LEAD1.0: Large-scale Annotated Dataset for Energy Anomaly Detection (arXiv:2203.17256, 2022)       |
| • Authors: Manoj Gulati (IIIT Delhi) & Pandarasamy Arjunan (Berkeley Lab / NUS)                   |
| • Tool: Custom R Shiny interactive application (AnomalyAnnotator/combine_anomaly.R)               |
| • Ground Truth: Manual & semi-automated tagging of logged power outage periods (outage_indices)    |
| • Scope: 1,413 building meter series evaluated across 24-hour daily aggregated profiles           |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| Tier 3: Immediate Experimental Dataset Package & Distribution Source                              |
| Kaggle Community Prediction Competition: Large-scale Energy Anomaly Detection (LEAD)             |
| • Organizers: Pandarasamy Arjunan (Samy) and Manoj Gulati                                         |
| • Release: Exact 200-building training subset (train.csv) and 206-building test subset (test.csv) |
| • Resolution: Hourly per-reading time-series (1,749,494 training rows covering full year 2016)    |
| • Target: Binary anomaly flag (1 = recorded outage, 0 = unmarked observation)                    |
| • Feature Bundle: Precomputed 57-column feature files (train_features.csv, test_features.csv)     |
+---------------------------------------------------------------------------------------------------+
```

### Verified Links:
* **Kaggle LEAD Competition:** [https://www.kaggle.com/competitions/energy-anomaly-detection](https://www.kaggle.com/competitions/energy-anomaly-detection)
* **LEAD Dataset & Code Repository:** [https://github.com/samy101/lead-dataset](https://github.com/samy101/lead-dataset)
* **LEAD1.0 Research Paper:** [https://arxiv.org/abs/2203.17256](https://arxiv.org/abs/2203.17256)
* **Upstream ASHRAE GEPIII Competition:** [https://www.kaggle.com/c/ashrae-energy-prediction](https://www.kaggle.com/c/ashrae-energy-prediction)

> [!NOTE]
> **Disproven Candidate Datasets:** Forensic auditing confirmed that the project data did **not** originate from the UCI Machine Learning Repository (*Individual Household Electric Power Consumption*). The UCI dataset covers a single French home from 2006–2010 with 1-minute sampling and zero anomaly labels, whereas this project evaluates 200 commercial facilities from calendar year 2016 at hourly resolution with 37,296 annotated power outages. Full forensic comparisons are documented in `DATASET_SOURCE_AUDIT.md`.

---

## Repository Structure

```
Anomaly-Detection/
│
├── 01_LEAD_Data_Exploration.ipynb   # Executable research notebook with complete pipeline
├── DATASET_SOURCE_AUDIT.md          # Comprehensive dataset provenance audit & forensic log
├── final_report.md                  # Complete academic project report (12-section MSc format)
├── README.md                        # Project documentation and reproducibility guide
├── SUBMISSION_CHECKLIST.md          # Verification status and instructor delivery checklist
├── requirements.txt                 # Pinned Python package dependencies
│
├── train.csv                        # Raw training dataset (200 buildings, 1,749,494 rows)
├── test.csv                         # Raw competition test dataset (206 buildings, 1,800,567 rows)
├── train_features.csv               # Precomputed 57-feature training file (audited and excluded)
├── test_features.csv                # Precomputed 57-feature test file (audited and excluded)
├── sample_submission.csv            # Kaggle competition submission template
│
└── figures/                         # High-resolution generated experimental figures
    ├── dataset_overview.png                          # Figure 1: Partitions, coverage & imbalance
    ├── methodology_flowchart.png                     # Figure 2: End-to-end experimental architecture
    ├── validation_scores_threshold.png               # Figure 3: Anomaly score distributions & thresholds
    ├── validation_anomaly_detection_timeseries.png   # Figure 4: Building 254 time-series case study
    ├── validation_precision_recall.png               # Figure 5: Standalone Precision-Recall curves
    ├── validation_confusion_matrices.png             # Figure 6: Validation confusion matrices
    ├── validation_model_comparison.png               # Figure 7: Validation metric comparison chart
    ├── holdout_confusion_matrices.png                # Figure 8: Final holdout confusion matrices
    └── validation_error_patterns.png                 # Figure 9: Error breakdowns by month and building
```

---

## Environment Setup and Installation

The project runs on standard Python 3.10+ environments.

### 1. Clone the Repository
```bash
git clone https://github.com/avantika9773-svg/Anomaly-Detection.git
cd Anomaly-Detection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run the Project

### Interactive Notebook
Open and inspect the research workflow in Jupyter Lab or VS Code:
```bash
jupyter lab 01_LEAD_Data_Exploration.ipynb
```

> [!IMPORTANT]
> **Holdout Integrity Protection:**
> * All development statistics and models are fitted on **Cells 11–18** (Development Partition: Jan–Aug 2016).
> * Validation threshold selection and diagnostic error analyses run on **Cells 19–22 and 26–27** (Validation Partition: Sep–Oct 2016).
> * **Cell 24** (`final-holdout-evaluation-once`) contains the protected final holdout evaluation (Nov–Dec 2016). To prevent data snooping, **do NOT rerun Cell 24** during iterative development. Its historical outputs are saved and verified.

---

## Implemented Models and Methodology

### Feature Transformation
All observations are transformed into log-space to handle heavy right-skewed energy magnitudes:
$$x = \log(1 + \text{meter\_reading})$$

### Model A: Robust Per-Building Median/MAD Baseline
For each building $b$, development readings $y_b$ are used to calculate the center and scale:
$$\text{Center}_b = \text{median}(\log(1 + y_b))$$
$$\text{Scale}_b = 1.4826 \times \text{median}\left(|\log(1 + y_b) - \text{Center}_b|\right)$$
The anomaly score is the modified Z-score:
$$\text{Score}_{\text{Baseline}}(y, b) = \frac{|\log(1 + y) - \text{Center}_b|}{\text{Scale}_b}$$
An observation is flagged as an anomaly if $\text{Score}_{\text{Baseline}} \ge \tau_{\text{Baseline}} = 7.0591$.

### Model B: One-Dimensional Isolation Forest
Fitted globally across all development log-readings ($n=200$ trees, `max_samples='auto'`, `contamination='auto'`, `random_state=42`):
$$\text{Score}_{\text{IF}}(y) = -\text{score\_samples}(\log(1 + y))$$
An observation is flagged as an anomaly if $\text{Score}_{\text{IF}} \ge \tau_{\text{IF}} = 0.6107$.

---

## Experimental Results Summary

### Validation Performance (Sep–Oct 2016, 272,698 Observed Readings)

| Metric | Robust Median/MAD Baseline | One-Dimensional Isolation Forest | Operational Advantage |
|---|---:|---:|:---:|
| **Optimal Threshold ($\tau$)** | `7.0591` | `0.6107` | Fixed for holdout |
| **Precision** | **0.5128** | 0.1663 | **+3.08× precision** |
| **Recall** | 0.2854 | **0.4458** | +859 outages detected |
| **F1 Score** | **0.3667** | 0.2423 | **+51.3% relative gain** |
| **ROC-AUC** | **0.7021** | 0.6617 | +0.0404 |
| **Average Precision (PR-AUC)** | **0.2808** | 0.0847 | **+3.32× ranking quality** |
| **False Positives (FP)** | **1,452** | 11,964 | **-87.9% false alarms** |
| **False Positive Rate (FPR)** | **0.543%** | 4.475% | **8.2× lower false alarm rate** |

### Final Holdout Performance (Nov–Dec 2016, 285,634 Observed Readings)

| Metric | Robust Median/MAD Baseline | One-Dimensional Isolation Forest |
|---|---:|---:|
| **Carried Threshold ($\tau$)** | `7.0591` | `0.6107` |
| **Holdout Precision** | **0.2926** | 0.1775 |
| **Holdout Recall** | 0.3961 | **0.4494** |
| **Holdout F1 Score** | **0.3366** | 0.2545 |
| **Holdout ROC-AUC** | **0.7587** | 0.6894 |
| **Holdout PR-AUC** | **0.2712** | 0.0971 |
| **Holdout False Positives** | **5,307** | 11,540 |

---

## Known Project Limitations

1. **Outage Annotations vs. True Normality:** Labels represent recorded power outages. An unmarked record (`label = 0`) denotes absence of a logged outage, not verified normal operation.
2. **Missing Meter Readings:** Exactly 100% of missing readings carry label 0. Excluding missing rows limits conclusions strictly to observed-reading rows.
3. **1D Feature Restriction:** Discarding precomputed features to prevent leakage restricted both models to 1D univariate consumption, excluding diurnal schedules and ambient weather.
4. **Winter Baseline Drift:** In the November–December holdout, baseline precision decreased from 0.5128 to 0.2926 due to unmodeled seasonal heating demand shifts.

---

## Academic References

1. Gulati, M., & Arjunan, P. (2022). *LEAD1.0: A Large-scale Annotated Dataset for Energy Anomaly Detection in Commercial Buildings*. arXiv preprint arXiv:2203.17256. [https://arxiv.org/abs/2203.17256](https://arxiv.org/abs/2203.17256)
2. Arjunan, P., & Gulati, M. (2022). *Large-scale Energy Anomaly Detection (LEAD) Dataset Repository and Annotation Tool*. GitHub: [https://github.com/samy101/lead-dataset](https://github.com/samy101/lead-dataset)
3. Arjunan, P., & Gulati, M. (2022). *Large-scale Energy Anomaly Detection (LEAD) Community Prediction Competition*. Kaggle: [https://www.kaggle.com/competitions/energy-anomaly-detection](https://www.kaggle.com/competitions/energy-anomaly-detection)
4. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. 2008 Eighth IEEE International Conference on Data Mining (ICDM), pp. 413–422. [https://doi.org/10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)
5. Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly Detection: A Survey*. ACM Computing Surveys, 41(3), Article 15. [https://doi.org/10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882)
6. ASHRAE & Kaggle. (2019). *ASHRAE - Great Energy Predictor III: Machine Learning Competition*. Kaggle: [https://www.kaggle.com/c/ashrae-energy-prediction](https://www.kaggle.com/c/ashrae-energy-prediction)
7. Miller, C., Kathirgamanathan, A., Picchetti, B., & Arjunan, P. (2020). *ASHRAE Great Energy Predictor III Solution Analysis Repository*. GitHub: [https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis)
8. BUDS Lab / National University of Singapore. (2022). *LEAD Competition 1st Place Winning Solution Codebase*. GitHub: [https://github.com/buds-lab/LEAD-1st-solution](https://github.com/buds-lab/LEAD-1st-solution)
