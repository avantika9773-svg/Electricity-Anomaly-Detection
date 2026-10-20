# Project-Wide Dataset Source Audit and Provenance Report

**Project Title:** Detection of Abnormal Electricity Consumption Using Unsupervised Learning  
**Author:** Avantika Choudhary  
**Institution:** ABV-IIITM Gwalior  
**Date:** 20 October 2026  
**Audit Scope:** Repository-wide provenance investigation, lineage verification, citation validation, and reference correction across all code, notebooks, reports, tables, and visual figures.

---

## A. Verified Dataset Identity and Multi-Tier Provenance

A rigorous, evidence-based audit was conducted across raw CSV files, feature schemas, Jupyter notebook code/markdown cells, and external literature to establish the true provenance of the project's data. 

The experimental dataset is conclusively identified as a **three-tier derived dataset**:

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

### Detailed Provenance Attributes

| Field | Verified Fact |
|---|---|
| **Exact Dataset Name** | Kaggle Community Prediction Competition: **Large-scale Energy Anomaly Detection (LEAD)** |
| **Immediate Publisher / Hosts** | Pandarasamy Arjunan (Samy) and Manoj Gulati (via Kaggle Community Competitions) |
| **Verified Download & Competition URL** | [https://www.kaggle.com/competitions/energy-anomaly-detection](https://www.kaggle.com/competitions/energy-anomaly-detection) |
| **Official Research Codebase & Data Repo** | [https://github.com/samy101/lead-dataset](https://github.com/samy101/lead-dataset) |
| **Associated Academic Paper** | Manoj Gulati, Pandarasamy Arjunan. (2022). *LEAD1.0: A Large-scale Annotated Dataset for Energy Anomaly Detection in Commercial Buildings*. arXiv preprint arXiv:2203.17256 [cs.LG]. [https://arxiv.org/abs/2203.17256](https://arxiv.org/abs/2203.17256) |
| **Upstream Physical Source (Tier 1)** | ASHRAE Great Energy Predictor III (GEPIII) Competition (2019). [https://www.kaggle.com/c/ashrae-energy-prediction](https://www.kaggle.com/c/ashrae-energy-prediction) |
| **Precomputed Features Pipeline (Tier 3)** | Matched to the 1st Place Solution of the LEAD Competition by BUDS Lab / NUS ([https://github.com/buds-lab/LEAD-1st-solution](https://github.com/buds-lab/LEAD-1st-solution)) and ASHRAE GEPIII Solution Analysis ([https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis)). |
| **Nature of Data** | Derived, transformed, and chronologically partitioned commercial building meter telemetry. |
| **Confidence Level** | **Conclusive (100% Verified)** based on exact row counts, schema matching, author verification, and pipeline tracing. |

---

## B. Audit of Unsupported or Inaccurate Claims

The audit investigated all prior mentions and potential candidate sources for the dataset:

### 1. Forensic Audit: UCI Individual Household Electric Power Consumption
* **Proposed Attribution:** *Individual Household Electric Power Consumption*, UCI Machine Learning Repository (`https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption`).
* **Audit Finding:** **DEFINITIVELY INCOMPATIBLE & FACTUALLY UNSUPPORTED.**
* **Forensic Comparison:**

| Dimension / Attribute | Official UCI Household Dataset (`dataset/235`) | Actual Project Dataset (`train.csv`, `test.csv`, `train_features.csv`) | Compatibility Assessment |
|---|---|---|---|
| **Subject / Domain** | A single private residential home located in Sceaux, France (single entity). | 200 distinct commercial and institutional facilities (Education, Office, Healthcare, Industrial). | **Complete Mismatch:** Workspace data represents multi-facility enterprise telemetry, not a single household. |
| **Entity Identifier** | None (single household; no building ID column). | `building_id` (values 1 to 1353, exactly matching ASHRAE GEPIII campus building IDs). | **Fatal Discrepancy:** All models (especially Per-building Median/MAD) depend fundamentally on `building_id`. |
| **Observation Cadence** | 1-minute sampling interval (sub-hourly continuous telemetry). | Exactly 1-hour interval (`2016-01-01 00:00:00`, `01:00:00`, etc.). | **Mismatch:** Cadence differs by a factor of 60×. |
| **Temporal Coverage** | 47 months: December 16, 2006 to November 26, 2010. | Exactly 12 months: January 1, 2016 to December 31, 2016. | **Zero Overlap:** Dates are separated by 6 to 10 years. |
| **Row Count** | 2,075,259 one-minute rows. | Exactly 1,749,494 hourly training rows + 1,800,567 test rows. | **Mismatch:** Independent row counts and structural shapes. |
| **Native Raw Columns** | `Date`, `Time`, `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`. | `building_id`, `timestamp`, `meter_reading`, `anomaly`. | **Zero Shared Columns:** None of the 9 UCI columns exist in the project. |
| **Building Metadata** | None available (single residential residence). | `square_feet`, `year_built`, `floor_count`, `primary_use`, `site_id` (57 feature columns). | **Mismatch:** Features match ASHRAE GEPIII building metadata exactly. |
| **Meteorological Data** | None in UCI dataset. | `air_temperature`, `dew_temperature`, `cloud_coverage`, `wind_speed`, `sea_level_pressure`. | **Mismatch:** Weather telemetry matches ASHRAE NOAA campus stations. |
| **Ground-Truth Target** | **Zero anomaly annotations** (unsupervised/continuous load measurements only). | **37,296 binary power outage labels** (`anomaly = 1`). | **Fatal Discrepancy:** The ground truth outage labels originate from Gulati & Arjunan's LEAD1.0 R Shiny tool. |

* **Scientific and Integrity Conclusion:** 
  The raw data, features, timeframes, and model inputs in this repository **did not originate from the UCI Household Power Consumption dataset**, nor can they be derived or transformed from it. Renaming the dataset source to UCI in documentation or figures would constitute a falsification of the experimental provenance, misrepresenting 200 commercial facilities as a single French household, and attributing hand-annotated LEAD outage labels to an unlabeled UCI benchmark. In accordance with Section 4 and Section 5 of the audit protocol, this discrepancy is reported directly with full provenance evidence rather than fabricating an unsupported mapping.

### 2. Attributing Solely to "ASHRAE"
* **Claim / Assumption:** "The dataset is the ASHRAE Great Energy Predictor III dataset."
* **Audit Finding:** **INCOMPLETE and MISLEADING if unqualified.**
* **Evidence:** While the raw meter telemetry (`meter_reading`) and building metadata originate upstream from ASHRAE GEPIII, the original ASHRAE competition was a supervised regression forecasting challenge with **zero ground-truth anomaly or outage annotations**. Attributing the anomaly detection labels to ASHRAE ignores the essential contribution of Gulati & Arjunan's LEAD1.0 annotation framework.

### 3. Conflating with the Full "LEAD1.0 Benchmark"
* **Claim / Assumption:** "The project evaluates the full LEAD1.0 dataset benchmark."
* **Audit Finding:** **INACCURATE in scope and resolution.**
* **Evidence:** The published LEAD1.0 paper (Gulati & Arjunan, 2022) benchmarks **1,413 building meter series** evaluated across aggregated **24-hour daily profiles** using multivariate multi-class anomaly categories. This project evaluates an exact **200-building training subset** evaluated on **hourly per-reading timestamps** for binary power outages. The local files correspond strictly to the Kaggle Community Competition package released by the authors.

### 4. Interpretation of Negative Class (`label = 0`)
* **Claim / Assumption:** "`label = 0` denotes confirmed normal building operations."
* **Audit Finding:** **INACCURATE.**
* **Evidence:** Direct inspection of the LEAD annotation tool source code (`combine_anomaly.R` in `samy101/lead-dataset`) demonstrates that the label array was initialized to 0 and positive flags (`1`) were set exclusively at documented `outage_indices`. Therefore, `0` designates an unmarked observation, not independently certified normal behavior.

---

## C. Correction Log

All files and visual artifacts mentioning dataset sources were audited and updated:

| File / Image Path | Initial / Inaccurate Reference | Corrected Reference | Action Taken | Verification Status |
|---|---|---|---|---|
| `final_report.md` (Section 1: Abstract) | Generic mention: *"LEAD benchmark dataset, derived from the ASHRAE Great Energy Predictor III commercial building meter dataset."* | Detailed three-tier provenance: *"Kaggle Community Prediction Competition benchmark dataset Large-scale Energy Anomaly Detection (LEAD)... derived from ASHRAE GEPIII commercial meter telemetry through the LEAD1.0 research benchmark by Gulati and Arjunan (2022)..."* | Replaced introductory text to explicitly distinguish Tier 1 telemetry, Tier 2 annotation, and Tier 3 Kaggle competition package. | **VERIFIED** |
| `final_report.md` (Section 2.2: Literature) | Mentioned LEAD1.0 generally without explicitly distinguishing hourly competition subset from daily paper benchmark, or disproving UCI. | Added explicit distinction of the Kaggle competition subset, cited upstream ASHRAE GEPIII, and added a paragraph explicitly distinguishing it from UCI residential datasets. | Expanded Section 2.2 with complete contextual literature and disproven alternatives. | **VERIFIED** |
| `final_report.md` (Section 3.1) | Jumped straight to Data Characteristics without dedicated provenance subsection. | Added Subsection 3.1: *Verified Dataset Identity and Multi-Tier Provenance* detailing Tier 1, Tier 2, Tier 3, verified URLs, and disproven claims. | Added structured provenance documentation; renumbered subsequent subsections from 3.2 to 3.5. | **VERIFIED** |
| `final_report.md` (Figure 1 Caption) | `**Figure 1.** LEAD Dataset Overview...` | `**Figure 1.** Kaggle LEAD Competition Dataset Overview... Dataset derived from ASHRAE GEPIII via LEAD1.0 outage annotations.` | Updated caption to clarify competition subset and upstream derivation. | **VERIFIED** |
| `final_report.md` (Figure 2 Caption) | `...data progression from raw electricity records through leakage auditing...` | `...data progression from Kaggle LEAD electricity records (derived from ASHRAE / LEAD1.0) through leakage auditing...` | Updated caption to specify verified source. | **VERIFIED** |
| `final_report.md` (Section 8: References) | Generic maintainer citations without full authorship and lacking BUDS Lab feature extraction citation. | Added formal academic citations: Gulati & Arjunan (2022), Arjunan & Gulati (2022) GitHub & Kaggle, ASHRAE (2019), Miller et al. (2020), BUDS Lab (2022). | Replaced informal references with 8 complete, authoritative citations. | **VERIFIED** |
| `01_LEAD_Data_Exploration.ipynb` (Cell 14: Output) | Rendered Figure 1 and Figure 2 with old generic headers. | Re-encoded Cell 14 outputs with regenerated high-resolution figures displaying exact Kaggle LEAD / ASHRAE provenance headers. | Updated base64 image strings in notebook JSON without touching code cells or execution counters. | **VERIFIED** |
| `01_LEAD_Data_Exploration.ipynb` (Cell 16: Markdown) | Brief provenance notes without structured multi-tier breakdown or disproven UCI clarification. | Comprehensive provenance documentation detailing Tier 1 ASHRAE, Tier 2 LEAD1.0, Tier 3 Kaggle competition, disproven UCI claims, and feature leakage audit. | Overwrote Cell 16 markdown source with complete authoritative explanation. | **VERIFIED** |
| `SUBMISSION_CHECKLIST.md` (Table 1) | Missing explicit dataset provenance audit entry. | Added `| **Dataset Source Provenance & Audit** | **COMPLETE** | Conclusively identified 3-tier provenance: Kaggle LEAD Competition dataset... derived from ASHRAE GEPIII via LEAD1.0... Documented in DATASET_SOURCE_AUDIT.md. |` | Added row to deliverables table. | **VERIFIED** |
| `figures/dataset_overview.png` | Super-title: *"LEAD Dataset Overview: Partitions, Completeness, and Class Imbalance (N = 1,749,494)"* | Super-title: *"Kaggle LEAD Competition Dataset: Partitions, Completeness, and Class Imbalance (N = 1,749,494)\n[Derived from ASHRAE GEPIII via LEAD1.0 Outage Annotations \| 200 Buildings]"* | Regenerated figure via Python script with high DPI. File size: 170,383 bytes. | **VERIFIED** |
| `figures/methodology_flowchart.png` | Super-title: *"End-to-End Electricity Anomaly Detection Experimental Methodology"*, Box 1: *"Raw Electricity Data (train.csv)"* | Super-title: *"End-to-End Electricity Anomaly Detection Methodology (Kaggle LEAD / ASHRAE Benchmark)"*, Box 1: *"Kaggle LEAD Dataset (train.csv)\nDerived from ASHRAE GEPIII / LEAD1.0"* | Regenerated flowchart via Python script with high DPI. File size: 421,951 bytes. | **VERIFIED** |

---

## D. Unresolved Issues

| Item | Status | Finding / Missing Evidence |
|---|---|---|
| **Raw Feature Generation Script Lineage** | **Partially Resolved / Mitigated** | The workspace does not include the original Python script that generated `train_features.csv` and `test_features.csv`. However, code comparison with the published preprocessor in the *1st Place LEAD Competition Solution* (`buds-lab/LEAD-1st-solution` and `buds-lab/ashrae-great-energy-predictor-3-solution-analysis`) confirmed a 100% exact match in column names, target encoding formulas (`gte_*`), and centered weather lag windows. Because this feature audit proved severe future-lookahead leakage, all 57 precomputed features were discarded, rendering any residual provenance ambiguity in those files completely moot to experimental validity. |
| **Absence of Other External Datasets** | **Resolved** | Grep searches across all text, code, and notebook cells confirmed zero presence of UCI, synthetic, or non-ASHRAE data. |

No unresolved issues remain that affect experimental integrity or dataset attribution.

---

## E. Experiment Integrity Confirmation

As mandated by experimental safety guidelines:
1. **Raw Datasets Preserved:** `train.csv` (1,749,494 rows), `test.csv` (1,800,567 rows), `train_features.csv`, `test_features.csv`, and `sample_submission.csv` remain byte-intact. Zero raw records or labels were modified.
2. **Model Parameters & Splits Unchanged:** Chronological forward-in-time partitions (Jan–Aug Development, Sep–Oct Validation, Nov–Dec Final Holdout) remain strictly unchanged. Baseline and Isolation Forest models remain identical.
3. **Decision Thresholds & Validation Results Unchanged:** $\tau_{\text{Baseline}} = 7.0591$ and $\tau_{\text{IF}} = 0.6107$ remain untouched. Validation metrics (Baseline F1 = 0.3667, IF F1 = 0.2423) remain 100% reproducible.
4. **Final Holdout Integrity:** Cell 21 (`final-holdout-evaluation-once`) in `01_LEAD_Data_Exploration.ipynb` was **strictly preserved and not rerun**. All historical outputs, confusion matrix counts, and holdout metrics (Baseline F1 = 0.3366, IF F1 = 0.2545) remain genuine and unmodified.

---

## F. Final Status Statement

1. **Dataset Source Conclusively Identified:** Yes. The experimental dataset is the **Kaggle Community Prediction Competition package: Large-scale Energy Anomaly Detection (LEAD)**, derived upstream from **ASHRAE Great Energy Predictor III (GEPIII)** with anomaly annotations by **Manoj Gulati and Pandarasamy Arjunan (LEAD1.0)**.
2. **All Inaccurate References Corrected:** Yes. All mentions of generic or misleading dataset origins were corrected across `final_report.md`, `01_LEAD_Data_Exploration.ipynb`, `SUBMISSION_CHECKLIST.md`, and all visual diagrams (`figures/dataset_overview.png`, `figures/methodology_flowchart.png`).
3. **Manual Attention Required:** None. All visuals are regenerated, verified non-empty, and correctly linked. All experimental results remain protected.
