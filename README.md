# EEG-BCI-Target-Classification
EEG BCI project for single-trial left vs right target classification.

# EEG-BCI-Target-Classification

This repository contains a reproduction of EEG-based **single-trial left vs right target classification** experiments, based on the paper:

> Matran-Fernandez, A., & Poli, R. (2017).  
> *Towards the automated localisation of targets in rapid image-sifting by collaborative brain-computer interfaces*.  
> PLoS ONE 12(5): e0178498. [https://doi.org/10.1371/journal.pone.0178498](https://doi.org/10.1371/journal.pone.0178498)

---

## 📑 Overview
The project focuses on detecting the **N2pc Event-Related Potential (ERP)** from EEG signals to classify whether a visual target appeared on the **left** or **right** side of the screen.  
The dataset used is the [LTRSVP dataset from PhysioNet](https://physionet.org/physiobank/database/ltrsvp/).

---

## 🧠 Methods

### 1. Data Preprocessing
- Band-pass filtering (0.15–28 Hz).  
- Downsampling to 64 Hz.  
- Artifact removal (eye blinks and movements).  
- Epoch extraction: 200–400 ms after stimulus onset (where N2pc is expected).  

### 2. Feature Extraction
- Selected electrode differences:  
  - PO7–PO8  
  - P7–P8  
  - PO3–PO4  
  - O1–O2  
- Each epoch produced a feature vector of **56 values (14 time points × 4 channels)**.  
- Contralateral vs ipsilateral representation used for N2pc convention.  

### 3. Classification
- **Classifier**: Linear SVM.  
- **Validation**: Double cross-validation setup:  
  - Outer loop: 75/25 train/test split, repeated 10 times.  
  - Inner loop: 10-fold stratified CV to tune SVM `C` parameter.  
- **Evaluation Metric**: AUC (Area Under ROC Curve).  

### 4. Results
- Single-user BCIs achieved median AUC ≈ **0.75**.  
- Some participants reached AUC ≥ 0.80 (good single-trial classification).  
- Results are consistent with findings in the original paper.  

---

## 📊 Example Results
ROC curve for left vs right classification:

![ROC Curve](results/roc_auc.png)

Confusion matrix:

![Confusion Matrix](results/confusion_matrix.png)

ERP waveform (N2pc component):

![ERP](results/erp_waveform.png)

---

## 📂 Project Structure
