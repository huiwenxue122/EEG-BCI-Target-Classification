# ROC Analysis for EEG BCI Target Classification

This script (`roc.py`) generates **ROC curves** and performance reports for single-user EEG-based left vs right target classification, based on the N2pc ERP.

---

## 🔍 What the Script Does
- Scans a data directory for files matching the pattern:  
  `rsvp_*Hz_*[ab].raw.fif`
- For each participant and stimulus rate, merges the two runs (`a/b`).  
- **Preprocessing**:  
  - Keeps 8 occipital/parietal-occipital electrodes + the event channel (`Status`).  
  - Downsamples to 64 Hz.  
- **Events**:  
  - Keeps only target events (IDs 1..640).  
  - Defines lateral targets using |x − 320| > 1.2° (~67 px).  
  - Labels: LVF = 0, RVF = 1.  
- **Epoching**:  
  - Time window: [-0.2, 0.4] s, baseline (None, 0).  
- **Features**:  
  - Contra−ipsi difference on four electrode pairs (PO7–PO8, P7–P8, PO3–PO4, O1–O2).  
  - Feature vector length = 56 (14 time points × 4 channels).  
- **Evaluation**:  
  - Outer loop: 10 × (75/25 split).  
  - Inner loop: fixed 10-fold CV to tune linear SVM (C).  
  - Metric: AUC.  
- **ROC Option**:  
  - K-fold ROC curves per participant.  
  - Outputs per-fold curves, averaged ROC, and confidence bands.

---

## ⚙️ Requirements
- Python 3.9+ (3.10/3.11 tested)  
- Packages: `mne`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`

Install with:
```bash
pip install mne numpy pandas scikit-learn matplotlib
---
## 📂 Data Naming

Files must follow the format:

rsvp_<RATE>Hz_<PID><RUN>.raw.fif


Examples:

rsvp_5Hz_02a.raw.fif
rsvp_10Hz_11b.raw.fif


Event channel must be named Status.
If your dataset uses a different event channel, edit the script:

STIM = 'Status'

🚀 Usage
Parameters
Argument	Description	Default
--data_dir	Data directory	.
--pattern	File matching pattern	rsvp_*Hz_*[ab].raw.fif
--plot	ROC plot mode	off / individual / average / both
--kfolds	Number of folds for ROC	10
--band	Type of band for averaged ROC	ci95 or std
--savedir	Output directory for figures	(same as --data_dir)
Example 1: Average ROC + 95% CI
python roc.py --plot average --band ci95 \
  --data_dir "C:/Users/huiwe/Desktop/BME Lab/new_RSVP/rsvp_mne_files/rsvp_mne_files"


Generates:

PIDxx_<rate>Hz_ROC_mean_band.png

rsvp_single_user_auc_10fold.csv

Example 2: Per-fold + Average ROC
python roc.py --plot both --kfolds 10 --band ci95 \
  --data_dir "C:/Users/huiwe/Desktop/BME Lab/new_RSVP/rsvp_mne_files/rsvp_mne_files" \
  --savedir "./roc_figs"


Generates:

PIDxx_<rate>Hz_ROC_per_fold.png

PIDxx_<rate>Hz_ROC_mean_band.png

Results CSV in --data_dir

📊 Output

Figures: ROC curves per fold, average ROC with confidence band.

CSV: rsvp_single_user_auc_10fold.csv with columns:

pid, rate_hz, n_epochs, auc_mean, auc_std, inner_mean


Console log: reports AUC, std, number of epochs, and inner CV mean.

🛠 Troubleshooting

No files matched → Check --data_dir and file naming.

stim 'Status' not found → Change STIM in the script to your event channel.

(no lateral target epochs) → Verify your event codes and lateral threshold.

K-fold error → Reduce --kfolds if one class has <10 samples.

No figures generated → Ensure --plot is not off, and --savedir is valid.

🔬 Experimental Notes

Outer loop: 10 × random 75/25 splits → auc_mean, auc_std.

Inner loop: fixed 10-fold CV on training set to select best linear SVM (C ∈ [0.01, 0.1, 1, 10, 100]).

ROC curves: stratified K-fold ROC across all samples.

Confidence bands:

ci95: mean ± 1.96 × SE

std: mean ± STD
pip install mne numpy pandas scikit-learn matplotlib
