from pathlib import Path
import numpy as np
import pandas as pd
import mne
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ========= CONFIG =========
FIF = Path(r"C:\Users\huiwe\Desktop\BME Lab\new_RSVP\rsvp_mne_files\rsvp_mne_files\rsvp_10Hz_02a.raw.fif")
FOCUS_CHS = ['PO7','PO8','P7','P8','PO3','PO4','O1','O2']
STIM = 'Status'
CENTER = 320.0
PX_PER_DEG = 640.0 / 11.5
CENTER_THR = 1.2 * PX_PER_DEG   # ≈66.8 px
EPOCH_TMIN, EPOCH_TMAX = -0.2, 0.4
BASELINE = (None, 0.0)
FEAT_TMIN, FEAT_TMAX = 0.2, 0.4
C_GRID = [0.01, 0.1, 1, 10, 100]
N_OUTER = 10
RANDOM_STATE = 42
# ==========================

def load_and_prep_raw(fif_path):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    # 只保留目标通道与事件通道
    picks = [ch for ch in FOCUS_CHS if ch in raw.ch_names]
    if STIM in raw.ch_names: picks += [STIM]
    raw.pick(picks)
    # 下采样到 64 Hz
    raw.resample(64, npad="auto")
    return raw

def get_events_targets(raw):
    events = mne.find_events(raw, stim_channel=STIM, output='onset', shortest_event=1, verbose=False)
    df = pd.DataFrame({'sample': events[:,0], 'id': events[:,2]})
    # 只要目标事件（1..640），并构造 LVF/RVF 标签
    df = df[(df['id']>=1) & (df['id']<=640)].copy()
    x = df['id'].astype(float)
    lvf = x < (CENTER - CENTER_THR)
    rvf = x > (CENTER + CENTER_THR)
    df = df[lvf | rvf].copy()
    y = np.where(df['id'] > (CENTER + CENTER_THR), 1, 0)  # RVF=1, LVF=0
    return df[['sample','id']].to_numpy(), y

def make_epochs(raw, events):
    # 事件矩阵需为 (n, 3) : [sample, 0, event_id]
    ev = np.column_stack([events[:,0], np.zeros(len(events), int), events[:,1]])
    epochs = mne.Epochs(raw, ev, event_id=None, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                        baseline=BASELINE, picks=FOCUS_CHS, preload=True, reject=None, verbose=False)
    # 只保留特征时间窗口
    epochs.crop(FEAT_TMIN, FEAT_TMAX)
    return epochs

def build_features(epochs, y):
    # 通道索引
    ch_idx = {ch:i for i,ch in enumerate(epochs.ch_names)}
    def pair_diff(a,b):  # a-b
        return epochs.get_data()[:, ch_idx[a], :] - epochs.get_data()[:, ch_idx[b], :]

    # 四对差分（先按“左-右”）
    diff_PO  = pair_diff('PO7','PO8')
    diff_P   = pair_diff('P7','P8')
    diff_PO3 = pair_diff('PO3','PO4')
    diff_O   = pair_diff('O1','O2')
    X_left_right = np.stack([diff_PO, diff_P, diff_PO3, diff_O], axis=1)  # (n,4,t)

    # 构造“对-同侧”：RVF 用(左-右)，LVF 需取(右-左)=-(左-右)
    sign = np.where(y==1, 1.0, -1.0)[:,None,None]
    X_contra_ipsi = sign * X_left_right
    # 展平到 56 维
    n, k, t = X_contra_ipsi.shape
    X = X_contra_ipsi.reshape(n, k*t)
    return X

def inner_cv_best_C(X, y, random_state):
    # 计算各类样本数，确定折数（至少 2 折）
    n_min = min(np.bincount(y))
    n_folds = max(2, min(10, n_min))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    best_auc, best_C = -np.inf, C_GRID[0]
    for C in C_GRID:
        aucs = []
        for tr, va in skf.split(X, y):
            clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=C))
            clf.fit(X[tr], y[tr])
            scores = clf.decision_function(X[va])
            aucs.append(roc_auc_score(y[va], scores))
        m = float(np.mean(aucs))
        if m > best_auc: best_auc, best_C = m, C
    return best_C

def outer_cv_auc(X, y):
    rng = np.random.RandomState(RANDOM_STATE)
    aucs = []
    for _ in range(N_OUTER):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=rng.randint(0,1<<31))
        C = inner_cv_best_C(X_tr, y_tr, random_state=rng.randint(0,1<<31))
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=C))
        clf.fit(X_tr, y_tr)
        scores = clf.decision_function(X_te)
        aucs.append(roc_auc_score(y_te, scores))
    return np.mean(aucs), np.std(aucs), aucs

# ===== run =====
raw = load_and_prep_raw(FIF)
events, y = get_events_targets(raw)         # 只含侧向目标
epochs = make_epochs(raw, events)
X = build_features(epochs, y)
mean_auc, std_auc, aucs = outer_cv_auc(X, y)
print(f"{FIF.name} | n_epochs={len(y)} | AUC = {mean_auc:.3f} ± {std_auc:.3f}")
