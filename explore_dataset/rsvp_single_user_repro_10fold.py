"""
Batch reproduction for LTRSVP (fixed 10-fold inner CV)
- 遍历 rsvp_*Hz_*[ab].raw.fif，按 (pid, rate) 合并 a/b
- 预处理：保留 8 通道+Status；重采样 64 Hz；FIF 已含 0.15–28 Hz
- 事件：目标 1..640；筛侧向 |x-320|>66.8 px；LVF=0, RVF=1
- Epoch：[-0.2,0.4] s，完整窗口做基线 [-0.2,0]；不裁剪
- 特征：[0.2,0.4] s 上四对 contra–ipsi 差分，56 维
- 评估：外层 10×(75/25)；内层固定 10 折（若任一类 <10，抛错）；指标 AUC
- 输出：rsvp_single_user_auc_10fold.csv（pid, rate_hz, n_epochs, auc_mean, auc_std）
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

# ========= CONFIG =========
DATA_DIR = Path(r"C:\Users\huiwe\Desktop\BME Lab\new_RSVP\rsvp_mne_files\rsvp_mne_files")
FILE_PATTERN = "rsvp_*Hz_*[ab].raw.fif"
OUTPUT_CSV = DATA_DIR / "rsvp_single_user_auc_10fold.csv"

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
VERBOSE_MNE = "ERROR"
# ==========================

def parse_meta(name: str):
    m = re.match(r"rsvp_(\d+)Hz_(\d+)([ab])\.raw\.fif$", name)
    if not m: return None
    return dict(rate_hz=int(m.group(1)), pid=int(m.group(2)), run=m.group(3))

def load_raw(fpath: Path):
    raw = mne.io.read_raw_fif(fpath, preload=True, verbose=VERBOSE_MNE)
    picks = [ch for ch in FOCUS_CHS if ch in raw.ch_names]
    if STIM in raw.ch_names:
        picks += [STIM]
    else:
        raise RuntimeError(f"{fpath.name}: stim '{STIM}' not found.")
    raw.pick(picks)
    raw.resample(64, npad="auto")
    return raw

def events_lateral(raw):
    ev = mne.find_events(raw, stim_channel=STIM, output='onset', shortest_event=1, verbose=VERBOSE_MNE)
    if len(ev)==0: return None, None
    mask = (ev[:,2]>=1) & (ev[:,2]<=640)
    ev = ev[mask]
    if len(ev)==0: return None, None
    x = ev[:,2].astype(float)
    lvf = x < (CENTER - CENTER_THR)
    rvf = x > (CENTER + CENTER_THR)
    keep = lvf | rvf
    ev = ev[keep]
    if len(ev)==0: return None, None
    y = np.where(ev[:,2] > (CENTER + CENTER_THR), 1, 0)  # RVF=1, LVF=0
    return ev[:,[0,2]], y

def epochs_from(raw, ev2):
    ev = np.column_stack([ev2[:,0], np.zeros(len(ev2), int), ev2[:,1]])
    epochs = mne.Epochs(
        raw, ev, event_id=None,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=BASELINE,   # 在完整窗上做基线
        picks=[ch for ch in FOCUS_CHS if ch in raw.ch_names],
        preload=True, reject=None, verbose=VERBOSE_MNE
    )
    return epochs   # 不裁剪

def features_from(epochs, y):
    times = epochs.times
    tmask = (times >= FEAT_TMIN) & (times <= FEAT_TMAX)
    data = epochs.get_data()[:, :, tmask]

    ch = {c:i for i,c in enumerate(epochs.ch_names)}
    def diff(a,b): return data[:, ch[a], :] - data[:, ch[b], :]
    d1 = diff('PO7','PO8'); d2 = diff('P7','P8'); d3 = diff('PO3','PO4'); d4 = diff('O1','O2')
    X_lr = np.stack([d1,d2,d3,d4], axis=1)
    sign = np.where(y==1, 1.0, -1.0)[:,None,None]
    return (sign * X_lr).reshape(len(y), -1)  # 56 维

def inner_best_C_10fold(X, y, seed):
    counts = np.bincount(y)
    if counts.min() < 10:
        raise RuntimeError(f"Inner CV requires >=10 samples per class, got {counts.tolist()}")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    best_auc, best_C = -1.0, C_GRID[0]
    for C in C_GRID:
        aucs = []
        for tr, va in skf.split(X, y):
            clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=C))
            clf.fit(X[tr], y[tr])
            scores = clf.decision_function(X[va])
            aucs.append(roc_auc_score(y[va], scores))
        m = float(np.mean(aucs))
        if m > best_auc:
            best_auc, best_C = m, C
    return best_C

def outer_auc(X, y, n_outer=N_OUTER, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_outer):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=rng.randint(0,1<<31))
        C = inner_best_C_10fold(Xtr, ytr, seed=rng.randint(0,1<<31))
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=C))
        clf.fit(Xtr, ytr)
        scores = clf.decision_function(Xte)
        aucs.append(roc_auc_score(yte, scores))
    return float(np.mean(aucs)), float(np.std(aucs)), len(y)

def process_group(files):
    raws, ev_list, y_list = [], [], []
    for f in files:
        raw = load_raw(f)
        ev2, y = events_lateral(raw)
        if ev2 is None: continue
        raws.append(raw); ev_list.append(ev2); y_list.append(y)
    if not ev_list: return None

    # 每个 run 先各自做 epochs，再合并
    epochs_all, y_all = [], []
    for raw, ev2, y in zip(raws, ev_list, y_list):
        ep = epochs_from(raw, ev2)
        epochs_all.append(ep); y_all.append(y)
    epochs = mne.concatenate_epochs(epochs_all, verbose=VERBOSE_MNE)
    y = np.concatenate(y_all, axis=0)

    X = features_from(epochs, y)
    return outer_auc(X, y)

def main():
    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    if not files:
        print("No files matched."); return

    # group by (pid, rate)
    groups = {}
    for f in files:
        meta = parse_meta(f.name)
        if not meta: continue
        key = (meta["pid"], meta["rate_hz"])
        groups.setdefault(key, []).append(f)

    rows = []
    print(f"Found {len(groups)} groups (participant × rate).")
    for (pid, rate), flist in sorted(groups.items()):
        flist = sorted(flist, key=lambda p: p.name)  # e.g., ['...02a...', '...02b...']
        print(f"\n== PID {pid:02d} | {rate} Hz | files: {[p.name for p in flist]}")
        try:
            res = process_group(flist)
            if res is None:
                print("  (no lateral target epochs)")
                continue
            mean_auc, std_auc, n_ep = res
            print(f"  AUC = {mean_auc:.3f} +/- {std_auc:.3f} | n_epochs={n_ep}")
            rows.append(dict(pid=pid, rate_hz=rate, n_epochs=n_ep,
                             auc_mean=mean_auc, auc_std=std_auc))
        except Exception as e:
            print(f"  ERROR: {e}")

    if not rows:
        print("\nNo results to save."); return

    df = pd.DataFrame(rows).sort_values(["rate_hz","pid"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved per-participant results -> {OUTPUT_CSV}")

    # per-rate summary (median like the paper)
    print("\n=== Per-rate summary (median AUC across participants) ===")
    for rate, sub in df.groupby("rate_hz"):
        med = sub["auc_mean"].median()
        mean = sub["auc_mean"].mean()
        print(f"{rate} Hz : median={med:.3f} | mean={mean:.3f} | n_participants={len(sub)}")
    print("\nDone.")

if __name__ == "__main__":
    main()
