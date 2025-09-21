# -*- coding: utf-8 -*-
"""
Batch reproduction for LTRSVP (fixed 10-fold inner CV) — Table 2-style output
+ ROC curve utilities (K-fold per-fold curves & mean curve with band)

- 遍历 rsvp_*Hz_*[ab].raw.fif，按 (pid, rate) 合并 a/b
- 预处理：保留 8 通道+Status；重采样 64 Hz；FIF 已含 0.15–28 Hz
- 事件：目标 1..640；筛侧向 |x-320|>66.8 px；LVF=0, RVF=1
- Epoch：[-0.2,0.4] s，baseline (None,0)；特征窗 [0.2,0.4] s
- 特征：四对 contra–ipsi 差分，56 维
- 评估：外层 10×(75/25)，内层固定 10 折（若任一类 <10 抛错）；AUC
- 输出：
    1) rsvp_single_user_auc_10fold.csv（长表：pid, rate_hz, n_epochs, auc_mean, auc_std, inner_mean）
    2) rsvp_auc_table_wide.csv（宽表：与论文 Table 2 一致；p 值=内外环配对双侧检验）
    3) ROC 图：
       - 多折单独曲线（每折一条） → *_per_fold.png
       - 平均 ROC 曲线 + 置信带（95% CI 或 ±1 SD） → *_mean_band.png
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
import argparse
import warnings
from typing import List, Tuple, Dict

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve

# ========= 默认配置 =========
FILE_PATTERN = "rsvp_*Hz_*[ab].raw.fif"

FOCUS_CHS = ['PO7','PO8','P7','P8','PO3','PO4','O1','O2']
STIM = 'Status'
CENTER = 320.0
PX_PER_DEG = 640.0 / 11.5
CENTER_THR = 1.2 * PX_PER_DEG   # ≈ 66.8 px

EPOCH_TMIN, EPOCH_TMAX = -0.2, 0.4
BASELINE = (None, 0.0)
FEAT_TMIN, FEAT_TMAX = 0.2, 0.4

C_GRID = [0.01, 0.1, 1, 10, 100]
N_OUTER = 10
RANDOM_STATE = 42
VERBOSE_MNE = "ERROR"  # 改 'INFO' 查看详细日志

# ===== ROC 可视化配置 =====
PLOT_MODE = 'off'  # 'off' | 'individual' | 'average' | 'both'
K_FOLDS_FOR_ROC = 10
CONF_BAND = 'ci95'  # 'ci95' | 'std'
FIG_DPI = 160
# =======================


# ====== 文件名元数据解析 ======
def parse_meta(name: str):
    m = re.match(r"rsvp_(\d+)Hz_(\d+)([ab])\.raw\.fif$", name)
    if not m:
        return None
    return {"rate_hz": int(m.group(1)), "pid": int(m.group(2)), "run": m.group(3)}


# ====== I/O 与预处理 ======
def load_and_prep_raw(fif_path: Path):
    if not fif_path.exists():
        raise FileNotFoundError(f"文件不存在：{fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=VERBOSE_MNE)
    # 只保留 8 通道 + 事件通道
    picks = [ch for ch in FOCUS_CHS if ch in raw.ch_names]
    if STIM in raw.ch_names:
        picks += [STIM]
    else:
        raise RuntimeError(f"{fif_path.name}: 找不到事件通道 '{STIM}'；可用通道：{raw.ch_names}")
    raw.pick(picks)
    # 下采样至 64 Hz（FIF 已含 0.15–28 Hz 带通）
    raw.resample(64, npad="auto")
    return raw


def get_events_lvf_rvf(raw):
    """返回事件矩阵 (n,2) [sample,id] 以及标签 y (RVF=1, LVF=0)；若无侧向目标返回 (None,None)"""
    ev = mne.find_events(raw, stim_channel=STIM, output='onset',
                         shortest_event=1, verbose=VERBOSE_MNE)
    if len(ev) == 0:
        return None, None
    # 目标事件 (1..640)
    mask = (ev[:,2] >= 1) & (ev[:,2] <= 640)
    ev = ev[mask]
    if len(ev) == 0:
        return None, None

    x = ev[:,2].astype(float)
    lvf = x < (CENTER - CENTER_THR)
    rvf = x > (CENTER + CENTER_THR)
    keep = lvf | rvf
    ev = ev[keep]
    if len(ev) == 0:
        return None, None

    y = np.where(ev[:,2] > (CENTER + CENTER_THR), 1, 0)  # RVF=1, LVF=0
    ev2 = ev[:, [0,2]]  # 保留 sample 与 id
    return ev2, y


def make_epochs(raw, events2):
    """构建 epochs（不裁剪，只 baseline），后续特征再切 [0.2,0.4]"""
    ev = np.column_stack([events2[:,0], np.zeros(len(events2), int), events2[:,1]])
    epochs = mne.Epochs(raw, ev, event_id=None, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                        baseline=BASELINE,
                        picks=[ch for ch in FOCUS_CHS if ch in raw.ch_names],
                        preload=True, reject=None, verbose=VERBOSE_MNE)
    return epochs


# ====== 特征抽取（56 维）======
def build_features(epochs, y):
    times = epochs.times
    tmask = (times >= FEAT_TMIN) & (times <= FEAT_TMAX)
    data = epochs.get_data()[:, :, tmask]  # (n, ch, t)

    ch = {c:i for i,c in enumerate(epochs.ch_names)}
    def diff(a,b): return data[:, ch[a], :] - data[:, ch[b], :]
    d1 = diff('PO7','PO8'); d2 = diff('P7','P8'); d3 = diff('PO3','PO4'); d4 = diff('O1','O2')
    X_lr = np.stack([d1,d2,d3,d4], axis=1)  # (n,4,t)
    # RVF：保持(左-右)；LVF：取相反 => 乘以 sign
    sign = np.where(y==1, 1.0, -1.0)[:, None, None]
    return (sign * X_lr).reshape(len(y), -1)  # (n, 56)


# ====== CV 搜参与评估（改：返回 inner 的均值）======
def inner_cv_best_C(X, y, seed):
    # 固定 10 折；若某类 <10 抛错（与论文一致）
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
    # 返回最佳 C 以及该 C 在 inner 10 折上的平均 AUC
    return best_C, best_auc


def outer_cv_auc(X, y, n_outer=N_OUTER, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    outer_aucs = []
    inner_means = []
    for _ in range(n_outer):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=rng.randint(0, 1<<31)
        )
        C, inner_mean = inner_cv_best_C(Xtr, ytr, seed=rng.randint(0, 1<<31))
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=C))
        clf.fit(Xtr, ytr)
        scores = clf.decision_function(Xte)
        outer_auc = roc_auc_score(yte, scores)
        outer_aucs.append(outer_auc)
        inner_means.append(inner_mean)
    return float(np.mean(outer_aucs)), float(np.std(outer_aucs)), len(y), float(np.mean(inner_means))


# ====== 新增：K 折 ROC 曲线计算（每折单独 + 平均/置信带）======
def kfold_roc_curves(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> Dict[str, List]:
    """对同一 (pid, rate) 在 K 折上绘制 ROC 所需数据。"""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fprs, tprs, aucs = [], [], []
    for tr, va in skf.split(X, y):
        C, _ = inner_cv_best_C(X[tr], y[tr], seed=seed)
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=C, probability=False))
        clf.fit(X[tr], y[tr])
        scores = clf.decision_function(X[va])
        fpr, tpr, _ = roc_curve(y[va], scores)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc_score(y[va], scores))
    return {'fpr': fprs, 'tpr': tprs, 'auc': aucs}


def _interp_mean_roc(fprs: List[np.ndarray], tprs: List[np.ndarray], method: str = 'ci95'):
    """插值到统一 FPR 网格并求均值与带宽。"""
    grid = np.linspace(0, 1, 200)
    interp_tprs = []
    for fpr, tpr in zip(fprs, tprs):
        interp = np.interp(grid, fpr, tpr)
        interp[0] = 0.0
        interp_tprs.append(interp)
    interp_tprs = np.vstack(interp_tprs)
    mean_tpr = interp_tprs.mean(axis=0)
    std_tpr = interp_tprs.std(axis=0, ddof=1) if interp_tprs.shape[0] > 1 else np.zeros_like(mean_tpr)
    band = 1.96 * std_tpr / max(1, np.sqrt(interp_tprs.shape[0])) if method == 'ci95' else std_tpr
    mean_tpr[-1] = 1.0
    return grid, mean_tpr, band


def plot_rocs(curves: Dict[str, List], title: str, save_path: Path,
              mode: str = 'individual', band_method: str = 'ci95', dpi: int = 160):
    import matplotlib.pyplot as plt
    fprs, tprs, aucs = curves['fpr'], curves['tpr'], curves['auc']

    if mode in ('individual', 'both'):
        plt.figure(figsize=(5, 5), dpi=dpi)
        for i, (fpr, tpr, aucv) in enumerate(zip(fprs, tprs, aucs)):
            plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC={aucv:.3f})")
        plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(title + " — per-fold ROC")
        plt.legend(loc='lower right', fontsize=8, ncol=1)
        plt.tight_layout()
        plt.savefig(save_path.with_name(save_path.stem + "_per_fold.png"))
        plt.close()

    if mode in ('average', 'both'):
        grid, mean_tpr, band = _interp_mean_roc(fprs, tprs, method=band_method)
        mean_auc = np.mean(aucs) if len(aucs) else float('nan')
        plt.figure(figsize=(5, 5), dpi=dpi)
        plt.plot(grid, mean_tpr, linewidth=2, label=f"Mean ROC (AUC={mean_auc:.3f})")
        upper = np.clip(mean_tpr + band, 0, 1)
        lower = np.clip(mean_tpr - band, 0, 1)
        label_band = '95% CI' if band_method == 'ci95' else '±1 SD'
        plt.fill_between(grid, lower, upper, alpha=0.2, label=label_band)
        plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(title + " — mean ROC with band")
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path.with_name(save_path.stem + "_mean_band.png"))
        plt.close()


def process_group(files, plot_mode=PLOT_MODE, k_folds=K_FOLDS_FOR_ROC, band_method=CONF_BAND, out_dir: Path = None):
    """合并 a/b，返回 (mean_auc, std_auc, n_epochs, inner_mean)，并可绘制 ROC。"""
    raws, ev_list, y_list = [], [], []
    for f in files:
        raw = load_and_prep_raw(Path(f))
        ev2, y = get_events_lvf_rvf(raw)
        if ev2 is None:
            continue
        raws.append(raw); ev_list.append(ev2); y_list.append(y)
    if not ev_list:
        return None

    epochs_all, y_all = [], []
    for raw, ev2, y in zip(raws, ev_list, y_list):
        ep = make_epochs(raw, ev2)
        epochs_all.append(ep); y_all.append(y)
    epochs = mne.concatenate_epochs(epochs_all, verbose=VERBOSE_MNE)
    y = np.concatenate(y_all, axis=0)

    X = build_features(epochs, y)
    mean_auc, std_auc, n_ep, inner_mean = outer_cv_auc(X, y)

    if plot_mode != 'off':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                curves = kfold_roc_curves(X, y, k=k_folds, seed=RANDOM_STATE)
                meta = parse_meta(Path(files[0]).name)
                pid, rate = meta['pid'], meta['rate_hz'] if meta else ("?", "?")
                out_dir = out_dir or Path.cwd()
                out_dir.mkdir(parents=True, exist_ok=True)
                base = out_dir / f"PID{pid:02d}_{rate}Hz_ROC"
                title = f"PID {pid:02d} | {rate} Hz | K={k_folds}"
                plot_rocs(curves, title=title, save_path=base, mode=plot_mode, band_method=band_method, dpi=FIG_DPI)
            except Exception as e:
                print(f"  [WARN] ROC 绘图跳过：{e}")

    return mean_auc, std_auc, n_ep, inner_mean


def main():
    parser = argparse.ArgumentParser(description='LTRSVP batch with ROC plotting')
    parser.add_argument('--data_dir', type=str, default=".", help='数据目录')
    parser.add_argument('--pattern', type=str, default=FILE_PATTERN, help='文件通配符')
    parser.add_argument('--plot', type=str, default=PLOT_MODE, choices=['off','individual','average','both'], help='ROC 绘图模式')
    parser.add_argument('--kfolds', type=int, default=K_FOLDS_FOR_ROC, help='用于 ROC 的 K 折数量')
    parser.add_argument('--band', type=str, default=CONF_BAND, choices=['ci95','std'], help='平均 ROC 的置信带类型')
    parser.add_argument('--savedir', type=str, default=None, help='ROC 图保存目录（默认数据目录或当前目录）')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pattern = args.pattern
    plot_mode = args.plot
    kfolds = args.kfolds
    band = args.band
    save_dir = Path(args.savedir) if args.savedir else data_dir

    files = sorted(data_dir.glob(pattern))
    if not files:
        print(f"No files matched in: {data_dir} with pattern: {pattern}")
        return

    groups: Dict[Tuple[int,int], List[Path]] = {}
    for f in files:
        meta = parse_meta(f.name)
        if not meta:
            continue
        key = (meta["pid"], meta["rate_hz"])
        groups.setdefault(key, []).append(f)

    rows = []
    print(f"Found {len(groups)} groups (participant × rate).")
    for (pid, rate), flist in sorted(groups.items()):
        flist = sorted(flist, key=lambda p: p.name)
        print(f"\n== PID {pid:02d} | {rate} Hz | files: {[p.name for p in flist]}")
        try:
            res = process_group(flist, plot_mode=plot_mode, k_folds=kfolds, band_method=band, out_dir=save_dir)
            if res is None:
                print("  (no lateral target epochs)")
                continue
            mean_auc, std_auc, n_ep, inner_mean = res
            print(f"  AUC = {mean_auc:.3f} +/- {std_auc:.3f} | n_epochs={n_ep} | inner-mean={inner_mean:.3f}")
            rows.append(dict(pid=pid, rate_hz=rate, n_epochs=n_ep,
                             auc_mean=mean_auc, auc_std=std_auc, inner_mean=inner_mean))
        except Exception as e:
            print(f"  ERROR: {e}")

    if not rows:
        print("\nNo results to save.")
        return

    output_csv = data_dir / "rsvp_single_user_auc_10fold.csv"
    df = pd.DataFrame(rows).sort_values(["rate_hz", "pid"])
    df.to_csv(output_csv, index=False)
    print(f"\nSaved per-participant results -> {output_csv}")

    # 小结 + 宽表（保持与之前一致）
    def _fmt_p(p):
        if p is None or (isinstance(p, float) and np.isnan(p)): return "–"
        return "{:.2e}".format(p) if p < 1e-3 else "{:.2f}".format(p)

    summary_rows = []
    all_rates = [5, 6, 10]
    for rate in all_rates:
        sub = df[df["rate_hz"] == rate].dropna(subset=["auc_mean", "inner_mean"])
        med_inner = float(sub["inner_mean"].median()) if len(sub) else np.nan
        med_outer = float(sub["auc_mean"].median())   if len(sub) else np.nan
        p = None
        if len(sub) >= 2:
            try:
                from scipy.stats import wilcoxon
                stat = wilcoxon(sub["auc_mean"].values,
                                sub["inner_mean"].values,
                                alternative="two-sided",
                                zero_method="pratt")
                p = float(stat.pvalue)
            except Exception:
                p = None
        summary_rows.append({"rate_hz": rate,
                             "median_inner": med_inner,
                             "median_outer": med_outer,
                             "p_value": p})
    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Inner vs Outer (per participant) summary ===")
    print(
        summary_df.assign(
            median_inner=lambda d: d["median_inner"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "–"),
            median_outer=lambda d: d["median_outer"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "–"),
            p_value=lambda d: d["p_value"].map(_fmt_p),
        ).rename(columns={
            "rate_hz": "Rate (Hz)",
            "median_inner": "Median (inner mean AUC)",
            "median_outer": "Median (outer mean AUC)",
            "p_value": "p value (paired, two-sided)"
        }).to_string(index=False)
    )
    summary_csv = output_csv.with_name("rsvp_inner_vs_outer_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved inner-vs-outer summary -> {summary_csv}")

        # === 宽表（动态按实际出现的 pid；不同 rate 人数可不同）===
    df["mean_std"] = df.apply(lambda r: f"{r['auc_mean']:.2f} ± {r['auc_std']:.2f}", axis=1)

    # 各 rate 实际出现的 pid 集合
    all_rates = [5, 6, 10]
    pids_by_rate = {r: sorted(df.loc[df["rate_hz"] == r, "pid"].unique().tolist()) for r in all_rates}
    # 全部出现过的 pid（并集，升序）
    all_pids = sorted(set().union(*pids_by_rate.values()))

    # 宽表（显示 “均值 ± 标准差”）
    wide = (
        df[df["rate_hz"].isin(all_rates)]
        .pivot(index="pid", columns="rate_hz", values="mean_std")
        .reindex(index=all_pids, columns=all_rates)   # 只按实际 pid 排列；缺失填 "–"
        .rename_axis(index=None, columns=None)
    ).fillna("–")

    # 计算“Median on test set”等行（对每个 rate 的参与者取 AUC 中位数）
    med = (
        df[df["rate_hz"].isin(all_rates)]
        .pivot(index="pid", columns="rate_hz", values="auc_mean")
        .reindex(index=all_pids, columns=all_rates)
    )
    med_row = med.median(axis=0).apply(lambda x: f"{x:.2f}")
    med_test_row = med_row.copy()

    # p 值来自前面 summary_df
    def _fmt_p(p):
        if p is None or (isinstance(p, float) and np.isnan(p)): return "–"
        return "{:.2e}".format(p) if p < 1e-3 else "{:.2f}".format(p)

    p_map = {int(r): _fmt_p(float(summary_df.loc[summary_df["rate_hz"] == r, "p_value"].values[0]))
             if (summary_df["rate_hz"] == r).any() else "–"
             for r in all_rates}
    p_row = pd.Series(p_map)

    table = wide.copy()
    table.loc["Median"] = med_row
    table.loc["Median on test set"] = med_test_row
    table.loc["p value"] = p_row

    # 列名美化
    table.columns = [f"{c} Hz" for c in table.columns]
    table.index = [str(i) if isinstance(i, int) else i for i in table.index]
    table.index.name = "Participant"

    out_wide = output_csv.with_name("rsvp_auc_table_wide.csv")
    table.to_csv(out_wide)
    print(f"Saved wide-format table -> {out_wide}")

    console_table = table.applymap(lambda s: s.replace("±", "+/-") if isinstance(s, str) else s)
    print("\n=== Cross-validation mean AUC +/- std by Participant × Rate ===")
    print(console_table.to_string())

    # 可选：打印每个 rate 实际覆盖的 pid，便于检查
    for r in all_rates:
        print(f"Rate {r} Hz participants: {pids_by_rate.get(r, [])}")


if __name__ == "__main__":
    main()
