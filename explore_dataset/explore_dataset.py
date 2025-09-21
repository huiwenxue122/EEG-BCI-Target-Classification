# -*- coding: utf-8 -*-
"""
Explore LTRSVP EEG dataset (.fif files) and save results into one txt file
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne

# ========= CONFIG =========
DATA_DIR = Path(r"C:\Users\huiwe\Desktop\BME Lab\new_RSVP\rsvp_mne_files\rsvp_mne_files")  # 改成你的数据路径
OUTPUT_FILE = DATA_DIR / "ltrsvp_summary_full.txt"
FILE_PATTERN = "rsvp_*Hz_*[ab].raw.fif"
FOCUS_CHANNELS = ['PO7', 'PO8', 'P7', 'P8', 'PO3', 'PO4', 'O1', 'O2']
STIM_CHANNEL = 'Status'
# ==========================


def parse_meta_from_name(fname: Path):
    m = re.match(r"rsvp_(\d+)Hz_(\d+)([ab])\.raw\.fif$", fname.name)
    if not m:
        return None
    return {"rate_hz": int(m.group(1)), "participant": int(m.group(2)), "run": m.group(3)}


def read_raw_info(raw: mne.io.BaseRaw, channels_subset=None):
    info = raw.info
    ch_names = info["ch_names"]
    if channels_subset is not None:
        sel = [ch for ch in channels_subset if ch in ch_names]
        raw.pick(sel + [STIM_CHANNEL] if STIM_CHANNEL in ch_names else sel)
        ch_names = raw.info["ch_names"]
    return {
        "sfreq": info["sfreq"],
        "n_channels": len(ch_names),
        "channels": ch_names,
        "highpass": info.get("highpass", None),
        "lowpass": info.get("lowpass", None)
    }


def find_and_label_events(raw: mne.io.BaseRaw, stim_channel=STIM_CHANNEL):
    events = mne.find_events(raw, stim_channel=stim_channel, output='onset', shortest_event=1)
    ids = events[:, 2].astype(int)
    times = events[:, 0] / raw.info["sfreq"]
    kind = np.where(ids == 1000, "burst_start",
            np.where(ids == 1001, "burst_end",
            np.where(ids == 1002, "non_target", "target")))
    target_x = np.where(kind == "target", ids, np.nan)
    df = pd.DataFrame({
        "sample": events[:, 0],
        "time_s": times,
        "event_id": ids,
        "kind": kind,
        "target_x": target_x
    })
    return df


def summarize_file(fpath: Path):
    meta = parse_meta_from_name(fpath)
    if meta is None:
        return None, None
    raw = mne.io.read_raw_fif(fpath, preload=False, verbose=False)
    info = read_raw_info(raw, channels_subset=FOCUS_CHANNELS)
    try:
        edf = find_and_label_events(raw, stim_channel=STIM_CHANNEL)
    except Exception:
        return meta, None

    counts = edf["kind"].value_counts()
    if (edf["kind"] == "target").any():
       
        # 建议改为（按 ±1.2° ≈ 66.8 像素）：
        # x: 只取 target 的像素位置
        x = pd.to_numeric(edf.loc[edf["kind"] == "target", "target_x"], errors="coerce").dropna()

        CENTER_PX = 320.0
        DEG_PER_IMG = 11.5          # 水平视角（图像宽 640 px）
        PX_PER_DEG = 640.0 / DEG_PER_IMG
        CENTER_HALF_WIDTH = 1.2 * PX_PER_DEG   # ≈ 66.8 px，不取整

        center_mask = (x >= CENTER_PX - CENTER_HALF_WIDTH) & (x <= CENTER_PX + CENTER_HALF_WIDTH)
        center = int(center_mask.sum())
        left   = int((x <  CENTER_PX - CENTER_HALF_WIDTH).sum())
        right  = int((x >  CENTER_PX + CENTER_HALF_WIDTH).sum())
    else:
        left = right = center = 0

    summary = {
        **meta,
        "file": fpath.name,
        "sfreq": info["sfreq"],
        "n_channels": info["n_channels"],
        "highpass": info["highpass"],
        "lowpass": info["lowpass"],
        "n_events_total": len(edf),
        "n_burst_start": int(counts.get("burst_start", 0)),
        "n_burst_end": int(counts.get("burst_end", 0)),
        "n_non_target": int(counts.get("non_target", 0)),
        "n_target": int(counts.get("target", 0)),
        "targets_left": int(left),
        "targets_right": int(right),
        "targets_center": int(center)
    }
    return edf, summary


def main():
    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    summaries = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        if not files:
            f.write("No files found.\n")
            return

        f.write(f"Found {len(files)} files.\n")
        for file in files:
            edf, summ = summarize_file(file)
            if summ is None:
                continue
            f.write("\n=== File: " + file.name + " ===\n")
            f.write(f"Participant: {summ['participant']:02d} | Run: {summ['run']} | Rate: {summ['rate_hz']} Hz\n")
            f.write(f"Sampling rate: {summ['sfreq']} Hz | Channels: {summ['n_channels']}\n")
            f.write("Events count:\n")
            f.write(edf["kind"].value_counts().to_string() + "\n")
            if summ["n_target"] > 0:
                tx = edf.loc[edf["kind"] == "target", "target_x"].astype(int)
                f.write(f"Target x: min={tx.min()}, max={tx.max()}, mean={tx.mean():.1f}, median={tx.median()}\n")
                f.write(f"Left={summ['targets_left']} | Center={summ['targets_center']} | Right={summ['targets_right']}\n")
            summaries.append(summ)

        if summaries:
            sum_df = pd.DataFrame(summaries).sort_values(["participant", "rate_hz", "run"])
            f.write("\n=== Overall Summary ===\n")
            f.write(sum_df.to_string(index=False))

    print("All results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
