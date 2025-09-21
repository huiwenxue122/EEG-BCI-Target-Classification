# import mne
# from pathlib import Path

# # 你的文件路径
# fpath = Path(r"C:\Users\huiwe\Desktop\BME Lab\new_RSVP\rsvp_mne_files\rsvp_mne_files\rsvp_5Hz_02a.raw.fif")

# # 读取原始数据
# raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)

# # 只挑 Status 通道
# status_data, times = raw[raw.ch_names.index("Status"), :]

# # 打印前1000个采样点（大部分是 0，事件时会跳到1000/1001/1002/目标编号）
# print("Status channel first 1000 samples:")
# print(status_data[0][:1000])

# # 也可以打印所有非零点（就是事件位置）
# nonzero_idx = (status_data[0] != 0).nonzero()[0]
# print("\nNumber of events in Status channel:", len(nonzero_idx))
# print("First 20 non-zero samples (sample index, value):")
# for i in nonzero_idx[:20]:
#     print(i, status_data[0][i])
# -*- coding: utf-8 -*-

import mne
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# 选择一个 EEG 文件
fpath = Path(r"C:\Users\huiwe\Desktop\BME Lab\new_RSVP\rsvp_mne_files\rsvp_mne_files\rsvp_5Hz_02a.raw.fif")

# 读文件
raw = mne.io.read_raw_fif(fpath, preload=False, verbose=False)

# 提取事件
events = mne.find_events(raw, stim_channel="Status", output="onset", shortest_event=1)

# 时间（秒）和事件编号
times = events[:, 0] / raw.info["sfreq"]
codes = events[:, 2]

# 分类：开始/结束/非目标/目标
burst_start_mask = codes == 1000
burst_end_mask   = codes == 1001
non_target_mask  = codes == 1002
target_mask      = (codes < 1000)  # 1..640

plt.figure(figsize=(12, 4))
plt.scatter(times[burst_start_mask], codes[burst_start_mask], color="red", label="Burst start (1000)", marker="|", s=200)
plt.scatter(times[burst_end_mask], codes[burst_end_mask], color="darkred", label="Burst end (1001)", marker="|", s=200)
plt.scatter(times[non_target_mask], codes[non_target_mask], color="gray", label="Non-target (1002)", marker="|", s=100, alpha=0.5)
plt.scatter(times[target_mask], codes[target_mask], color="blue", label="Target (1..640)", marker="|", s=150)

plt.xlabel("Time (s)")
plt.ylabel("Event code")
plt.title(f"Events in Status channel — {fpath.name}")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 打印前 20 个事件（便于核对）
print("First 20 events (time_s, code):")
for t, c in zip(times[:20], codes[:20]):
    print(f"{t:.3f} s -> {c}")
