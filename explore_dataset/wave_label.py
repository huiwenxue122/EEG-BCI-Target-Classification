import os  # 文件和路径操作 / File and path operations
import glob  # 文件模式匹配 / Filename pattern matching
import numpy as np  # 数值计算 / Numerical computing
import mne  # EEG 数据处理 / EEG data processing
import matplotlib.pyplot as plt  # 绘图 / Plotting

# --- 全局参数设置 / Global parameters ---
data_root    = r"C:\Users\huiwe\Desktop\RSVP0420\eeg-signals-from-an-rsvp-task-1.0.0"
rates        = ['5-Hz', '6-Hz', '10-Hz']  # 三种频率条件 / Three flicker rates
epoch_tmin   = 0.2    # 片段起始时间 s / Epoch start time (s)
epoch_tmax   = 0.4    # 片段结束时间 s / Epoch end time (s)
threshold_px = 67     # 侧向阈值像素 / Pixel threshold for lateral target
image_width  = 640    # 图片宽度 px / Image width (px)

# EEG 通道列表，顺序需与原始数据一致 / List of EEG channels in order
eeg_chs = [
    'EEG PO7','EEG PO8','EEG P7','EEG P8',
    'EEG PO3','EEG PO4','EEG O1','EEG O2',
]
# 配对通道，用于计算对侧-同侧差值 / Channel pairs for contra-ipsi difference
paired_chs = [
    ('EEG PO7','EEG PO8'),
    ('EEG P7','EEG P8'),
    ('EEG PO3','EEG PO4'),
    ('EEG O1','EEG O2'),
]

def extract_features(raw):
    """
    从原始 MNE Raw 对象中提取 Contralateral-Ipsilateral 特征矩阵 X 和标签 y。
    Extract the contra-ipsi difference features X and labels y from a Raw object.
    """
    sfreq = raw.info['sfreq']  # 采样率 / Sampling frequency
    center_px = image_width // 2  # 图像中心位置 / Center pixel

    events, labels = [], []
    # 遍历所有注释, 仅保留 T=1 且侧向的 trial / Keep only lateral T=1 trials
    for desc, onset in zip(raw.annotations.description, raw.annotations.onset):
        if not desc.startswith("T=1,"):
            continue
        x_px = int(desc.split('x=')[1])
        dx = x_px - center_px
        if abs(dx) < threshold_px:
            continue
        events.append([int(onset * sfreq), 0, 1 if dx > 0 else 0])
        labels.append(1 if dx > 0 else 0)

    y = np.array(labels, dtype=int)
    # 如果没有标签, 返回空矩阵 / If no trials, return empty
    if y.size == 0:
        n_feats = len(paired_chs) * int((epoch_tmax - epoch_tmin) * sfreq)
        return np.empty((0, n_feats)), y

    # 创建 Epochs 并获取数据 / Create Epochs and get data
    epochs = mne.Epochs(raw, np.array(events), event_id=None,
                        tmin=epoch_tmin, tmax=epoch_tmax,
                        baseline=None, preload=True, verbose=False)
    data = epochs.get_data()  # 形状 (n_trials, n_ch, n_times)

    # 计算对侧-同侧差异特征 / Compute contra-ipsi difference features
    X = []
    for trial, label in zip(data, y):
        feats = []
        for chL, chR in paired_chs:
            iL = eeg_chs.index(chL)
            iR = eeg_chs.index(chR)
            diff = (trial[iL] - trial[iR]) if label == 1 else (trial[iR] - trial[iL])
            feats.append(diff)
        X.append(np.hstack(feats))
    return np.vstack(X), y  # 返回特征矩阵 X 和标签 y / Return X, y

def plot_example_waveforms(X_all, y_all, rates, paired_chs, epoch_tmin, epoch_tmax, sfreq, n_examples=1):
    for rate in rates:
        X = X_all[rate]  # shape (n_trials, n_feats)
        y = y_all[rate]
        if X.size == 0:
            continue

        n_pairs = len(paired_chs)
        n_feats = X.shape[1]
        n_times = n_feats // n_pairs  # 自动计算时间点数
        times = np.linspace(epoch_tmin, epoch_tmax, n_times)

        left_idx  = np.where(y == 0)[0][:n_examples]
        right_idx = np.where(y == 1)[0][:n_examples]

        plt.figure(figsize=(8,4))
        for i, idx in enumerate(left_idx):
            mat = X[idx].reshape(n_pairs, n_times)
            # avg_wave = mat.mean(axis=0)
            # plt.plot(times, avg_wave,
            #          color='blue', alpha=0.7,
            #          label='Left example' if i==0 else None)
            avg_wave = mat.mean(axis=0)  # 单位是 V
            avg_wave_uV = avg_wave * 1e6  # 转成 µV
            plt.plot(times, avg_wave_uV,
             color='blue', alpha=0.7,
             label='Left example' if i==0 else None)


        for i, idx in enumerate(right_idx):
            mat = X[idx].reshape(n_pairs, n_times)
            # avg_wave = mat.mean(axis=0)
            # plt.plot(times, avg_wave,
            #          color='red', alpha=0.7,
            #          label='Right example' if i==0 else None)
            avg_wave = mat.mean(axis=0)
            avg_wave_uV = avg_wave * 1e6
            plt.plot(times, avg_wave_uV,
             color='red', alpha=0.7,
             label='Right example' if i==0 else None)

        plt.axhline(0, color='black', linewidth=0.5)
        plt.xlabel('Time (s) relative to stimulus onset')
        plt.ylabel('Contra – Ipsi voltage (µV)')
        plt.title(f'Example EEG waveforms ({rate})')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 1) 提取所有文件的特征，并存入字典 / Extract features for all files into dicts
    X_all = {rate: [] for rate in rates}
    y_all = {rate: [] for rate in rates}

    for rate in rates:
        edfs = glob.glob(os.path.join(data_root, rate, '*.edf'))
        for edf in edfs:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
            raw.pick_channels(eeg_chs)
            raw.filter(0.15, 28., fir_design='firwin', verbose=False)
            raw.resample(64, npad='auto', verbose=False)

            X, y = extract_features(raw)
            if X.size == 0:
                continue
            X_all[rate].append(X)
            y_all[rate].append(y)

        # 将各个 EDF 的结果拼接为单个数组 / Concatenate per-rate lists into arrays
        if X_all[rate]:
            X_all[rate] = np.vstack(X_all[rate])
            y_all[rate] = np.hstack(y_all[rate])
        else:
            X_all[rate] = np.empty((0, len(paired_chs)*int((epoch_tmax-epoch_tmin)*64)))
            y_all[rate] = np.array([], dtype=int)

    # 2) 绘制示例波形 / Plot example waveforms
    sfreq = 64  # 采样率 / Sampling rate
    plot_example_waveforms(
        X_all, y_all, rates, paired_chs,
        epoch_tmin, epoch_tmax, sfreq,
        n_examples=1  # 每个速率各画 1 条 Left 和 1 条 Right
    )
