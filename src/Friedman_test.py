import numpy as np
import pandas as pd
from scipy import stats

data_new = {
    "Participant": [2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14],
    "AUC_5Hz": [0.84, 0.80, 0.85, 0.79, 0.82, 0.76, 0.69, 0.74, 0.88, 0.68, 0.82],
    "AUC_6Hz": [0.77, 0.74, 0.90, 0.82, 0.72, 0.54, 0.69, 0.78, 0.88, 0.51, 0.90],
    "AUC_10Hz": [0.62, 0.67, 0.86, 0.88, 0.69, 0.65, 0.63, 0.85, 0.96, 0.51, np.nan]  # 14号没有10Hz
}

df_new = pd.DataFrame(data_new)

# 只保留完整的数据（三个频率都有）
auc_values_new = df_new[["AUC_5Hz", "AUC_6Hz", "AUC_10Hz"]].dropna()

# Friedman 检验
friedman_stat, friedman_p = stats.friedmanchisquare(
    auc_values_new["AUC_5Hz"],
    auc_values_new["AUC_6Hz"],
    auc_values_new["AUC_10Hz"]
)

print("Friedman test statistic:", friedman_stat)
print("p-value:", friedman_p)
