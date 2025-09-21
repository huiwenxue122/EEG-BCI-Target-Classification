import numpy as np
import pandas as pd
from scipy import stats

# ===== 1) 录入新数据 =====
data_new = {
    "Participant": [2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14],
    "AUC_5Hz": [0.84, 0.80, 0.85, 0.79, 0.82, 0.76, 0.69, 0.74, 0.88, 0.68, 0.82],
    "AUC_6Hz": [0.77, 0.74, 0.90, 0.82, 0.72, 0.54, 0.69, 0.78, 0.88, 0.51, 0.90],
    "AUC_10Hz": [0.62, 0.67, 0.86, 0.88, 0.69, 0.65, 0.63, 0.85, 0.96, 0.51, np.nan]  # 14号缺 10Hz
}
df_new = pd.DataFrame(data_new)

# 只保留三列都非空的行（保证成对比较是配对的）
auc_values_new = df_new[["AUC_5Hz", "AUC_6Hz", "AUC_10Hz"]].dropna()

# ===== 2) Wilcoxon 成对比较 =====
pairs = [("AUC_5Hz", "AUC_6Hz"),
         ("AUC_5Hz", "AUC_10Hz"),
         ("AUC_6Hz", "AUC_10Hz")]

rows = []
for a, b in pairs:
    paired = auc_values_new[[a, b]].dropna()  # 这一步是稳妥写法
    stat, p = stats.wilcoxon(paired[a], paired[b])
    rows.append([a, b, stat, p])

res = pd.DataFrame(rows, columns=["Cond1", "Cond2", "Wilcoxon_stat", "p_raw"])

# ===== 3) Bonferroni 校正 =====
m = len(pairs)
res["p_bonferroni"] = (res["p_raw"] * m).clip(upper=1.0)

print(res)
