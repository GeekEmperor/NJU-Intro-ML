import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 绘制PR曲线
data = pd.read_csv("data.csv", index_col=0)
data = data.sort_values('output', ascending=False)
thresholds = data['output'].unique()
n = len(thresholds)
P = np.zeros(n)
R = np.zeros(n)
TPR = np.zeros(n)
FPR = np.zeros(n)
for i in range(n):
    data['pred'] = data['output'] >= thresholds[i]
    TP = np.sum((data['label'] == 1) & (data['pred'] == 1))
    FP = np.sum((data['label'] == 0) & (data['pred'] == 1))
    TN = np.sum((data['label'] == 0) & (data['pred'] == 0))
    FN = np.sum((data['label'] == 1) & (data['pred'] == 0))
    P[i] = TP / (TP + FP)
    R[i] = TP / (TP + FN)
    TPR[i] = TP / (TP + FN)
    FPR[i] = FP / (FP + TN)
plt.plot(R, P)
plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P-R Curve')
plt.show()

# 绘制ROC曲线，计算AUC
plt.plot(FPR, TPR)
plt.axis([0, 1, 0, 1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()
AUC = np.sum((FPR[1:] - FPR[:-1]) * (TPR[:-1] + TPR[1:])) / 2
print("AUC is", AUC)