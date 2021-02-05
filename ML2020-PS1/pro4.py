import numpy as np
import pandas as pd
from scipy.stats import f

data = pd.DataFrame(np.array([[2, 3, 1, 5, 4],
                              [5, 4, 2, 3, 1],
                              [4, 5, 1, 2, 3],
                              [2, 3, 1, 5, 4],
                              [3, 4, 1, 5, 2]]),
                    index=['D1', 'D2', 'D3', 'D4', 'D5'],
                    columns=['A', 'B', 'C', 'D', 'E'])
# Friedman检验
r = data.mean()
n, k = data.shape
chi2 = 12 * n / k / (k + 1) * (np.sum(r ** 2) - (k * (k + 1) ** 2) / 4)
fvalue = (n - 1) * chi2 / (n * (k - 1) - chi2)
if fvalue < f.ppf(0.95, 4, 16):
    print("所有算法性能相同")
# Nemenyi检验
else:
    cd = 2.728 * np.sqrt(k * (k + 1) / 6 / n)
    for i in range(k):
        for j in range(i + 1, k):
            if abs(r[i] - r[j]) > cd:
                print(f'算法{r.index[i]}与{r.index[j]}性能显著不同')
    