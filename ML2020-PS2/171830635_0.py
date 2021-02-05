# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# 读取数据
X_train = pd.read_csv('train_feature.csv')
y_train = pd.read_csv('train_target.csv')
X_val = pd.read_csv('val_feature.csv')
y_val = pd.read_csv('val_target.csv')
X_test = pd.read_csv('test_feature.csv')

# 增加一列
X_train['ones'] = 1
X_val['ones'] = 1
X_test['ones'] = 1

# 训练模型
X = X_train.values
y = y_train.values
beta = np.linalg.pinv(X.T @ X) @ X.T @ y

# 评估模型
theta = 0.55
z = X_val.values @ beta
f = 1 / (1 + np.exp(-z))
y_pred = f > theta
tp = np.sum((y_pred == 1) & (y_val.values == 1))
acc = np.mean(y_pred == y_val.values)
precision = tp / np.sum(y_pred == 1)
recall = tp / np.sum(y_val.values == 1)
print('accuracy:', acc)
print('precision:', precision)
print('recall:', recall)

# 预测输出
z = X_test.values @ beta
f = 1 / (1 + np.exp(-z))
y_pred = f > theta
pd.DataFrame(y_pred, dtype=int).to_csv('171830635_0.csv', index=False)
# np.sum(pd.read_csv('171830635_0.csv') == pd.read_csv('171830635_1.csv'))