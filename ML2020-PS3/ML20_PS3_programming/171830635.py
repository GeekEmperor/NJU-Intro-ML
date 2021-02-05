import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
from itertools import combinations

# (1)
X_train = pd.read_csv('./X_train.csv', header=None).values
y_train = pd.read_csv('./y_train.csv', header=None).values
X_test = pd.read_csv('./X_test.csv', header=None).values
y_train[y_train == 0] = -1
y_train = y_train.astype(float)

def QP(X, y):
    m, n = X.shape
    # 求解QP问题
    P = matrix((y @ y.T) * (X @ X.T) + np.diag(np.full(m, 1e-7)))
    q = matrix(-np.ones(m))
    G = matrix(-np.eye(m))
    h = matrix(np.zeros(m))
    A = matrix(y.T)
    b = matrix(0.0)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(sol['x'])
    w = X.T @ (alpha * y)
    # 支持向量索引
    indice = np.ravel(alpha > 1)
    b = np.mean(y[indice] - X[indice] @ w)
    return w, b

w, b = QP(X_train, y_train)
y_pred = np.sign(X_train @ w + b)
print("the accuracy using QP:", np.mean(y_pred == y_train))

# (2)
def SMO(X, y):
    m, n = X.shape
    # 两两距离
    D = np.sum(np.square(X[np.newaxis,:,:] - X[:,np.newaxis,:]), -1)
    D = y @ y.T * D
    # 最远异类
    D = np.argmin(D, 1)
    alpha = np.random.rand(m, 1) * 0.02
    for k in range(1000):
        w = X.T @ (alpha * y)
        indice = np.ravel(alpha > 0.01)
        b = np.mean(y[indice] - X[indice] @ w)
        # KKT违背程度
        err = np.ravel(y * (X @ w + b) - 1)
        err[indice] **= 2
        err[~indice & (err >= 0)] = 0
        err[~indice & (err < 0)] **= 2 
        i = np.argmax(err)
        j = D[i]
        # 更新alpha[i]和alpha[j]
        c = alpha[i] * y[i] + alpha[j] * y[j] - alpha.T @ y
        p = np.sum(alpha * y * X @ X[i].T)
        q = np.sum(alpha * y * X @ X[j].T)
        r = X[i] @ X[j].T
        t = 1 - y[i] * y[j] - y[i] * p + y[i] * q + y[i] * c * r
        t /= 2 * r
        alpha[i] = max([0, t])
        alpha[j] = c * y[j] - y[i] * y[j] * alpha[i]
    return w, b

w, b = SMO(X_train, y_train)
y_pred = np.sign(X_train @ w + b)
print("the accuracy using SMO:", np.mean(y_pred == y_train))

# (3)
def Kernel(X, y, sigma = 0.1):
    m, n = X.shape
    D = np.exp(np.sum(np.square(X[np.newaxis,:,:] - X[:,np.newaxis,:]), -1) \
               / (-2 * sigma ** 2))
    P = matrix(y @ y.T * D + np.diag(np.full(m, 1e-7)))
    q = matrix(-np.ones(m))
    G = matrix(-np.eye(m))
    h = matrix(np.zeros(m))
    A = matrix(y.T)
    b = matrix(0.0)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(sol['x'])
    b = np.mean(y.T - np.sum(alpha * y * D, 0))
    return alpha, b

# 核技巧
sigma = 0.1
alpha, b = Kernel(X_train, y_train, sigma)
D = np.exp(np.sum(np.square(X_train[np.newaxis,:,:] - X_train[:,np.newaxis,:]), -1) \
           / (-2 * sigma ** 2))
y_pred = np.sign(np.sum(alpha * y_train * D, 0) + b)
print("the accuracy using Kernel:", np.mean(y_pred == y_train))

# 特征工程
# 取倒数
d = {}
for i in range(1, 6):
    for j in combinations(range(5), i):
        X_temp = X_train.copy()
        X_temp[:, j] = 1 / X_temp[:, j]
        w, b = QP(X_temp, y_train)
        y_pred = np.sign(X_temp @ w + b)
        d[j] = np.mean(y_pred == y_train)
print("the accuracy using inverse of feature:", max(d.items(), key=lambda x: x[1]))

# 取指数
d = {}
for i in range(1, 6):
    for j in combinations(range(5), i):
        X_temp = X_train.copy()
        X_temp[:, j] = np.exp(X_temp[:, j])
        w, b = QP(X_temp, y_train)
        y_pred = np.sign(X_temp @ w + b)
        d[j] = np.mean(y_pred == y_train)
print("the accuracy using exp of feature:", max(d.items(), key=lambda x: x[1]))

# 输出文件
w, b = QP(X_train, y_train)
y_pred = np.sign(X_test @ w + b)
y_pred[y_pred < 0] = 0
pd.DataFrame(y_pred.astype(int)).to_csv('171830635_俞星凯.csv', header=False, index=False)
