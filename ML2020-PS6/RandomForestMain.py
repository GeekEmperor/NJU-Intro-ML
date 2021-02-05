import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate

class RandomForest:
    def __init__(self, n, **kwargs):
        self.n = n
        self.kwargs = kwargs
    
    def get_params(self, deep=True):
        return {'n': self.n}
            
    def fit(self, X, y):
        self.clfs = []
        for i in range(self.n):
            clf = DecisionTreeClassifier(max_features='log2', **self.kwargs)
            idx = np.random.choice(np.arange(y.shape[0]), y.shape[0])
            clf.fit(X[idx], y[idx])
            self.clfs.append(clf)
        return self
    
    def predict(self, X):
        y = np.zeros(X.shape[0])
        for clf in self.clfs:
            y += clf.predict(X)
        return np.sign(y)
    
    def predict_proba(self, X):
        y = np.zeros((X.shape[0], 2))
        for clf in self.clfs:
            y += clf.predict_proba(X)
        return y / y.sum(1, keepdims=True)

X_train = np.loadtxt('adult_dataset/adult_train_feature.txt')
y_train = np.loadtxt('adult_dataset/adult_train_label.txt', dtype=int)
X_test = np.loadtxt('adult_dataset/adult_test_feature.txt')
y_test = np.loadtxt('adult_dataset/adult_test_label.txt', dtype=int)
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
nums = range(5, 300, 5)
train_aucs = []
test_aucs = []
for n in nums:
    scores = cross_validate(RandomForest(n), X_train, y_train, 
                            scoring='roc_auc', cv=5, return_train_score=True)
    train_aucs.append(scores['train_score'].mean())
    test_aucs.append(scores['test_score'].mean())
plt.plot(nums, train_aucs, 'b', label='Train')
plt.plot(nums, test_aucs, 'r', label='Test')
plt.xlabel('Number Of Base Learners')
plt.ylabel('Area Under ROC Curve')
plt.legend()
plt.savefig('randomforest.jpg')
clf = RandomForest(130)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('n_estimators = {}, auc = {}'.format(130, auc))