from __future__ import print_function
import numpy as np

class KNN():
   
    #k: int,最近邻个数.
    def __init__(self, k=5):
        self.k = k

    # 此处需要填写，建议欧式距离，计算一个样本与训练集中所有样本的距离
    def distance(self, one_sample, X_train):
        return np.sum(np.square(X_train - one_sample), 1)
    
    # 此处需要填写，获取k个近邻的类别标签
    def get_k_neighbor_labels(self, distances, y_train):
        return y_train[np.argsort(distances)[:self.k]]
    
    # 此处需要填写，标签统计，票数最多的标签即该测试样本的预测标签
    def vote(self, one_sample, X_train, y_train):
        distances = self.distance(one_sample, X_train)
        labels = self.get_k_neighbor_labels(distances, y_train)
        labels = list(labels)
        return max(set(labels), key=labels.count)
    
    # 此处需要填写，对测试集进行预测
    def predict(self, X_test, X_train, y_train):
        n = X_test.shape[0]
        y_pred = np.zeros(n)
        for i in range(n):
            y_pred[i] = self.vote(X_test[i], X_train, y_train)
        return y_pred
  

def main():
    clf = KNN(k=5)
    train_data = np.genfromtxt('./data/train_data.csv', delimiter=' ')
    train_labels = np.genfromtxt('./data/train_labels.csv', delimiter=' ')
    test_data = np.genfromtxt('./data/test_data.csv', delimiter=' ')

    #将预测值存入y_pred(list)内   
    y_pred = clf.predict(test_data, train_data, train_labels)
    np.savetxt("171830635_ypred.csv", y_pred, delimiter=' ')
  


if __name__ == "__main__":
    main()