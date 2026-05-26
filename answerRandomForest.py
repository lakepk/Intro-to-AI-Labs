from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 20     # 树的数量
ratio_data = 0.5   # 采样的数据比例
ratio_feat = 0.9 # 采样的特征比例
hyperparams = {
    "depth":6, 
    "purity_bound":1,
    "gainfunc": negginiDA
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    n, d = X.shape
    trees = []
    n_sample = int(n * ratio_data)
    n_feat = int(d * ratio_feat)
    for _ in range(num_tree):
        # 样本扰动：bootstrap采样
        indices = np.random.choice(n, n_sample, replace=True)
        X_sample = X[indices]
        Y_sample = Y[indices]
        # 属性扰动：随机选择特征子集
        feat_indices = np.random.choice(d, n_feat, replace=False)
        unused = list(feat_indices)
        # 构建决策树
        tree = buildTree(X_sample, Y_sample, unused, **hyperparams)
        trees.append(tree)
    return trees

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
