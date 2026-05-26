import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 5e-3   # 学习率
wd1 = 0     # L1正则化
wd2 = 1e-4  # L2正则化
batchsize = 128
hidden_dim = 256

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    nodes = [
        StdScaler(mnist.mean_X, mnist.std_X),
        Linear(mnist.num_feat, hidden_dim),
        relu(),
        Linear(hidden_dim, mnist.num_class),
        LogSoftmax(),
        NLLLoss(Y)
    ]
    return Graph(nodes)
