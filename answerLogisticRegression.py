import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 3.0  # 学习率
wd = 0.005  # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    X = X / 255.0
    return X @ weight + bias
    raise NotImplementedError

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    X = X / 255.0
    haty = X @ weight + bias
    z = -Y * haty
    loss = np.mean(np.maximum(0, z) + np.log(1 + np.exp(-np.abs(z)))) + wd * np.sum(weight ** 2)
    d = -Y * sigmoid(-Y*haty)
    grad_weight = X.T @ d / X.shape[0] + 2 * wd * weight
    grad_bias = d.mean()
    weight -= lr * grad_weight
    bias -= lr * grad_bias
    return (haty, loss, weight, bias)
    raise NotImplementedError
