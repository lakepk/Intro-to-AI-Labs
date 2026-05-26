import numpy as np
import modelLogisticRegression as LR
import modelTree as Tree
import modelRandomForest as Forest
import modelSoftmaxRegression as SR
import modelMultiLayerPerceptron as MLP
import YourTraining
import YourTraining2  # 使CNN的Conv2D/Flatten类可被pickle反序列化
import pickle
from scipy.ndimage import rotate, shift
from autograd.BaseGraph import Graph

# 将CNN类注册到__main__以使pickle能反序列化(训练时模块名为__main__)
import __main__
__main__.Conv2D = YourTraining2.Conv2D
__main__.Flatten = YourTraining2.Flatten

class NullModel:

    def __init__(self):
        pass

    def __call__(self, figure):
        return 0


class LRModel:
    def __init__(self) -> None:
        with open(LR.save_path, "rb") as f:
            self.weight, self.bias = pickle.load(f)

    def __call__(self, figure):
        pred = figure @self.weight + self.bias
        return 0 if pred > 0 else 1

class TreeModel:
    def __init__(self) -> None:
        with open(Tree.save_path, "rb") as f:
            self.root = pickle.load(f)
    
    def __call__(self, figure):
        return Tree.inferTree(self.root, Tree.discretize(figure.flatten()))


class ForestModel:
    def __init__(self) -> None:
        with open(Forest.save_path, "rb") as f:
            self.roots = pickle.load(f)
    
    def __call__(self, figure):
        return Forest.infertrees(self.roots, Forest.discretize(figure.flatten()))


class SRModel:
    def __init__(self) -> None:
        with open(SR.save_path, "rb") as f:
            graph = pickle.load(f)
        self.graph = graph
        self.graph.eval()

    def __call__(self, figure):
        self.graph.flush()
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        return np.argmax(pred, axis=-1)
    
class MLPModel:
    def __init__(self) -> None:
        with open(MLP.save_path, "rb") as f:
            graph = pickle.load(f)
        self.graph = graph
        self.graph.eval()

    def __call__(self, figure):
        self.graph.flush()
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        return np.argmax(pred, axis=-1)

class YourModel:
    def __init__(self) -> None:
        with open(YourTraining.save_path, "rb") as f:
            data = pickle.load(f)
        # Graph是List子类，需先判断Graph
        if isinstance(data, Graph):
            self.graphs = [data]
        else:
            self.graphs = data
        for g in self.graphs:
            g.eval()
        # 检测是否为CNN（第一个节点有ks属性=conv kernel_size）
        self.is_cnn = hasattr(self.graphs[0][0], 'ks')

    def __call__(self, figure):
        if figure.ndim == 2:
            figure = figure.reshape(-1)
        x = (figure - YourTraining.GLOBAL_MEAN) / (YourTraining.GLOBAL_STD + 1e-6)
        if self.is_cnn:
            x = x.reshape(1, 1, 28, 28)
        preds = []
        for g in self.graphs:
            g.flush()
            pred = g.forward(x, removelossnode=True)[-1]
            preds.append(pred)
        avg_pred = np.mean(preds, axis=0)
        return int(np.argmax(avg_pred, axis=-1).flat[0])

modeldict = {
    "Null": NullModel,
    "LR": LRModel,
    "Tree": TreeModel,
    "Forest": ForestModel,
    "SR": SRModel,
    "MLP": MLPModel,
    "Your": YourModel
}

