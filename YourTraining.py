import numpy as np
import mnist
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
import pickle
from util import setseed
from scipy.ndimage import rotate, shift
import time

save_path = "model/your.npy"
TIME_LIMIT = 585

lr_start = 1e-3
lr_end = 1e-4
wd2 = 1e-4
batchsize = 256
n_aug_per_sample = 10

GLOBAL_MEAN = float(np.mean(mnist.trn_X))
GLOBAL_STD = float(np.std(mnist.trn_X))

def normalize(X):
    return (X - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)

def augment_image(img_flat, level=0):
    img = img_flat.reshape(28, 28)
    angles = [12, 20, 30]
    shifts = [2, 4, 6]
    angle = np.random.uniform(-angles[level], angles[level])
    shift_range = shifts[level]
    img = rotate(img, angle, reshape=False)
    dx, dy = np.random.uniform(-shift_range, shift_range, 2)
    img = shift(img, (dy, dx))
    img = np.clip(img, 0, 255)
    return img.flatten()

if __name__ == "__main__":
    t_start = time.time()
    print(f"Global norm: mean={GLOBAL_MEAN:.1f} std={GLOBAL_STD:.1f}")

    # 单模型：最佳架构 + 全量数据
    graph = Graph([
        Linear(mnist.num_feat, 1024), relu(), Dropout(0.2),
        Linear(1024, 512), relu(), Dropout(0.2),
        Linear(512, 256), relu(), Dropout(0.1),
        Linear(256, mnist.num_class),
        LogSoftmax(), NLLLoss(np.zeros(1, dtype=np.int64))
    ])

    # 数据：trn + aug + ALL val
    trn_X_norm = normalize(mnist.trn_X)
    val_X_norm = normalize(mnist.val_X)
    n_trn = trn_X_norm.shape[0]
    n_aug = n_trn * n_aug_per_sample
    print(f"Precomputing {n_aug} augmented samples (3 levels)...")
    X_aug = np.zeros((n_aug, mnist.num_feat))
    Y_aug = np.zeros(n_aug, dtype=np.int64)
    for i in range(n_trn):
        for j in range(n_aug_per_sample):
            level = min(j * 3 // n_aug_per_sample, 2)
            X_aug[i * n_aug_per_sample + j] = normalize(augment_image(mnist.trn_X[i], level))
            Y_aug[i * n_aug_per_sample + j] = mnist.trn_Y[i]
    print(f"Aug done in {time.time()-t_start:.0f}s")

    X_data = np.concatenate([trn_X_norm, X_aug, val_X_norm], axis=0)
    Y_data = np.concatenate([mnist.trn_Y, Y_aug, mnist.val_Y], axis=0)
    n = X_data.shape[0]
    print(f"Total samples: {n}")

    graph.train()
    best_acc = 0
    best_graph = None
    epoch = 0

    while time.time() - t_start < TIME_LIMIT:
        epoch += 1
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT:
            break

        progress = min(1.0, elapsed / TIME_LIMIT)
        lr = lr_start + (lr_end - lr_start) * progress

        perm = np.random.permutation(n)
        losses, hatys, ys = [], [], []

        for start in range(0, n, batchsize):
            if time.time() - t_start > TIME_LIMIT:
                break
            end = min(start + batchsize, n)
            idx = perm[start:end]
            Xb = np.ascontiguousarray(X_data[idx])
            Yb = Y_data[idx]

            graph[-1].y = Yb; graph.flush()
            pred, loss = graph.forward(Xb)[-2:]
            hatys.append(np.argmax(pred, axis=1)); ys.append(Yb); losses.append(loss)
            graph.backward(); graph.optimstep(lr, 0, wd2)

        acc = np.average(np.concatenate(hatys) == np.concatenate(ys))
        elapsed = time.time() - t_start
        print(f"ep{epoch:2d} lr={lr:.2e} acc {acc:.4f} t{elapsed:.0f}s")

        if acc > best_acc:
            best_acc = acc
            best_graph = pickle.loads(pickle.dumps(graph))

    if best_graph is not None:
        with open(save_path, "wb") as f:
            pickle.dump(best_graph, f)
    print(f"Done. Best acc: {best_acc:.4f}, total: {time.time()-t_start:.0f}s")
