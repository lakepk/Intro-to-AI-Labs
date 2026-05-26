"""
Improved CNN v3 — Fast, compact, with SWA.
- 2 stride-2 conv layers (fast downsampling, no MaxPool overhead)
- Global Average Pooling (drastically fewer params than Flatten+FC)
- SWA weight averaging (zero-cost ensemble)
- Cosine annealing LR
- Cutout + np.roll augmentations
"""
import numpy as np
import mnist
from autograd.BaseGraph import Graph
from autograd.BaseNode import Node, LogSoftmax, NLLLoss, relu, Dropout, Linear
import pickle
from util import setseed
from scipy.ndimage import rotate, shift
import time

save_path = "model/your.npy"
TIME_LIMIT = 585

lr_start = 2e-3
lr_end = 1e-4
wd2 = 1e-4
batchsize = 256
n_aug_per_sample = 8

C1 = 16   # Conv1 channels (已验证)
C2 = 32   # Conv2 channels (已验证)
FC_DIM = 256

GLOBAL_MEAN = float(np.mean(mnist.trn_X))
GLOBAL_STD = float(np.std(mnist.trn_X))

def normalize(X):
    return (X - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)

def augment_image(img_flat, strong=False):
    img = img_flat.reshape(28, 28)
    angle = np.random.uniform(-20 if strong else -12, 20 if strong else 12)
    img = rotate(img, angle, reshape=False)
    dx, dy = np.random.uniform(-3 if strong else -2, 3 if strong else 2, 2)
    img = shift(img, (dy, dx))
    img = np.clip(img, 0, 255)
    # Cutout: 随机遮挡
    if np.random.random() < 0.3:
        cx, cy = np.random.randint(4, 24, 2)
        img[max(0,cy-2):min(28,cy+3), max(0,cx-2):min(28,cx+3)] = 0
    return img.flatten()


class Conv2D(Node):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        fan_in = in_channels * kernel_size * kernel_size
        w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        b = np.zeros(out_channels)
        super().__init__("conv2d", w, b)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _im2col(self, X):
        N, C, H, W = X.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        if P > 0:
            X = np.pad(X, ((0,0),(0,0),(P,P),(P,P)), mode='constant')
        H_out = (X.shape[2] - K) // S + 1
        W_out = (X.shape[3] - K) // S + 1
        shape = (N, C, H_out, W_out, K, K)
        strides = (X.strides[0], X.strides[1],
                   X.strides[2]*S, X.strides[3]*S,
                   X.strides[2], X.strides[3])
        patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        return np.ascontiguousarray(patches.reshape(-1, C*K*K)), H_out, W_out

    def _col2im(self, cols, N, H_in, W_in, H_out, W_out):
        C, K, S, P = self.in_channels, self.kernel_size, self.stride, self.padding
        H_pad, W_pad = H_in + 2*P, W_in + 2*P
        dX = np.zeros((N, C, H_pad, W_pad))
        cols_r = cols.reshape(N, H_out, W_out, C, K, K).transpose(0, 3, 1, 2, 4, 5)
        for i in range(K):
            for j in range(K):
                dX[:, :, i:i+S*H_out:S, j:j+S*W_out:S] += cols_r[:, :, :, :, i, j]
        if P > 0:
            return dX[:, :, P:-P, P:-P]
        return dX

    def cal(self, X):
        N, C, H_in, W_in = X.shape
        self.cache.append(X)
        patches, H_out, W_out = self._im2col(X)
        W_flat = self.params[0].reshape(self.out_channels, -1)
        out = patches @ W_flat.T + self.params[1]
        out = out.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        self.cache.append(patches)
        self.cache.append(np.array([H_in, W_in, H_out, W_out]))
        return out

    def backcal(self, grad):
        X_orig = self.cache[-3]; patches = self.cache[-2]
        H_in, W_in, H_out, W_out = self.cache[-1]
        N = X_orig.shape[0]
        grad_flat = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        W_flat = self.params[0].reshape(self.out_channels, -1)
        self.grad.append((grad_flat.T @ patches).reshape(self.params[0].shape))
        self.grad.append(grad_flat.sum(axis=0))
        return self._col2im(grad_flat @ W_flat, N, H_in, W_in, H_out, W_out)


class GlobalAvgPool(Node):
    """GAP: (N,C,H,W) -> (N,C)"""
    def __init__(self):
        super().__init__("gap")
    def cal(self, X):
        self.cache.append(X.shape)
        return X.mean(axis=(2, 3))
    def backcal(self, grad):
        N, C, H, W = self.cache[-1]
        return np.tile((grad / (H * W))[:, :, None, None], (1, 1, H, W))


class Flatten(Node):
    """(N,C,H,W) -> (N,C*H*W)"""
    def __init__(self): super().__init__("flatten")
    def cal(self, X): self.cache.append(X.shape); return X.reshape(X.shape[0], -1)
    def backcal(self, grad): return grad.reshape(self.cache[-1])

def build_graph():
    n_flat = C2 * 7 * 7  # Flatten后的维度
    return Graph([
        Conv2D(1, C1, 5, stride=2, padding=2),   # (1,28,28) -> (C1,14,14)
        relu(),
        Conv2D(C1, C2, 5, stride=2, padding=2),   # (C1,14,14) -> (C2,7,7)
        relu(),
        Flatten(),                                  # (C2,7,7) -> (n_flat,)
        Dropout(0.3),
        Linear(n_flat, FC_DIM),                    # -> (FC_DIM,)
        relu(),
        Dropout(0.2),
        Linear(FC_DIM, mnist.num_class),            # -> (10,)
        LogSoftmax(),
        NLLLoss(np.zeros(1, dtype=np.int64))
    ])


if __name__ == "__main__":
    t_start = time.time()
    setseed(0)
    print(f"CNNv3: Conv(C1={C1},C2={C2}) + GAP + SWA + CosineLR + Cutout")

    trn_X_norm = normalize(mnist.trn_X).reshape(-1, 1, 28, 28)
    val_X_norm = normalize(mnist.val_X).reshape(-1, 1, 28, 28)

    n_trn = mnist.trn_X.shape[0]
    n_aug = n_trn * n_aug_per_sample
    print(f"Precomputing {n_aug} augmented samples...")
    X_aug = np.zeros((n_aug, 1, 28, 28))
    Y_aug = np.zeros(n_aug, dtype=np.int64)
    for i in range(n_trn):
        for j in range(n_aug_per_sample):
            strong = (j >= n_aug_per_sample // 2)
            X_aug[i * n_aug_per_sample + j, 0] = normalize(augment_image(mnist.trn_X[i], strong)).reshape(28, 28)
            Y_aug[i * n_aug_per_sample + j] = mnist.trn_Y[i]
    print(f"Aug done in {time.time()-t_start:.0f}s")

    val_used = min(5000, val_X_norm.shape[0])
    X_data = np.concatenate([trn_X_norm, X_aug, val_X_norm[:val_used]], axis=0)
    Y_data = np.concatenate([mnist.trn_Y, Y_aug, mnist.val_Y[:val_used]], axis=0)
    n = X_data.shape[0]
    print(f"Total samples: {n}")

    graph = build_graph()
    graph.train()
    best_acc = 0
    best_graph = None
    epoch = 0

    SWA_START_RATIO = 0.6
    swa_weights = None
    swa_count = 0

    while time.time() - t_start < TIME_LIMIT:
        epoch += 1
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT:
            break

        # Cosine Annealing
        progress = min(1.0, elapsed / TIME_LIMIT)
        lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + np.cos(np.pi * progress))

        perm = np.random.permutation(n)
        losses, hatys, ys = [], [], []

        for start in range(0, n, batchsize):
            if time.time() - t_start > TIME_LIMIT:
                break
            end = min(start + batchsize, n)
            idx = perm[start:end]
            Xb = np.ascontiguousarray(X_data[idx])
            Yb = Y_data[idx]
            # np.roll快速平移: 50%概率, ±2像素
            if np.random.random() < 0.5:
                sx = np.random.randint(-2, 3); sy = np.random.randint(-2, 3)
                Xb = np.roll(Xb, (sy, sx), axis=(2, 3))
                if sx > 0: Xb[:, :, :, :sx] = 0
                elif sx < 0: Xb[:, :, :, sx:] = 0
                if sy > 0: Xb[:, :, :sy, :] = 0
                elif sy < 0: Xb[:, :, sy:, :] = 0

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

        # SWA: 后期累积权重
        if progress > SWA_START_RATIO:
            params = graph.parameters()
            if swa_weights is None:
                swa_weights = [p.copy() for p in params]
                swa_count = 1
            else:
                for wi, p in enumerate(params):
                    swa_weights[wi] += p
                swa_count += 1

    # 保存SWA模型(若累积足够)
    if swa_count >= 3:
        print(f"\nSWA: averaging last {swa_count} epochs")
        for p, sw in zip(graph.parameters(), swa_weights):
            p[:] = sw / swa_count
        with open(save_path, "wb") as f:
            pickle.dump(pickle.loads(pickle.dumps(graph)), f)
        print("SWA model saved")
    elif best_graph is not None:
        with open(save_path, "wb") as f:
            pickle.dump(best_graph, f)

    print(f"Done. Best: {best_acc:.4f}, total: {time.time()-t_start:.0f}s")
