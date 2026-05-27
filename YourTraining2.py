"""
2-CNN ensemble: M0(wide, 50000 val, 350s) + M1(heavy aug, 5000 val, 240s)
互补策略：广度覆盖 + 增强鲁棒性
"""
import numpy as np; import mnist
from autograd.BaseGraph import Graph
from autograd.BaseNode import Node, LogSoftmax, NLLLoss, relu, Dropout, Linear
import pickle; from util import setseed; from scipy.ndimage import rotate, shift; import time

save_path = "model/your.npy"

lr_start = 2e-3; lr_end = 1e-4; wd2 = 1e-4; batchsize = 256
n_aug = 6

GLOBAL_MEAN = float(np.mean(mnist.trn_X))
GLOBAL_STD = float(np.std(mnist.trn_X))
def N(X): return (X - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)

def aug_image(img_flat, strong=False, focus='both'):
    """focus: 'rotate'重旋转 / 'shift'重平移 / 'both'均衡"""
    img = img_flat.reshape(28, 28)
    if focus == 'rotate':
        angle = np.random.uniform(-30 if strong else -15, 30 if strong else 15)
        img = rotate(img, angle, reshape=False)
        dx, dy = np.random.uniform(-1, 1, 2); img = shift(img, (dy, dx))
    elif focus == 'shift':
        angle = np.random.uniform(-8, 8); img = rotate(img, angle, reshape=False)
        dx, dy = np.random.uniform(-5 if strong else -3, 5 if strong else 3, 2)
        img = shift(img, (dy, dx))
    else:
        angle = np.random.uniform(-20 if strong else -12, 20 if strong else 12)
        img = rotate(img, angle, reshape=False)
        dx, dy = np.random.uniform(-3 if strong else -2, 3 if strong else 2, 2)
        img = shift(img, (dy, dx))
    img = np.clip(img, 0, 255)
    if np.random.random() < 0.3:
        cx, cy = np.random.randint(4, 24, 2)
        img[max(0,cy-2):min(28,cy+3), max(0,cx-2):min(28,cx+3)] = 0
    return img.flatten()

class Conv2D(Node):
    def __init__(self, ic, oc, ks, stride=1, pad=0):
        fan = ic * ks * ks; w = np.random.randn(oc, ic, ks, ks) * np.sqrt(2.0/fan)
        super().__init__("conv2d", w, np.zeros(oc))
        self.ks, self.S, self.P = ks, stride, pad; self.ic, self.oc = ic, oc
    def _im2col(self, X):
        N,C,H,W=X.shape; K,S,P=self.ks,self.S,self.P
        if P>0: X=np.pad(X,((0,0),(0,0),(P,P),(P,P)),mode='constant')
        Ho=(X.shape[2]-K)//S+1; Wo=(X.shape[3]-K)//S+1
        sh=(N,C,Ho,Wo,K,K); st=(X.strides[0],X.strides[1],X.strides[2]*S,X.strides[3]*S,X.strides[2],X.strides[3])
        p=np.lib.stride_tricks.as_strided(X,shape=sh,strides=st)
        return np.ascontiguousarray(p.reshape(-1,C*K*K)),Ho,Wo
    def _col2im(self, cols, N, Hi, Wi, Ho, Wo):
        C,K,S,P=self.ic,self.ks,self.S,self.P; Hp,Wp=Hi+2*P,Wi+2*P
        dX=np.zeros((N,C,Hp,Wp)); cr=cols.reshape(N,Ho,Wo,C,K,K).transpose(0,3,1,2,4,5)
        for i in range(K):
            for j in range(K): dX[:,:,i:i+S*Ho:S,j:j+S*Wo:S]+=cr[:,:,:,:,i,j]
        if P>0: return dX[:,:,P:-P,P:-P]
        return dX
    def cal(self, X):
        N,C,Hi,Wi=X.shape; self.cache.append(X)
        patches,Ho,Wo=self._im2col(X); Wf=self.params[0].reshape(self.oc,-1)
        out=patches@Wf.T+self.params[1]; out=out.reshape(N,Ho,Wo,self.oc).transpose(0,3,1,2)
        self.cache.append(patches); self.cache.append(np.array([Hi,Wi,Ho,Wo])); return out
    def backcal(self, grad):
        Xo=self.cache[-3]; patches=self.cache[-2]; Hi,Wi,Ho,Wo=self.cache[-1]; N=Xo.shape[0]
        gf=grad.transpose(0,2,3,1).reshape(-1,self.oc); Wf=self.params[0].reshape(self.oc,-1)
        self.grad.append((gf.T@patches).reshape(self.params[0].shape)); self.grad.append(gf.sum(axis=0))
        return self._col2im(gf@Wf,N,Hi,Wi,Ho,Wo)

class Flatten(Node):
    def __init__(self): super().__init__("flatten")
    def cal(self,X): self.cache.append(X.shape); return X.reshape(X.shape[0],-1)
    def backcal(self,grad): return grad.reshape(self.cache[-1])

def build_graph(C1, C2):
    return Graph([
        Conv2D(1,C1,5,stride=2,pad=2), relu(),
        Conv2D(C1,C2,5,stride=2,pad=2), relu(),
        Flatten(), Dropout(0.3),
        Linear(C2*7*7,256), relu(), Dropout(0.2),
        Linear(256,mnist.num_class), LogSoftmax(),
        NLLLoss(np.zeros(1,dtype=np.int64))])

def prepare_data(val_count, focus='both'):
    trn=N(mnist.trn_X).reshape(-1,1,28,28); val=N(mnist.val_X).reshape(-1,1,28,28)
    nt=mnist.trn_X.shape[0]; na=nt*n_aug
    Xa=np.zeros((na,1,28,28)); Ya=np.zeros(na,dtype=np.int64)
    for i in range(nt):
        for j in range(n_aug):
            strong=(j>=n_aug//2)
            Xa[i*n_aug+j,0]=N(aug_image(mnist.trn_X[i],strong,focus)).reshape(28,28)
            Ya[i*n_aug+j]=mnist.trn_Y[i]
    vu=min(val_count,val.shape[0])
    X=np.concatenate([trn,Xa,val[:vu]],axis=0); Y=np.concatenate([mnist.trn_Y,Ya,mnist.val_Y[:vu]],axis=0)
    return X,Y

def train_one(seed, t_start, deadline, C1, C2, val_count, focus='both'):
    setseed(seed); g=build_graph(C1,C2); g.train()
    X,Y=prepare_data(val_count, focus); n=X.shape[0]
    t_model=time.time(); budget=deadline-time.time()
    best_acc=0; best_g=None; ep=0; swa_w=None; swa_n=0
    print(f"  S{seed} C1={C1} C2={C2} val={val_count} n={n} budget={budget:.0f}s")

    while time.time()<deadline:
        ep+=1
        if time.time()>deadline: break
        p=min(1.0,(time.time()-t_model)/budget)
        lr=lr_end+0.5*(lr_start-lr_end)*(1+np.cos(np.pi*p))
        perm=np.random.permutation(n); losses,haty,ys=[],[],[]
        for s in range(0,n,batchsize):
            if time.time()>deadline: break
            e=min(s+batchsize,n); idx=perm[s:e]
            Xb=np.ascontiguousarray(X[idx]); Yb=Y[idx]
            if np.random.random()<0.5:
                sx=np.random.randint(-2,3); sy=np.random.randint(-2,3)
                Xb=np.roll(Xb,(sy,sx),axis=(2,3))
                if sx>0: Xb[:,:,:,:sx]=0
                elif sx<0: Xb[:,:,:,sx:]=0
                if sy>0: Xb[:,:,:sy,:]=0
                elif sy<0: Xb[:,:,sy:,:]=0
            g[-1].y=Yb; g.flush()
            pred,loss=g.forward(Xb)[-2:]
            haty.append(np.argmax(pred,axis=1)); ys.append(Yb); losses.append(loss)
            g.backward(); g.optimstep(lr,0,wd2)
        acc=np.average(np.concatenate(haty)==np.concatenate(ys))
        elapsed=time.time()-t_start
        print(f"  S{seed} ep{ep:2d} lr={lr:.2e} acc {acc:.4f} t{elapsed:.0f}s")
        if acc>best_acc: best_acc=acc; best_g=pickle.loads(pickle.dumps(g))
        if p>0.6:
            params=g.parameters()
            if swa_w is None: swa_w=[w.copy() for w in params]; swa_n=1
            else:
                for i,w in enumerate(params): swa_w[i]+=w
                swa_n+=1
    if swa_n>=3:
        print(f"  S{seed} SWA: avg {swa_n} epochs")
        for w,sw in zip(g.parameters(),swa_w): w[:]=sw/swa_n
        return pickle.loads(pickle.dumps(g))
    return best_g

if __name__=="__main__":
    t0=time.time()
    # M0: wider, more val data (breadth) — 350s
    # M1: standard, less val but same aug ratio (robustness) — 240s
    # M0: 旋转为主 (±30°), M1: 平移为主 (±5px). 互补增强多样性
    configs = [(0, 295, 16, 32, 5000, 'rotate'), (42, 590, 16, 32, 5000, 'shift')]
    graphs=[]
    for seed, dl, c1, c2, vc, focus in configs:
        g=train_one(seed,t0,t0+dl,c1,c2,vc,focus)
        if g is not None: graphs.append(g)
    with open(save_path,"wb") as f: pickle.dump(graphs,f)
    print(f"\n{len(graphs)} models saved. Total: {time.time()-t0:.0f}s")
