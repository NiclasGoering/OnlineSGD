# relu_learning_curve.py  –  fixed 2025‑07‑21
import math, argparse, pathlib, functools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1.  Exact Boolean–Hermite coefficient  c_n  for ReLU  (log‑safe)
# ------------------------------------------------------------------
LOG_MIN = -745.0                                      # smallest double

@functools.lru_cache(maxsize=None)
def c_relu(n: int, sigma_v: float = 1.0) -> float:
    if n & 1:                       # odd parity → 0
        return 0.0
    q      = n // 2
    log_c  = (n + 1) * math.log(sigma_v)            \
           - (n + 2) * math.log(2.0)                \
           - math.lgamma(2*q + 2)                   \
           + 0.5 * math.log(2.0 / math.pi)
    return math.exp(max(log_c, LOG_MIN))

# ------------------------------------------------------------------
# 2.  χ∞  and  Υ(d,φ)  (no over‑counting of σ_v)
# ------------------------------------------------------------------
def χ_inf(n, d, N, γ, κ, σa, σv):
    cn = c_relu(n, σv)
    return (N ** (1 - γ)) * σa**2 * cn**2 / (κ * d**(n + 1))        # <-- FIX A‑1

def Υ(d, n_star, σv):
    tot = 0.0
    for ℓ in range(d + 1):
        cℓ = c_relu(ℓ, σv)
        tot += math.comb(d, ℓ) * cℓ**2 / d                          # <-- FIX A‑2
    c_star = c_relu(n_star, σv)
    tot -= math.comb(d, n_star) * c_star**2 / d
    return tot

def first_unstable(k, d, N, γ, κ, σa, σv):
    start = k if k % 2 == 0 else k + 1
    for n in range(start, d + 1, 2):
        cn = c_relu(n, σv)
        if cn == 0.0:
            continue
        χ = χ_inf(n, d, N, γ, κ, σa, σv)
        if χ > 1.0:
            return n, cn, χ
    raise RuntimeError("χ_inf ≤ 1 for every mode")

# ------------------------------------------------------------------
# 3.  Landau constants
# ------------------------------------------------------------------
def constants(hp, device="cpu"):
    d,k,N,γ,κ,σa,σv = (hp[x] for x in
                       ("d","k","N","gamma","kappa","sigma_a","sigma_v"))
    n_star, c_star, χ = first_unstable(k,d,N,γ,κ,σa,σv)
    ζ   = (σa**2/κ) * Υ(d, n_star, σv) / d**n_star
    Pc  = ζ / (χ - 1.0)
    α   = (χ - 1.0) / (2 * N * κ)

    J2  = N * c_star**2 * (σv**2 / d)**n_star                       # <-- FIX A‑3
    B0  = 3 * σa**4 / (4 * N**2 * κ**4) * J2**2

    # variance of ReLU output → c0
    M = 40000
    w = torch.randn(M, d, device=device) * σv / math.sqrt(d)
    x = torch.randint(0, 2, (M, d), device=device).float().mul_(2).sub_(1)
    c0 = σa / κ * math.sqrt(F.relu((w * x).sum(1)).var(unbiased=True).item())
    return Pc, α, B0, c0

# ------------------------------------------------------------------
# 4.  m*(P) root  &  theory classification error
# ------------------------------------------------------------------
def m_star(P, Pc, α, B0, c0):
    if P <= Pc:
        return 0.0
    A, C = α * (1 - Pc/P), c0 / math.sqrt(P)
    roots = np.roots([4*B0, 0.0, 2*A, -C])
    real = [r.real for r in roots if abs(r.imag)<1e-8 and r.real>0]
    return min(real) if real else 0.0

# ------------------------------------------------------------------
# 5.  SGLD model & helpers
# ------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, d, N, σv, σa):
        super().__init__()
        self.h = nn.Linear(d, N, False)
        self.o = nn.Linear(N, 1, False)
        nn.init.normal_(self.h.weight, 0, σv/math.sqrt(d))
        nn.init.normal_(self.o.weight, 0, σa/math.sqrt(N))
    def forward(self,x): return self.o(F.relu(self.h(x)))



# No decorator at the top
def sgld_step(net, X, y, lr, κ, λ):
    # 1. Calculate gradients (requires grad context)
    net.zero_grad(set_to_none=True)
    pred = net(X)
    loss = (0.5 * ((pred - y)**2).mean())
    loss.backward()

    # 2. Manually update weights (does NOT require grad context)
    with torch.no_grad():
        std = math.sqrt(2 * lr * κ)
        for p in net.parameters():
            p.add_(-lr * (p.grad + λ * p) + std * torch.randn_like(p))

def make_data(P,d,k,device):
    X = torch.randint(0,2,(P,d),device=device).float().mul_(2).sub_(1)
    y = X[:,:k].prod(1,keepdim=True)
    return X,y

def empirical_err(P,hp,lr,steps,λ,device):
    d,k,N,σv,σa,κ = (hp[x] for x in
                     ("d","k","N","sigma_v","sigma_a","kappa"))
    net = Net(d,N,σv,σa).to(device)
    Xtr,ytr = make_data(P,d,k,device)
    Xte,yte = make_data(4000,d,k,device)
    for _ in range(steps):
        sgld_step(net,Xtr,ytr,lr,κ,λ)
    return ((net(Xte).sign()!=yte.sign()).float().mean().item())

# ------------------------------------------------------------------
# 6.  driver
# ------------------------------------------------------------------
def run(
    d=20, k=3, N=8096,
    sigma_a=1.0, sigma_v=4.0,
    kappa=5e-9, lam=5e-2,
    gamma=0.0,
    lr=1e-5, steps=4000,
    P_grid=np.geomspace(32, 4e4, 18),
    device="cuda",
    save_path=None,
):
    hp = dict(d=d,k=k,N=N,sigma_a=sigma_a,sigma_v=sigma_v,
              kappa=kappa,gamma=gamma)
    Pc, α, B0, c0 = constants(hp, device)
    theory = np.array([(1 - m_star(P,Pc,α,B0,c0))/2 for P in P_grid]) # <-- FIX A‑4
    emp    = np.array([empirical_err(int(P),hp,lr,steps,lam,device)
                       for P in P_grid])

    fig,ax = plt.subplots(figsize=(7,4.5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(P_grid, theory, lw=2, label="theory")
    ax.scatter(P_grid, emp, color="crimson", zorder=3, label="empirical")
    ax.axvline(Pc, ls="--", color="k", lw=1, label="$P_c$")
    ax.set_xlabel("training samples $P$"); ax.set_ylabel("0‑1 test error")
    ax.set_title(f"Theory vs. SGLD  (ReLU, {k}-parity)")
    ax.grid(alpha=.3, which="both"); ax.legend()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print("saved →", save_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    run(save_path=args.save)
