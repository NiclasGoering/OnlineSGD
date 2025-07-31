#!/usr/bin/env python
# Parity-learning sweep + κ-tracking (compatible with torch ≥2.1)

# ── USER GRID ─────────────────────────────────────────────────────
Ps         = [1000, 5000, 10000, 25000, 50000, 100000,500000,1000000]
lrs        = [1e-3,1e-4]
widths     = [1024]
gammas     = [0.0, 1.0]
sigma_as   = [0.5,1.0,2.0]
sigma_ws   = [0.5,1.0,2.0]

# ── STATIC PARAMS ─────────────────────────────────────────────────
D           = 60
k           = 4
batch_size  = 1024             # GPU-friendly; lower if OOM
test_sz     = 10_000
ensemble    = 4
max_epochs  = 500_000
loss_tol    = 1e-3
EVAL_EVERY  = 20              # epochs between κ measurements
out_dir     = '/home/goring/OnlineSGD/empirical_tests/results_604'

# ── IMPORTS ───────────────────────────────────────────────────────
import os, json, math, time, itertools
from pathlib import Path
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.func import grad, vmap, functional_call

# ── HELPERS ───────────────────────────────────────────────────────
def hms(t):
    h, r = divmod(int(t), 3600)
    m, s = divmod(r, 60)
    return f'{h:02}:{m:02}:{s:02}'

class ParityDataset(Dataset):
    def __init__(self, P: int, d: int, k_: int):
        x = torch.randint(0, 2, (P, d), dtype=torch.float32)
        x[x == 0] = -1.0
        self.x, self.y = x, torch.prod(x[:, :k_], 1, keepdim=True)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class TwoLayer(nn.Module):
    def __init__(self, d, width, gamma, sigma_w, sigma_a):
        super().__init__()
        self.fc1 = nn.Linear(d, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.fc1.weight, 0., sigma_w / math.sqrt(d))
        nn.init.normal_(self.fc2.weight, 0.,
                        sigma_a / (width ** ((1 + gamma) / 2)))
    def forward(self, x):  # ReLU network
        return self.fc2(torch.relu(self.fc1(x)))

@torch.no_grad()
def generalisation_err(net: nn.Module, device: torch.device):
    x = torch.randint(0, 2, (test_sz, D), dtype=torch.float32)
    x[x == 0] = -1.
    y = torch.prod(x[:, :k], 1, keepdim=True)
    x, y = x.to(device), y.to(device)
    return (torch.sign(net(x)) != y).float().mean().item()

# ── σ_g² AND κ ────────────────────────────────────────────────────
def sigma_g2(model: nn.Module, loader: DataLoader,
             device: torch.device, batches: int = 4):
    """Estimate σ_g² = E‖∇ℓ‖² using up to `batches` mini-batches."""
    params  = dict(model.named_parameters())
    mse     = nn.MSELoss()

    def loss(p, xb, yb):
        return mse(functional_call(model, p, (xb,)), yb)

    g_fn  = grad(loss)
    norms = []
    for b, (x, y) in enumerate(loader):
        if b >= batches: break
        x, y = x.to(device), y.to(device)
        per  = vmap(g_fn, in_dims=(None, 0, 0))(params, x, y)
        flat = torch.cat([g.flatten(1) for g in per.values()], 1)
        norms.append((flat ** 2).sum(1))
    return torch.cat(norms).mean().item() if norms else 0.

# ── SINGLE TRAINING RUN ───────────────────────────────────────────
def train_once(cfg, device):
    P, lr, width, gamma, sw, sa = cfg
    ds = ParityDataset(P, D, k)
    B  = min(batch_size, P) if P else 1
    dl = DataLoader(ds, batch_size=B, shuffle=True, num_workers=0)
    net = TwoLayer(D, width, gamma, sw, sa).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    mse = nn.MSELoss()

    kappa_curve = []
    epoch, running = 0, float('inf')

    while epoch < max_epochs and running > loss_tol and P > 0:
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mse(net(xb), yb).backward()
            opt.step()
        epoch += 1

        if epoch % EVAL_EVERY == 0:
            σ2 = sigma_g2(net, dl, device)
            κ  = lr * σ2 / B
            kappa_curve.append((epoch, κ))
            running = mse(net(xb), yb).item()

    err      = generalisation_err(net, device)
    κ_final  = kappa_curve[-1][1] if kappa_curve else 0.
    return err, κ_final, kappa_curve

# ── THEORY (unchanged) ────────────────────────────────────────────
def lambda_S_theory(width, gamma, σw, σa):
    n_star, C = k, 1.0
    return (width ** (1 - gamma)) * (σw ** n_star * σa) ** 2 * C * (D ** -n_star)

def err_theory(lam, κ): return 0.5 if lam + κ == 0 else 0.5 * (1 - lam / (lam + κ))

# ── MAIN (parallel friendly) ──────────────────────────────────────
def main():
    rank       = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    device     = torch.device(
        f'cuda:{rank % torch.cuda.device_count()}'
        if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)

    if rank == 0: Path(out_dir).mkdir(parents=True, exist_ok=True)

    sweep   = list(itertools.product(Ps, lrs, widths, gammas, sigma_ws, sigma_as))
    per     = (len(sweep) + world_size - 1) // world_size
    tasks   = sweep[rank * per: (rank + 1) * per]

    print(f"[rank {rank}] {len(tasks)} configs on {device}")

    for idx, cfg in enumerate(tasks, 1):
        tag  = '_'.join(str(x) for x in cfg)
        jfp  = Path(out_dir) / f"{tag}.json"
        if jfp.exists(): continue            # skip completed

        t0 = time.time()
        errs, κs = [], []

        for e in range(ensemble):
            err, κ_fin, κ_curve = train_once(cfg, device)
            errs.append(err); κs.append(κ_fin)

            # plot κ(t)
            if κ_curve:
                epochs, vals = zip(*κ_curve)
                plt.figure()
                plt.plot(epochs, vals, '-o')
                plt.title(f"κ-track  {tag}  (ensemble {e})")
                plt.xlabel("epoch"); plt.ylabel("κ"); plt.grid(True)
                plt.tight_layout()
                plt.savefig(Path(out_dir) / f"{tag}_kappa_e{e}.png")
                plt.close()

        err_mean, err_std = float(torch.tensor(errs).mean()), float(torch.tensor(errs).std())
        κ_mean            = float(torch.tensor(κs).mean())
        lam               = lambda_S_theory(*cfg[2:])  # width, γ, σw, σa
        err_th            = err_theory(lam, κ_mean)

        with open(jfp, 'w') as f:
            json.dump({'cfg': list(cfg),
                       'err_mean': err_mean,
                       'err_std': err_std,
                       'kappa_mean': κ_mean,
                       'lambda_theory': lam,
                       'err_theory': err_th},
                      f, indent=2)

        dt = hms(time.time() - t0)
        print(f"[rank {rank}] {idx}/{len(tasks)}  {tag}  "
              f"err={err_mean:.3f}±{err_std:.3f}  κ={κ_mean:.3e}  ({dt})")

if __name__ == '__main__':
    main()
