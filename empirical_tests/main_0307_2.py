#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple, Robust, Parallel Parity-Learning Sweep (v7 – “works-out-of-the-box”)
"""

from __future__ import annotations 

# ────────────────────────  USER CONFIG  ────────────────────────
Ps         = [100, 1000, 5000, 10000, 25000, 50000]
lrs        = [1e-3]
widths     = [1024]
gammas     = [0.0, 1.0]
sigma_as   = [1.0]
sigma_ws   = [1.0]

# static
D, k           = 37, 4
batch_size     = 512          # memory friendly on H100
test_sz        = 10_000
ensemble       = 3
max_epochs     = 100_000
loss_tol       = 1e-3
out_dir        = '/home/goring/OnlineSGD/empirical_tests/results_374'
# ───────────────────────────────────────────────────────────────


import os, time, json, math, itertools
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.func import vmap, grad, functional_call

torch.backends.cudnn.benchmark = True   # fastest algorithms

# -------- small utils ---------------------------------------------------------
def hms(sec: float) -> str:
    h, rem  = divmod(sec, 3600)
    m, s    = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{s:05.2f}"

def cfg_tag(cfg: Tuple) -> str:
    """File-system safe unique id."""
    return "_".join(str(v).replace('.', 'p') for v in cfg)
# ------------------------------------------------------------------------------

# ----------------------------  data + model  ----------------------------------
class ParityDataset(Dataset):
    def __init__(self, P: int, d: int, k_: int):
        x = torch.randint(0, 2, (P, d), dtype=torch.float32)
        x[x == 0] = -1.0
        self.x = x
        self.y = torch.prod(x[:, :k_], dim=1, keepdim=True)

    def __len__(self):  return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]

class TwoLayer(nn.Module):
    def __init__(self, d: int, width: int,
                 gamma: float, sigma_w: float, sigma_a: float):
        super().__init__()
        self.fc1 = nn.Linear(d, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.fc1.weight, 0.0, sigma_w / math.sqrt(d))
        nn.init.normal_(self.fc2.weight, 0.0,
                        sigma_a / (width ** ((1 + gamma) / 2)))
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x)))
# ------------------------------------------------------------------------------

# -------------------- helpers: generalisation + σg² ---------------------------
@torch.no_grad()
def generalisation_err(net: nn.Module, device: torch.device) -> float:
    net.eval()
    x = torch.randint(0, 2, (test_sz, D), dtype=torch.float32, device=device)
    x[x == 0] = -1.0
    y  = torch.prod(x[:, :k], dim=1, keepdim=True)
    pred = torch.sign(net(x))
    return (pred != y).float().mean().item()

def sigma_g2(model: nn.Module, loader: DataLoader,
             device: torch.device) -> float:
    """
    Per-sample gradient ‖g‖² averaged over *one* mini-batch.
    Only the very first batch is used; enough for κ estimate.
    """
    try:
        x, y = next(iter(loader))
    except StopIteration:
        return 0.0
    x, y = x.to(device), y.to(device)

    params: Dict[str, torch.Tensor] = {k: v.clone().detach().requires_grad_(True)
                                       for k, v in model.named_parameters()}

    def loss_fn(prm: Dict[str, torch.Tensor],
                single_x: torch.Tensor, single_y: torch.Tensor):
        out = functional_call(model, prm, (single_x.unsqueeze(0),), {})
        return ((out - single_y.unsqueeze(0)) ** 2).mean()

    grad_fn  = grad(loss_fn)
    g_per_ex = vmap(grad_fn, in_dims=(None, 0, 0))(params, x, y)

    # flatten every parameter, square, sum over params and features
    flat_sq  = [p.view(p.shape[0], -1).pow(2) for p in g_per_ex.values()]
    norm_sq  = torch.cat(flat_sq, dim=1).sum(dim=1)
    return norm_sq.mean().item()
# ------------------------------------------------------------------------------

# -------------------------------  training  ----------------------------------
def train_once(cfg: Tuple, device: torch.device) -> Tuple[float, float]:
    P, lr, width, gamma, sw, sa = cfg

    ds = ParityDataset(P, D, k)
    effective_bs = max(1, min(batch_size, P))
    dl = DataLoader(ds, batch_size=effective_bs, shuffle=True,
                    drop_last=False, num_workers=0, pin_memory=False)

    net = TwoLayer(D, width, gamma, sw, sa).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    mse = nn.MSELoss()

    # κ estimate from *initial* gradient noise
    sigma2 = sigma_g2(net, dl, device)
    kappa  = lr * sigma2 / effective_bs

    running, epoch = float('inf'), 0
    while epoch < max_epochs and running > loss_tol and P > 0:
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = mse(net(x), y)
            loss.backward()
            opt.step()
        running = loss.item()
        epoch  += 1

    err = generalisation_err(net, device)
    return err, kappa
# ------------------------------------------------------------------------------

# ---------------- theoretical λS and error prediction -------------------------
def lambda_S_theory(width: int, gamma: float,
                    sigma_w: float, sigma_a: float) -> float:
    n_star = k            # first non-vanishing Taylor coefficient order
    c_n_sq = 1.0          # ReLU ⇒ c_k = 1 up to a constant factor
    return (width ** (1 - gamma)) * \
           ((sigma_w ** n_star) * sigma_a) ** 2 * \
           c_n_sq * D ** (-n_star)

def err_theory(lam: float, kap: float) -> float:
    return 0.5 if lam + kap == 0 else 0.5 * (1 - lam / (lam + kap))
# ------------------------------------------------------------------------------

def main() -> None:
    rank       = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))

    # ----- device placement -----
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device_idx = rank % n_gpus
        torch.cuda.set_device(device_idx)
        device = torch.device(f'cuda:{device_idx}')
    else:
        device = torch.device('cpu')

    # mkdir exactly once
    if rank == 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ----- distribute grid search -----
    job_grid     = list(itertools.product(Ps, lrs, widths,
                                          gammas, sigma_ws, sigma_as))
    total_jobs   = len(job_grid)
    jobs_per_rnk = (total_jobs + world_size - 1) // world_size
    my_jobs      = job_grid[rank * jobs_per_rnk : (rank + 1) * jobs_per_rnk]

    print(f"[Rank {rank}] running on {device}. "
          f"Assigned {len(my_jobs)} / {total_jobs} configs.")

    for j, cfg in enumerate(my_jobs, 1):
        tag      = cfg_tag(cfg)
        out_path = Path(out_dir) / f"{tag}.json"
        if out_path.exists():
            print(f"[Rank {rank}] skip (done) {tag}")
            continue

        t0   = time.time()
        errs, kaps = [], []

        for run in range(ensemble):
            try:
                e, k = train_once(cfg, device)
            except Exception as exc:
                print(f"[Rank {rank}] !!! run failed: {cfg} – {exc}")
                e, k = 0.5, 0.0
            errs.append(e); kaps.append(k)

        errs_t  = torch.tensor(errs)
        kap_m   = float(torch.tensor(kaps).mean())
        err_mu  = float(errs_t.mean())
        err_sd  = float(errs_t.std(unbiased=False))

        # theory
        P_, lr_, width, gamma, sw, sa = cfg
        lam   = lambda_S_theory(width, gamma, sw, sa)
        err_th= err_theory(lam, kap_m)

        print(f"[Rank {rank}] {tag}  "
              f"emp={err_mu:.4f}±{err_sd:.4f}  "
              f"theory={err_th:.4f}  "
              f"(λ={lam:.3e}, κ̄={kap_m:.3e})  "
              f"time={hms(time.time()-t0)}")

        with out_path.open('w') as fh:
            json.dump({
                "cfg"          : list(cfg),
                "err_mean"     : err_mu,
                "err_std"      : err_sd,
                "kappa_mean"   : kap_m,
                "lambda_theory": lam,
                "err_theory"   : err_th
            }, fh, indent=2)

    print(f"### Rank {rank} finished ({len(my_jobs)} jobs).")

if __name__ == "__main__":
    main()
