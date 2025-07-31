# parity_phase_transition.py (v1.4 – summary plots)
"""
Sweep over several dataset sizes for k‑sparse parity **and produce two kinds of
graphics**:

1. *Per‑size summary plot*  – one PNG per training‑set size that overlays
   • the histogram of the teacher order parameter `m_S` (red) and
   • the combined histogram of all the other `m_A` (grey).
   This makes it obvious when `m_S` separates from the noise bulk.
2. *Global trend*  – a single PNG `mean_mS_vs_n.png` that plots ⟨m_S⟩ ± σ across
   dataset size, so you can see the phase‑transition onset.

Live progress prints (added in v1.3) are intact.

Tested on PyTorch 1.9 and 2.1; GPU optional.
"""

from __future__ import annotations

import argparse, math, pathlib, time
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
#  Walsh‑parity helpers
# ----------------------------------------------------------------------------

def chi_S(x: torch.Tensor, S: Tuple[int, ...]) -> torch.Tensor:
    """Walsh parity χ_S(x)=∏_{i∈S} x_i."""
    return torch.prod(x[..., list(S)], dim=-1)


def make_parity_dataset(d: int, k: int, n: int, *, device, rng):
    X = torch.randint(0, 2, (n, d), device=device, dtype=torch.float32, generator=rng)
    X.mul_(2).sub_(1)  # map {0,1} → {−1,+1}
    S = torch.randperm(d, device=device, generator=rng)[:k]
    y = chi_S(X, tuple(S.tolist()))
    return X, y, tuple(S.tolist())

# ----------------------------------------------------------------------------
#  Two‑layer network
# ----------------------------------------------------------------------------
class TwoLayerNet(nn.Module):
    def __init__(self, d, N, *, phi, sigma_v2, sigma_w2, gamma, device):
        super().__init__()
        self.d, self.N, self.phi = d, N, phi
        self.W = nn.Parameter(torch.randn(N, d, device=device) * math.sqrt(sigma_v2 / d))
        self.a = nn.Parameter(torch.randn(N, device=device) * math.sqrt(sigma_w2 / (N ** gamma)))

    def forward(self, x):
        return self.phi(x @ self.W.T) @ self.a

    def activs(self, x):
        return self.phi(x @ self.W.T)

# ----------------------------------------------------------------------------
#  Langevin SGD step (noise creation compatible with all PT 1.x versions)
# ----------------------------------------------------------------------------

def langevin_step(model: TwoLayerNet, X, y, *, kappa, lr, sigma_w2, gamma, eta):
    model.zero_grad(set_to_none=True)
    f = model(X)
    loss = 0.5 / kappa * (f - y).pow(2).mean()
    loss += (model.N ** gamma) / (2 * sigma_w2) * model.a.pow(2).sum() / X.size(0)
    loss += model.d / (2 * sigma_w2) * model.W.pow(2).sum() / X.size(0)
    loss.backward()

    coeff = math.sqrt(2 * lr * eta)
    for p in model.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * coeff - lr * p.grad)
    return loss.item()

# ----------------------------------------------------------------------------
#  Compute J_A and m_A
# ----------------------------------------------------------------------------

def walsh_stats(model: TwoLayerNet, walsh_X, subsets):
    chi = torch.stack([chi_S(walsh_X, A) for A in subsets])            # (S, n)
    J = (chi @ model.activs(walsh_X)) / walsh_X.size(0)               # (S, N)
    m = (J @ model.a) / model.N                                       # (S,)
    return m.cpu()

# ----------------------------------------------------------------------------
#  One dataset size, many runs → returns teacher mean & std
# ----------------------------------------------------------------------------

def sweep_size(n_train, *, cfg, device, outdir):
    print(f"\n=== DATASET n={n_train} ===")
    m_samples: Dict[Tuple[int, ...], List[float]] = defaultdict(list)
    rng_global = torch.Generator(device).manual_seed(0)
    walsh_X = torch.randint(0, 2, (cfg.walsh_n, cfg.d), device=device, dtype=torch.float32, generator=rng_global)
    walsh_X.mul_(2).sub_(1)

    for run in range(cfg.runs_per_size):
        run_t0 = time.time()
        rng = torch.Generator(device).manual_seed(10_000 + 31 * run)
        X, y, S_teacher = make_parity_dataset(cfg.d, cfg.k, n_train, device=device, rng=rng)
        subsets = [S_teacher] + [(i,) for i in range(cfg.d)]

        model = TwoLayerNet(cfg.d, cfg.N, phi=torch.relu, sigma_v2=cfg.sigma_v2,
                             sigma_w2=cfg.sigma_w2, gamma=cfg.gamma, device=device)
        kappa = cfg.kappa0 * (cfg.N ** (1 - cfg.gamma))
        milestones = {int(cfg.epochs * f) for f in (0.25, 0.5, 0.75, 1.0)}
        print(f"[run {run+1}/{cfg.runs_per_size}]", flush=True)
        for ep in range(1, cfg.epochs + 1):
            loss = langevin_step(model, X, y, kappa=kappa, lr=cfg.lr, sigma_w2=cfg.sigma_w2, gamma=cfg.gamma, eta=cfg.eta)
            if ep in milestones:
                pct = int(100 * ep / cfg.epochs)
                print(f"  {pct:3d}%  loss={loss:.3e}  t={time.time()-run_t0:.1f}s", flush=True)

        m = walsh_stats(model, walsh_X, subsets)
        for idx, A in enumerate(subsets):
            m_samples[A].append(float(m[idx]))

    # ---------------- summary histogram ----------------
    mS_vals = m_samples[S_teacher]
    other_vals = [v for A, lst in m_samples.items() if A != S_teacher for v in lst]

    plt.figure(figsize=(5, 3))
    plt.hist(other_vals, bins=30, color="lightgrey", alpha=0.6, label="other m_A")
    plt.hist(mS_vals, bins=30, color="tab:red", alpha=0.8, label="m_S (teacher)")
    plt.legend(frameon=False)
    plt.title(f"n={n_train},  d={cfg.d}, N={cfg.N}")
    plt.xlabel("m value")
    plt.tight_layout()
    plt.savefig(outdir / f"summary_m_n{n_train}.png", dpi=170)
    plt.close()

    mu = torch.tensor(mS_vals).mean().item()
    sd = torch.tensor(mS_vals).std().item()
    print(f"m_S: mean={mu:+.3e}  std={sd:.3e}")
    return mu, sd

# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def parse_cli():
    p = argparse.ArgumentParser("Parity phase transition sweep (summary plots)")
    p.add_argument("--outdir", type=pathlib.Path, required=True)
    p.add_argument("--dataset_sizes", nargs="*", type=int, default=[256, 1024, 4096, 8192])
    p.add_argument("--runs_per_size", type=int, default=5)
    p.add_argument("--d", type=int, default=30)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--sigma_v2", type=float, default=1.0)
    p.add_argument("--sigma_w2", type=float, default=1.0)
    p.add_argument("--kappa0", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--walsh_n", type=int, default=8192)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()

# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = parse_cli()
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    print("Using device:", device)

    means, stds = [], []
    start_all = time.time()
    for n in cfg.dataset_sizes:
        mu, sd = sweep_size(n, cfg=cfg, device=device, outdir=cfg.outdir)
        means.append(mu); stds.append(sd)

    # ---- global trend plot ----
    plt.figure(figsize=(4.8, 3.2))
    plt.errorbar(cfg.dataset_sizes, means, yerr=stds, marker="o", lw=1.4, capsize=3)
    plt.xscale("log")
    plt.xlabel("dataset size n")
    plt.ylabel(r"$\langle m_S \rangle \pm \sigma$")
    plt.title("Teacher coupling vs. data size")
    plt.tight_layout()
    plt.savefig(cfg.outdir / "mean_mS_vs_n.png", dpi=180)
    plt.close()

    print(f"\nAll done in {time.time() - start_all:.1f}s  →  results in {cfg.outdir.resolve()}")
