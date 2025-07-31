#!/usr/bin/env python
"""
Simple, Robust, Parallel Parity-Learning Sweep
-----------------------------------------------
* V6 UPDATE:
* 1. SIGNIFICANTLY INCREASED BATCH SIZE: To better utilize powerful GPUs (H100s)
* and reduce the relative overhead of Python, speeding up the runs.
* 2. ADDED VARIANCE REPORTING: Now calculates and saves the standard
* deviation of the generalization error across the ensemble.
* 3. ADDED THEORETICAL PREDICTION: Calculates and saves the theoretical
* error for each configuration, making analysis much easier.
"""

# --- USER CONFIG ---
Ps         = [100, 1000, 5000, 10000, 25000, 50000, 100000]
lrs        = [1e-3]
widths     = [1024]
gammas     = [0.0, 1.0]
sigma_as   = [1.0]
sigma_ws   = [1.0]

# --- Static Parameters ---
D = 35
k = 5
# INCREASED BATCH SIZE FOR GPU EFFICIENCY. Adjust if you run into memory issues.
batch_size = 512
test_sz    = 10000
ensemble   = 3 # Increased ensemble size for more stable variance estimates
max_epochs = 100000
loss_tol   = 1e-3
out_dir    = '/home/goring/OnlineSGD/empirical_tests/results_355'

# --- IMPLEMENTATION BELOW ---
import os
import time
import math
import json
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.func import vmap, grad, functional_call

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h:02}:{m:02}:{s:05.2f}"

class ParityDataset(Dataset):
    def __init__(self, P, d, k):
        x = torch.randint(0, 2, (P, d), dtype=torch.float32); x[x == 0] = -1.0
        self.x, self.y = x, torch.prod(x[:, :k], dim=1, keepdim=True)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class TwoLayer(nn.Module):
    def __init__(self, d, width, gamma, sigma_w, sigma_a):
        super().__init__()
        self.fc1 = nn.Linear(d, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.fc1.weight, 0.0, sigma_w / math.sqrt(d))
        nn.init.normal_(self.fc2.weight, 0.0, sigma_a / (width**((1 + gamma) / 2.0)))
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x)))

def generalisation_err(net, device):
    net.eval()
    with torch.no_grad():
        x = torch.randint(0, 2, (test_sz, D), dtype=torch.float32); x[x == 0] = -1.0
        x, y = x.to(device), torch.prod(x[:, :k], dim=1, keepdim=True).to(device)
        pred = torch.sign(net(x))
        err = (pred != y).float().mean().item()
    net.train()
    return err

def sigma_g2(model, loader, device):
    params = dict(model.named_parameters())
    try: x, y = next(iter(loader))
    except StopIteration: return 0.0
    x, y = x.to(device), y.to(device)
    def compute_loss(current_params, single_x, single_y):
        out = functional_call(model, current_params, (single_x.unsqueeze(0),))
        return ((out - single_y.unsqueeze(0)) ** 2).mean()
    grad_fn = grad(compute_loss)
    per_sample_grads = vmap(grad_fn, in_dims=(None, 0, 0))(params, x, y)
    all_grads_sq = [p.view(p.shape[0], -1)**2 for p in per_sample_grads.values()]
    per_sample_grad_norms_sq = torch.cat(all_grads_sq, dim=1).sum(dim=1)
    return per_sample_grad_norms_sq.mean().item()

def train_once(cfg, device):
    P, lr, width, gamma, sw, sa = cfg
    ds = ParityDataset(P, D, k)
    effective_batch_size = min(batch_size, P) if P > 0 else 1
    dl = DataLoader(ds, batch_size=effective_batch_size, shuffle=True, num_workers=0)
    net = TwoLayer(D, width, gamma, sw, sa).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    sigma2 = sigma_g2(net, dl, device)
    kappa = lr * sigma2 / effective_batch_size if effective_batch_size > 0 else 0
    epoch, running = 0, 1e9
    net.train()
    while epoch < max_epochs and running > loss_tol:
        if P == 0: break
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = mse(net(x), y); loss.backward(); opt.step()
        running = loss.item()
        epoch += 1
    final_err = generalisation_err(net, device)
    return final_err, kappa

# --- THEORETICAL PREDICTION FUNCTIONS ---
def lambda_S_theory(width, gamma, sigma_w, sigma_a):
    n_star = k
    C_K_CONST = 1.0
    c_n_sq = C_K_CONST**2
    return (width**(1 - gamma)) * (sigma_w**n_star * sigma_a)**2 * c_n_sq * (D**(-n_star))

def calculate_theoretical_error(lam, kappa):
    if lam + kappa == 0: return 0.5
    return 0.5 * (1 - lam / (lam + kappa))

def main():
    rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device(f'cuda:{rank % num_gpus}')
        torch.cuda.set_device(device)
        print(f"INFO: Rank {rank} is assigned to device: {device}")
    else:
        device = torch.device('cpu')
        print(f"INFO: Rank {rank} is running on CPU")

    if rank == 0:
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        print(f"INFO: Results will be saved to: {out_dir}")

    all_params = [Ps, lrs, widths, gammas, sigma_ws, sigma_as]
    full_job_list = list(itertools.product(*all_params))
    
    num_jobs = len(full_job_list)
    jobs_per_rank = (num_jobs + world_size - 1) // world_size
    start_idx = rank * jobs_per_rank
    end_idx = min(start_idx + jobs_per_rank, num_jobs)
    
    my_tasks = full_job_list[start_idx:end_idx]

    print(f"INFO: Rank {rank} has {len(my_tasks)} tasks to process.")

    for i, cfg in enumerate(my_tasks):
        print(f"==> Rank {rank} starting task {i+1}/{len(my_tasks)}: {cfg}")
        tag = '_'.join(f'{v}' for v in cfg)
        filepath = Path(out_dir) / f'{tag}.json'
        
        if filepath.exists():
            print(f"INFO: Rank {rank} skipping already completed task: {cfg}")
            continue

        start_time = time.time()
        errs, kaps = [], []
        for ens_i in range(ensemble):
            try:
                e, k = train_once(cfg, device)
                errs.append(e)
                kaps.append(k)
            except Exception as e:
                print(f"!!!!!! ERROR: Rank {rank} FAILED on config {cfg} with error: {e} !!!!!!")
                errs.append(0.5); kaps.append(0.0)
        
        # --- NEW: Calculate mean and standard deviation ---
        errs_tensor = torch.tensor(errs)
        err_mean = errs_tensor.mean().item()
        err_std = errs_tensor.std().item() if len(errs) > 1 else 0.0
        kap_mean = torch.tensor(kaps).mean().item()
        
        # --- NEW: Calculate theoretical prediction ---
        p, lr, width, gamma, sw, sa = cfg
        lam_theory = lambda_S_theory(width, gamma, sw, sa)
        err_theory = calculate_theoretical_error(lam_theory, kap_mean)

        elapsed = hms_string(time.time() - start_time)
        
        print(f"<-- Rank {rank} FINISHED Cfg {cfg} in {elapsed}")
        print(f"    --> Empirical Error: {err_mean:.4f} ± {err_std:.4f}")
        print(f"    --> Theoretical Error: {err_theory:.4f} (using λ={lam_theory:.3e}, κ={kap_mean:.3e})")

        # --- NEW: Save all data to JSON ---
        with open(filepath, 'w') as f:
            json.dump({
                'cfg': list(cfg),
                'err_mean': err_mean,
                'err_std': err_std,
                'kappa_mean': kap_mean,
                'lambda_theory': lam_theory,
                'err_theory': err_theory
            }, f, indent=2)

    print(f"########## Rank {rank} has completed all its assigned tasks. ##########")

if __name__ == '__main__':
    main()