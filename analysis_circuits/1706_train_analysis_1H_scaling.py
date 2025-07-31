import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import time
import itertools
import math
from math import factorial
import json
import pandas as pd
from multiprocessing import Pool, cpu_count
from matplotlib.colors import SymLogNorm

# --- Model Definition ---
class OneLayerTanhNet(torch.nn.Module):
    """A one-hidden-layer neural network with a ReLU activation and a linear output layer."""
    def __init__(self, input_dim, hidden_width, W1_grad=True, a_grad=True):
        super(OneLayerTanhNet, self).__init__()
        # Initialize layers with scaling
        self.W1 = torch.nn.Parameter(torch.randn(hidden_width, input_dim) / np.sqrt(input_dim))
        self.a = torch.nn.Parameter(torch.randn(hidden_width) / np.sqrt(hidden_width))
        
        # Control which weights are trainable
        self.W1.requires_grad = W1_grad
        self.a.requires_grad = a_grad

    def activation(self, x):
        return torch.relu(x)

    def forward(self, x):
        h1 = self.activation(torch.matmul(x, self.W1.t()))
        output = torch.matmul(h1, self.a)
        return output

    def get_activations(self, x):
        return self.activation(torch.matmul(x, self.W1.t()))


# --- Main Experiment Class (Streamlined) ---
class EnsembleParityExperiment:
    """
    Manages training and analysis for an ensemble of one-layer networks.
    Focus is on training, Hermite analysis, and stiffness (g_l) calculation.
    """
    def __init__(self, d=30, k=6, M1=512, learning_rate=0.01, max_epochs=50000,
                 batch_size=512, n_ensemble=10, tracker=100,
                 weight_decay_W1=0.0, weight_decay_a=0.0,
                 device_id=None, save_dir="parity_run", L_max=30):
        
        # --- Hyperparameters ---
        self.d = d; self.k = k; self.M1 = M1; self.learning_rate = learning_rate
        self.max_epochs = max_epochs; self.batch_size = batch_size; self.n_ensemble = n_ensemble
        self.tracker = tracker
        self.weight_decay_W1 = weight_decay_W1; self.weight_decay_a = weight_decay_a
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # --- Device Configuration ---
        if isinstance(device_id, int) and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")
        print(f"Experiment(d={d},k={k},N={M1}) initialized on {self.device}")
        
        # --- Ensemble Initialization ---
        self.models = [OneLayerTanhNet(d, M1).to(self.device) for _ in range(n_ensemble)]
        self.optimizers = [self._create_optimizer(model) for model in self.models]
        self.criterion = torch.nn.MSELoss()

        # --- Data and State Storage (Streamlined) ---
        self.is_converged = [False] * self.n_ensemble
        self.metrics = [{'epochs': [], 'correlation': [], 'phase_transition_epoch': None} for _ in range(n_ensemble)]
        self.macro_observables = {'epochs': [], 'qu_mean': [], 'qu_std': [], 'qv_mean': [], 'qv_std': []}
        
        self.L_max = L_max
        self.hermite_evolution_stats = {
            'epochs': [],
            'coeffs_mean': [],
            'coeffs_std': []
        }
        
        # --- Fixed Test Dataset ---
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(2000, d)), dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)

    # --- Static Methods for Hermite Analysis ---
    @staticmethod
    def _relu_d(l):
        if l < 0: return 0
        return 1 / (np.sqrt(np.pi) * 2**l) * factorial(l)

    @staticmethod
    def _hermite(n, z):
        if n == 0: return torch.ones_like(z)
        if n == 1: return z
        Hnm2, Hnm1 = torch.ones_like(z), z
        for k_iter in range(2, n + 1):
            Hn = z * Hnm1 - (k_iter - 1) * Hnm2
            Hnm2, Hnm1 = Hnm1, Hn
        return Hn

    @staticmethod
    def _orthonormal_hermite(n, z):
        H = EnsembleParityExperiment._hermite(n, z)
        norm = np.sqrt(float(math.factorial(n)))
        return H / norm

    # --- Core Calculation Methods ---
    def _hermite_coeff_centered(self, a, W1, x_batch, ell):
        B = x_batch.shape[0]
        z = (W1 @ x_batch.T)
        H_tilde = self._orthonormal_hermite(ell, z)
        N = W1.shape[0]
        f = (a[:, None] * torch.relu(z)).sum(0) / math.sqrt(N)
        f -= f.mean()
        H_bar = H_tilde.mean(0)
        coeff = torch.dot(f, H_bar) / B
        d_ell = self._relu_d(ell)
        return (coeff / d_ell).item() if d_ell != 0 else 0.0

    def _create_optimizer(self, model):
        param_groups = [{'params': model.W1, 'weight_decay': self.weight_decay_W1}, 
                        {'params': model.a, 'weight_decay': self.weight_decay_a}]
        return torch.optim.Adam(param_groups, lr=self.learning_rate)

    def _target_function(self, x):
        return torch.prod(x[:, :self.k], dim=1)

    def _compute_macro_observables(self, epoch):
        qu_vals, qv_vals = [], []
        N = float(self.M1)
        for model in self.models:
            model.eval()
            with torch.no_grad():
                W1 = model.W1
                if self.k > 0:
                    u = W1[:, :self.k]
                    qu_vals.append(torch.mean(torch.sum(u**2, dim=1)).cpu().item())
                else:
                    qu_vals.append(0.0)
                if self.d > self.k:
                    v = W1[:, self.k:]
                    qv_vals.append(torch.mean(torch.sum(v**2, dim=1)).cpu().item())
                else:
                    qv_vals.append(0.0)
        self.macro_observables['epochs'].append(epoch)
        self.macro_observables['qu_mean'].append(np.mean(qu_vals))
        self.macro_observables['qu_std'].append(np.std(qu_vals))
        self.macro_observables['qv_mean'].append(np.mean(qv_vals))
        self.macro_observables['qv_std'].append(np.std(qv_vals))

    def _compute_hermite_evolution(self, epoch):
        all_coeffs = torch.zeros(self.n_ensemble, self.L_max + 1, device=self.device)
        for i, model in enumerate(self.models):
            model.eval()
            a, W1 = model.a.detach(), model.W1.detach()
            for l in range(self.L_max + 1):
                all_coeffs[i, l] = self._hermite_coeff_centered(a, W1, self.X_test, l)
        self.hermite_evolution_stats['epochs'].append(epoch)
        self.hermite_evolution_stats['coeffs_mean'].append(torch.mean(all_coeffs, dim=0).cpu().numpy())
        self.hermite_evolution_stats['coeffs_std'].append(torch.std(all_coeffs, dim=0).cpu().numpy())

    def _compute_metrics_step(self, epoch):
        # This is a focused metrics calculation step for training progress
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                preds = model(self.X_test).squeeze()
                corr = torch.corrcoef(torch.stack([preds, self.y_test.squeeze()]))[0, 1].item() if preds.numel() > 1 and torch.var(preds) > 1e-6 else 0.0
            self.metrics[i]['epochs'].append(epoch)
            self.metrics[i]['correlation'].append(corr)

        # Compute expensive metrics only at tracker intervals
        self._compute_macro_observables(epoch)
        self._compute_hermite_evolution(epoch)
        
    def train(self, early_stop_corr=0.995):
        print(f"[{self.device}] Starting training for d={self.d}, k={self.k}, N={self.M1}...")
        self._compute_metrics_step(0)
        
        epoch = 0
        with tqdm(total=self.max_epochs, desc=f"d={self.d},k={self.k},N={self.M1}", position=int(str(self.device)[-1])) as pbar:
            while not all(self.is_converged) and epoch < self.max_epochs:
                epoch += 1
                
                # Check for active models to train
                active_indices = [i for i, conv in enumerate(self.is_converged) if not conv]
                if not active_indices:
                    break

                X_batch = torch.randn(self.batch_size, self.d, device=self.device).sign()
                y_batch = self._target_function(X_batch)
                
                for model_idx in active_indices:
                    model, optimizer = self.models[model_idx], self.optimizers[model_idx]
                    model.train()
                    loss = self.criterion(model(X_batch), y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if epoch % self.tracker == 0 or epoch == self.max_epochs:
                    self._compute_metrics_step(epoch)
                    for i in range(self.n_ensemble):
                        if not self.is_converged[i] and self.metrics[i]['correlation'] and self.metrics[i]['correlation'][-1] > early_stop_corr:
                            self.is_converged[i] = True
                    
                    num_converged = sum(self.is_converged)
                    avg_corr = np.mean([m['correlation'][-1] for m in self.metrics if m['correlation']])
                    pbar.set_description(f"d={self.d},k={self.k},N={self.M1} | Conv: {num_converged}/{self.n_ensemble} | Corr: {avg_corr:.4f}")
                
                pbar.update(1)
        
        print(f"[{self.device}] Training complete for d={self.d}, k={self.k}, N={self.M1} at epoch {epoch}.")

    # --- Final Analysis Methods ---
    def _K_theory(self, ell: int, q_u: float, q_v: float) -> float:
        """Analytic kernel eigenvalue K_l(q_u, q_v)."""
        return math.factorial(ell) * q_u**(ell / 2.0)

    def calculate_and_save_g_theory(self, max_ell: int = 15, gamma: float = 1.0):
        if not self.macro_observables['epochs']:
            print("Cannot compute g_theory – train the ensemble first.")
            return

        qu = self.macro_observables['qu_mean'][-1]
        qv = self.macro_observables['qv_mean'][-1]
        N = float(self.M1)

        print(f"\n[{self.device}] Theoretical large-N stiffness g_l (d={self.d},k={self.k},N={self.M1})")
        g_dict = {}
        for ell in range(1, max_ell + 1):
            if qu > 1e-9: # Avoid math domain error if q_u is zero or negative
                K_ell = self._K_theory(ell, qu, qv)
                if K_ell > 1e-9: # Avoid division by zero
                    g = N**((1.0 - gamma) / 2.0) / math.sqrt(K_ell)
                    g_dict[ell] = g
                    print(f"  l = {ell:<2d} : g = {g: .3e}")
            else:
                g_dict[ell] = float('inf')


        # Save to JSON in the specified directory
        results_path = os.path.join(self.save_dir, "g_theory_values.json")
        with open(results_path, "w") as fp:
            json.dump(g_dict, fp, indent=2)
        print(f"[{self.device}] Saved g_l values to {results_path}")

    def plot_hermite_evolution_heatmap(self):
        print(f"[{self.device}] Plotting Hermite coefficient evolution heatmap...")
        stats = self.hermite_evolution_stats
        if not stats['epochs']:
            print("Hermite evolution data not available.")
            return

        heatmap_data = np.stack(stats['coeffs_mean'], axis=0)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        img = ax.imshow(
            heatmap_data,
            aspect='auto',
            cmap='bwr',
            norm=SymLogNorm(linthresh=1e-6, vmin=-np.max(np.abs(heatmap_data)), vmax=np.max(np.abs(heatmap_data))),
            origin='lower',
            extent=[-0.5, self.L_max + 0.5, stats['epochs'][0], stats['epochs'][-1]]
        )
        
        ax.axvline(x=self.k, color='red', linestyle='--', label=f'$k={self.k}$')
        ax.set_xlabel('Hermite Order $\ell$')
        ax.set_ylabel('Epoch')
        ax.set_yscale('log')
        ax.set_title(f'Hermite Coefficients Evolution (d={self.d}, k={self.k}, N={self.M1})')
        fig.colorbar(img, ax=ax, label='Coefficient Value')
        ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "hermite_evolution_heatmap.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[{self.device}] Saved Hermite heatmap to {plot_path}")


# --- Worker Function for Multiprocessing ---
def run_single_experiment(config):
    """
    Wrapper function to instantiate and run a single experiment.
    This function is mapped to a process pool.
    """
    try:
        experiment = EnsembleParityExperiment(**config)
        experiment.train(early_stop_corr=0.99)
        experiment.calculate_and_save_g_theory()
        experiment.plot_hermite_evolution_heatmap()
        # Save the config used for this run for later analysis
        with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
            # device is not serializable
            config.pop('device_id', None)
            json.dump(config, f, indent=4)

    except Exception as e:
        print(f"!!!!!!!!!!!!!!\nERROR in worker for d={config.get('d')}, k={config.get('k')}, N={config.get('M1')}: {e}\n!!!!!!!!!!!!!!")
        import traceback
        traceback.print_exc()

# --- Post-Processing and Plotting ---
def aggregate_and_plot_g_l(base_save_dir):
    """
    Collects all g_l results from the grid search and plots them.
    """
    print("\n" + "="*80)
    print("Aggregating all results and generating final plots...")
    
    results = []
    # Walk through the directories created by the grid search
    for root, dirs, files in os.walk(base_save_dir):
        if 'g_theory_values.json' in files and 'config.json' in files:
            with open(os.path.join(root, 'config.json'), 'r') as f:
                config = json.load(f)
            with open(os.path.join(root, 'g_theory_values.json'), 'r') as f:
                g_values = json.load(f)
            
            for l_str, g in g_values.items():
                results.append({
                    'd': config['d'],
                    'k': config['k'],
                    'N': config['M1'],
                    'l': int(l_str),
                    'g_l': g
                })
    
    if not results:
        print("No results found to plot. Make sure experiments ran and saved their data.")
        return

    df = pd.DataFrame(results)
    
    # --- Plot g_l vs. d for fixed k ---
    k_vals = df['k'].unique()
    l_vals_to_plot = sorted(df['l'].unique())

    for k_val in k_vals:
        plt.figure(figsize=(12, 8))
        sub_df = df[df['k'] == k_val]
        
        for l_val in l_vals_to_plot:
            l_df = sub_df[sub_df['l'] == l_val]
            if not l_df.empty:
                plt.plot(l_df['d'], l_df['g_l'], marker='o', linestyle='-', label=f'$l={l_val}$')
        
        plt.xlabel('Input Dimension (d)')
        plt.ylabel('Stiffness ($g_l$)')
        plt.title(f'Stiffness $g_l$ vs. Dimension $d$ (for k={k_val})')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(base_save_dir, f'g_vs_d_for_k_{k_val}.png'), dpi=300)
        plt.close()

    # --- Plot g_l vs. k for fixed d ---
    d_vals = df['d'].unique()

    for d_val in d_vals:
        plt.figure(figsize=(12, 8))
        sub_df = df[df['d'] == d_val]
        
        for l_val in l_vals_to_plot:
            l_df = sub_df[sub_df['l'] == l_val]
            if not l_df.empty:
                plt.plot(l_df['k'], l_df['g_l'], marker='o', linestyle='-', label=f'$l={l_val}$')
        
        plt.xlabel('Parity Dimension (k)')
        plt.ylabel('Stiffness ($g_l$)')
        plt.title(f'Stiffness $g_l$ vs. Parity $k$ (for d={d_val})')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(base_save_dir, f'g_vs_k_for_d_{d_val}.png'), dpi=300)
        plt.close()
        
    print(f"Final plots saved in {base_save_dir}")
    print("="*80)


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Grid Search Configuration ---
    d_values = [25,50,75,100,125,150,200,250]
    k_values = [2, 4,6, 8]
    N_values = [128,256, 512,1024] 
    
    # --- Fixed Hyperparameters ---
    base_config = {
        'max_epochs': 500000, # Set a reasonable max for grid search
        'batch_size': 1024,
        'learning_rate': 0.008,
        'tracker': 1000,
        'n_ensemble': 4, # Smaller ensemble for faster grid search runs
        'weight_decay_W1': 1e-5,
        'weight_decay_a': 1e-5,
        'L_max': 20, # Max Hermite order to compute
    }
    
    base_save_dir = f"/home/goring/OnlineSGD/results_ana/parity_grid_search_1706_{int(time.time())}"
    os.makedirs(base_save_dir, exist_ok=True)
    print(f"Base save directory: {base_save_dir}")

    # --- Prepare all experiment configurations for the grid search ---
    tasks = []
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("WARNING: No GPUs found. Running all tasks on CPU sequentially.")
    
    gpu_idx = 0
    for d in d_values:
        for k in k_values:
            for N in N_values:
                if k >= d: continue # k must be smaller than d
                
                config = base_config.copy()
                config['d'] = d
                config['k'] = k
                config['M1'] = N
                config['save_dir'] = os.path.join(base_save_dir, f'd_{d}_k_{k}_N_{N}')
                
                if num_gpus > 0:
                    config['device_id'] = gpu_idx % num_gpus
                    gpu_idx += 1
                else:
                    config['device_id'] = 'cpu'

                tasks.append(config)

    # --- Run experiments in parallel ---
    # Determine number of parallel processes
    # Limited by GPUs if available, otherwise by CPU cores
    num_workers = num_gpus if num_gpus > 0 else cpu_count()
    
    if num_workers > 0 and len(tasks) > 0:
        print(f"\nStarting {len(tasks)} experiments on {num_workers} parallel workers...")
        with Pool(processes=num_workers) as pool:
            pool.map(run_single_experiment, tasks)
        print("\nAll training runs have completed.")
    elif len(tasks) > 0:
        print("\nRunning tasks sequentially on CPU...")
        for task in tasks:
            run_single_experiment(task)
        print("\nAll training runs have completed.")
    else:
        print("No tasks to run.")

    # --- Aggregate results and plot ---
    aggregate_and_plot_g_l(base_save_dir)
    
    print("\nAll analyses complete.")