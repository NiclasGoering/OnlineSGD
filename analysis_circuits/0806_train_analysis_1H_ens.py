import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import time
import itertools
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

# --- Model Definition ---
class OneLayerTanhNet(torch.nn.Module):
    """A one-hidden-layer neural network with a Tanh activation and a linear output layer."""
    def __init__(self, input_dim, hidden_width, W1_grad=True, a_grad=True):
        super(OneLayerTanhNet, self).__init__()
        # Initialize layers with scaling
        self.W1 = torch.nn.Parameter(torch.randn(hidden_width, input_dim) / np.sqrt(input_dim))
        self.a = torch.nn.Parameter(torch.randn(hidden_width) / np.sqrt(hidden_width))
        
        # Control which weights are trainable
        self.W1.requires_grad = W1_grad
        self.a.requires_grad = a_grad

    def activation(self, x):
        return torch.tanh(x)

    def forward(self, x):
        h1 = self.activation(torch.matmul(x, self.W1.t()))
        output = torch.matmul(h1, self.a)
        return output

    def get_activations(self, x):
        return self.activation(torch.matmul(x, self.W1.t()))

# --- Main Experiment Class ---
class EnsembleParityExperiment:
    """
    Manages training and analysis for an ensemble of one-layer networks learning a parity function.
    Each network is trained until individual convergence.
    """
    def __init__(self, d=30, k=6, M1=512, learning_rate=0.01, max_epochs=50000,
                 batch_size=512, n_ensemble=10, n_parallel=5, tracker=100,
                 weight_decay_W1=0.0, weight_decay_a=0.0, top_n_neurons=100, 
                 kernel_n_samples=100000, device_id=None, save_dir="parity_ensemble_analysis"):
        
        # --- Hyperparameters ---
        self.d = d; self.k = k; self.M1 = M1; self.learning_rate = learning_rate
        self.max_epochs = max_epochs; self.batch_size = batch_size; self.n_ensemble = n_ensemble
        self.n_parallel = min(n_parallel, n_ensemble); self.tracker = tracker
        self.weight_decay_W1 = weight_decay_W1; self.weight_decay_a = weight_decay_a
        self.top_n_neurons = top_n_neurons; self.kernel_n_samples = kernel_n_samples
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # --- Device Configuration ---
        if device_id is not None and torch.cuda.is_available(): self.device = torch.device(f"cuda:{device_id}")
        else: self.device = torch.device("cpu")
        print(f"EnsembleParityExperiment initialized on {self.device}")
        
        # --- Ensemble Initialization ---
        self.models = [OneLayerTanhNet(d, M1).to(self.device) for _ in range(n_ensemble)]
        self.optimizers = [self._create_optimizer(model) for model in self.models]
        self.criterion = torch.nn.MSELoss()

        # --- Data and State Storage ---
        self.is_converged = [False] * self.n_ensemble
        self.final_epochs = [0] * self.n_ensemble
        self.metrics = [{'epochs': [], 'correlation': [], 'phase_transition_epoch': None} for _ in range(n_ensemble)]
        self.final_weights = {'W1': [], 'a': []}
        
        # Ensemble-level metrics
        self.replica_stats = {'epochs': [], 'q0_rel_mean': [], 'q0_rel_std': [], 'q1_rel_mean': [], 'q1_rel_std': [], 'q0_irr_mean': [], 'q0_irr_std': [], 'q1_irr_mean': [], 'q1_irr_std': []}
        self.replica_stats_active = {'epochs': [], 'q0_rel_mean': [], 'q0_rel_std': [], 'q1_rel_mean': [], 'q1_rel_std': [], 'q0_irr_mean': [], 'q0_irr_std': [], 'q1_irr_mean': [], 'q1_irr_std': []}
        self.final_performance = {'full_model_corr': [], 'sparse_model_corr': []}
        
        # Representative model metrics (re-integrated)
        self.representative_snapshots = {'epochs': [], 'W1': [], 'a': []}
        self.representative_feature_importance = {'epochs': [], 'ratio': []}
        self.representative_gradient_stats = {'epochs': [], 'relevant_grad_mean': [], 'irrelevant_grad_mean': [], 'w1_grad_norm': []}
        self.kernel_stats = {'epochs': [], 'top_eigenvalues': []}
        self.centered_kernel_stats = {'epochs': [], 'top_eigenvalues': []}
        
        # *** MODIFIED: Kernel stats now store mean and std across the ensemble ***
        self.kernel_stats = {'epochs': [], 'top_eigenvalues_mean': [], 'top_eigenvalues_std': []}
        self.centered_kernel_stats = {'epochs': [], 'top_eigenvalues_mean': [], 'top_eigenvalues_std': []}
        self.final_kernel_eigenvalues = []

        # --- Fixed Datasets ---
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(2000, d)), dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)
        self.X_kernel = torch.tensor(np.random.choice([-1, 1], size=(self.kernel_n_samples, d)), dtype=torch.float32).to(self.device)

    def _create_optimizer(self, model):
        param_groups = [{'params': model.W1, 'weight_decay': self.weight_decay_W1}, {'params': model.a, 'weight_decay': self.weight_decay_a}]
        return torch.optim.SGD(param_groups, lr=self.learning_rate)

    def _target_function(self, x):
        return torch.prod(x[:, :self.k], dim=1)

    def train(self, early_stop_corr=0.995):
        print(f"Starting training for up to {self.max_epochs} epochs...")
        self._compute_all_metrics(0)
        epoch = 0
        with tqdm(total=self.max_epochs, desc="Training Ensemble") as pbar:
            while not all(self.is_converged) and epoch < self.max_epochs:
                epoch += 1
                active_indices = [i for i, conv in enumerate(self.is_converged) if not conv]
                for i in range(0, len(active_indices), self.n_parallel):
                    chunk_indices = active_indices[i:i+self.n_parallel]
                    X_batch = torch.randn(self.batch_size, self.d, device=self.device).sign()
                    y_batch = self._target_function(X_batch)
                    for model_idx in chunk_indices:
                        model, optimizer = self.models[model_idx], self.optimizers[model_idx]
                        model.train()
                        loss = self.criterion(model(X_batch), y_batch)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                if epoch % self.tracker == 0 or epoch == self.max_epochs:
                    self._compute_all_metrics(epoch)
                    for i in range(self.n_ensemble):
                        if not self.is_converged[i] and self.metrics[i]['correlation'] and self.metrics[i]['correlation'][-1] > early_stop_corr:
                            self.is_converged[i], self.final_epochs[i] = True, epoch
                            print(f"\n--- Model {i} converged at epoch {epoch} ---")
                    num_converged = sum(self.is_converged)
                    avg_corr = np.mean([m['correlation'][-1] for m in self.metrics if m['correlation']])
                    pbar.set_description(f"Converged: {num_converged}/{self.n_ensemble} | Avg Corr: {avg_corr:.4f}")
                pbar.update(1)
        self.final_weights = {'W1': [m.W1.detach().cpu().numpy() for m in self.models], 'a': [m.a.detach().cpu().numpy() for m in self.models]}
        print("\n" + "="*80 + "\nTraining complete!")

    def _compute_all_metrics(self, epoch):
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                preds = model(self.X_test).squeeze()
                corr = torch.corrcoef(torch.stack([preds, self.y_test.squeeze()]))[0, 1].item() if preds.numel() > 1 and torch.var(preds) > 1e-6 else 0.0
            self.metrics[i]['epochs'].append(epoch); self.metrics[i]['correlation'].append(corr)
            if corr > 0.9 and self.metrics[i]['phase_transition_epoch'] is None: self.metrics[i]['phase_transition_epoch'] = epoch
            
            # --- Detailed analysis for the representative model (model 0) ---
            if i == 0:
                self._compute_representative_model_metrics(epoch, model)

        self._compute_replica_metrics(epoch)
        self._compute_replica_metrics_active(epoch)
        self._compute_kernels(epoch)
        
    def _compute_representative_model_metrics(self, epoch, model):
        """Computes and stores all detailed metrics for the representative model."""
        # Weight snapshots
        w1_snap = model.W1.detach().cpu().numpy()
        a_snap = model.a.detach().cpu().numpy()
        self.representative_snapshots['epochs'].append(epoch)
        self.representative_snapshots['W1'].append(w1_snap)
        self.representative_snapshots['a'].append(a_snap)
        
        # Feature importance
        rel_imp = np.mean(np.abs(w1_snap[:, :self.k])) if self.k > 0 else 0
        irrel_imp = np.mean(np.abs(w1_snap[:, self.k:])) if self.d > self.k else 0
        self.representative_feature_importance['epochs'].append(epoch)
        self.representative_feature_importance['ratio'].append(rel_imp / (irrel_imp + 1e-9))
        
        # Gradient statistics
        if not self.is_converged[0]:
            model.train()
            X_grad = torch.randn(self.batch_size, self.d, device=self.device).sign()
            y_grad = self._target_function(X_grad)
            loss = self.criterion(model(X_grad), y_grad)
            self.optimizers[0].zero_grad()
            loss.backward()
            if model.W1.grad is not None:
                grads = model.W1.grad.detach().cpu().numpy()
                self.representative_gradient_stats['epochs'].append(epoch)
                self.representative_gradient_stats['relevant_grad_mean'].append(np.mean(np.abs(grads[:, :self.k])))
                self.representative_gradient_stats['irrelevant_grad_mean'].append(np.mean(np.abs(grads[:, self.k:])))
                self.representative_gradient_stats['w1_grad_norm'].append(np.linalg.norm(grads))
            self.optimizers[0].zero_grad()

    
    def _compute_kernel_metrics(self, epoch):
        # 1. Conjugate Kernel Eigenvalue Evolution
        all_top_eigs, all_top_eigs_centered = [], []
        # 2. Explained Variance
        all_cumulative_ratios = []
        y_test_np = self.y_test.cpu().numpy()
        total_y_variance = np.sum(y_test_np**2)

        for model in self.models:
            model.eval()
            with torch.no_grad():
                # Conjugate Kernel on large dataset
                H_conj = model.get_activations(self.X_conjugate_kernel)
                K_conj = (H_conj.T @ H_conj) / float(self.kernel_n_samples)
                try:
                    eigs = torch.linalg.eigvalsh(K_conj).cpu().numpy(); all_top_eigs.append(np.sort(eigs)[-20:][::-1])
                    K_conj_centered = K_conj - torch.diag(torch.diag(K_conj))
                    eigs_c = torch.linalg.eigvalsh(K_conj_centered).cpu().numpy(); all_top_eigs_centered.append(np.sort(eigs_c)[-20:][::-1])
                except torch.linalg.LinAlgError: pass

                # Explained Variance on fixed test set
                H_test = model.get_activations(self.X_test)
                K_test = (H_test @ H_test.T).cpu().numpy()
                try:
                    e_vals, e_vecs = np.linalg.eigh(K_test)
                    sorted_indices = np.argsort(e_vals)[::-1]
                    sorted_e_vecs = e_vecs[:, sorted_indices]
                    proj_coeffs = y_test_np.T @ sorted_e_vecs
                    variance_per_eigenvector = proj_coeffs**2
                    cumulative_ratio = np.cumsum(variance_per_eigenvector) / total_y_variance
                    all_cumulative_ratios.append(cumulative_ratio)
                except np.linalg.LinAlgError: pass
        
        # Store stats for conjugate kernel
        if all_top_eigs:
            self.kernel_stats['epochs'].append(epoch)
            self.kernel_stats['top_eigenvalues_mean'].append(np.mean(all_top_eigs, axis=0))
            self.kernel_stats['top_eigenvalues_std'].append(np.std(all_top_eigs, axis=0))
        if all_top_eigs_centered:
            self.centered_kernel_stats['epochs'].append(epoch)
            self.centered_kernel_stats['top_eigenvalues_mean'].append(np.mean(all_top_eigs_centered, axis=0))
            self.centered_kernel_stats['top_eigenvalues_std'].append(np.std(all_top_eigs_centered, axis=0))
        
        # Store stats for explained variance
        if all_cumulative_ratios:
            self.explained_variance_stats['epochs'].append(epoch)
            self.explained_variance_stats['mean_ratios'].append(np.mean(all_cumulative_ratios, axis=0))
            self.explained_variance_stats['std_ratios'].append(np.std(all_cumulative_ratios, axis=0))

    def _compute_kernels(self, epoch):
        """Computes the conjugate kernel and its eigenvalues for ALL models in the ensemble."""
        all_top_eigs = []
        all_top_eigs_centered = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                H = model.get_activations(self.X_kernel)
                K = (H.T @ H) / float(self.kernel_n_samples)
                
                # Regular Kernel
                try:
                    eigs = torch.linalg.eigvalsh(K).cpu().numpy()
                    all_top_eigs.append(np.sort(eigs)[-100:][::-1]) # Get top 20
                except torch.linalg.LinAlgError:
                    all_top_eigs.append(np.zeros(100)) # Append zeros on failure
                    print(f"Warning: Eigendecomposition failed for standard kernel at epoch {epoch}.")

                # Centered Kernel
                try:
                    K_centered = K - torch.diag(torch.diag(K))
                    eigs_centered = torch.linalg.eigvalsh(K_centered).cpu().numpy()
                    all_top_eigs_centered.append(np.sort(eigs_centered)[-100:][::-1])
                except torch.linalg.LinAlgError:
                    all_top_eigs_centered.append(np.zeros(100))
                    print(f"Warning: Eigendecomposition failed for centered kernel at epoch {epoch}.")
        
        # Store mean and std dev across the ensemble for the standard kernel
        if all_top_eigs:
            self.kernel_stats['epochs'].append(epoch)
            self.kernel_stats['top_eigenvalues_mean'].append(np.mean(all_top_eigs, axis=0))
            self.kernel_stats['top_eigenvalues_std'].append(np.std(all_top_eigs, axis=0))

        # Store mean and std dev for the centered kernel
        if all_top_eigs_centered:
            self.centered_kernel_stats['epochs'].append(epoch)
            self.centered_kernel_stats['top_eigenvalues_mean'].append(np.mean(all_top_eigs_centered, axis=0))
            self.centered_kernel_stats['top_eigenvalues_std'].append(np.std(all_top_eigs_centered, axis=0))

    def _compute_replica_metrics(self, epoch):
        weights_list = [m.W1.detach().cpu().numpy() for m in self.models]
        self._calculate_and_store_q_params(epoch, weights_list, self.replica_stats, use_best_match=False)

    def _compute_replica_metrics_active(self, epoch):
        if self.top_n_neurons <= 0: return
        effective_weights_list = []
        for model in self.models:
            a, w1 = model.a.detach().cpu().numpy(), model.W1.detach().cpu().numpy()
            indices = np.argsort(np.abs(a))[-self.top_n_neurons:]
            effective_weights_list.append(w1[indices, :] * np.sign(a[indices])[:, np.newaxis])
        self._calculate_and_store_q_params(epoch, effective_weights_list, self.replica_stats_active, use_best_match=True)

    def _calculate_and_store_q_params(self, epoch, weights_list, storage_dict, use_best_match=False):
        q0r, q1r, q0i, q1i = [], [], [], []
        for w in weights_list:
            v, u = w[:,:self.k], w[:,self.k:]
            if self.k > 0: q0r.append(np.mean(np.sum(v*v, axis=1) / self.k))
            if self.d > self.k: q0i.append(np.mean(np.sum(u*u, axis=1)/(self.d-self.k)))
        for wa, wb in itertools.combinations(weights_list, 2):
            va, ua = wa[:,:self.k], wa[:,self.k:]; vb, ub = wb[:,:self.k], wb[:,self.k:]
            if use_best_match:
                if self.k > 0: q1r.append(np.mean(np.max(np.abs((va@vb.T)/self.k), axis=1)))
                if self.d > self.k: q1i.append(np.mean(np.max(np.abs((ua@ub.T)/(self.d-self.k)), axis=1)))
            else:
                if self.k > 0: q1r.append(np.mean(np.sum(va*vb, axis=1)/self.k))
                if self.d > self.k: q1i.append(np.mean(np.sum(ua*ub, axis=1)/(self.d-self.k)))
        storage_dict['epochs'].append(epoch)
        for name, vals in [('q0_rel',q0r),('q1_rel',q1r),('q0_irr',q0i),('q1_irr',q1i)]:
            storage_dict[f'{name}_mean'].append(np.mean(vals) if vals else 0)
            storage_dict[f'{name}_std'].append(np.std(vals) if vals else 0)

    def evaluate_final_performance(self, test_set_size=10000):
        """
        Evaluates the final performance of both the full and sparse (top_n) models
        on a large, independent test set. Also computes the final kernel eigenvalue spectrum.
        """
        print(f"\nEvaluating final performance on {test_set_size} samples...")
        X_final = torch.randn(test_set_size, self.d, device=self.device).sign()
        y_final = self._target_function(X_final)
        
        for m in self.models:
            m.eval()
            with torch.no_grad():
                # --- 1. Correlation Performance ---
                preds_full = m(X_final).squeeze()
                self.final_performance['full_model_corr'].append(torch.corrcoef(torch.stack([preds_full, y_final]))[0,1].item())
                a, w1 = m.a.detach(), m.W1.detach()
                indices = torch.argsort(torch.abs(a))[-self.top_n_neurons:]
                h_sparse = torch.tanh(X_final @ w1[indices,:].T)
                preds_sparse = (h_sparse @ a[indices]).squeeze()
                self.final_performance['sparse_model_corr'].append(torch.corrcoef(torch.stack([preds_sparse, y_final]))[0,1].item())

                # --- 2. Final Kernel Eigenvalue Spectrum ---
                H = m.get_activations(self.X_kernel)
                K = (H.T @ H) / float(self.kernel_n_samples)
                try:
                    eigs = torch.linalg.eigvalsh(K).cpu().numpy()
                    self.final_kernel_eigenvalues.append(eigs)
                except torch.linalg.LinAlgError:
                    print(f"Warning: Final eigenvalue decomposition failed for a model.")
                    self.final_kernel_eigenvalues.append(np.zeros(self.M1)) # Append zeros on failure

        print(f"Avg correlation (Full): {np.mean(self.final_performance['full_model_corr']):.4f}")
        print(f"Avg correlation (Sparse, Top {self.top_n_neurons}): {np.mean(self.final_performance['sparse_model_corr']):.4f}")

    def plot_final_kernel_eigenvalue_histograms(self):
        """
        Plots overlaid log-log histograms of the final kernel eigenvalue spectrum for each model.
        This helps visualize the distribution and identify any dominant 'spikes'.
        """
        print("Plotting final kernel eigenvalue spectrum histograms...")
        if not self.final_kernel_eigenvalues:
            print("Final kernel eigenvalue data not available. Run evaluate_final_performance() first.")
            return

        plt.figure(figsize=(12, 8))
        
        # Use a colormap to distinguish models if there aren't too many
        colors = cm.get_cmap('viridis', self.n_ensemble)
        
        for i, eigs in enumerate(self.final_kernel_eigenvalues):
            # We plot the absolute values because log scale can't handle negatives,
            # which can appear from numerical precision issues for very small eigenvalues.
            eigs_to_plot = np.abs(eigs[eigs != 0])
            if eigs_to_plot.size > 0:
                 sns.histplot(eigs_to_plot, color=colors(i), alpha=0.15, log_scale=(True, True), element="step")

        plt.title(f'Final Kernel Eigenvalue Spectrum (Overlay of {self.n_ensemble} Models)')
        plt.xlabel('Eigenvalue (log scale)')
        plt.ylabel('Count (log scale)')
        plt.grid(True, which="both", linestyle=':')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/final_kernel_eigenvalue_histograms.png", dpi=300)
        plt.close()

    def plot_explained_variance_evolution(self):
        print("Plotting explained variance evolution...")
        stats = self.explained_variance_stats
        if not stats['epochs']: return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        epochs_to_plot = {0: "Start"}
        avg_pt_epoch = np.mean([m['phase_transition_epoch'] for m in self.metrics if m['phase_transition_epoch'] is not None])
        if not np.isnan(avg_pt_epoch):
             pt_idx = np.argmin(np.abs(np.array(stats['epochs']) - avg_pt_epoch))
             epochs_to_plot[pt_idx] = f"Phase Transition (Epoch ~{stats['epochs'][pt_idx]})"
        
        final_idx = len(stats['epochs']) - 1
        epochs_to_plot[final_idx] = f"End (Epoch {stats['epochs'][final_idx]})"
        
        n_sqrt = np.sqrt(self.n_ensemble)
        num_eigenvectors = len(stats['mean_ratios'][0])
        x_axis = np.arange(1, num_eigenvectors + 1)

        for idx, label in epochs_to_plot.items():
            mean_ratio = stats['mean_ratios'][idx]
            std_ratio = stats['std_ratios'][idx]
            sem = std_ratio / n_sqrt
            line, = ax.plot(x_axis, mean_ratio, marker='', linestyle='-', label=label)
            ax.fill_between(x_axis, mean_ratio - sem, mean_ratio + sem, color=line.get_color(), alpha=0.2)
            
        ax.set_xscale('log')
        ax.set_xlabel('Number of Eigenvectors (log scale)')
        ax.set_ylabel('Fraction of Explained Variance')
        ax.set_title('Cumulative Explained Variance of Target by Kernel Eigenvectors')
        ax.grid(True, which="both", linestyle=':')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/explained_variance_evolution.png", dpi=300)
        plt.close()

    # --- Plotting Functions ---
    def plot_all(self):
        print("\n" + "="*80 + "\nGenerating all plots...")
        self.plot_overlaid_training_curves()
        self.plot_replica_evolution_with_errorbars()
        self.plot_replica_evolution_active_neurons() 
        self.plot_kernel_eigenvalue_evolution()
        self.plot_phase_transition_histogram()
        self.plot_overlaid_weight_histograms()
        self.plot_final_w1_transpose_w1_heatmaps()
        self.plot_sparse_vs_full_performance()
        self.plot_representative_feature_importance()
        self.plot_representative_w1_snapshots()
        self.plot_representative_gradient_evolution()
        self.plot_representative_weight_histograms()
        self.plot_final_kernel_eigenvalue_histograms() # *** NEWLY ADDED ***
        self.plot_final_kernel_eigenvalue_scatter() # *** NEWLY ADDED ***
        self.plot_explained_variance_evolution()


        print("All plots saved successfully!")

    
    def plot_replica_evolution_with_errorbars(self):
        """Plots the replica symmetry metrics vs. epoch for all neurons."""
        print("Plotting replica evolution for ALL neurons...")
        self._plot_replica_evolution(self.replica_stats, "all_neurons", "Replica Evolution for All Neurons")

    # def plot_replica_evolution_active_neurons(self):
    #     """Plots the replica symmetry metrics vs. epoch for the top N active neurons."""
    #     print(f"Plotting replica evolution for top {self.top_n_neurons} active neurons...")
    #     title = f'Replica Evolution for Top {self.top_n_neurons} Active Neurons'
    #     self._plot_replica_evolution(self.replica_stats_active, "active_neurons", title)

    # def _plot_replica_evolution(self, stats, filename_suffix, suptitle):
    #     """Helper to plot replica evolution with ribbons for the standard error of the mean."""
    #     if not stats['epochs']: return
    #     fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    #     epochs = stats['epochs']
    #     fig.suptitle(suptitle, fontsize=16)

    #     n_q0 = self.n_ensemble
    #     n_q1 = self.n_ensemble * (self.n_ensemble - 1) / 2
        
    #     ax = axs[0]
    #     if n_q0 > 0:
    #         mean, std = np.array(stats['q0_rel_mean']), np.array(stats['q0_rel_std'])
    #         sem = std / np.sqrt(n_q0)
    #         line, = ax.plot(epochs, mean, '-o', markersize=3, label='$q_{0}^{\\rm rel}$ (Self-Overlap)')
    #         ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
    #     if n_q1 > 0:
    #         mean, std = np.array(stats['q1_rel_mean']), np.array(stats['q1_rel_std'])
    #         sem = std / np.sqrt(n_q1)
    #         line, = ax.plot(epochs, mean, '--s', markersize=3, label='$q_{1}^{\\rm rel}$ (Cross-Overlap)')
    #         ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
    #     ax.set_ylabel('Mean Norm Value'); ax.set_title('Relevant Feature Norms (dims $1$ to $k$)'); ax.legend(); ax.set_yscale('log')

    #     ax = axs[1]
    #     if n_q0 > 0:
    #         mean, std = np.array(stats['q0_irr_mean']), np.array(stats['q0_irr_std'])
    #         sem = std / np.sqrt(n_q0)
    #         line, = ax.plot(epochs, mean, '-o', markersize=3, label='$q_{0}^{\\rm irr}$ (Self-Overlap)')
    #         ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
    #     if n_q1 > 0:
    #         mean, std = np.array(stats['q1_irr_mean']), np.array(stats['q1_irr_std'])
    #         sem = std / np.sqrt(n_q1)
    #         line, = ax.plot(epochs, mean, '--s', markersize=3, label='$q_{1}^{\\rm irr}$ (Cross-Overlap)')
    #         ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
    #     ax.set_ylabel('Mean Norm Value'); ax.set_title('Irrelevant Feature Norms (dims $k+1$ to $d$)'); ax.legend(); ax.set_yscale('log')
        
    #     for ax_i in axs:
    #         ax_i.set_xlabel('Epoch'); ax_i.grid(True, which='both', linestyle=':')
        
    #     plt.tight_layout(rect=[0, 0, 1, 0.95])
    #     plt.savefig(f"{self.save_dir}/replica_evolution_{filename_suffix}.png", dpi=300)
    #     plt.close()

  


    def _plot_replica_evolution(self, stats, filename_suffix, suptitle):
        if not stats['epochs']: return
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True); fig.suptitle(suptitle, fontsize=16)
        epochs = stats['epochs']; n_q0, n_q1 = self.n_ensemble, self.n_ensemble*(self.n_ensemble-1)/2
        for i, part in enumerate(['rel', 'irr']):
            ax = axs[i]; title_part = 'Relevant' if part == 'rel' else 'Irrelevant'
            if n_q0>0:
                mean,std=np.array(stats[f'q0_{part}_mean']),np.array(stats[f'q0_{part}_std']); sem=std/np.sqrt(n_q0)
                line,=ax.plot(epochs,mean,'-o',markersize=3,label='$q_{0}$'); ax.fill_between(epochs,mean-sem,mean+sem,color=line.get_color(),alpha=0.2)
            if n_q1>0:
                mean,std=np.array(stats[f'q1_{part}_mean']),np.array(stats[f'q1_{part}_std']); sem=std/np.sqrt(n_q1)
                line,=ax.plot(epochs,mean,'--s',markersize=3,label='$q_{1}$'); ax.fill_between(epochs,mean-sem,mean+sem,color=line.get_color(),alpha=0.2)
            ax.set_title(f'{title_part} Norms'); ax.set_ylabel('Mean Norm Value'); ax.legend(); ax.set_yscale('log'); ax.grid(True,which='both',linestyle=':')
        axs[1].set_xlabel('Epoch'); plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(f"{self.save_dir}/replica_evolution_{filename_suffix}.png",dpi=300); plt.close()

    def plot_kernel_eigenvalue_evolution(self):
        """
        Plots the evolution of the top 20 eigenvalues for the conjugate kernel and its centered version,
        showing the mean across the ensemble with a ribbon for the standard error of the mean.
        """
        print("Plotting kernel eigenvalue evolution...")
        if not self.kernel_stats['epochs']:
            print("Kernel data not available.")
            return

        fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        epochs = self.kernel_stats['epochs']
        colors = cm.get_cmap('viridis', 100)
        n_sqrt = np.sqrt(self.n_ensemble)

        # --- Plot 1: Standard Kernel ---
        ax = axs[0]
        means = np.stack(self.kernel_stats['top_eigenvalues_mean'], axis=0)
        stds = np.stack(self.kernel_stats['top_eigenvalues_std'], axis=0)
        
        for i in range(100):
            mean_line = means[:, i]
            sem = stds[:, i] / n_sqrt
            line, = ax.plot(epochs, mean_line, color=colors(i), lw=1.5)
            ax.fill_between(epochs, mean_line - sem, mean_line + sem, color=line.get_color(), alpha=0.15)

        ax.set_yscale('log')
        ax.set_ylabel("Mean Eigenvalue")
        ax.set_title("Evolution of Top 20 Conjugate Kernel Eigenvalues (Ensemble Mean)")
        ax.grid(True, which='both', linestyle=':')

        # --- Plot 2: Centered Kernel (Diagonal Subtracted) ---
        ax = axs[1]
        means_centered = np.stack(self.centered_kernel_stats['top_eigenvalues_mean'], axis=0)
        stds_centered = np.stack(self.centered_kernel_stats['top_eigenvalues_std'], axis=0)
        
        for i in range(100):
            mean_line = means_centered[:, i]
            sem = stds_centered[:, i] / n_sqrt
            line, = ax.plot(epochs, mean_line, color=colors(i), lw=1.5)
            ax.fill_between(epochs, mean_line - sem, mean_line + sem, color=line.get_color(), alpha=0.15)
        
        ax.set_yscale('log')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Eigenvalue")
        ax.set_title("Evolution of Top 20 Centered Kernel Eigenvalues (Ensemble Mean)")
        ax.grid(True, which='both', linestyle=':')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/kernel_eigenvalue_evolution.png", dpi=300)
        plt.close()


    def plot_overlaid_training_curves(self):
        print("Plotting overlaid training curves...")
        if not self.metrics[0]['epochs']: return
        fig, ax = plt.subplots(1, 1, figsize=(12, 7)); colors = cm.get_cmap('viridis', self.n_ensemble)
        for i, metric in enumerate(self.metrics):
            if metric['epochs']: ax.plot(metric['epochs'], metric['correlation'], color=colors(i), alpha=0.7, label=f'Model {i}' if self.n_ensemble<=10 else None)
        ax.set_title('Test Correlation Evolution'); ax.set_xlabel('Epoch'); ax.set_ylabel('Test Correlation'); ax.grid(True,ls=':'); ax.axhline(y=0.995,color='r',ls='--',label='Threshold');
        if self.n_ensemble<=10: ax.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/overlaid_training_curves.png",dpi=300); plt.close()

    def plot_replica_evolution_active_neurons(self):
        print(f"Plotting replica evolution for top {self.top_n_neurons} active neurons...")
        self._plot_replica_evolution(self.replica_stats_active, "active_neurons", f'Replica Evolution for Top {self.top_n_neurons} Active Neurons')
    
    def plot_phase_transition_histogram(self):
        print("Plotting phase transition histogram...")
        pt_epochs = [m['phase_transition_epoch'] for m in self.metrics if m['phase_transition_epoch'] is not None]
        if not pt_epochs: return
        plt.figure(figsize=(10,6)); sns.histplot(pt_epochs, bins=max(10,self.n_ensemble//5), kde=True)
        plt.title('Phase Transition Epoch Distribution'); plt.xlabel('Epoch (Correlation > 0.9)'); plt.ylabel('Count')
        plt.grid(True,ls=':'); plt.tight_layout(); plt.savefig(f"{self.save_dir}/phase_transition_histogram.png",dpi=300); plt.close()

    def plot_final_w1_transpose_w1_heatmaps(self):
        print("Plotting final W1.T @ W1 heatmaps...")
        if not self.final_weights['W1']: return
        n, ncols = self.n_ensemble, int(np.ceil(np.sqrt(self.n_ensemble))); nrows = int(np.ceil(n/ncols))
        fig,axs = plt.subplots(nrows,ncols,figsize=(3*ncols,3*nrows),squeeze=False)
        vmax = max(np.max(w.T@w) for w in self.final_weights['W1']); vmin = min(np.min(w.T@w) for w in self.final_weights['W1'])
        for i,w1 in enumerate(self.final_weights['W1']):
            r,c=divmod(i,ncols); ax=axs[r,c]; im=ax.imshow(w1.T@w1,cmap='viridis',aspect='auto',vmin=vmin,vmax=vmax)
            ax.set_title(f"Model {i}",fontsize=8)
            if self.k>0: ax.axvline(x=self.k-0.5,color='r',ls='--',lw=0.8); ax.axhline(y=self.k-0.5,color='r',ls='--',lw=0.8)
            ax.set_xticks([]); ax.set_yticks([])
        for i in range(n,nrows*ncols): axs.flat[i].axis('off')
        fig.colorbar(im,ax=axs.ravel().tolist(),shrink=0.6,location='right'); fig.suptitle('$W_1^T W_1$ at End of Training',fontsize=16)
        plt.tight_layout(rect=[0,0,0.9,0.96]); plt.savefig(f"{self.save_dir}/final_w1t_w1_heatmaps.png",dpi=300); plt.close()

    def plot_sparse_vs_full_performance(self):
        print("Plotting sparse vs. full performance...")
        if not self.final_performance['full_model_corr']: return
        plt.figure(figsize=(10,6));
        sns.histplot(self.final_performance['full_model_corr'],color="blue",label=f'Full Model ({self.M1} Neurons)',kde=True,stat='density',alpha=0.6)
        sns.histplot(self.final_performance['sparse_model_corr'],color="red",label=f'Sparse Model (Top {self.top_n_neurons} Neurons)',kde=True,stat='density',alpha=0.6)
        plt.title('Final Test Correlation Distribution'); plt.xlabel('Test Correlation'); plt.ylabel('Density'); plt.legend(); plt.grid(True,ls=':'); plt.tight_layout()
        plt.yscale(
            'log'   # Use log scale for large datasets
        )
        plt.savefig(f"{self.save_dir}/sparse_vs_full_performance_hist.png",dpi=300); plt.close()

    def plot_representative_feature_importance(self):
        print("Plotting feature importance for representative model...")
        data = self.representative_feature_importance
        if not data['epochs']: return
        plt.figure(figsize=(12, 8)); plt.semilogy(data['epochs'], data['ratio'], marker='o', label='Feature Importance Ratio')
        pt_epoch = self.metrics[0]['phase_transition_epoch']
        if pt_epoch: plt.axvline(x=pt_epoch, color='r', ls='--', label=f"PT @ {pt_epoch}")
        plt.xlabel('Epoch'); plt.ylabel('Relevant/Irrelevant Ratio (log)'); plt.title('Feature Importance (Model 0)'); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(f"{self.save_dir}/representative_feature_importance.png", dpi=300); plt.close()

    def plot_representative_w1_snapshots(self):
        print("Plotting W1 snapshots for representative model...")
        snaps = self.representative_snapshots
        if not snaps['epochs']: return
        epochs = np.array(snaps['epochs']); pt_epoch = self.metrics[0]['phase_transition_epoch']; final_epoch = self.final_epochs[0]
        indices, labels = {0: "Start"}, {}
        if final_epoch > 0: indices[np.argmin(np.abs(epochs - final_epoch))] = "End"
        if pt_epoch: indices[np.argmin(np.abs(epochs - pt_epoch))] = "Phase Transition"
        sorted_indices = sorted(indices.keys()); plot_labels = [indices[i] for i in sorted_indices]; num_plots = len(sorted_indices)
        fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), squeeze=False)
        for i, snap_idx in enumerate(sorted_indices):
            epoch, W1, ax = snaps['epochs'][snap_idx], snaps['W1'][snap_idx], axs[0, i]
            im = ax.imshow(W1, cmap='RdBu_r', aspect='auto'); fig.colorbar(im, ax=ax)
            ax.set_title(f"W1 at Epoch {epoch}\n({plot_labels[i]})"); ax.set_xlabel("Input Feature")
            if i == 0: ax.set_ylabel("Neuron Index")
            if self.k > 0: ax.axvline(x=self.k - 0.5, color='g', ls='--')
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/representative_w1_snapshots.png", dpi=300); plt.close()

    def plot_representative_gradient_evolution(self):
        print("Plotting gradient evolution for representative model...")
        stats = self.representative_gradient_stats
        if not stats['epochs']: return
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].semilogy(stats['epochs'], stats['relevant_grad_mean'], 'o-', label='Relevant Mean Mag')
        axs[0].semilogy(stats['epochs'], stats['irrelevant_grad_mean'], 'x--', label='Irrelevant Mean Mag')
        axs[0].set_title('Gradient Magnitude (Model 0)'); axs[0].set_ylabel('Mean Gradient Magnitude (log)'); axs[0].legend()
        axs[1].semilogy(stats['epochs'], stats['w1_grad_norm'], 'o-', label='W1 Grad Norm')
        axs[1].set_title('Overall W1 Gradient Norm (Model 0)'); axs[1].set_ylabel('Norm (log)'); axs[1].legend()
        for ax in axs: ax.set_xlabel('Epoch'); ax.grid(True);
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/representative_gradient_evolution.png", dpi=300); plt.close()

    def plot_overlaid_weight_histograms(self):
        """Plots overlaid histograms of the final weights for W1 and a."""
        print("Plotting overlaid weight histograms...")
        if not self.final_weights['W1']: return
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        for w1 in self.final_weights['W1']: sns.histplot(w1.flatten(), bins=50, ax=axs[0], alpha=0.1, color='blue', stat='density')
        axs[0].set_title(f'Final $W_1$ Distributions (Overlay of {self.n_ensemble} models)'); axs[0].set_xlabel('Weight Value')
        for a in self.final_weights['a']: sns.histplot(a.flatten(), bins=50, ax=axs[1], alpha=0.1, color='green', stat='density')
        axs[1].set_title(f'Final $a$ Distributions (Overlay of {self.n_ensemble} models)'); axs[1].set_xlabel('Weight Value')
        for ax in axs: ax.grid(True, linestyle=':')
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/overlaid_weight_histograms.png", dpi=300); plt.close()

    def plot_final_kernel_eigenvalue_scatter(self):
        """
        Plots an overlaid scatter plot of the final kernel eigenvalue spectrum for each model.
        Eigenvalues are plotted against their rank on a log-log scale to visualize the distribution.
        """
        print("Plotting final kernel eigenvalue spectrum scatter plot...")
        if not self.final_kernel_eigenvalues:
            print("Final kernel eigenvalue data not available. Run evaluate_final_performance() first.")
            return

        plt.figure(figsize=(12, 8))
        colors = cm.get_cmap('viridis', self.n_ensemble)

        for i, eigs in enumerate(self.final_kernel_eigenvalues):
            # Sort eigenvalues in descending order for plotting against their rank
            # We use absolute value to handle potential small negatives from numerical precision
            sorted_eigs = np.sort(np.abs(eigs))[::-1]
            ranks = np.arange(1, len(sorted_eigs) + 1)
            
            # Use scatter with low alpha to see the density of points
            plt.scatter(ranks, sorted_eigs, color=colors(i), alpha=0.3, s=15, edgecolor='none')

        plt.yscale('log')
        plt.xscale('log') # A log-log plot is standard for viewing spectra
        plt.title(f'Final Kernel Eigenvalue Spectrum (Overlay of {self.n_ensemble} Models)')
        plt.xlabel('Rank (log scale)')
        plt.ylabel('Eigenvalue (log scale)')
        plt.grid(True, which="both", linestyle=':')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/final_kernel_eigenvalue_scatter.png", dpi=300)
        plt.close()

    def plot_representative_weight_histograms(self):
        print("Plotting weight histograms for representative model...")
        snaps = self.representative_snapshots
        if len(snaps['W1']) < 2: return
        start_W1, end_W1 = snaps['W1'][0].flatten(), snaps['W1'][-1].flatten()
        start_a, end_a = snaps['a'][0].flatten(), snaps['a'][-1].flatten()
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].hist(start_W1, bins=50, density=True, alpha=0.7); axs[0, 0].set_title("W1 Start (Model 0)")
        axs[0, 1].hist(end_W1, bins=50, density=True, alpha=0.7, color='C1'); axs[0, 1].set_title("W1 End (Model 0)")
        axs[1, 0].hist(start_a, bins=50, density=True, alpha=0.7); axs[1, 0].set_title("'a' Start (Model 0)")
        axs[1, 1].hist(end_a, bins=50, density=True, alpha=0.7, color='C1'); axs[1, 1].set_title("'a' End (Model 0)")
        for ax in axs.flat: ax.set_xlabel("Weight Value"); ax.set_ylabel("Density"); ax.grid(True)
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/representative_weight_histograms.png", dpi=300); plt.close()

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Experiment Configuration ---
    config = {
        'd': 30,
        'k': 5,
        'M1': 512,
        'max_epochs': 1000000,
        'batch_size': 4096,
        'learning_rate': 0.01,
        'tracker': 500,
        'n_ensemble': 24,
        'n_parallel': 8,
        'weight_decay_W1': 1e-5,
        'weight_decay_a': 1e-5,
        'top_n_neurons': 100, 
        'kernel_n_samples': 100000,
        'device_id': 0,
        'save_dir': "/home/goring/OnlineSGD/results_ana/parity_ensemble_run_3005_histo10"
    }

    # --- Run Experiment ---
    experiment = EnsembleParityExperiment(**config)
    experiment.train(early_stop_corr=0.995)
    experiment.evaluate_final_performance()
    experiment.plot_all()
