import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import time
from scipy import stats
import pandas as pd
from scipy.linalg import subspace_angles
import itertools
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

class OneLayerReLUNet(torch.nn.Module):
    """A one-hidden-layer neural network with a ReLU activation and a linear output layer."""
    def __init__(self, input_dim, hidden_width, W1_grad=True, a_grad=True, W1_init='random', k=None):
        super(OneLayerReLUNet, self).__init__()
        # Initialize layers
        if W1_init == 'random':
            self.W1 = torch.nn.Parameter(torch.randn(hidden_width, input_dim) / np.sqrt(input_dim))
        elif W1_init == 'sparse' and k is not None:
            W1_init_val = torch.zeros(hidden_width, input_dim)
            W1_init_val[:, :k] = torch.randn(hidden_width, k) / np.sqrt(input_dim)
            self.W1 = torch.nn.Parameter(W1_init_val)
        else:
            self.W1 = torch.nn.Parameter(torch.randn(hidden_width, input_dim) / np.sqrt(input_dim))

        self.a = torch.nn.Parameter(torch.randn(hidden_width) / np.sqrt(hidden_width))

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

class OneLayerParityNetAnalyzer:
    """Analyzes a one-hidden-layer network learning a parity function."""
    def __init__(self, d=30, k=6, M1=512, learning_rate=0.01,
                 batch_size=512, device_id=None, save_dir="parity_analysis_one_layer",
                 tracker=20, W1_grad=True, a_grad=True, W1_init='random', kernel_num=100000):
        self.d = d
        self.k = k
        self.M1 = M1
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tracker = tracker
        self.W1_grad = W1_grad
        self.a_grad = a_grad
        self.W1_init = W1_init
        self.kernel_num = kernel_num

        # Set device
        if device_id is not None:
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Initialize model and optimizer
        self.model = self._create_model()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(trainable_params, lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

        # Initialize all history and statistics dictionaries
        self.loss_history = []
        self.correlation_history = []
        
        self.kernel_stats = {'epochs': [], 'kernels': [], 'eigs': []}

        self.feature_correlations = {
            'epochs': [], 'relevant_mean': [], 'irrelevant_mean': []
        }
        
        self.gradient_stats = {
            'epochs': [], 'relevant_grad_mean': [], 'irrelevant_grad_mean': [], 'w1_grad_norm': []
        }
        
        self.weight_snapshots = {'epochs': [], 'W1': [], 'a': []}
        
        self.feature_importance = {
            'epochs': [], 'relevant_importance': [], 'irrelevant_importance': [], 'ratio': []
        }
        
        self.phase_transition = {
            'detected': False, 'epoch': None, 'correlation': None, 'final_epoch': None
        }
        
        self.activation_stats = {'epochs': [], 'percent_active': []}
        
        self.hidden_target_correlations = {'epochs': [], 'corr_sum': []}

        # *** NEWLY ADDED ***
        # Dictionary to store replica symmetry metrics
        self.replica_stats = {
            'epochs': [], 'q0_rel': [], 'q1_rel': [], 'q0_irr': [], 'q1_irr': []
        }
        # *** END OF ADDITION ***
        self.explained_variance_stats = {'epochs': [], 'mean_ratios': [], 'std_ratios': []}
        # Fixed test and analysis sets
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, d)), dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)
        self.X_analysis = torch.tensor(np.random.choice([-1, 1], size=(5000, d)), dtype=torch.float32).to(self.device)
        self.y_analysis = self._target_function(self.X_analysis)

        self.neuron_activation_evolution = {'epochs': [], 'activation_ratios': []}
        self.neuron_target_correlation_evolution = {'epochs': [], 'neuron_correlations': []}

        # Print initialization summary
        print(f"OneLayerParityNetAnalyzer initialized on {self.device}")
        print(f"Analyzing {k}-parity function in {d} dimensions with Tanh activation")
        print(f"Network: {M1} -> 1")
        print(f"Batch size: {batch_size}, Tracking metrics every {tracker} epochs")
        print(f"W1 trainable: {W1_grad}, a trainable: {a_grad}")
        print(f"W1 initialization: {W1_init}")

    def _create_model(self):
        """Creates and returns a one-layer ReLU network model."""
        return OneLayerReLUNet(
            self.d, self.M1, W1_grad=self.W1_grad, a_grad=self.a_grad,
            W1_init=self.W1_init, k=self.k
        ).to(self.device)

    def _target_function(self, x):
        """Computes the k-sparse parity function."""
        return torch.prod(x[:, :self.k], dim=1)

    def compute_all_metrics(self, epoch):
        """Computes and stores all tracked metrics for a given epoch."""
        self.take_weight_snapshot(epoch)
        self.compute_kernel(epoch)
        self.compute_feature_importance(epoch)
        self.compute_feature_correlations(epoch)
        self.compute_gradient_statistics(epoch)
        self.compute_activation_statistics(epoch)
        self.compute_hidden_target_correlations(epoch)
        self.track_individual_neuron_target_correlations(epoch)
        # *** NEWLY ADDED ***
        self.compute_replica_symmetry_metrics(epoch)
        # *** END OF ADDITION ***

    def take_weight_snapshot(self, epoch):
        """Stores the current model weights."""
        self.weight_snapshots['epochs'].append(epoch)
        self.weight_snapshots['W1'].append(self.model.W1.detach().cpu().numpy())
        self.weight_snapshots['a'].append(self.model.a.detach().cpu().numpy())

    def compute_replica_symmetry_metrics(self, epoch):
        """
        Computes replica symmetry order parameters by comparing current weights (replica 'a')
        with the initial weights (replica 'b').
        """
        # Ensure at least the initial weight snapshot exists
        if not self.weight_snapshots['W1']: return

        # Replica 'a' is the current model state
        W1_a = self.model.W1.detach().cpu().numpy()
        
        # Replica 'b' is the initial model state (at epoch 0)
        W1_b = self.weight_snapshots['W1'][0]

        # Split weights into relevant (v) and irrelevant (u) parts
        v_a = W1_a[:, :self.k]
        u_a = W1_a[:, self.k:]
        v_b = W1_b[:, :self.k]
        u_b = W1_b[:, self.k:]

        # Calculate relevant norms, q_rel
        if self.k > 0:
            # q0_rel: Mean squared norm of relevant weights (within replica a)
            q0_rel = np.mean(np.sum(v_a * v_a, axis=1) / self.k)
            # q1_rel: Mean overlap of relevant weights (between replicas a and b)
            q1_rel = np.mean(np.sum(v_a * v_b, axis=1) / self.k)
        else:
            q0_rel, q1_rel = 0.0, 0.0

        # Calculate irrelevant norms, q_irr
        if self.d - self.k > 0:
            # q0_irr: Mean squared norm of irrelevant weights (within replica a)
            q0_irr = np.mean(np.sum(u_a * u_a, axis=1) / (self.d - self.k))
            # q1_irr: Mean overlap of irrelevant weights (between replicas a and b)
            q1_irr = np.mean(np.sum(u_a * u_b, axis=1) / (self.d - self.k))
        else:
            q0_irr, q1_irr = 0.0, 0.0
            
        # Store the computed values
        self.replica_stats['epochs'].append(epoch)
        self.replica_stats['q0_rel'].append(q0_rel)
        self.replica_stats['q1_rel'].append(q1_rel)
        self.replica_stats['q0_irr'].append(q0_irr)
        self.replica_stats['q1_irr'].append(q1_irr)

    def compute_kernel(self, epoch):
        """Computes the conjugate kernel over hidden units and its eigenvalues."""
        self.model.eval()
        X = torch.tensor(np.random.choice([-1, 1], size=(self.kernel_num, self.d)), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            H1 = self.model.get_activations(X)
            K1 = (H1.t() @ H1) / float(self.kernel_num)
        try:
            eigs = torch.linalg.eigvalsh(K1).cpu().numpy()
        except (torch.linalg.LinAlgError, RuntimeError):
            K1_cpu = K1.cpu().numpy()
            try:
                eigs = np.linalg.eigvalsh(K1_cpu)
            except Exception:
                eigs = np.zeros(K1_cpu.shape[0], dtype=float)
        self.kernel_stats['epochs'].append(epoch)
        self.kernel_stats['kernels'].append(K1.cpu().numpy())
        self.kernel_stats['eigs'].append(eigs)

    def compute_feature_importance(self, epoch):
        W1 = self.model.W1.detach().cpu().numpy()
        relevant_importance = np.mean(np.abs(W1[:, :self.k])) if self.k > 0 else 0
        irrelevant_importance = np.mean(np.abs(W1[:, self.k:])) if self.d > self.k else 0
        ratio = relevant_importance / (irrelevant_importance + 1e-9)
        self.feature_importance['epochs'].append(epoch)
        self.feature_importance['relevant_importance'].append(relevant_importance)
        self.feature_importance['irrelevant_importance'].append(irrelevant_importance)
        self.feature_importance['ratio'].append(ratio)
        
    def compute_feature_correlations(self, epoch):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_analysis)
            error = y_pred - self.y_analysis
        corrs = [torch.mean(error * self.X_analysis[:, l]).item() for l in range(self.d)]
        self.feature_correlations['epochs'].append(epoch)
        self.feature_correlations['relevant_mean'].append(np.mean(np.abs(corrs[:self.k])))
        self.feature_correlations['irrelevant_mean'].append(np.mean(np.abs(corrs[self.k:])))
        
    def compute_gradient_statistics(self, epoch):
        if not self.W1_grad: return
        self.model.train()
        X = torch.tensor(np.random.choice([-1, 1], size=(self.batch_size, self.d)), dtype=torch.float32).to(self.device)
        y = self._target_function(X)
        self.optimizer.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        loss.backward()
        if self.model.W1.grad is not None:
            W1_grads = self.model.W1.grad.detach().cpu().numpy()
            relevant_grads = W1_grads[:, :self.k]
            irrelevant_grads = W1_grads[:, self.k:]
            self.gradient_stats['epochs'].append(epoch)
            self.gradient_stats['relevant_grad_mean'].append(np.mean(np.abs(relevant_grads)))
            self.gradient_stats['irrelevant_grad_mean'].append(np.mean(np.abs(irrelevant_grads)))
            self.gradient_stats['w1_grad_norm'].append(np.linalg.norm(W1_grads))
        self.optimizer.zero_grad()

    def plot_replica_symmetry_metrics(self):
        """Plots the evolution of replica symmetry order parameters."""
        print("Plotting replica symmetry metrics...")
        if not self.replica_stats['epochs']: return
        
        fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
        epochs = self.replica_stats['epochs']
        
        # Plot for relevant norms
        ax1 = axs[0]
        ax1.plot(epochs, self.replica_stats['q0_rel'], 'o-', label='$q_{0}^{\\rm rel}$ (Self-Overlap)', markersize=4)
        ax1.plot(epochs, self.replica_stats['q1_rel'], 'x--', label='$q_{1}^{\\rm rel}$ (Overlap with Init)', markersize=4)
        ax1.set_title('Relevant Feature Norms (dims $1$ to $k$)')
        ax1.set_ylabel('Mean Normalized Norm')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Plot for irrelevant norms
        ax2 = axs[1]
        ax2.plot(epochs, self.replica_stats['q0_irr'], 'o-', label='$q_{0}^{\\rm irr}$ (Self-Overlap)', markersize=4)
        ax2.plot(epochs, self.replica_stats['q1_irr'], 'x--', label='$q_{1}^{\\rm irr}$ (Overlap with Init)', markersize=4)
        ax2.set_title('Irrelevant Feature Norms (dims $k+1$ to $d$)')
        ax2.set_ylabel('Mean Normalized Norm')
        ax2.legend()
        ax2.set_yscale('log')

        # Add common labels and phase transition line
        for ax in axs:
            ax.set_xlabel('Epoch')
            ax.grid(True, which='both', linestyle=':')
            if self.phase_transition['detected']:
                ax.axvline(x=self.phase_transition['epoch'], color='r', linestyle='--', label=f"PT @ {self.phase_transition['epoch']}")
            ax.legend() # Re-call legend to show the PT line label
        
        fig.suptitle('Evolution of Replica Symmetry Order Parameters', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.save_dir}/replica_symmetry_metrics.png", dpi=300)
        plt.close()



    def compute_activation_statistics(self, epoch, num_samples=10000):
        self.model.eval()
        X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            h1 = self.model.get_activations(X)
            self.activation_stats['epochs'].append(epoch)
            self.activation_stats['percent_active'].append(torch.mean((h1 > 0).float()).item() * 100)
            self.neuron_activation_evolution['epochs'].append(epoch)
            self.neuron_activation_evolution['activation_ratios'].append(torch.mean((h1 > 0).float(), dim=0).cpu().numpy())

    def compute_hidden_target_correlations(self, epoch):
        self.model.eval()
        with torch.no_grad():
            h1 = self.model.get_activations(self.X_analysis)
            corrs = [torch.mean(h1[:, i] * self.y_analysis.float()).item() for i in range(self.M1)]
            self.hidden_target_correlations['epochs'].append(epoch)
            self.hidden_target_correlations['corr_sum'].append(np.sum(np.abs(corrs)))

    def track_individual_neuron_target_correlations(self, epoch):
        self.model.eval()
        with torch.no_grad():
            h1 = self.model.get_activations(self.X_analysis)
            corrs = [torch.mean(h1[:, i] * self.y_analysis.float()).item() for i in range(self.M1)]
            self.neuron_target_correlation_evolution['epochs'].append(epoch)
            self.neuron_target_correlation_evolution['neuron_correlations'].append(np.array(corrs))

    def train(self, n_epochs=10000, early_stop_corr=0.995):
        """Trains the model and collects data at specified intervals."""
        print(f"Starting training for {n_epochs} epochs...")
        start_time = time.time()

        self.compute_all_metrics(0)

        for epoch in tqdm(range(1, n_epochs + 1), desc="Training Progress"):
            self.model.train()
            X_batch = torch.tensor(np.random.choice([-1, 1], size=(self.batch_size, self.d)), dtype=torch.float32).to(self.device)
            y_batch = self._target_function(X_batch)
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())

            if epoch % self.tracker == 0 or epoch == n_epochs:
                self.model.eval()
                with torch.no_grad():
                    # *** BUG FIX APPLIED HERE ***
                    preds = self.model(self.X_test)
                    test_loss = self.criterion(preds, self.y_test).item()
                    preds_s, y_s = preds.squeeze(), self.y_test.squeeze()
                    if preds_s.numel() > 1 and torch.var(preds_s) > 0 and torch.var(y_s) > 0:
                        correlation = torch.corrcoef(torch.stack([preds_s, y_s]))[0, 1].item()
                    else:
                        correlation = 0.0
                self.correlation_history.append((epoch, correlation))
                
                self.compute_all_metrics(epoch)

                print(f"\nEpoch {epoch}: MSE={test_loss:.6f}, Corr={correlation:.4f}, Time={time.time() - start_time:.1f}s")

                if not self.phase_transition['detected'] and correlation > 0.9:
                    self.phase_transition['detected'] = True
                    self.phase_transition['epoch'] = epoch
                    print(f"Phase transition detected at epoch {epoch} with correlation {correlation:.4f}")

                if correlation > early_stop_corr:
                    print(f"Early stopping at epoch {epoch} with correlation {correlation:.4f}")
                    self.phase_transition['final_epoch'] = epoch
                    break
        
        if not self.phase_transition.get('final_epoch'):
             self.phase_transition['final_epoch'] = n_epochs

        print("Training completed!")

    def _get_snapshot_indices(self):
        """Determines the indices for plotting based on key training events."""
        epochs = self.weight_snapshots['epochs']
        if not epochs: return [], []
        
        indices, labels = {0}, {0: "Start"}
        pt_epoch = self.phase_transition.get('epoch')
        final_epoch = self.phase_transition.get('final_epoch')
        
        if final_epoch:
            final_idx = np.argmin(np.abs(np.array(epochs) - final_epoch))
            if final_idx not in labels:
                indices.add(final_idx); labels[final_idx] = "End"
        if pt_epoch:
            transition_window = int(0.05 * final_epoch) if final_epoch else int(0.05 * epochs[-1])
            pt_idx = np.argmin(np.abs(np.array(epochs) - pt_epoch))
            if pt_idx not in labels:
                indices.add(pt_idx); labels[pt_idx] = "Phase Transition"
            before_pt_idx = np.argmin(np.abs(np.array(epochs) - max(0, pt_epoch - transition_window)))
            if before_pt_idx not in labels:
                indices.add(before_pt_idx); labels[before_pt_idx] = "Before PT"
            after_pt_idx = np.argmin(np.abs(np.array(epochs) - min(epochs[-1], pt_epoch + transition_window)))
            if after_pt_idx not in labels:
                indices.add(after_pt_idx); labels[after_pt_idx] = "After PT"

        sorted_indices = sorted(list(indices))
        return sorted_indices, [labels[i] for i in sorted_indices]

    def plot_all(self):
        """Generates all plots."""
        print("\n" + "="*80); print("Generating all plots...")
        if not self.weight_snapshots['epochs']:
            print("No data to plot."); return
        indices, labels = self._get_snapshot_indices()
        print(f"Plotting for epochs: {[self.weight_snapshots['epochs'][i] for i in indices]}")
        
        # Call all plotting functions
        self.plot_training_progress()
        self.plot_feature_correlations()
        self.plot_feature_importance_evolution()
        self.plot_gradient_statistics()
        self.plot_neuron_target_correlation_evolution()
        self.plot_weight_matrices(indices, labels)
        self.plot_w1_transpose_w1(indices, labels)
        self.plot_weight_histograms()
        self.plot_squared_weight_difference_evolution()
        self.plot_kernel_heatmaps(indices, labels)
        self.plot_kernel_top_eigenvalues_evolution()
        self.plot_explained_variance_evolution()
        # *** NEWLY ADDED ***
        self.plot_replica_symmetry_metrics()
        # *** END OF ADDITION ***
        
        print("All plots generated!"); print("="*80 + "\n")

    def plot_training_progress(self):
        print("Plotting training progress...")
        if not self.correlation_history: return
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        corr_epochs, corr_values = zip(*self.correlation_history)
        ax.plot(corr_epochs, corr_values, marker='.', linestyle='-', label="Test Correlation")
        if self.phase_transition['detected']:
            ax.axvline(x=self.phase_transition['epoch'], color='r', linestyle='--', label=f"PT @ {self.phase_transition['epoch']}")
        ax.set_xlabel('Epoch'); ax.set_ylabel('Correlation'); ax.set_title('Training Progress')
        ax.grid(True); ax.legend(); plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_progress.png", dpi=300); plt.close()

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

    def plot_feature_correlations(self):
        print("Plotting feature correlations...")
        if not self.feature_correlations['epochs']: return
        plt.figure(figsize=(12, 8))
        plt.semilogy(self.feature_correlations['epochs'], self.feature_correlations['relevant_mean'], marker='o', label='Relevant Features Mean $|C_l|$')
        plt.semilogy(self.feature_correlations['epochs'], self.feature_correlations['irrelevant_mean'], marker='x', linestyle='--', label='Irrelevant Features Mean $|C_l|$')
        if self.phase_transition['detected']:
            plt.axvline(x=self.phase_transition['epoch'], color='r', linestyle='--', label=f"PT @ {self.phase_transition['epoch']}")
        plt.xlabel('Epoch'); plt.ylabel('Mean Absolute Correlation $|C_l|$ (log scale)'); plt.title('Feature Correlations $C_l$ Evolution')
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_correlations.png", dpi=300); plt.close()

    def plot_feature_importance_evolution(self):
        print("Plotting feature importance evolution...")
        if not self.feature_importance['epochs']: return
        plt.figure(figsize=(12, 8))
        plt.semilogy(self.feature_importance['epochs'], self.feature_importance['ratio'], marker='o', label='Feature Importance Ratio')
        if self.phase_transition['detected']:
            plt.axvline(x=self.phase_transition['epoch'], color='r', linestyle='--', label=f"PT @ {self.phase_transition['epoch']}")
        plt.xlabel('Epoch'); plt.ylabel('Relevant/Irrelevant Feature Importance Ratio (log)'); plt.title('Evolution of Feature Importance Ratio')
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_importance_evolution.png", dpi=300); plt.close()

    def plot_gradient_statistics(self):
        print("Plotting gradient statistics...")
        if not self.gradient_stats['epochs']: return
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].semilogy(self.gradient_stats['epochs'], self.gradient_stats['relevant_grad_mean'], 'o-', label='Relevant Mean Mag')
        axs[0].semilogy(self.gradient_stats['epochs'], self.gradient_stats['irrelevant_grad_mean'], 'x--', label='Irrelevant Mean Mag')
        axs[0].set_title('Gradient Magnitude Evolution'); axs[0].set_ylabel('Mean Gradient Magnitude (log)'); axs[0].legend()
        axs[1].semilogy(self.gradient_stats['epochs'], self.gradient_stats['w1_grad_norm'], 'o-', label='W1 Grad Norm')
        axs[1].set_title('Overall W1 Gradient Norm'); axs[1].set_ylabel('Norm (log)'); axs[1].legend()
        for ax in axs:
            ax.set_xlabel('Epoch'); ax.grid(True)
            if self.phase_transition['detected']: ax.axvline(x=self.phase_transition['epoch'], color='r', linestyle='--')
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/gradient_statistics.png", dpi=300); plt.close()

    def plot_neuron_target_correlation_evolution(self):
        print("Plotting neuron-target correlation evolution...")
        if not self.neuron_target_correlation_evolution['epochs']: return
        data = np.array(self.neuron_target_correlation_evolution['neuron_correlations']).T
        epochs = self.neuron_target_correlation_evolution['epochs']
        vmax = np.max(np.abs(data))
        plt.figure(figsize=(15, 8))
        im = plt.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        extent=[epochs[0], epochs[-1], self.M1, 0])
        plt.colorbar(im, label='Correlation with Target')
        if self.phase_transition['detected']:
            plt.axvline(x=self.phase_transition['epoch'], color='k', linestyle='--', label=f"PT @ {self.phase_transition['epoch']}")
        plt.xlabel('Epoch'); plt.ylabel('Neuron Index'); plt.title('Neuron-Target Correlation Evolution'); plt.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/neuron_target_correlation_evolution.png", dpi=300); plt.close()

    def plot_weight_matrices(self, indices, labels):
        print("Plotting weight matrices...")
        num_plots = len(indices)
        if num_plots == 0: return
        fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), squeeze=False)
        for i, snap_idx in enumerate(indices):
            epoch = self.weight_snapshots['epochs'][snap_idx]
            W1 = self.weight_snapshots['W1'][snap_idx]
            ax = axs[0, i]
            im = ax.imshow(W1, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"W1 at Epoch {epoch}\n({labels[i]})")
            ax.set_xlabel("Input Feature")
            if i == 0: ax.set_ylabel("Neuron Index")
            if self.k > 0: ax.axvline(x=self.k - 0.5, color='g', linestyle='--')
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/W1_snapshots.png", dpi=300); plt.close()

    def plot_w1_transpose_w1(self, indices, labels):
        print("Plotting W1.T @ W1...")
        num_plots = len(indices)
        if num_plots == 0: return
        fig, axs = plt.subplots(1, num_plots, figsize=(5.5 * num_plots, 5), squeeze=False)
        for i, snap_idx in enumerate(indices):
            epoch = self.weight_snapshots['epochs'][snap_idx]
            W1 = self.weight_snapshots['W1'][snap_idx]
            W1T_W1 = W1.T @ W1
            ax = axs[0, i]
            im = ax.imshow(W1T_W1, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"$W_1^T W_1$ at Epoch {epoch}\n({labels[i]})")
            ax.set_xlabel("Input Feature"); 
            if i == 0: ax.set_ylabel("Input Feature")
            if self.k > 0: 
                ax.axvline(x=self.k - 0.5, color='r', linestyle='--')
                ax.axhline(y=self.k - 0.5, color='r', linestyle='--')
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/W1T_W1_snapshots.png", dpi=300); plt.close()

    def plot_weight_histograms(self):
        print("Plotting weight histograms...")
        if len(self.weight_snapshots['W1']) < 2: return
        start_W1, end_W1 = self.weight_snapshots['W1'][0].flatten(), self.weight_snapshots['W1'][-1].flatten()
        start_a, end_a = self.weight_snapshots['a'][0].flatten(), self.weight_snapshots['a'][-1].flatten()
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].hist(start_W1, bins=50, density=True, alpha=0.7); axs[0, 0].set_title("W1 Distribution at Start")
        axs[0, 1].hist(end_W1, bins=50, density=True, alpha=0.7, color='C1'); axs[0, 1].set_title("W1 Distribution at End")
        axs[1, 0].hist(start_a, bins=50, density=True, alpha=0.7); axs[1, 0].set_title("'a' Distribution at Start")
        axs[1, 1].hist(end_a, bins=50, density=True, alpha=0.7, color='C1'); axs[1, 1].set_title("'a' Distribution at End")
        for ax in axs.flat: ax.set_xlabel("Weight Value"); ax.set_ylabel("Density"); ax.grid(True)
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/weight_histograms.png", dpi=300); plt.close()

    def plot_squared_weight_difference_evolution(self):
        print("Plotting squared weight difference evolution...")
        epochs = self.weight_snapshots['epochs']
        if not epochs: return
        rel_w_sq, irrel_w_sq = [], []
        for w1_snap in self.weight_snapshots['W1']:
            if w1_snap[:, :self.k].size > 0: rel_w_sq.append(np.mean(w1_snap[:, :self.k]**2))
            if w1_snap[:, self.k:].size > 0: irrel_w_sq.append(np.mean(w1_snap[:, self.k:]**2))
        if not rel_w_sq or not irrel_w_sq: return
        rel_w_sq, irrel_w_sq = np.array(rel_w_sq), np.array(irrel_w_sq)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(epochs, rel_w_sq, label='Mean Sq. Relevant Weights', color='blue')
        ax.plot(epochs, irrel_w_sq, label='Mean Sq. Irrelevant Weights', color='orange')
        ax.plot(epochs, rel_w_sq - irrel_w_sq, label='Difference (Rel - Irrel)', color='green', linestyle='--')
        if self.phase_transition['detected']: ax.axvline(x=self.phase_transition['epoch'], color='r', linestyle=':', label=f"PT @ {self.phase_transition['epoch']}")
        ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Squared Weight Value'); ax.set_title('Evolution of Squared Weights')
        ax.grid(True); ax.legend(); ax.set_yscale('log'); plt.tight_layout()
        plt.savefig(f"{self.save_dir}/squared_weight_evolution.png", dpi=300); plt.close()

    def plot_kernel_heatmaps(self, indices, labels):
        print("Plotting kernel heatmaps...")
        num_plots = len(indices)
        if num_plots == 0: return
        fig, axs = plt.subplots(1, num_plots, figsize=(5.5 * num_plots, 5), squeeze=False)
        for i, snap_idx in enumerate(indices):
            epoch = self.kernel_stats['epochs'][snap_idx]
            K = self.kernel_stats['kernels'][snap_idx]
            ax, vmax = axs[0, i], K.max()
            vmin = max(vmax * 1e-5, 1e-9)
            im = ax.imshow(K, cmap='viridis', aspect='auto', norm=LogNorm(vmin=vmin, vmax=vmax))
            plt.colorbar(im, ax=ax); ax.set_title(f"Kernel $H^T H$ at Epoch {epoch}\n({labels[i]})")
            ax.set_xlabel("Neuron Index")
            if i == 0: ax.set_ylabel("Neuron Index")
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/kernel_heatmaps.png", dpi=300); plt.close()

    def plot_kernel_top_eigenvalues_evolution(self):
        print("Plotting kernel eigenvalue evolution...")
        if not self.kernel_stats['epochs']: return
        eigs_all = np.stack(self.kernel_stats['eigs'], axis=0)
        epochs = self.kernel_stats['epochs']
        eigs_desc = np.sort(eigs_all, axis=1)[:, ::-1]
        top20 = eigs_desc[:, :20]
        norm0 = top20[0, 0] if top20.shape[0] > 0 and top20[0, 0] > 0 else 1.0
        top20_norm = top20 / (norm0 + 1e-12)
        colors = cm.get_cmap('viridis')(np.linspace(0, 1, 20))
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(20): ax.plot(epochs, top20_norm[:, i], color=colors[i], linewidth=1.5)
        ax.set_yscale('log'); ax.set_xlabel("Epoch"); ax.set_ylabel("Normalized Eigenvalue (λ / λ_max_initial)")
        ax.set_title(f"Kernel Top 20 Eigenvalues (Normalized)"); ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        if self.phase_transition['detected']:
            ax.axvline(x=self.phase_transition['epoch'], color='r', linestyle=':', label=f"PT @ {self.phase_transition['epoch']}"); ax.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/kernel_top20_eigenvalues.png", dpi=300); plt.close()

def run_one_layer_experiment(d, k, M1, learning_rate, batch_size, n_epochs, num_ex,
                               device_id=None, save_dir="parity_analysis_one_layer",
                               tracker=20, W1_grad=True, a_grad=True, W1_init='random'):
    """Sets up and runs a single experiment."""
    run_name = f"d{d}_k{k}_M1_{M1}_bs{batch_size}_lr{learning_rate}"
    run_specific_dir = os.path.join(save_dir, run_name)
    analyzer = OneLayerParityNetAnalyzer(
        d=d, k=k, M1=M1, learning_rate=learning_rate, batch_size=batch_size,
        device_id=device_id, save_dir=run_specific_dir, tracker=tracker,
        W1_grad=W1_grad, a_grad=a_grad, W1_init=W1_init, kernel_num=num_ex
    )
    analyzer.train(n_epochs=n_epochs)
    analyzer.plot_all()
    print(f"Experiment completed. Results saved to: {run_specific_dir}")
    return analyzer

if __name__ == "__main__":
    d = 30
    k = 5
    M1 = 1024
    batch_size = 4096
    learning_rate = 0.005
    n_epochs = 500000 
    tracker = 200
    num_ex = 100000
    base_save_dir = "parity_1H_0906_305_replica"

    device_id = 0 if torch.cuda.is_available() else None
    print(f"Using device: {'cuda:' + str(device_id) if device_id is not None else 'cpu'}")
    
    analyzer = run_one_layer_experiment(
        d=d, k=k, M1=M1, batch_size=batch_size, learning_rate=learning_rate,
        n_epochs=n_epochs, tracker=tracker, device_id=device_id, 
        save_dir=base_save_dir, num_ex=num_ex
    )
    print("One-layer experiment finished.")