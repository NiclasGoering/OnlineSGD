import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import time
import matplotlib.cm as cm
from matplotlib.colors import SymLogNorm

# --- Deep Model Definition ---
class DeepNet(nn.Module):
    """A fully-connected network with an arbitrary number of hidden layers."""
    def __init__(self, input_dim, hidden_widths, activation_fn=torch.relu):
        super(DeepNet, self).__init__()
        self.hidden_widths = [input_dim] + hidden_widths
        self.depth = len(hidden_widths)
        self.activation_fn = activation_fn
        self.hidden_layers = nn.ModuleList()
        for i in range(self.depth):
            self.hidden_layers.append(nn.Linear(self.hidden_widths[i], self.hidden_widths[i+1]))
        self.readout_a = nn.Parameter(torch.randn(self.hidden_widths[-1]))

    def forward(self, x):
        h = x
        hidden_activations = []
        for i in range(self.depth):
            z = self.hidden_layers[i](h)
            h = self.activation_fn(z)
            hidden_activations.append(h)
        final_representation = h
        output = torch.matmul(final_representation, self.readout_a) / np.sqrt(self.hidden_widths[-1])
        return output, hidden_activations

# --- Main Experiment Class for Deep Network ---
class DeepParityExperiment:
    """
    Manages training and analysis for an ensemble of deep networks learning a parity function.
    Focuses on tracking the layer-wise order parameters m_A for a variety of Walsh functions.
    """
    def __init__(self, d=100, k=4, depth=3, hidden_widths=None,
                 learning_rate=0.01, max_epochs=50000, batch_size=512,
                 n_ensemble=10, tracker=100, weight_decay=1e-5,
                 device_id=0, save_dir="deep_parity_analysis"):
        
        # --- Hyperparameters ---
        self.d = d
        self.k = k
        self.depth = depth
        self.hidden_widths = hidden_widths if hidden_widths is not None else [512] * depth
        if len(self.hidden_widths) != self.depth:
            raise ValueError("Length of hidden_widths must equal depth.")
            
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_ensemble = n_ensemble
        self.tracker = tracker
        self.weight_decay = weight_decay
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # --- Device Configuration ---
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")
        print(f"DeepParityExperiment initialized on {self.device} for d={d}, k={k}, depth={depth}")

        # --- Walsh Basis Setup ---
        print("Generating Walsh projection subsets...")
        self.walsh_n_samples = 100000
        self.X_walsh = torch.tensor(np.random.choice([-1, 1], size=(self.walsh_n_samples, d)), dtype=torch.float32).to(self.device)
        self.walsh_param_sets = self._generate_walsh_subsets()
        self.walsh_param_labels = self._generate_walsh_labels()
        
        # --- Pre-compute Walsh basis functions on the test set ---
        self.all_chi_A_walsh = self._precompute_walsh_basis()

        # --- Ensemble Initialization ---
        self.models = [DeepNet(d, self.hidden_widths).to(self.device) for _ in range(n_ensemble)]
        self.optimizers = [torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) for model in self.models]
        self.criterion = nn.MSELoss()

        # --- Data Storage ---
        self.layerwise_projection_stats = {
            'epochs': [],
            'projection_mean': [], # List of 3D arrays (epoch, layer, subset)
            'projection_std': []
        }
        self.correlation_stats = {'epochs': [], 'corr_mean': [], 'corr_std': []}

    def _generate_walsh_subsets(self):
        """Generates d+1 subsets: the teacher S, and d random subsets of sizes 1..d."""
        all_indices = list(range(self.d))
        subsets = [list(range(self.k))] # Start with teacher subset
        existing_subsets = {tuple(sorted(subsets[0]))}
        
        for l in range(1, self.d + 1):
            for _ in range(100): # Failsafe to prevent infinite loop
                new_subset = sorted(np.random.choice(all_indices, size=l, replace=False).tolist())
                if tuple(new_subset) not in existing_subsets:
                    subsets.append(new_subset)
                    existing_subsets.add(tuple(new_subset))
                    break
        print(f"Generated {len(subsets)} unique subsets for Walsh analysis.")
        return subsets

    def _generate_walsh_labels(self):
        """Generates descriptive labels for each Walsh subset for plotting."""
        labels = []
        teacher_subset = self.walsh_param_sets[0]
        for subset in self.walsh_param_sets:
            is_teacher = (len(subset) == len(teacher_subset) and all(x == y for x, y in zip(subset, teacher_subset)))
            if is_teacher:
                labels.append(f'$S$ (k={len(subset)})')
            else:
                labels.append(f'|A|={len(subset)}')
        return labels
    
    def _target_function_general(self, x, A):
        """Computes the Walsh function chi_A(x) for subset indices A."""
        if not A: return torch.ones(x.shape[0], device=self.device)
        return torch.prod(x[:, torch.tensor(A, device=self.device, dtype=torch.long)], dim=1)

    def _precompute_walsh_basis(self):
        """Pre-calculates all chi_A(x) basis functions on the test set."""
        num_subsets = len(self.walsh_param_sets)
        all_chi_A = torch.zeros(num_subsets, self.walsh_n_samples, device=self.device)
        for i, A in enumerate(self.walsh_param_sets):
            all_chi_A[i, :] = self._target_function_general(self.X_walsh, A)
        return all_chi_A

    def _compute_layerwise_projections(self, epoch):
        """
        Calculates m_A^(l) for all layers l and all subsets A.
        m_A^(l) = < E_x[ phi(z_i^(l)) * chi_A(x) ] >_{i in layer l}
        """
        num_subsets = len(self.walsh_param_sets)
        # Store results from all models: shape (n_ensemble, depth, num_subsets)
        all_projections_ensemble = torch.zeros(self.n_ensemble, self.depth, num_subsets, device=self.device)

        with torch.no_grad():
            for model_idx, model in enumerate(self.models):
                model.eval()
                _, hidden_activations = model(self.X_walsh)
                
                for l in range(self.depth):
                    h_l = hidden_activations[l]  # Shape: (walsh_n_samples, width_l)
                    
                    # Project each neuron's activation vector onto every Walsh basis function
                    # Shape: (num_subsets, width_l)
                    neuron_projections_matrix = torch.matmul(self.all_chi_A_walsh, h_l) / self.walsh_n_samples
                    
                    # Average projections over all neurons in the layer
                    # This gives the final m_A vector for this layer
                    # Shape: (num_subsets,)
                    m_A_vector_for_layer = torch.mean(neuron_projections_matrix, dim=1)
                    
                    all_projections_ensemble[model_idx, l, :] = m_A_vector_for_layer

        # Store the mean and std dev across the ensemble
        self.layerwise_projection_stats['epochs'].append(epoch)
        self.layerwise_projection_stats['projection_mean'].append(torch.mean(all_projections_ensemble, dim=0).cpu().numpy())
        self.layerwise_projection_stats['projection_std'].append(torch.std(all_projections_ensemble, dim=0).cpu().numpy())

    def _compute_correlation(self, epoch):
        """Computes the correlation of the final output with the target."""
        correlations = []
        y_target = self._target_function_general(self.X_walsh, self.walsh_param_sets[0])
        with torch.no_grad():
            for model in self.models:
                model.eval()
                preds, _ = model(self.X_walsh)
                preds = preds.squeeze()
                if preds.numel() > 1 and torch.var(preds) > 1e-6:
                    correlations.append(torch.corrcoef(torch.stack([preds, y_target]))[0, 1].item())
                else:
                    correlations.append(0.0)
        
        self.correlation_stats['epochs'].append(epoch)
        self.correlation_stats['corr_mean'].append(np.mean(correlations))
        self.correlation_stats['corr_std'].append(np.std(correlations))

    def train(self, early_stop_corr=0.98):
        """Main training loop."""
        print(f"Starting training for {self.n_ensemble} deep networks...")
        self._compute_correlation(0)
        self._compute_layerwise_projections(0)

        for epoch in tqdm(range(1, self.max_epochs + 1), desc="Training Ensemble"):
            X_batch = torch.tensor(np.random.choice([-1, 1], size=(self.batch_size, self.d)), dtype=torch.float32).to(self.device)
            y_batch = self._target_function_general(X_batch, self.walsh_param_sets[0])
            
            for i in range(self.n_ensemble):
                self.models[i].train()
                outputs, _ = self.models[i](X_batch)
                loss = self.criterion(outputs, y_batch)
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()
            
            if epoch % self.tracker == 0 or epoch == self.max_epochs:
                self._compute_correlation(epoch)
                self._compute_layerwise_projections(epoch)
                last_corr = self.correlation_stats['corr_mean'][-1]
                tqdm.write(f"Epoch {epoch}: Avg Correlation = {last_corr:.4f}")
                if last_corr > early_stop_corr:
                    print(f"\nEnsemble has converged with average correlation {last_corr:.4f} > {early_stop_corr}.")
                    break

        print("\n" + "="*80 + "\nTraining complete!")
        self.plot_all()

    def plot_layerwise_projection_heatmaps(self):
        """Plots the evolution of m_A for all subsets A as a heatmap for each layer."""
        print("Plotting layer-wise m_A projection heatmaps...")
        stats = self.layerwise_projection_stats
        if not stats['epochs']:
            print("No projection data available to plot.")
            return

        # Shape: (num_epochs, depth, num_subsets)
        projection_data = np.stack(stats['projection_mean'], axis=0)
        epochs = stats['epochs']
        num_subsets = projection_data.shape[2]

        # Determine global color scale for consistency
        vmax = np.max(np.abs(projection_data))
        if vmax < 1e-9: vmax = 1e-9
        norm = SymLogNorm(linthresh=1e-4, vmin=-vmax, vmax=vmax, base=10)

        fig, axs = plt.subplots(self.depth, 1, figsize=(15, 5 * self.depth), sharex=True, squeeze=False)
        axs = axs.flatten()

        for l in range(self.depth):
            ax = axs[l]
            # Data for this layer: (num_epochs, num_subsets) -> transpose for plotting
            layer_data = projection_data[:, l, :].T
            
            img = ax.imshow(layer_data, aspect='auto', cmap='bwr', norm=norm, origin='lower',
                            extent=[epochs[0], epochs[-1], -0.5, num_subsets - 0.5])
            
            ax.set_title(f'Layer {l+1} Projection Evolution')
            ax.set_ylabel('Walsh Basis Function $\chi_A$')
            ax.set_yticks(np.arange(num_subsets))
            ax.set_yticklabels(self.walsh_param_labels, fontsize=8)

        fig.colorbar(img, ax=axs, label=r'Signal Strength $m_A^{(\ell)}$', shrink=0.8)
        axs[-1].set_xlabel('Epoch')
        axs[-1].set_xscale('log')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.suptitle('Evolution of Layer-wise Projections onto Walsh Basis', fontsize=16)
        plt.savefig(os.path.join(self.save_dir, "layerwise_projection_heatmaps.png"), dpi=300)
        plt.close()

    def plot_correlation_evolution(self):
        """Plots the evolution of the final output correlation."""
        stats = self.correlation_stats
        if not stats['epochs']: return
        fig, ax = plt.subplots(figsize=(12, 8))
        mean, std = np.array(stats['corr_mean']), np.array(stats['corr_std'])
        sem = std / np.sqrt(self.n_ensemble)
        ax.plot(stats['epochs'], mean, color='blue', lw=2, label='Ensemble Mean Correlation')
        ax.fill_between(stats['epochs'], mean - sem, mean + sem, color='blue', alpha=0.2)
        ax.set_title('Final Output Correlation with Target')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Pearson Correlation')
        ax.set_xscale('log'); ax.grid(True, which="both", linestyle=':'); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "correlation_evolution.png"), dpi=300)
        plt.close()

    def plot_all(self):
        print("\n" + "="*80 + "\nGenerating all plots...")
        self.plot_correlation_evolution()
        self.plot_layerwise_projection_heatmaps()
        print("All plots saved successfully!")

# --- Main Execution Block ---
if __name__ == "__main__":
    config = {
        'd': 25,
        'k': 4,
        'depth': 4,
        'hidden_widths': [256, 256, 256, 256],
        'max_epochs': 40000,
        'batch_size': 1024,
        'learning_rate': 0.01,
        'tracker': 200,
        'n_ensemble': 8,
        'weight_decay': 1e-5,
        'device_id': 0,
        'save_dir': "/home/goring/OnlineSGD/results_ana/2006_parity_relu_deepSGD_35_4_d4_256_2"
    }
    experiment = DeepParityExperiment(**config)
    experiment.train(early_stop_corr=0.995)
    print("\nAll analyses complete.")