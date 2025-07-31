import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import matplotlib.cm as cm


# --- Deep Model Definition (Corrected) ---
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
        """
        Performs the forward pass and returns both the final output and all
        intermediate representations that serve as INPUTS to the next layer.
        """
        h = x
        # pre_activations will store [x, h^(0), h^(1), ..., h^(depth-1)]
        pre_activations = [x] 
        for i in range(self.depth):
            z = self.hidden_layers[i](h)
            h = self.activation_fn(z)
            pre_activations.append(h)
            
        final_representation = h
        output = torch.matmul(final_representation, self.readout_a) / np.sqrt(self.hidden_widths[-1])
        
        # --- THIS IS THE CORRECTED LINE ---
        # The full list has length (depth + 1) and is what the analysis function needs.
        return output, pre_activations
    
# --- New Experiment Class for Local "RG-Step" Analysis ---
class DeepRGExperiment:
    """
    Manages training and analysis for a deep network ensemble.
    Focuses on the local, layer-to-layer "RG-step" analysis by projecting
    h^(l) onto the eigenbasis of the kernel of h^(l-1).
    """
    def __init__(self, d=100, k=4, depth=3, hidden_widths=None,
                 learning_rate=0.001, max_epochs=50000, batch_size=1024,
                 n_ensemble=8, tracker=500, weight_decay=1e-5,
                 device_id=0, save_dir="deep_rg_analysis"):

        # --- Hyperparameters ---
        self.d, self.k, self.depth = d, k, depth
        self.hidden_widths = hidden_widths if hidden_widths is not None else [512] * depth
        self.learning_rate, self.max_epochs, self.batch_size = learning_rate, max_epochs, batch_size
        self.n_ensemble, self.tracker, self.weight_decay = n_ensemble, tracker, weight_decay
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # --- Analysis Parameters ---
        self.analysis_batch_size = 4096  # Batch size for the expensive kernel calculation
        self.k_top_eigenvectors = 100    # Number of top eigenvectors to track

        # --- Device Configuration ---
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        print(f"DeepRGExperiment initialized on {self.device} for d={d}, k={k}, depth={depth}")

        # --- Fixed Dataset for Analysis ---
        self.X_analysis = torch.tensor(np.random.choice([-1, 1], size=(self.analysis_batch_size, d)), dtype=torch.float32).to(self.device)

        # --- Ensemble Initialization ---
        self.models = [DeepNet(d, self.hidden_widths).to(self.device) for _ in range(n_ensemble)]
        self.optimizers = [torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) for model in self.models]
        self.criterion = nn.MSELoss()

        # --- Data Storage for Local Projections ---
        # A list of dictionaries, one for each interface l-1 -> l
        self.local_projection_stats = [{'epochs': [], 'projection_mean': [], 'projection_std': []} for _ in range(self.depth)]

    def _target_function(self, x):
        """The k-sparse parity teacher function."""
        return torch.prod(x[:, :self.k], dim=1)

    def _compute_local_projections(self, epoch):
        """
        The core "RG-step" analysis. For each layer l, projects its representation
        h^(l) onto the eigenbasis of the kernel of the previous layer's representation h^(l-1).
        """
        print(f"\n--- Epoch {epoch}: Starting local projection analysis ---")
        
        with torch.no_grad():
            for l in range(self.depth): # Iterate over interfaces l-1 -> l
                print(f"Analyzing interface: Layer {l} -> Layer {l+1}")
                # Store projections for this interface from all models
                interface_projections_ensemble = torch.zeros(self.n_ensemble, self.k_top_eigenvectors, device=self.device)

                for model_idx, model in enumerate(self.models):
                    model.eval()
                    # Get representations h^(l-1) and h^(l)
                    _, prev_layer_reps = model(self.X_analysis)
                    h_prev = prev_layer_reps[l] # Input to layer l, output of l-1
                    h_curr = prev_layer_reps[l+1] # Output of layer l
                    
                    # 1. Compute the kernel of the PREVIOUS layer's representation
                    # Normalize by width for stability
                    K_prev = torch.matmul(h_prev, h_prev.T) / h_prev.shape[1]

                    # 2. Eigendecompose the kernel to get the adaptive basis
                    try:
                        eigvals, eigvecs = torch.linalg.eigh(K_prev)
                        # Sort eigenvectors by eigenvalue in descending order
                        sorted_indices = torch.argsort(eigvals, descending=True)
                        basis_vectors = eigvecs[:, sorted_indices[:self.k_top_eigenvectors]]
                    except torch.linalg.LinAlgError:
                        print(f"Warning: Eigendecomposition failed for model {model_idx}, layer {l}. Skipping.")
                        continue # Use zeros for this entry

                    # 3. Project the CURRENT layer's representation onto this basis.
                    # The representation h_curr is (batch, width). We want to see how the
                    # activation of each neuron (a column vector) projects onto the basis.
                    # Projection matrix shape: (k_top, width_curr)
                    projection_matrix = torch.matmul(basis_vectors.T, h_curr)

                    # 4. Average these projections across all neurons in the current layer
                    # This gives the final strength vector for this model at this interface
                    avg_projections = torch.mean(projection_matrix, dim=1)
                    interface_projections_ensemble[model_idx, :] = avg_projections

                # Store the mean and std dev across the ensemble for this interface
                stats = self.local_projection_stats[l]
                stats['epochs'].append(epoch)
                stats['projection_mean'].append(torch.mean(interface_projections_ensemble, dim=0).cpu().numpy())
                stats['projection_std'].append(torch.std(interface_projections_ensemble, dim=0).cpu().numpy())

    def train(self, early_stop_corr=0.999):
        """Main training loop."""
        print(f"Starting training for {self.n_ensemble} deep networks...")
        # Initial measurement at epoch 0
        self._compute_local_projections(0)

        for epoch in tqdm(range(1, self.max_epochs + 1), desc="Training Ensemble"):
            X_batch = torch.tensor(np.random.choice([-1, 1], size=(self.batch_size, self.d)), dtype=torch.float32).to(self.device)
            y_batch = self._target_function(X_batch)
            
            for i in range(self.n_ensemble):
                self.models[i].train()
                outputs, _ = self.models[i](X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()
            
            if epoch % self.tracker == 0 or epoch == self.max_epochs:
                self._compute_local_projections(epoch)

        print("\n" + "="*80 + "\nTraining complete!")
        self.plot_all()

    def analyze_final_state_relevance(self):
        """
        Performs a post-hoc analysis on the final trained models to connect
        local feature utilization with global task relevance.
        """
        print("\n" + "="*80)
        print("Starting post-hoc analysis: Feature Relevance vs. Utilization...")
        
        # We will analyze the first model in the ensemble as a representative
        model = self.models[0]
        model.eval()

        # Get the final representations from the trained model
        with torch.no_grad():
            _, final_reps = model(self.X_analysis)
            y_target = self._target_function(self.X_analysis)

        all_interfaces_data = []

        # We analyze the interface l-1 -> l, so we loop l from 1 to depth
        for l in range(1, self.depth):
            print(f"Analyzing interface: Layer {l} -> Layer {l+1}")
            h_prev = final_reps[l]    # Output of layer l
            h_curr = final_reps[l+1]  # Output of layer l+1
            
            # 1. Get the Basis: Eigendecompose the kernel of the previous layer
            with torch.no_grad():
                K_prev = torch.matmul(h_prev, h_prev.T) / h_prev.shape[1]
                try:
                    eigvals, eigvecs = torch.linalg.eigh(K_prev)
                    sorted_indices = torch.argsort(eigvals, descending=True)
                    basis_vectors = eigvecs[:, sorted_indices[:self.k_top_eigenvectors]]
                except torch.linalg.LinAlgError:
                    print(f"Warning: Eigendecomposition failed for layer {l}. Skipping interface.")
                    continue

                # 2. Calculate Relevance of each basis vector to the final target
                # Shape: (k_top,)
                relevance = torch.abs(torch.matmul(basis_vectors.T, y_target))

                # 3. Calculate Utilization of each basis vector by the next layer
                projection_matrix = torch.matmul(basis_vectors.T, h_curr)
                utilization = torch.abs(torch.mean(projection_matrix, dim=1))

            all_interfaces_data.append({
                'relevance': relevance.cpu().numpy(),
                'utilization': utilization.cpu().numpy(),
                'rank': np.arange(self.k_top_eigenvectors)
            })

        # Pass the collected data to the new plotting function
        self.plot_relevance_vs_utilization(all_interfaces_data)

    def plot_relevance_vs_utilization(self, analysis_data):
        """
        Creates a scatter plot of feature relevance vs. utilization for each layer interface.
        """
        print("Plotting feature relevance vs. utilization...")
        num_interfaces = len(analysis_data)
        if num_interfaces == 0:
            return
            
        fig, axs = plt.subplots(1, num_interfaces, figsize=(7 * num_interfaces, 6), squeeze=False)
        axs = axs.flatten()
        
        # Find global max values for consistent axis scaling
        max_rel = max(np.max(data['relevance']) for data in analysis_data) * 1.1
        max_util = max(np.max(data['utilization']) for data in analysis_data) * 1.1

        for i, data in enumerate(analysis_data):
            ax = axs[i]
            # Use the rank of the eigenvector for coloring
            colors = data['rank'] 
            
            sc = ax.scatter(data['relevance'], data['utilization'], c=colors, cmap='viridis', alpha=0.7)
            
            # Fit and plot a regression line to guide the eye
            m, b = np.polyfit(data['relevance'], data['utilization'], 1)
            ax.plot(np.array([0, max_rel]), m*np.array([0, max_rel]) + b, color='red', linestyle='--', label=f'Fit: y={m:.2f}x+{b:.2f}')

            ax.set_title(f'Layer {i+1} -> Layer {i+2} Interface')
            ax.set_xlabel(f'Relevance of Layer {i+1} Feature\n(Correlation with Final Target)')
            ax.set_ylabel(f'Utilization by Layer {i+2}\n(Projection Strength)')
            ax.set_xlim(0, max_rel)
            ax.set_ylim(0, max_util)
            ax.set_yscale('symlog', linthresh=1e-4)
            ax.set_yscale('symlog', linthresh=1e-4)
            ax.grid(True, linestyle=':')
            ax.legend()

        fig.colorbar(sc, ax=axs, label='Feature Rank (Eigenvector Index)')
        fig.suptitle('Network Intelligence: Connecting Local Processing to Global Goals', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.save_dir, "relevance_vs_utilization.png"), dpi=300)
        plt.close()

    def plot_local_projection_evolution(self):
        """Plots the evolution of the local, layer-to-layer projection strengths."""
        print("Plotting local 'RG-step' projection evolution...")
        
        fig, axs = plt.subplots(self.depth, 1, figsize=(15, 7 * self.depth), sharex=True, squeeze=False)
        axs = axs.flatten()
        colors = cm.get_cmap('viridis', self.k_top_eigenvectors)
        n_sqrt = np.sqrt(self.n_ensemble)

        for l in range(self.depth):
            stats = self.local_projection_stats[l]
            if not stats['epochs']: continue
            
            ax = axs[l]
            epochs = stats['epochs']
            # Shape: (num_epochs, k_top)
            means = np.array(stats['projection_mean'])
            stds = np.array(stats['projection_std'])

            for i in range(self.k_top_eigenvectors):
                mean_line = means[:, i]
                sem = stds[:, i] / n_sqrt
                ax.plot(epochs, mean_line, color=colors(i), lw=1.5)
                ax.fill_between(epochs, mean_line - sem, mean_line + sem, color=colors(i), alpha=0.1)

            layer_name = "Input" if l == 0 else f"Layer {l}"
            ax.set_title(f'Projection of Layer {l+1} onto Eigenbasis of {layer_name} Kernel')
            ax.set_ylabel('Projection Strength')
            ax.set_xscale('log')
            ax.set_yscale('symlog', linthresh=1e-4)
            ax.grid(True, which="both", linestyle=':')

        axs[-1].set_xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "local_projection_evolution.png"), dpi=300)
        plt.close()

    def plot_all(self):
        print("\n" + "="*80 + "\nGenerating all plots...")
        self.plot_local_projection_evolution()
        print("All plots saved successfully!")

# --- Main Execution Block ---
if __name__ == "__main__":
    config = {
        'd': 30,
        'k': 4,
        'depth': 4,
        'hidden_widths': [256, 256, 256,256],
        'max_epochs': 10000,
        'batch_size': 1024,
        'learning_rate': 0.005,
        'tracker': 500, # Track less frequently due to expensive analysis
        'n_ensemble': 16, # Use a smaller ensemble for faster testing
        'weight_decay': 1e-5,
        'device_id': 0,
        'save_dir': "/home/goring/OnlineSGD/results_ana/2006_parity_relu_deepSGD_30_4_d4_256_2_RG"
    }

    experiment = DeepRGExperiment(**config)
    experiment.train()
    experiment.analyze_final_state_relevance()

    print("\nAll analyses complete.")