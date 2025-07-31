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
import math
from matplotlib.colors import LogNorm, SymLogNorm
from math import factorial
from matplotlib.colors import SymLogNorm
import copy


# --- Model Definition ---
# --- Model Definition ---
class OneLayerTanhNet(torch.nn.Module):
    """A one-hidden-layer neural network with a Tanh activation and a linear output layer."""
    # CHANGED: Added a_init_scale parameter
    def __init__(self, input_dim, hidden_width, a_init_scale=1.0, W1_grad=True, a_grad=True):
        super(OneLayerTanhNet, self).__init__()
        # Initialize layers with scaling
        self.W1 = torch.nn.Parameter(torch.randn(hidden_width, input_dim) / np.sqrt(input_dim))
        
        # CHANGED: Use a_init_scale to control the initial variance of 'a'
        self.a = torch.nn.Parameter(torch.randn(hidden_width) * a_init_scale / np.sqrt(hidden_width))
        
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


# --- START: Paste this entire block into the EnsembleParityExperiment class ---


# --- Main Experiment Class ---
class EnsembleParityExperiment:
    """
    Manages training and analysis for an ensemble of one-layer networks learning a parity function.
    Each network is trained until individual convergence.
    """
    def __init__(self, d=30, k=6, M1=512, learning_rate=0.01, max_epochs=50000,
                 batch_size=512, n_ensemble=10, n_parallel=5, tracker=100,
                 weight_decay_W1=0.0, weight_decay_a=0.0, top_n_neurons=100, 
                 kernel_n_samples=100000, device_id=None, save_dir="parity_ensemble_analysis",
                 L_max=50, gamma=1.0, a_init_scale=1.0, test_set_size=20000, 
                 train_set_size=50000):
        
        # --- Hyperparameters ---
        self.d = d; self.k = k; self.M1 = M1; self.learning_rate = learning_rate
        self.max_epochs = max_epochs; self.batch_size = batch_size; self.n_ensemble = n_ensemble
        self.n_parallel = min(n_parallel, n_ensemble); self.tracker = tracker
        self.weight_decay_W1 = weight_decay_W1; self.weight_decay_a = weight_decay_a
        self.top_n_neurons = top_n_neurons; self.kernel_n_samples = kernel_n_samples
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.raw_projection_stats = {'epochs': [], 'projections_mean': [], 'projections_std': []}
        self.gamma = gamma
        self.a_init_scale = a_init_scale
        self.train_set_size = train_set_size # <-- NEW: Store the hyperparameter

        # --- Device Configuration ---
        if device_id is not None and torch.cuda.is_available(): self.device = torch.device(f"cuda:{device_id}")
        else: self.device = torch.device("cpu")
        print(f"EnsembleParityExperiment initialized on {self.device}")
        
        # --- Ensemble Initialization ---
        self.models = [OneLayerTanhNet(d, M1, a_init_scale=self.a_init_scale).to(self.device) for _ in range(n_ensemble)]
        self.optimizers = [self._create_optimizer(model) for model in self.models]
        self.criterion = torch.nn.MSELoss()
        
        self.loss_stats = {
            'epochs': [],
            'train_loss_mean': [], 'train_loss_std': [],
            'test_loss_mean': [], 'test_loss_std': []
        }

        # --- Data and State Storage ---
        self.is_converged = [False] * self.n_ensemble
        self.final_epochs = [0] * self.n_ensemble
        self.metrics = [{'epochs': [], 'correlation': [], 'phase_transition_epoch': None} for _ in range(n_ensemble)]
        self.final_weights = {'W1': [], 'a': []}

        # ... after the self.macro_observables definition
        self.drift_diagnostics = {
            'epochs': [],
            'drift_I_mean': [], 'drift_I_std': [],
            'drift_J_mean': [], 'drift_J_std': [],
        }
        self.I0_vals = None  # To store initial values of I = q_u + q_v
        self.J0_vals = None  # To store initial values of J = r / sqrt(q_u*q_v)
                
        self.L_max = L_max
        self.hermite_evolution_stats = {
        'epochs': [],
        'coeffs_mean': [], # Will store a list of numpy arrays (one array per epoch)
        'coeffs_std': []
    }
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
        self.explained_variance_snapshots = {'start': [], 'end': []}
        # --- ADD THIS ---
        self.kernel_diag_offdiag_stats = {'epochs': [], 'diag_norm_mean': [], 'diag_norm_std': [], 'offdiag_norm_mean': [], 'offdiag_norm_std': []}
        # --- END ADD ---
        self.eigenvector_projection_analysis = {
            'projection_matrix_mean': None, # Will store the final projection matrix, averaged over the ensemble
            'projection_matrix_std': None
        }
        # --- Fixed Datasets ---
        

        print(f"Creating fixed training set with {self.train_set_size} samples.")
        self.X_train = torch.tensor(np.random.choice([-1, 1], size=(self.train_set_size, d)), dtype=torch.float32).to(self.device)
        self.y_train = self._target_function(self.X_train)
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(test_set_size, d)), dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)
        self.X_kernel = torch.tensor(np.random.choice([-1, 1], size=(self.kernel_n_samples, d)), dtype=torch.float32).to(self.device)


        self.macro_observables = {
    'epochs': [],
    'm_mean': [], 'm_std': [],
    'qu_mean': [], 'qu_std': [],
    'qv_mean': [], 'qv_std': [],
    'r_mean': [], 'r_std': [],
    's_mean': [], 's_std': []
        }

        # In __init__
        self.hermite_evolution_stats = {
            'epochs': [],
            'coeffs_mean': [],
            'coeffs_std': [],
            'coeffs_all_models': [] # <<<--- ADD THIS LINE
        }



        # --- NEW: Walsh Order Parameter (`m_A`) Analysis ---
        print("Generating Walsh projection subsets...")
        self.walsh_n_samples = 200000  # Large sample count for low variance, as requested
        self.X_walsh = torch.tensor(np.random.choice([-1, 1], size=(self.walsh_n_samples, d)), dtype=torch.float32).to(self.device)
        self.walsh_param_sets = self._generate_walsh_subsets()
        self.walsh_param_labels = self._generate_walsh_labels()

        # Storage for single-model m_A values
        self.walsh_single_model_stats = [{'epochs': [], 'm_A': []} for _ in range(self.n_ensemble)]
        # Storage for ensemble-averaged m_A values
        self.walsh_ensemble_stats = {'epochs': [], 'm_A_mean': [], 'm_A_std': []}

        self.neuron_level_stats = [
            {'epochs': [], 'activation_correlations': [], 'a_values': []}
            for _ in range(self.n_ensemble)
        ]
        # --- END NEW ---

        # --- START: Paste this entire block into the EnsembleParityExperiment class ---

    # --- In __init__, ADD these lines for Cavity Method Diagnostics ---
    # (This can be placed with the other metric initializations)
        self.initial_models = None
        # Use a set to avoid duplicates if self.k is in the list
        # --- Find this block in your __init__ method and apply the modification ---

        # --- Cavity Method Diagnostics ---
    
        # --- MODIFICATION: Calculate for all k up to L_max or d ---
        max_k_chi = min(self.d, self.L_max) 
        self.k_values_for_chi = list(range(1, max_k_chi + 1))
        # --- END MODIFICATION ---
        self.num_chi_subsets = 30 # Number of random subsets to average over for each k
        self.chi_X_samples = 10000 # Number of x samples for Monte Carlo integration of J_A
        self.X_chi = torch.tensor(np.random.choice([-1, 1], size=(self.chi_X_samples, d)), dtype=torch.float32).to(self.device)
        self.susceptibility_subsets = {}
        self.susceptibility_stats = {
            'epochs': [],
            'k_values': self.k_values_for_chi,
            'chi_k_current_mean': [], 'chi_k_current_std': [],
            'chi_k_initial_mean': None, 'chi_k_initial_std': None
        }
    # --- END of __init__ additions ---

    # --- In train(), ADD a call to initialize diagnostics ---
    
        
    # --- In _compute_all_metrics(), ADD a call to the new computation ---

    def _compute_loss_metrics(self, epoch):
        """
        Calculates and stores the train and test MSE loss for the entire ensemble.
        Uses fixed, large evaluation sets for stable measurements.
        """
        train_losses, test_losses = [], []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                # --- THIS IS THE FIX ---
                # Calculate training loss on the fixed training set
                # Changed self.X_train_eval -> self.X_train
                # Changed self.y_train_eval -> self.y_train
                train_preds = model(self.X_train)
                train_loss = self.criterion(train_preds.squeeze(), self.y_train)
                train_losses.append(train_loss.item())
                # --- END FIX ---

                # Calculate test loss on the fixed test set
                test_preds = model(self.X_test)
                test_loss = self.criterion(test_preds.squeeze(), self.y_test)
                test_losses.append(test_loss.item())

        # Store the mean and standard deviation of the losses across the ensemble
        self.loss_stats['epochs'].append(epoch)
        self.loss_stats['train_loss_mean'].append(np.mean(train_losses))
        self.loss_stats['train_loss_std'].append(np.std(train_losses))
        self.loss_stats['test_loss_mean'].append(np.mean(test_losses))
        self.loss_stats['test_loss_std'].append(np.std(test_losses))

    def plot_train_test_loss_evolution(self):
        """
        Plots the evolution of the mean training and testing loss for the ensemble,
        with ribbons representing the standard error of the mean (SEM).
        """
        print("Plotting train and test loss evolution...")
        stats = self.loss_stats
        if not stats['epochs']:
            print("Loss data not available to plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = stats['epochs']
        n_sqrt = np.sqrt(self.n_ensemble)

        # --- Plot Training Loss ---
        mean_train = np.array(stats['train_loss_mean'])
        std_train = np.array(stats['train_loss_std'])
        sem_train = std_train / n_sqrt
        
        line_train, = ax.plot(epochs, mean_train, marker='', linestyle='-', label='Train Loss')
        ax.fill_between(epochs, mean_train - sem_train, mean_train + sem_train, color=line_train.get_color(), alpha=0.2)
        
        # --- Plot Test Loss ---
        mean_test = np.array(stats['test_loss_mean'])
        std_test = np.array(stats['test_loss_std'])
        sem_test = std_test / n_sqrt

        line_test, = ax.plot(epochs, mean_test, marker='', linestyle='--', label='Test Loss')
        ax.fill_between(epochs, mean_test - sem_test, mean_test + sem_test, color=line_test.get_color(), alpha=0.2)

        # --- Formatting ---
        ax.set_title('Ensemble Mean Train & Test Loss (MSE)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Squared Error (MSE)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle=':')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/train_test_loss_evolution.png", dpi=300)
        plt.close()
    
        
    def _compute_neuron_level_metrics(self, epoch):
        """
        Calculates and stores per-neuron diagnostics for each model, specifically:
        1. The correlation of each neuron's activation phi(w_i^T*x) with the target y.
        2. The value of each output weight a_i.
        """
        # A smaller, fixed set of samples for this potentially slow calculation
        X_corr_test = self.X_test 
        y_corr_test = self.y_test

        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                # Get neuron activations on the test set. Shape: (n_samples, M1)
                activations = model.get_activations(X_corr_test)

                # Store the current output weights
                self.neuron_level_stats[i]['a_values'].append(model.a.detach().cpu().numpy().copy())
                self.neuron_level_stats[i]['epochs'].append(epoch)

                # Calculate correlation for each neuron
                correlations = []
                for j in range(self.M1):
                    neuron_activation = activations[:, j]
                    # Check for zero variance to avoid errors
                    if torch.var(neuron_activation) > 1e-8:
                        corr_matrix = torch.corrcoef(torch.stack([neuron_activation, y_corr_test]))
                        correlations.append(corr_matrix[0, 1].item())
                    else:
                        correlations.append(0.0)
                
                self.neuron_level_stats[i]['activation_correlations'].append(correlations)

    
    def plot_neuron_correlation_and_a_weight_heatmaps(self):
        """
        For each model, plots three heatmaps:
        1. The correlation of each neuron's activation with the target function over epochs.
        2. The absolute value of each neuron's output weight 'a' over epochs.
        3. The product of (1) and (2), showing the 'effective contribution' of each neuron.
        """
        print("Plotting neuron-level correlation, 'a' weight, and contribution heatmaps...")
        for i in range(self.n_ensemble):
            stats = self.neuron_level_stats[i]
            if not stats['epochs']:
                print(f"No neuron-level data to plot for model {i}.")
                continue

            # CHANGED: Create a figure with 3 subplots instead of 2
            fig, axs = plt.subplots(1, 3, figsize=(28, 8), sharey=True)
            fig.suptitle(f'Model {i}: Neuron-Level Diagnostics', fontsize=16)

            epochs = stats['epochs']
            
            # --- Prepare Data ---
            # Data shape: (num_epochs, M1) -> Transpose to (M1, num_epochs) for plotting
            corr_data = np.array(stats['activation_correlations']).T
            a_data_abs = np.abs(np.array(stats['a_values']).T)

            # --- Heatmap 1: Activation-Target Correlation ---
            ax = axs[0]
            img1 = ax.imshow(corr_data, aspect='auto', cmap='bwr', vmin=-1.0, vmax=1.0, origin='lower',
                             extent=[epochs[0], epochs[-1], 0, self.M1])
            ax.set_title('Activation-Target Correlation $\\mathrm{corr}(\\phi(w_i^T x), y)$')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Neuron Index')
            fig.colorbar(img1, ax=ax, label='Correlation', shrink=0.8)

            # --- Heatmap 2: Absolute 'a' weights ---
            ax = axs[1]
            # Use LogNorm to see both large and small weights clearly. Add a small epsilon to avoid log(0).
            img2 = ax.imshow(a_data_abs, aspect='auto', cmap='viridis', norm=LogNorm(vmin=1e-3), origin='lower',
                             extent=[epochs[0], epochs[-1], 0, self.M1])
            ax.set_title('Output Weight Magnitude $|a_i|$')
            ax.set_xlabel('Epoch')
            fig.colorbar(img2, ax=ax, label='Weight Magnitude (log scale)', shrink=0.8)

            # --- NEW: Heatmap 3: Effective Contribution (Correlation * |a|) ---
            ax = axs[2]
            effective_contribution = corr_data * a_data_abs
            
            # Find a symmetric color limit for the diverging colormap
            vmax = np.max(np.abs(effective_contribution))
            
            img3 = ax.imshow(effective_contribution, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax, origin='lower',
                             extent=[epochs[0], epochs[-1], 0, self.M1])
            ax.set_title('Effective Contribution $\\mathrm{corr}(\\phi, y) \\times |a_i|$')
            ax.set_xlabel('Epoch')
            fig.colorbar(img3, ax=ax, label='Contribution Strength', shrink=0.8)
            # --- END NEW ---

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # CHANGED: Updated filename to reflect new content
            plt.savefig(f"{self.save_dir}/neuron_diagnostics_heatmap_model_{i}.png", dpi=300)
            plt.close(fig)


    def plot_initial_vs_final_weight_dist_per_model(self):
        """
        For each model, plots overlaid histograms of the initial and final
        distributions for both the W1 and 'a' weights, using a log scale
        on the y-axis to emphasize outliers.
        """
        print("Plotting initial vs. final weight distributions for each model...")
        if self.initial_models is None or not self.final_weights['W1']:
            print("Cannot plot weight distributions: initial or final weights not available.")
            return

        for i in range(self.n_ensemble):
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Model {i}: Initial vs. Final Weight Distributions', fontsize=16)

            # --- Get Weights ---
            initial_W1 = self.initial_models[i].W1.detach().cpu().numpy().flatten()
            initial_a = self.initial_models[i].a.detach().cpu().numpy().flatten()
            final_W1 = self.final_weights['W1'][i].flatten()
            final_a = self.final_weights['a'][i].flatten()

            # --- Plot 1: W1 Weights ---
            ax = axs[0]
            sns.histplot(initial_W1, bins=100, ax=ax, color='blue', alpha=0.6, label='Initial', stat='density')
            sns.histplot(final_W1, bins=100, ax=ax, color='red', alpha=0.6, label='Final', stat='density')
            ax.set_yscale('log')
            ax.set_title(f'$W_1$ Weights (N={len(initial_W1)})')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density (log scale)')
            ax.grid(True, linestyle=':')
            ax.legend()

            # --- Plot 2: 'a' Weights ---
            ax = axs[1]
            sns.histplot(initial_a, bins=50, ax=ax, color='blue', alpha=0.6, label='Initial', stat='density')
            sns.histplot(final_a, bins=50, ax=ax, color='red', alpha=0.6, label='Final', stat='density')
            ax.set_yscale('log')
            ax.set_title(f'$a$ Weights (N={len(initial_a)})')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density (log scale)')
            ax.grid(True, linestyle=':')
            ax.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"{self.save_dir}/weight_distribution_model_{i}.png", dpi=300)
            plt.close(fig)


    # --- ADD THE FOLLOWING NEW METHODS ANYWHERE INSIDE THE CLASS ---

    def _initialize_cavity_method_diagnostics(self):
        """Initializes objects needed for cavity method analysis."""
        print("Initializing cavity method diagnostics (storing initial weights, generating subsets)...")
        # Store a deep copy of the initial models to represent P_0
        self.initial_models = [copy.deepcopy(m) for m in self.models]
        # Generate the random subsets for susceptibility calculation
        self._generate_susceptibility_subsets()
        # Pre-compute the initial susceptibility chi_k(0)
        self._compute_initial_susceptibility()

    def _generate_susceptibility_subsets(self):
        """
        Generates and stores a list of random subsets A for each cardinality k
        that we want to measure chi_k for.
        """
        all_indices = list(range(self.d))
        for k_val in self.k_values_for_chi:
            self.susceptibility_subsets[k_val] = []
            # Ensure we don't try to generate more unique subsets than exist
            max_possible = math.comb(self.d, k_val)
            num_to_gen = min(self.num_chi_subsets, max_possible)
            
            existing_subsets = set()
            while len(self.susceptibility_subsets[k_val]) < num_to_gen:
                new_subset = tuple(sorted(np.random.choice(all_indices, size=k_val, replace=False).tolist()))
                if new_subset not in existing_subsets:
                    existing_subsets.add(new_subset)
                    self.susceptibility_subsets[k_val].append(torch.tensor(new_subset, device=self.device, dtype=torch.long))
        print(f"Generated random subsets for susceptibility k values: {self.k_values_for_chi}")


    # --- Replace the old plot_susceptibility_evolution_heatmap() with this robust version ---

    def plot_susceptibility_evolution_heatmap(self):
        """
        Plots the evolution of the susceptibility chi_k(t) as a heatmap.
        This version uses robust plotting logic to handle tricky axes and color scaling.
        """
        print("Plotting susceptibility (chi_k) evolution as a heatmap...")
        stats = self.susceptibility_stats
        if not stats['epochs'] or len(stats['epochs']) < 2:
            print("Susceptibility data not available or insufficient to plot heatmap.")
            return

        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Data has shape (num_epochs, num_k_values)
        heatmap_data = np.stack(stats['chi_k_current_mean'], axis=0)
        
        # Transpose for plotting: (num_k_values, num_epochs)
        heatmap_data = heatmap_data.T 

        # Use SymLogNorm for robustness around zero and to avoid log(0) errors.
        # It creates a linear region for small values and a log scale for large ones.
        vmax = np.max(np.abs(heatmap_data))
        linthresh = max(1e-9, vmax / 5000) # Set a small linear threshold
        # Since chi_k is non-negative, we set vmin=0
        norm = SymLogNorm(linthresh=linthresh, vmin=0, vmax=vmax, base=10) 

        epochs = stats['epochs']
        k_values = stats['k_values']
        
        # Plotting against integer indices (0, 1, 2...) for robustness
        img = ax.imshow(
            heatmap_data,
            aspect='auto',
            cmap='magma',
            norm=norm,
            origin='lower',
        )
        
        # Manually set the ticks and labels for the axes
        # Y-axis (k-values)
        ax.set_yticks(np.arange(len(k_values)))
        ax.set_yticklabels(k_values)
        
        # X-axis (Epochs) - Create human-readable, non-overlapping labels
        tick_indices = np.linspace(0, len(epochs) - 1, num=6, dtype=int)
        tick_labels = [f"{epochs[i]:.1e}" for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels)

        # Highlight the true k value with a horizontal line
        # Find the integer index corresponding to the teacher k
        if self.k in k_values:
            try:
                teacher_k_index = k_values.index(self.k)
                ax.axhline(y=teacher_k_index, color='cyan', linestyle='--', label=f'Teacher k = {self.k}')
            except ValueError:
                pass # k is not in the list of k_values being plotted
    
        ax.set_title(r'Evolution of Susceptibility $\chi_k$')
        ax.set_ylabel('Subset Size $k$')
        ax.set_xlabel('Epoch')
        
        fig.colorbar(img, ax=ax, label=r'Susceptibility Value $\chi_k$ (SymLog scale)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/susceptibility_evolution_heatmap.png", dpi=300)
        plt.close()



    def _compute_single_model_chi_k(self, model):
        """
        Helper function to compute chi_k for all specified k values for a single model.
        This is the core computational part of the susceptibility calculation.
        
        Returns a numpy array of chi_k values, one for each k in self.k_values_for_chi.
        """
        model.eval()
        a, W1 = model.a.detach(), model.W1.detach()
        
        # Calculate neuron activations phi(w_i^T * x) on the dedicated test set X_chi
        # Shape: (chi_X_samples, M1)
        all_phi = model.get_activations(self.X_chi)
        
        chi_k_values_for_model = []
        with torch.no_grad():
            for k_val in self.k_values_for_chi:
                chi_A_values_for_k = []
                # Loop over the pre-generated random subsets A of size k_val
                for A_subset in self.susceptibility_subsets[k_val]:
                    # 1. Calculate J_A(w_i) = E_x[phi(w_i^T*x) * chi_A(x)] for all neurons i
                    # chi_A(x) for the current subset A
                    chi_A_x = self._target_function_general(self.X_chi, A_subset) # Shape: (chi_X_samples,)
                    
                    # J_A(w_i) for all neurons i. Shape: (M1,)
                    J_A_neurons = torch.einsum('ni,n->i', all_phi, chi_A_x) / self.chi_X_samples
                    
                    # 2. Calculate chi_A = M1 * <a^2 * J_A(w)^2>_neurons
                    term = (a**2 * J_A_neurons**2).mean()
                    chi_A = self.M1 * term
                    chi_A_values_for_k.append(chi_A.cpu().item())
                
                # 3. Average over all the random subsets A of size k_val
                # This gives the final estimate for chi_k for this model.
                mean_chi_k = np.mean(chi_A_values_for_k)
                chi_k_values_for_model.append(mean_chi_k)
                
        return np.array(chi_k_values_for_model)

    def _compute_initial_susceptibility(self):
        """Calculates chi_k(0) once using the initial model weights."""
        all_runs_chi_k_initial = []
        for initial_model in self.initial_models:
            chi_k_values = self._compute_single_model_chi_k(initial_model)
            all_runs_chi_k_initial.append(chi_k_values)
            
        # Average over the ensemble
        stacked_results = np.stack(all_runs_chi_k_initial, axis=0)
        self.susceptibility_stats['chi_k_initial_mean'] = np.mean(stacked_results, axis=0)
        self.susceptibility_stats['chi_k_initial_std'] = np.std(stacked_results, axis=0)
        print(f"Calculated initial susceptibility chi_k(0): {self.susceptibility_stats['chi_k_initial_mean']}")

    def _compute_susceptibility(self, epoch):
        """Calculates chi_k(t) at the current epoch for all models."""
        all_runs_chi_k_current = []
        for model in self.models:
            chi_k_values = self._compute_single_model_chi_k(model)
            all_runs_chi_k_current.append(chi_k_values)
            
        # Average over the ensemble
        stacked_results = np.stack(all_runs_chi_k_current, axis=0)
        
        # Store results
        self.susceptibility_stats['epochs'].append(epoch)
        self.susceptibility_stats['chi_k_current_mean'].append(np.mean(stacked_results, axis=0))
        self.susceptibility_stats['chi_k_current_std'].append(np.std(stacked_results, axis=0))

    def plot_susceptibility_evolution(self):
        """
        Plots the evolution of the susceptibility chi_k(t) and compares it
        to the initial value chi_k(0).
        """
        print("Plotting susceptibility (chi_k) evolution...")
        stats = self.susceptibility_stats
        if not stats['epochs']:
            print("Susceptibility data not available.")
            return

        fig, ax = plt.subplots(figsize=(14, 8))
        epochs = stats['epochs']
        n_sqrt = np.sqrt(self.n_ensemble)
        
        # Stack the list of arrays into a 2D matrix for easier indexing
        # Shape: (num_epochs, num_k_values)
        chi_k_means = np.stack(stats['chi_k_current_mean'], axis=0)
        chi_k_stds = np.stack(stats['chi_k_current_std'], axis=0)
        
        num_k_values = chi_k_means.shape[1]
        colors = cm.get_cmap('plasma', num_k_values)

        for i in range(num_k_values):
            k_val = self.k_values_for_chi[i]
            label_current = fr'$\chi_{k_val}(t)$ (current weights)'
            label_initial = fr'$\chi_{k_val}(0)$ (initial weights)'
            
            # Plot evolving chi_k(t)
            mean = chi_k_means[:, i]
            sem = chi_k_stds[:, i] / n_sqrt # Standard Error of the Mean
            line, = ax.plot(epochs, mean, marker='', linestyle='-', lw=2, color=colors(i), label=label_current)
            ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.15)
            
            # Plot horizontal line for initial chi_k(0)
            if stats['chi_k_initial_mean'] is not None:
                initial_mean = stats['chi_k_initial_mean'][i]
                ax.axhline(y=initial_mean, color=colors(i), linestyle='--', lw=2, label=label_initial)

        ax.set_title(r'Evolution of Susceptibility $\chi_k$ (Ensemble Mean ± SEM)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'Susceptibility Value $\chi_k$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle=':')
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/susceptibility_evolution.png", dpi=300)
        plt.close()



    # --- NEW: Method to generate subsets for Walsh projections ---
    def _generate_walsh_subsets(self):
        """
        Generates all non-empty subsets of the teacher subset S = {0, ..., k-1}
        for computing m_A. The subsets are sorted by size, then lexicographically.
        """
        teacher_indices = list(range(self.k))
        all_subsets = []

        # Generate all combinations for each possible size from 1 to k
        for l in range(1, self.k + 1):
            # itertools.combinations gives tuples, convert them to lists
            for subset_tuple in itertools.combinations(teacher_indices, l):
                all_subsets.append(list(subset_tuple))
                
        # The list is already naturally sorted by size and then content.
        # This ensures the full teacher subset S is always the last one.
        print(f"Generated {len(all_subsets)} subsets of the teacher set S for Walsh analysis.")
        # e.g., for k=3: [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        
        return [torch.tensor(s, device=self.device, dtype=torch.long) for s in all_subsets]

    # --- NEW: Method to create labels for the Walsh plots ---
    def _generate_walsh_labels(self):
        """Generates descriptive LaTeX labels for each Walsh subset."""
        labels = []
        teacher_subset_list = list(range(self.k))
        
        for subset_tensor in self.walsh_param_sets:
            subset = subset_tensor.cpu().numpy().tolist()
            
            # Check if the current subset is the full teacher set
            if subset == teacher_subset_list:
                # Special, more descriptive label for the main target function
                labels.append(f'$m_S$ (k={self.k})')
            else:
                # Create a label like m_{0,2,3} for lower-order subsets
                subset_str = ",".join(map(str, subset))
                labels.append(f'$m_{{{subset_str}}}$')
                
        return labels

    # --- Replace the old plot_walsh_power_spectrum_snapshots() with this robust version ---
    def plot_walsh_power_spectrum_snapshots(self):
        """
        Plots the Walsh power spectrum (m_A^2) for each network at the start
        and end of training. This version uses a robust 'symlog' scale and descriptive labels
        for all subsets of the teacher set.
        """
        print("Plotting Walsh power spectrum snapshots for each model...")
        if not self.walsh_single_model_stats[0]['epochs']:
            print("Walsh order parameter data not available.")
            return
            
        num_models = self.n_ensemble
        ncols = int(np.ceil(np.sqrt(num_models)))
        nrows = int(np.ceil(num_models / ncols))
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True, squeeze=False)
        axs = axs.flatten()
        
        x_pos = np.arange(len(self.walsh_param_sets))
        
        # --- Find the index of the full teacher subset 'S' to highlight it. ---
        teacher_idx = -1
        try:
            teacher_subset_list = list(range(self.k))
            for i, s_tensor in enumerate(self.walsh_param_sets):
                if s_tensor.cpu().numpy().tolist() == teacher_subset_list:
                    teacher_idx = i
                    break
        except Exception as e:
            print(f"Warning: Could not locate teacher subset S in Walsh list: {e}")

        for i in range(num_models):
            ax = axs[i]
            stats = self.walsh_single_model_stats[i]
            if not stats['m_A']: continue
            
            m_A_start = stats['m_A'][0]
            m_A_end = stats['m_A'][-1]
            
            power_start = m_A_start**2
            power_end = m_A_end**2
            
            width = 0.4
            
            bars_start = ax.bar(x_pos - width/2, power_start, width, label='Start (t=0)', color='orange', alpha=0.7)
            bars_end = ax.bar(x_pos + width/2, power_end, width, label='End (final)', color='darkblue', alpha=0.9)
            
            # Highlight the bar corresponding to the full teacher subset S
            if teacher_idx != -1:
                bars_start[teacher_idx].set_color('red')
                bars_start[teacher_idx].set_edgecolor('black')
                bars_end[teacher_idx].set_color('red')
                bars_end[teacher_idx].set_edgecolor('black')
            
            ax.set_title(f'Model {i}')
            ax.set_yscale('symlog', linthresh=1e-7)  
            ax.grid(True, which="both", linestyle=':')

        # --- Configure shared axes and legend using the new, descriptive labels ---
        labels = self.walsh_param_labels
                
        axs[0].set_xticks(x_pos)
        # To avoid clutter, only label a subset of ticks if k is large
        if len(labels) > 25:
            step = max(1, len(labels) // 15)
            tick_indices = list(range(0, len(labels), step))
            # Ensure the last tick (the full teacher subset) is always shown
            if len(labels)-1 not in tick_indices:
                tick_indices.append(len(labels)-1)

            axs[0].set_xticks(tick_indices)
            axs[0].set_xticklabels([labels[j] for j in tick_indices], rotation=90, fontsize=8)
        else:
            axs[0].set_xticklabels(labels, rotation=90, fontsize=8)

        axs[0].legend()
        
        fig.supxlabel('Walsh Basis Function $\\chi_A$', fontsize=14)
        fig.supylabel(r'Power ($m_A^2$) (SymLog Scale)', fontsize=14)
        fig.suptitle('Walsh Power Spectrum Snapshots (Start vs. End of Training)', fontsize=18, y=0.98)
        
        for i in range(num_models, len(axs)):
            axs[i].set_visible(False)
            
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(f"{self.save_dir}/walsh_power_spectrum_snapshots.png", dpi=300)
        plt.close()



    
    @staticmethod
    def _relu_d(ell):
        """Calculates the correct Hermite coefficient d_ell for phi(z)=ReLU(z)."""
        if ell < 0:
            return 0
        if ell == 0:
            return 1 / np.sqrt(2 * np.pi)
        if ell == 1:
            return 0.5
        if ell % 2 != 0:  # Odd l > 1
            return 0.0
        
        # Even l >= 2
        m = ell // 2
        # Note: requires 'from math import factorial'
        # The (-1) factor is often ignored as we look at c_l^2, but we include it for correctness.
        numerator = (-1)**(m - 1)
        denominator = np.sqrt(2 * np.pi) * (2**ell) * factorial(m) * (ell - 1)
        return numerator / denominator

   

    # --- Learning Mechanism Diagnostics (Corrected for Centering) ---
    @staticmethod
    def _hermite(n, z):
        """Probabilists' Hermite polynomials H_n(z) via recursion."""
        if n == 0: return torch.ones_like(z)
        if n == 1: return z
        Hnm2, Hnm1 = torch.ones_like(z), z
        for k_iter in range(2, n + 1):
            Hn = z * Hnm1 - (k_iter - 1) * Hnm2
            Hnm2, Hnm1 = Hnm1, Hn
        return Hn

    @staticmethod
    def _orthonormal_hermite(n, z):
        """Orthonormal version of the Hermite polynomials: H_n(z) / sqrt(n!)."""
        # Note: Requires 'import math' at the top of your script.
        H = EnsembleParityExperiment._hermite(n, z)
        norm = np.sqrt(float(math.factorial(n)))
        return H / norm

    # ------------------------------------------------------------------
#  Correct centred Hermite coefficient  c_ell
# ------------------------------------------------------------------
    # --- Replace your _hermite_coeff_centered with this corrected version ---

    def _hermite_coeff_centered(self, a, W1, x_batch, ell):
        """
        CORRECTLY calculates the Monte-Carlo estimate of the projection:
            c_ell ≈ (1/d_ell) * < f - <f> ,  H_tilde_ell(x_S) >_x
        """
        N = W1.shape[0]
        B = x_batch.shape[0]
        k = self.k

        # 1. Calculate the full student function f(x) on the batch
        z_preactivation = W1 @ x_batch.T  # Shape: (N, B)
        f_raw = (a[:, None] * torch.relu(z_preactivation)).sum(0) / math.sqrt(N) # Shape: (B,)
        
        # 2. Center the function by subtracting its empirical mean
        f = f_raw - f_raw.mean()

        # 3. Define the basis function H_ell based *only* on the relevant inputs x_S
        # The argument to the Hermite polynomial is the normalized projection of x onto the S subspace
        if k > 0:
            one_S = torch.ones(k, device=self.device)
            z_S = (x_batch[:, :k] @ one_S) / np.sqrt(k)
        else: # For k=0, the only relevant basis is H_0(0) = 1
             z_S = torch.zeros(B, device=self.device)
             
        H_basis_tilde = self._orthonormal_hermite(ell, z_S) # Shape: (B,)

        # 4. Compute the projection <f, H_basis> via a dot product, averaged over the batch
        # This is the raw projection, let's call it p_ell
        p_ell = torch.dot(f, H_basis_tilde) / B
        
        # 5. Renormalize by the theoretical activation coefficient d_ell
        d_ell = self._relu_d(ell)
        
        c_ell = p_ell / d_ell if d_ell != 0 else 0.0
        
        return c_ell.item() if torch.is_tensor(c_ell) else c_ell


    
    

    def _m_overlap(self, a, W1):
        oneS = torch.ones(self.k, device=self.device)
        return torch.sum(a * (W1[:, :self.k] @ oneS)) / (len(a) * np.sqrt(self.k))

    def _s_single(self, a, W1):
        g = W1.T @ a
        term1 = torch.sum(g**2)
        term2 = torch.sum(a**2 * torch.sum(W1**2, 1))
        N = len(a)
        return (term1 - term2) / N**2

    def _hidden_pair_kernel_eigs(self, a, W1):
        G = (a[:, None] * a[None, :]) * (W1 @ W1.T)
        eigvals = torch.linalg.eigvalsh(G).cpu().numpy()
        return eigvals[::-1]

    def _run_parity_accuracy_sanity_check(self):
        """NEW: Checks test accuracy after removing the DC offset."""
        print("Running Sanity Check: Parity Accuracy of Centered Function...")
        model = self.models[0]
        a, W1 = model.a.detach(), model.W1.detach()
        
        with torch.no_grad():
            y_test_target = self._target_function(self.X_test)
            f_raw = (a[:,None] * torch.relu(W1 @ self.X_test.T)).sum(0)
            
            # Remove bias and binarise the prediction
            f_pred_centered = torch.sign(f_raw - f_raw.mean())
            
            # Handle potential zeros from sign function
            f_pred_centered[f_pred_centered == 0] = 1 
            
            accuracy = (f_pred_centered == y_test_target).float().mean().item()
            print("-" * 60)
            print(f"Test accuracy of centered and binarized function: {accuracy:.4f}")
            print("-" * 60)

    def _plot_diagnostic_A_m_histogram(self):
        print("Running Diagnostic A: Plotting histogram of 'm' overlap...")
        m_vals = [self._m_overlap(model.a.detach(), model.W1.detach()).cpu().item() for model in self.models]
        plt.figure(figsize=(10, 6)); sns.histplot(m_vals, bins=max(10, self.n_ensemble // 2), kde=True)
        plt.title('Diagnostic A: Distribution of Overlap $m$ Across Seeds'); plt.xlabel('Value of $m$'); plt.ylabel('Count')
        plt.grid(True, linestyle=':'); plt.tight_layout(); plt.savefig(f"{self.save_dir}/diagnostic_A_m_histogram.png", dpi=300); plt.close()

    def _plot_diagnostic_B_hermite_projection(self):
        print("Running Diagnostic B: Plotting CENTERED Hermite projection bar chart...")
        model = self.models[0]
        a, W1 = model.a.detach(), model.W1.detach()
        Nmax = self.k * 2 + 4
        coeffs = torch.zeros(Nmax + 1)
        for n in range(Nmax + 1):
            coeffs[n] = self._hermite_coeff_centered(a, W1, self.X_test, n)

        plt.figure(figsize=(12, 7)); plt.bar(range(Nmax + 1), coeffs.cpu().numpy())
        plt.yscale('symlog', linthresh=1e-6); plt.title(f'Diagnostic B: CENTERED Orthonormal Hermite Projection (Model 0, k={self.k})')
        plt.xlabel('Hermite Order $n$'); plt.ylabel(r'Centered Coefficient $\langle f - \langle f \rangle, \tilde{H}_n \rangle$')
        plt.axvline(x=self.k, color='red', linestyle='--', label=f'$k={self.k}$')
        plt.axvline(x=2 * self.k, color='magenta', linestyle='--', label=f'$2k={2*self.k}$')
        plt.grid(True, which='major', linestyle='-'); plt.grid(True, which='minor', linestyle=':'); plt.legend(); plt.tight_layout()
        plt.savefig(f"{self.save_dir}/diagnostic_B_hermite_projection_centered_barchart.png", dpi=300);
        plt.close()
       

    def _plot_diagnostic_B_hermite_projection(self):
        print("Running Diagnostic B: Plotting CENTERED Hermite projection bar chart...")
        model = self.models[0]
        a, W1 = model.a.detach(), model.W1.detach()
        Nmax = self.k * 2 + 4
        coeffs = torch.zeros(Nmax + 1)
        for n in range(Nmax + 1):
            coeffs[n] = self._hermite_coeff_centered(a, W1, self.X_test, n)

        plt.figure(figsize=(12, 7)); plt.bar(range(Nmax + 1), coeffs.cpu().numpy())
        plt.yscale('symlog', linthresh=1e-6); plt.title(f'Diagnostic B: CENTERED Orthonormal Hermite Projection (Model 0, k={self.k})')
        plt.xlabel('Hermite Order $n$'); plt.ylabel(r'Centered Coefficient $\langle f - \langle f \rangle, \tilde{H}_n \rangle$')
        plt.axvline(x=self.k, color='red', linestyle='--', label=f'$k={self.k}$')
        plt.axvline(x=2 * self.k, color='magenta', linestyle='--', label=f'$2k={2*self.k}$')
        plt.grid(True, which='major', linestyle='-'); plt.grid(True, which='minor', linestyle=':'); plt.legend(); plt.tight_layout()
        plt.savefig(f"{self.save_dir}/diagnostic_B_hermite_projection_centered_barchart.png", dpi=300); plt.close()


    

    def _plot_diagnostic_C_kernel_and_s(self):
        print("Running Diagnostic C: Plotting kernel spectrum and 's' distribution...")
        fig, axs = plt.subplots(1, 2, figsize=(18, 7)); fig.suptitle('Diagnostic C: Hidden-Pair Structure', fontsize=16)
        ax = axs[0]; colors = plt.cm.viridis(np.linspace(0, 1, self.n_ensemble))
        for i, model in enumerate(self.models):
            a, W1 = model.a.detach(), model.W1.detach()
            eigs = self._hidden_pair_kernel_eigs(a, W1)
            ranks = np.arange(1, len(eigs) + 1)
            ax.scatter(ranks, eigs, color=colors[i], alpha=0.3, s=15, edgecolor='none')
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1e-3)
        ax.set_title(r'Spectrum of Hidden-Pair Kernel $G_{ij} = a_i a_j w_i^\top w_j$'); ax.set_xlabel('Rank (log scale)'); ax.set_ylabel('Eigenvalue (symlog scale)')
        ax.grid(True, which="both", linestyle=':')
        ax = axs[1]; s_vals = [self._s_single(model.a.detach(), model.W1.detach()).cpu().item() for model in self.models]
        sns.histplot(s_vals, bins=max(10, self.n_ensemble // 2), kde=True, ax=ax)
        ax.set_title(r'Distribution of Single-Run $s$ Across Seeds'); ax.set_xlabel(r'Value of $s$'); ax.set_ylabel('Count'); ax.grid(True, linestyle=':')
        plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f"{self.save_dir}/diagnostic_C_kernel_and_s.png", dpi=300); plt.close()

    # --- Paste these two new methods anywhere inside the class ---

    def _compute_eigenvector_projections(self):
        """
        For each model in the final ensemble, computes the kernel eigenvectors
        and projects them onto the pure Hermite basis functions.
        Averages the resulting projection matrices across the ensemble.
        """
        print("Running final analysis: Projecting learned eigenvectors onto Hermite basis...")
        if not self.final_weights['W1']:
            print("Cannot run, final weights not available.")
            return

        # Let's project the top 15 eigenvectors onto the first 15 Hermite modes
        num_top_eigs = 25
        L_max_proj = 25
        
        # Store the squared projection matrix for each run
        all_proj_matrices = []

        # Use the fixed test set for this analysis
        x_test = self.X_test
        P_test = x_test.shape[0]

        # Pre-compute all required Hermite basis functions on the test set
        # H_basis will be a matrix of shape (P_test, L_max + 1)
        H_basis = torch.zeros(P_test, L_max_proj + 1, device=self.device)
        if self.k > 0:
            one_S = torch.ones(self.k, device=self.device)
            z_S = (x_test[:, :self.k] @ one_S) / np.sqrt(self.k)
        else:
            z_S = torch.zeros(P_test, device=self.device)

        for l in range(L_max_proj + 1):
            # We use the ORTHONORMAL basis for a clean projection
            H_basis[:, l] = self._orthonormal_hermite(l, z_S)

        # --- Loop over each trained model in the ensemble ---
        for model in self.models:
            model.eval()
            with torch.no_grad():
                # 1. Get the learned kernel's eigenvectors (v_i)
                H_activations = model.get_activations(x_test) # Shape (P_test, N)
                # This is the Gram kernel K_munu, shape (P_test, P_test)
                K_eff = (H_activations @ H_activations.T) / P_test 
                
                try:
                    # Eigenvectors are the COLUMNS of e_vecs
                    e_vals, e_vecs = torch.linalg.eigh(K_eff)
                    # Sort eigenvectors by descending eigenvalue
                    sorted_indices = torch.argsort(e_vals, descending=True)
                    sorted_e_vecs = e_vecs[:, sorted_indices]
                except torch.linalg.LinAlgError:
                    print("Warning: Eigendecomposition failed for one model. Skipping.")
                    continue

                # 2. Compute the projections
                # Matrix to store |<v_i, H_l>|^2 for this model
                proj_matrix = torch.zeros(num_top_eigs, L_max_proj + 1, device=self.device)
                for i in range(num_top_eigs):
                    v_i = sorted_e_vecs[:, i] # The i-th most important feature
                    for l in range(L_max_proj + 1):
                        H_l = H_basis[:, l] # The l-th pure basis function
                        # The projection is the squared dot product
                        projection_squared = (torch.dot(v_i, H_l))**2
                        proj_matrix[i, l] = projection_squared
                
                all_proj_matrices.append(proj_matrix)

        # 3. Average the results over the entire ensemble
        if all_proj_matrices:
            stacked_matrices = torch.stack(all_proj_matrices)
            self.eigenvector_projection_analysis['projection_matrix_mean'] = stacked_matrices.mean(dim=0).cpu().numpy()
            self.eigenvector_projection_analysis['projection_matrix_std'] = stacked_matrices.std(dim=0).cpu().numpy()


    def plot_eigenvector_projection_heatmap(self):
        """
        Visualizes the projection matrix showing the composition of each learned
        eigenvector in terms of the pure Hermite basis.
        """
        print("Plotting eigenvector projection heatmap...")
        proj_matrix = self.eigenvector_projection_analysis['projection_matrix_mean']

        if proj_matrix is None:
            print("Projection data not available. Run _compute_eigenvector_projections() first.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use LogNorm to see both large and small projection values clearly
        img = ax.imshow(proj_matrix, cmap='magma', norm=LogNorm(vmin=1e-4, vmax=1.0))

        ax.set_title('Composition of Learned Features (Kernel Eigenvectors)\nEnsemble Mean')
        ax.set_xlabel('Hermite Order $\ell$')
        ax.set_ylabel('Eigenvector Rank $i$ (Sorted by Importance)')
        
        ax.set_xticks(np.arange(0, proj_matrix.shape[1]))
        ax.set_yticks(np.arange(0, proj_matrix.shape[0]))
        
        # Add vertical lines for theoretically important channels
        ax.axvline(x=self.k, color='cyan', linestyle='--', label=f'$k={self.k}$')
        ax.axvline(x=2 * self.k, color='lime', linestyle='--', label=f'$2k={2*self.k}$')
        ax.legend()
        
        fig.colorbar(img, ax=ax, label='Squared Projection Power $|\langle v_i, \\tilde{H}_\\ell \\rangle|^2$')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/eigenvector_projection_heatmap.png", dpi=300)
        plt.close()

    def run_and_plot_final_diagnostics(self):
        """
        Runs a suite of diagnostic tests after training to determine *how* the
        parity function was learned by the ensemble.
        """
        if not self.final_weights['W1']:
            print("Cannot run final diagnostics. Final weights not available.")
            print("Please ensure model training has completed.")
            return
            
        print("\n" + "="*80 + "\nRunning Final Learning Mechanism Diagnostics...")
        
        # Run all diagnostic checks and plots
        self._run_parity_accuracy_sanity_check()
        self._plot_diagnostic_A_m_histogram()
        self._plot_diagnostic_B_hermite_projection()   # For the single-model bar chart
        self._plot_diagnostic_B_hermite_heatmap()      # For the all-models heatmap
        self._plot_diagnostic_C_kernel_and_s()
        
        print("All diagnostic plots saved successfully!")

    def _create_optimizer(self, model):
        param_groups = [{'params': model.W1, 'weight_decay': self.weight_decay_W1}, {'params': model.a, 'weight_decay': self.weight_decay_a}]
        return torch.optim.SGD(param_groups, lr=self.learning_rate)  #SGD

    def _target_function(self, x):
        return torch.prod(x[:, :self.k], dim=1)
    
        # --- NEW: General version of the target function for arbitrary subsets A ---
    def _target_function_general(self, x, A):
        """Computes the Walsh function chi_A(x) = product_{j in A} x_j."""
        if len(A) == 0:
            return torch.ones(x.shape[0], device=self.device)
        return torch.prod(x[:, A], dim=1)

    # --- Replace your existing train() method with this corrected version ---


    def train(self, loss_threshold=1e-3, patience=10000):
        """
        Main training loop for the ensemble using a fixed dataset (offline training).

        Trains all models in mini-batches sampled from a fixed training set.
        A model is considered converged if its training loss on the *entire*
        fixed training set is below `loss_threshold` for `patience` consecutive checks.
        """
        print(f"Starting offline training for up to {self.max_epochs} epochs.")
        print(f"Convergence criteria: Train Loss < {loss_threshold} for {patience // self.tracker} checks ({patience} epochs).")
        
        self._initialize_cavity_method_diagnostics()
        
        # Keep track of how many consecutive times each model has met the loss threshold
        convergence_counters = [0] * self.n_ensemble
        
        # Initial metric calculation at epoch 0
        self._compute_all_metrics(0)
        
        epoch = 0
        active_model_indices = list(range(self.n_ensemble))

        with tqdm(total=self.max_epochs, desc="Training Ensemble") as pbar:
            while epoch < self.max_epochs:
                if not active_model_indices:
                    print("\nAll models have converged. Stopping training early.")
                    break
                
                epoch += 1

                # --- Mini-batch Training Step ---
                # Shuffle the dataset indices for this epoch's batches
                epoch_indices = torch.randperm(self.train_set_size, device=self.device)
                
                for i in range(0, self.train_set_size, self.batch_size):
                    batch_indices = epoch_indices[i:i+self.batch_size]
                    X_batch = self.X_train[batch_indices]
                    y_batch = self.y_train[batch_indices]
                    y_batch_scaled = self.gamma * y_batch

                    # Train only the models that have not yet converged
                    for model_idx in active_model_indices:
                        model, optimizer = self.models[model_idx], self.optimizers[model_idx]
                        model.train()
                        loss = self.criterion(model(X_batch).squeeze(), y_batch_scaled)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # --- Metrics and Convergence Check ---
                if epoch % self.tracker == 0:
                    self._compute_all_metrics(epoch)
                    
                    # Check convergence for each active model
                    for model_idx in active_model_indices:
                        model = self.models[model_idx]
                        model.eval()
                        with torch.no_grad():
                            # Calculate loss on the *entire* fixed training set
                            full_train_preds = model(self.X_train).squeeze()
                            current_train_loss = self.criterion(full_train_preds, self.y_train).item()

                        if current_train_loss < loss_threshold:
                            convergence_counters[model_idx] += 1
                        else:
                            # Reset counter if loss goes above threshold
                            convergence_counters[model_idx] = 0
                        
                        # Check if patience has been met
                        if convergence_counters[model_idx] * self.tracker >= patience:
                            if not self.is_converged[model_idx]:
                                self.is_converged[model_idx] = True
                                self.final_epochs[model_idx] = epoch
                                print(f"\n--- Model {model_idx} converged at epoch {epoch} (loss: {current_train_loss:.2e}) ---")
                    
                    # Update the list of active models
                    active_model_indices = [i for i, conv in enumerate(self.is_converged) if not conv]
                    
                    # Update progress bar
                    num_converged = sum(self.is_converged)
                    last_test_loss = self.loss_stats['test_loss_mean'][-1] if self.loss_stats['test_loss_mean'] else float('nan')
                    pbar.set_description(f"Converged: {num_converged}/{self.n_ensemble} | Avg Test Loss: {last_test_loss:.4f}")
                
                pbar.update(1)

        self.final_weights = {'W1': [m.W1.detach().cpu().numpy() for m in self.models], 'a': [m.a.detach().cpu().numpy() for m in self.models]}
        print("\n" + "="*80 + "\nTraining complete!")

    def _compute_all_metrics(self, epoch):
        # --- Compute metrics that need to be tracked at every step ---
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                preds = model(self.X_test).squeeze()
                corr = torch.corrcoef(torch.stack([preds, self.y_test.squeeze()]))[0, 1].item() if preds.numel() > 1 and torch.var(preds) > 1e-6 else 0.0
            self.metrics[i]['epochs'].append(epoch)
            self.metrics[i]['correlation'].append(corr)
            if corr > 0.9 and self.metrics[i]['phase_transition_epoch'] is None:
                self.metrics[i]['phase_transition_epoch'] = epoch
            
            # Detailed analysis for the representative model (model 0)
            if i == 0:
                self._compute_representative_model_metrics(epoch, model)

        # These are computed from model weights and are relatively fast
        self._compute_replica_metrics(epoch)
        self._compute_replica_metrics_active(epoch)
        self._compute_hermite_evolution(epoch)
        self._compute_susceptibility(epoch)
        # This now only computes the fast conjugate kernel eigenvalues
        self._compute_conjugate_kernel_eigs(epoch)
        self._compute_macro_observables(epoch)
        self._compute_drift_diagnostics(epoch)
        self._compute_raw_projection_evolution(epoch)
        self._compute_walsh_order_parameters(epoch)
        self._compute_neuron_level_metrics(epoch)
        self._compute_loss_metrics(epoch)
        
    def _compute_conjugate_kernel_eigs(self, epoch):
        """Computes the conjugate kernel and its eigenvalues for ALL models in the ensemble."""
        all_top_eigs, all_top_eigs_centered = [], []
        
        # --- ADD THIS ---
        all_diag_norms, all_offdiag_norms = [], []
        # --- END ADD ---

        for model in self.models:
            model.eval()
            with torch.no_grad():
                H = model.get_activations(self.X_kernel)
                K = (H.T @ H) / float(self.kernel_n_samples)

                # --- ADD THIS BLOCK ---
                # Calculate the Frobenius norm of the diagonal and off-diagonal elements
                diag_elements = torch.diag(K)
                off_diag_elements = K - torch.diag(diag_elements)

                # --- MODIFIED BLOCK ---
                num_neurons = K.shape[0]
                num_off_diag = num_neurons #* num_neurons - num_neurons

                diag_elements = torch.diag(K)
                off_diag_elements = K - torch.diag(diag_elements)
                
                # Calculate the raw norms
                diag_norm_raw = torch.linalg.norm(diag_elements)
                offdiag_norm_raw = torch.linalg.norm(off_diag_elements)

                # Normalize to get the RMS value per element
                # This makes the comparison fair
                diag_norm_rms = (diag_norm_raw / np.sqrt(num_neurons)).cpu().item()
                if num_off_diag > 0:
                    offdiag_norm_rms = (offdiag_norm_raw / np.sqrt(num_off_diag)).cpu().item()
                else:
                    offdiag_norm_rms = 0.0 # Avoid division by zero if M1=1
                
                all_diag_norms.append(diag_norm_rms)
                all_offdiag_norms.append(offdiag_norm_rms)
       

                try:
                    eigs = torch.linalg.eigvalsh(K).cpu().numpy(); all_top_eigs.append(np.sort(eigs)[-100:][::-1])
                    K_centered = K - torch.diag(torch.diag(K))
                    eigs_centered = torch.linalg.eigvalsh(K_centered).cpu().numpy(); all_top_eigs_centered.append(np.sort(eigs_centered)[-100:][::-1])
                except torch.linalg.LinAlgError:
                    all_top_eigs.append(np.zeros(100)); all_top_eigs_centered.append(np.zeros(100))
        
        if all_top_eigs:
            self.kernel_stats['epochs'].append(epoch)
            self.kernel_stats['top_eigenvalues_mean'].append(np.mean(all_top_eigs, axis=0))
            self.kernel_stats['top_eigenvalues_std'].append(np.std(all_top_eigs, axis=0))
        if all_top_eigs_centered:
            self.centered_kernel_stats['epochs'].append(epoch)
            self.centered_kernel_stats['top_eigenvalues_mean'].append(np.mean(all_top_eigs_centered, axis=0))
            self.centered_kernel_stats['top_eigenvalues_std'].append(np.std(all_top_eigs_centered, axis=0))

        # --- ADD THIS BLOCK ---
        # Store the computed norm statistics
        if all_diag_norms:
            self.kernel_diag_offdiag_stats['epochs'].append(epoch)
            self.kernel_diag_offdiag_stats['diag_norm_mean'].append(np.mean(all_diag_norms))
            self.kernel_diag_offdiag_stats['diag_norm_std'].append(np.std(all_diag_norms))
            self.kernel_diag_offdiag_stats['offdiag_norm_mean'].append(np.mean(all_offdiag_norms))
            self.kernel_diag_offdiag_stats['offdiag_norm_std'].append(np.std(all_offdiag_norms))
        # --- END ADD ---

    # --- Add this new method to your EnsembleParityExperiment class ---

    def _raw_hermite_projection_centered(self, a, W1, x_batch, ell):
        """
        CORRECTLY and STABLY calculates the raw projection:
            p_ell = < f - <f> ,  H_tilde_ell(x_S) >_x
        This avoids the numerically unstable division by d_ell.
        """
        N = W1.shape[0]
        B = x_batch.shape[0]
        k = self.k
        
        # 1. Calculate the centered student function f(x)
        z_preactivation = W1 @ x_batch.T
        f_raw = (a[:, None] * torch.relu(z_preactivation)).sum(0) / math.sqrt(N)
        f = f_raw - f_raw.mean()

        # 2. Define the basis function H_ell based only on relevant inputs x_S
        if k > 0:
            one_S = torch.ones(k, device=self.device)
            # Ensure argument to Hermite is on the same device
            z_S = (x_batch[:, :k] @ one_S) / np.sqrt(k)
        else:
            z_S = torch.zeros(B, device=self.device)
            
        H_basis_tilde = self._orthonormal_hermite(ell, z_S)

        # 3. Compute the projection <f, H_basis>
        p_ell = torch.dot(f, H_basis_tilde) / B
        
        return p_ell.item()

        # New compute function to be called from _compute_all_metrics:
    def _compute_raw_projection_evolution(self, epoch):
        all_projs = torch.zeros(self.n_ensemble, self.L_max + 1, device=self.device)
        for i, model in enumerate(self.models):
            a, W1 = model.a.detach(), model.W1.detach()
            for l in range(self.L_max + 1):
                # Use the new STABLE calculation
                all_projs[i, l] = self._raw_hermite_projection_centered(a, W1, self.X_test, l)
                
        self.raw_projection_stats['epochs'].append(epoch)
        self.raw_projection_stats['projections_mean'].append(torch.mean(all_projs, dim=0).cpu().numpy())
        self.raw_projection_stats['projections_std'].append(torch.std(all_projs, dim=0).cpu().numpy())

    def plot_raw_projection_heatmap(self):
        """
        Plots the evolution of the STABLE raw Hermite projections p_l as a heatmap.
        This version uses the correct Matplotlib API for SymLogNorm.
        """
        print("Plotting STABLE raw Hermite projection heatmap...")
        stats = self.raw_projection_stats
        if not stats['epochs']:
            print("Raw projection data not available.")
            return

        heatmap_data = np.stack(stats['projections_mean'], axis=0)
        
        fig, ax = plt.subplots(figsize=(14, 8))

        # --- SAFETY CHECK for all-zero data ---
        vmax = np.max(np.abs(heatmap_data))
        
        if vmax < 1e-9: # If data is essentially all zero, use a simple linear scale
            print("Warning: Raw projection data is all zero. Plotting with a linear scale.")
            img = ax.imshow(
                heatmap_data, 
                aspect='auto', 
                cmap='bwr', 
                vmin=-1e-9, vmax=1e-9, # Use a tiny range around zero
                origin='lower', 
                extent=[-0.5, self.L_max + 0.5, stats['epochs'][0], stats['epochs'][-1]]
            )
        else:
            # --- CORRECT API USAGE for SymLogNorm ---
            # 1. Define the linear threshold around zero
            linthresh = max(1e-9, vmax / 1000)
            
            # 2. Create the normalization object with all relevant parameters
            log_norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
            
            # 3. Pass the norm OBJECT to imshow. Do NOT pass linthresh/vmin/vmax again.
            img = ax.imshow(
                heatmap_data, 
                aspect='auto', 
                cmap='bwr', 
                norm=log_norm, # Pass the created object here
                origin='lower', 
                extent=[-0.5, self.L_max + 0.5, stats['epochs'][0], stats['epochs'][-1]]
            )

        # --- The rest of the plotting code is the same ---
        ax.axvline(x=self.k, color='red', linestyle='--', label=f'$k={self.k}$')
        ax.axvline(x=2 * self.k, color='magenta', linestyle='--', label=f'$2k={2*self.k}$')
        
        ax.set_xlabel('Hermite Order $\ell$')
        ax.set_ylabel('Epoch')
        ax.set_title('Evolution of Raw Hermite Projections $p_\ell = \langle f - \\langle f \\rangle, \\tilde{H}_\\ell \\rangle$')
        plt.colorbar(img, ax=ax, label='Projection Value $p_\ell$')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/raw_projection_heatmap_STABLE.png", dpi=300)
        plt.close()
    # --- Replace your plot_raw_projection_heatmap with this robust version ---




    def _calculate_explained_variance_snapshot(self, snapshot_key='start'):
        """Helper to run the expensive explained variance calculation and store it."""
        print(f"Calculating '{snapshot_key}' explained variance snapshot...")
        all_cumulative_ratios = []
        y_test_np = self.y_test.cpu().numpy()
        total_y_variance = np.sum(y_test_np**2)
        for model in self.models:
            model.eval()
            with torch.no_grad():
                H_test = model.get_activations(self.X_test)
                K_test = (H_test @ H_test.T).cpu().numpy()
                try:
                    e_vals, e_vecs = np.linalg.eigh(K_test)
                    sorted_indices = np.argsort(e_vals)[::-1]
                    sorted_e_vecs = e_vecs[:, sorted_indices]
                    proj_coeffs = y_test_np.T @ sorted_e_vecs
                    all_cumulative_ratios.append(np.cumsum(proj_coeffs**2) / total_y_variance)
                except np.linalg.LinAlgError:
                    all_cumulative_ratios.append(np.zeros(len(y_test_np)))
        self.explained_variance_snapshots[snapshot_key] = all_cumulative_ratios


    def _compute_drift_diagnostics(self, epoch):
        """
        Calculates and records the drift of the quasi-conserved scale charge (I)
        and the alignment-to-norm ratio (J) from their initial values.
        """
        I_vals, J_vals = [], []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                a = model.a
                W1 = model.W1

                # Calculate q_u, q_v, and r for the current model
                qu_val = 0.0
                if self.k > 0:
                    u = W1[:, :self.k]
                    qu_val = torch.mean(torch.sum(u**2, dim=1)).cpu().item()

                qv_val = 0.0
                if self.d > self.k:
                    v = W1[:, self.k:]
                    qv_val = torch.mean(torch.sum(v**2, dim=1)).cpu().item()
                
                r_val = torch.mean(a**2).cpu().item()

                # Calculate I and J for the current model
                I_current = qu_val + qv_val
                # Add a small epsilon to prevent division by zero
                J_current = r_val / (np.sqrt(qu_val * qv_val) + 1e-9)
                
                I_vals.append(I_current)
                J_vals.append(J_current)

        # If it's the first logging step, store the initial values
        if self.I0_vals is None or self.J0_vals is None:
            self.I0_vals = np.array(I_vals)
            self.J0_vals = np.array(J_vals)
            # Drift at epoch 0 is zero by definition
            drift_I_mean, drift_I_std = 0.0, 0.0
            drift_J_mean, drift_J_std = 0.0, 0.0
        else:
            # Calculate the drift from the stored initial values
            drift_I_vals = np.abs(np.array(I_vals) - self.I0_vals)
            drift_J_vals = np.abs(np.array(J_vals) - self.J0_vals)
            
            drift_I_mean, drift_I_std = np.mean(drift_I_vals), np.std(drift_I_vals)
            drift_J_mean, drift_J_std = np.mean(drift_J_vals), np.std(drift_J_vals)
            
        # Store the computed statistics
        self.drift_diagnostics['epochs'].append(epoch)
        self.drift_diagnostics['drift_I_mean'].append(drift_I_mean)
        self.drift_diagnostics['drift_I_std'].append(drift_I_std)
        self.drift_diagnostics['drift_J_mean'].append(drift_J_mean)
        self.drift_diagnostics['drift_J_std'].append(drift_J_std)



    def plot_drift_diagnostics(self):
        """
        Plots the evolution of the drift for the I and J diagnostic quantities.
        """
        print("Plotting drift diagnostics...")
        stats = self.drift_diagnostics
        if not stats['epochs'] or len(stats['epochs']) < 2:
            print("Drift diagnostic data not available or insufficient to plot.")
            return

        fig, axs = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Evolution of Diagnostic Drifts from Initial Values', fontsize=16)
        
        epochs = stats['epochs'][1:] # Exclude epoch 0 where drift is zero
        n_sqrt = np.sqrt(self.n_ensemble)

        # --- Plot Drift of I = q_u + q_v ---
        ax = axs[0]
        mean = np.array(stats['drift_I_mean'][1:])
        std = np.array(stats['drift_I_std'][1:])
        sem = std / n_sqrt
        
        line, = ax.plot(epochs, mean, marker='.', linestyle='-', markersize=4)
        ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
        
        ax.set_title('Drift of Quasi-Scale Charge $I$')
        ax.set_ylabel(r'Drift $|I - I_0|$ (log scale)')
        ax.set_xlabel('Epoch (log scale)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle=':')

        # --- Plot Drift of J = r / sqrt(q_u*q_v) ---
        ax = axs[1]
        mean = np.array(stats['drift_J_mean'][1:])
        std = np.array(stats['drift_J_std'][1:])
        sem = std / n_sqrt
        
        line, = ax.plot(epochs, mean, marker='.', linestyle='-', markersize=4, color='C1')
        ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)

        ax.set_title('Drift of Alignment-to-Norm Ratio $J$')
        ax.set_ylabel(r'Drift $|J - J_0|$ (log scale)')
        ax.set_xlabel('Epoch (log scale)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle=':')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{self.save_dir}/drift_diagnostics_evolution.png", dpi=300)
        plt.close()

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
                H_conj = model.get_activations(self.X_kernel)
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

    # --- CORRECTED: Method to compute Walsh order parameters m_A ---
    def _compute_walsh_order_parameters(self, epoch):
        """
        Calculates the Walsh order parameter m_A for each model in the ensemble.
        This implementation correctly follows the theoretical definition:
        1. For each neuron i, calculate J_A(w_i) = E_x[phi(w_i^T * x) * chi_A(x)]
        2. For each model, calculate m_A = <a * J_A(w)>_P = (1/M1) * sum_i(a_i * J_A(w_i))
        """
        num_subsets = len(self.walsh_param_sets)
        # Array to hold the final m_A values for all models at this epoch
        all_m_A_epoch = torch.zeros(self.n_ensemble, num_subsets, device=self.device)

        with torch.no_grad():
            # 1. Pre-calculate all basis functions chi_A(x) on the large Walsh test set.
            # Shape: (num_subsets, walsh_n_samples)
            all_chi_A = torch.zeros(num_subsets, self.walsh_n_samples, device=self.device)
            for i, A in enumerate(self.walsh_param_sets):
                all_chi_A[i, :] = self._target_function_general(self.X_walsh, A)

            # 2. Loop over each model in the ensemble to calculate its m_A values.
            for model_idx, model in enumerate(self.models):
                model.eval()
                a, W1 = model.a, model.W1

                # 3. Calculate all neuron activations phi(w_i^T * x) at once.
                # Shape: (walsh_n_samples, M1)
                all_phi = model.get_activations(self.X_walsh)

                # 4. Calculate the J_A(w_i) matrix.
                # This performs the expectation E_x[...] for all A and all neurons i at once.
                # J_matrix[k, i] = E_x[phi(w_i^T * x) * chi_{A_k}(x)]
                # Shape: (num_subsets, M1)
                J_matrix = torch.matmul(all_chi_A, all_phi) / self.walsh_n_samples

                # 5. Calculate the final m_A values for this model.
                # This performs the average over the neuron population <a * J_A(w)>_P.
                # It is equivalent to (1/M1) * sum_i(a_i * J_A(w_i)) for each A.
                # Shape: (num_subsets,)
                m_A_values = torch.matmul(J_matrix, a) / self.M1

                # Store the results for the ensemble and for the single model
                all_m_A_epoch[model_idx, :] = m_A_values
                self.walsh_single_model_stats[model_idx]['epochs'].append(epoch)
                self.walsh_single_model_stats[model_idx]['m_A'].append(m_A_values.cpu().numpy())

        # 6. Now, calculate and store the ensemble average and std dev across models.
        self.walsh_ensemble_stats['epochs'].append(epoch)
        self.walsh_ensemble_stats['m_A_mean'].append(torch.mean(all_m_A_epoch, dim=0).cpu().numpy())
        self.walsh_ensemble_stats['m_A_std'].append(torch.std(all_m_A_epoch, dim=0).cpu().numpy())

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


    


    def _compute_macro_observables(self, epoch):
        """Computes and stores the macroscopic observables for the ensemble."""
        m_vals, qu_vals, qv_vals, r_vals, s_vals = [], [], [], [], []
        N = float(self.M1)
        k = float(self.k)
        d = float(self.d)

        # Vector of ones for the relevant dimensions, used in m_hat calculation
        if self.k > 0:
            one_S = torch.ones(self.k, device=self.device)

        for model in self.models:
            model.eval()
            with torch.no_grad():
                a = model.a
                W1 = model.W1

                # r_hat = (1/N) * sum_i(a_i^2)
                r_val = torch.mean(a**2)
                r_vals.append(r_val.cpu().item())

                # Split weights into relevant (u) and irrelevant (v) parts
                if self.k > 0:
                    u = W1[:, :self.k]
                    # m_hat = (1 / (N * sqrt(k))) * sum_i(a_i * u_i^T * 1_S)
                    m_val = torch.sum(a * torch.matmul(u, one_S)) / (N * np.sqrt(k))
                    m_vals.append(m_val.cpu().item())

                    # qu_hat = (1/N) * sum_i(||u_i||^2)
                    qu_val = torch.mean(torch.sum(u**2, dim=1))
                    qu_vals.append(qu_val.cpu().item())
                else:
                    # If no relevant features, m and qu are zero
                    m_vals.append(0.0)
                    qu_vals.append(0.0)

                if self.d > self.k:
                    v = W1[:, self.k:]
                    # qv_hat = (1/N) * sum_i(||v_i||^2)
                    qv_val = torch.mean(torch.sum(v**2, dim=1))
                    qv_vals.append(qv_val.cpu().item())
                else:
                    # If no irrelevant features, qv is zero
                    qv_vals.append(0.0)

                # s_hat = (1/N^2) * sum_{i!=j}(a_i * a_j * w_i^T * w_j)
                # This is calculated efficiently as (1/N^2) * ( ||W1^T a||^2 - sum_i(a_i^2 ||w_i||^2) )
                g = W1.t() @ a
                term1 = torch.sum(g**2)
                term2 = torch.sum(a**2 * torch.sum(W1**2, dim=1))
                s_val = (term1 - term2) / (N**2)
                s_vals.append(s_val.cpu().item())

        # Store the calculated means and standard deviations for the ensemble
        self.macro_observables['epochs'].append(epoch)
        for name, vals in [('m', m_vals), ('qu', qu_vals), ('qv', qv_vals), ('r', r_vals), ('s', s_vals)]:
            self.macro_observables[f'{name}_mean'].append(np.mean(vals) if vals else 0)
            self.macro_observables[f'{name}_std'].append(np.std(vals) if vals else 0)

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
        Evaluates final correlation, kernel spectrum, and explained variance.
        """
        # *** MODIFIED: This now also triggers the final explained variance calculation ***
        #self._calculate_explained_variance_snapshot(snapshot_key='end')

        print(f"\nEvaluating final performance on {test_set_size} samples...")
        X_final = torch.randn(test_set_size, self.d, device=self.device).sign()
        y_final = self._target_function(X_final)
        
        for m in self.models:
            m.eval()
            with torch.no_grad():
                # Correlation Performance
                preds_full = m(X_final).squeeze()
                self.final_performance['full_model_corr'].append(torch.corrcoef(torch.stack([preds_full, y_final]))[0,1].item())
                a, w1 = m.a.detach(), m.W1.detach()
                indices = torch.argsort(torch.abs(a))[-self.top_n_neurons:]
                h_sparse = torch.relu(X_final @ w1[indices,:].T)
                preds_sparse = (h_sparse @ a[indices]).squeeze()
                self.final_performance['sparse_model_corr'].append(torch.corrcoef(torch.stack([preds_sparse, y_final]))[0,1].item())

                # Final Kernel Eigenvalue Spectrum
                H = m.get_activations(self.X_kernel)
                K = (H.T @ H) / float(self.kernel_n_samples)
                try:
                    self.final_kernel_eigenvalues.append(torch.linalg.eigvalsh(K).cpu().numpy())
                except torch.linalg.LinAlgError:
                    self.final_kernel_eigenvalues.append(np.zeros(self.M1))

        print(f"Avg correlation (Full): {np.mean(self.final_performance['full_model_corr']):.4f}")
        print(f"Avg correlation (Sparse, Top {self.top_n_neurons}): {np.mean(self.final_performance['sparse_model_corr']):.4f}")

       

    def plot_explained_variance_snapshots(self):
        """
        Plots the cumulative explained variance at the start and end of training,
        averaged over the ensemble with standard error ribbons.
        """
        print("Plotting explained variance snapshots (start vs. end)...")
        stats = self.explained_variance_snapshots
        if not stats['start'] or not stats['end']:
            print("Start or end explained variance data not available.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        n_sqrt = np.sqrt(self.n_ensemble)
        
        # --- Start of Training ---
        start_ratios = np.array(stats['start'])
        mean_start = np.mean(start_ratios, axis=0)
        sem_start = np.std(start_ratios, axis=0) / n_sqrt
        num_eigenvectors = len(mean_start)
        x_axis = np.arange(1, num_eigenvectors + 1)
        
        line, = ax.plot(x_axis, mean_start, marker='', linestyle='--', label=f'Start (Epoch 0)', color='orange')
        ax.fill_between(x_axis, mean_start - sem_start, mean_start + sem_start, color=line.get_color(), alpha=0.2)

        # --- End of Training ---
        end_ratios = np.array(stats['end'])
        mean_end = np.mean(end_ratios, axis=0)
        sem_end = np.std(end_ratios, axis=0) / n_sqrt

        line, = ax.plot(x_axis, mean_end, marker='', linestyle='-', label='End (Final Epoch)', color='blue')
        ax.fill_between(x_axis, mean_end - sem_end, mean_end + sem_end, color=line.get_color(), alpha=0.2)
            
        ax.set_xscale('log')
        ax.set_xlabel('Number of Eigenvectors (log scale)')
        ax.set_ylabel('Fraction of Explained Variance')
        ax.set_title('Cumulative Explained Variance of Target by Kernel Eigenvectors')
        ax.grid(True, which="both", linestyle=':')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/explained_variance_snapshots.png", dpi=300)
        plt.close()

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


    def plot_macro_observables(self):
        """Plots the evolution of the five macroscopic observables over epochs."""
        print("Plotting macroscopic observables...")
        stats = self.macro_observables
        if not stats['epochs']:
            print("Macroscopic observable data not available.")
            return

        fig, axs = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
        axs = axs.flatten()
        fig.suptitle('Evolution of Macroscopic Observables (Ensemble Mean)', fontsize=18)
        epochs = stats['epochs']
        n_sqrt = np.sqrt(self.n_ensemble)

        observables_map = {
            'm':  {'ax': axs[0], 'label': r'$\hat m$', 'yscale': 'linear'},
            'qu': {'ax': axs[1], 'label': r'$\hat q_u$', 'yscale': 'log'},
            'qv': {'ax': axs[2], 'label': r'$\hat q_v$', 'yscale': 'log'},
            'r':  {'ax': axs[3], 'label': r'$\hat r$', 'yscale': 'log'},
            's':  {'ax': axs[4], 'label': r'$\hat s$', 'yscale': 'linear'},
        }
        
        axs[5].axis('off') # Hide the unused subplot

        for obs, details in observables_map.items():
            ax = details['ax']
            mean = np.array(stats[f'{obs}_mean'])
            std = np.array(stats[f'{obs}_std'])
            
            if len(mean) == 0:
                ax.text(0.5, 0.5, 'Not Applicable', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Evolution of {details["label"]}')
                continue

            sem = std / n_sqrt # Standard Error of the Mean
            
            line, = ax.plot(epochs, mean, marker='', linestyle='-', label='Mean')
            ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
            
            ax.set_ylabel('Value')
            ax.set_title(f'Evolution of {details["label"]}')
            ax.grid(True, which="both", linestyle=':')
            ax.set_xscale('log')
            
            if details['yscale'] == 'log' and np.all(mean > 0):
                ax.set_yscale('log')
            else:
                ax.set_yscale('symlog', linthresh=1e-5)

        # Add x-axis labels to the bottom-row plots
        for ax in [axs[2], axs[3], axs[4]]:
            ax.set_xlabel('Epoch')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.save_dir}/macro_observables_evolution.png", dpi=300)
        plt.close()



    def run_and_plot_final_diagnostics(self):
        """
        Runs a suite of diagnostic tests after training to determine *how* the
        parity function was learned by the ensemble.
        """
        if not self.final_weights['W1']:
            print("Cannot run final diagnostics. Final weights not available.")
            print("Please ensure model training has completed.")
            return
            
        print("\n" + "="*80 + "\nRunning Final Learning Mechanism Diagnostics...")
        
        # Run diagnostics for each of the three proposed mechanisms
        self._plot_diagnostic_A_m_histogram()
        self._plot_diagnostic_B_hermite_projection()
        self._plot_diagnostic_C_kernel_and_s()
        
        print("Diagnostic plots saved successfully!")

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
        #self.plot_explained_variance_snapshots() # *** This is the new plot call ***
        self.plot_kernel_diag_vs_offdiag_norm()
        self.plot_macro_observables() # <<<--- ADD THIS LINE
        self.plot_drift_diagnostics()
        self.plot_hermite_evolution_heatmap()
        self.plot_individual_hermite_evolution_heatmaps()
        self.plot_all_coeff_evolution_ribbon()
        self.plot_raw_projection_heatmap()
        self.plot_eigenvector_projection_heatmap()
        self.plot_raw_projection_evolution() 
        self.plot_walsh_order_parameters_ensemble()
        self.plot_walsh_order_parameters_single()
        self.plot_susceptibility_evolution_heatmap()
        self.plot_train_test_loss_evolution()
       
        self.plot_susceptibility_evolution() # <<<--- ADD THIS LINE
        self.plot_walsh_power_spectrum_snapshots() # <<<--- ADD THIS LINE
        self.plot_walsh_order_parameters_single()
        self.plot_neuron_correlation_and_a_weight_heatmaps()
        self.plot_initial_vs_final_weight_dist_per_model()


    # --- NEW: Plot for ensemble-averaged m_A ---
    def plot_walsh_order_parameters_ensemble(self):
        """
        Plots the evolution of the ensemble-averaged Walsh order parameters m_A.
        """
        print("Plotting ensemble-averaged Walsh order parameter (m_A) evolution...")
        stats = self.walsh_ensemble_stats
        if not stats['epochs']:
            print("Walsh order parameter data not available.")
            return

        fig, ax = plt.subplots(figsize=(14, 9))
        epochs = stats['epochs']
        
        # Stack the list of arrays into a 2D matrix for easier indexing
        # Shape: (num_epochs, num_subsets)
        m_A_means = np.stack(stats['m_A_mean'], axis=0)
        m_A_stds = np.stack(stats['m_A_std'], axis=0)
        n_sqrt = np.sqrt(self.n_ensemble)

        num_subsets = m_A_means.shape[1]
        colors = cm.get_cmap('turbo', num_subsets)

        for i in range(num_subsets):
            label = self.walsh_param_labels[i]
            mean = m_A_means[:, i]
            sem = m_A_stds[:, i] / n_sqrt # Standard Error of the Mean
            
            # Highlight the teacher function
            lw = 3.0 if 'm_S' in label else 1.5
            ls = '-' if 'm_S' in label else '--'
            alpha = 1.0 if 'm_S' in label else 0.8
            
            line, = ax.plot(epochs, mean, marker='', linestyle=ls, lw=lw, color=colors(i), label=label, alpha=alpha)
            ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.15)
        
        ax.set_title('Evolution of Walsh Order Parameters (Ensemble Mean ± SEM)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'Projection Value $m_A = \langle f, \chi_A \rangle$')
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.grid(True, which="both", linestyle=':')
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/walsh_mA_evolution_ensemble.png", dpi=300)
        plt.close()

    # --- NEW: Plot for single-model m_A ---
    def plot_walsh_order_parameters_single(self):
        """
        Plots the evolution of Walsh order parameters m_A for each model in a separate subplot.
        """
        print("Plotting single-model Walsh order parameter (m_A) evolution...")
        if not self.walsh_single_model_stats[0]['epochs']:
            print("Walsh order parameter data not available.")
            return

        num_models = self.n_ensemble
        ncols = int(np.ceil(np.sqrt(num_models)))
        nrows = int(np.ceil(num_models / ncols))
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharex=True, sharey=True, squeeze=False)
        axs = axs.flatten()
        
        num_subsets = len(self.walsh_param_sets)
        colors = cm.get_cmap('turbo', num_subsets)

        for i in range(num_models):
            ax = axs[i]
            stats = self.walsh_single_model_stats[i]
            epochs = stats['epochs']
            # Shape: (num_epochs, num_subsets)
            m_A_values = np.stack(stats['m_A'], axis=0)

            for j in range(num_subsets):
                label = self.walsh_param_labels[j]
                lw = 2.0 if 'm_S' in label else 1.0
                ls = '-' if 'm_S' in label else '--'
                ax.plot(epochs, m_A_values[:, j], lw=lw, ls=ls, color=colors(j), label=label)
            
            ax.set_title(f'Model {i}')
            ax.set_xscale('log')
            ax.set_yscale('symlog', linthresh=1e-4)
            ax.grid(True, which="both", linestyle=':')

        # Add shared labels and a single legend
        fig.supxlabel('Epoch')
        fig.supylabel(r'Projection Value $m_A = \langle f, \chi_A \rangle$')
        fig.suptitle('Evolution of Walsh Order Parameters for Each Model')
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=1, fontsize=9)
        
        # Hide unused subplots
        for i in range(num_models, len(axs)):
            axs[i].set_visible(False)
            
        plt.tight_layout(rect=[0, 0, 0.88, 0.95]) # Adjust rect to make space for legend
        plt.savefig(f"{self.save_dir}/walsh_mA_evolution_single_models.png", dpi=300)
        plt.close()
   
    def plot_kernel_diag_vs_offdiag_norm(self):
        """
        Plots the RMS value of the diagonal vs. the off-diagonal elements of the kernel matrix over time.
        """
        print("Plotting kernel diagonal vs. off-diagonal RMS evolution...")
        stats = self.kernel_diag_offdiag_stats
        if not stats['epochs']:
            print("Kernel norm data not available.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = stats['epochs']
        n_sqrt = np.sqrt(self.n_ensemble)

        # --- Plot Diagonal RMS ---
        mean_diag = np.array(stats['diag_norm_mean'])
        std_diag = np.array(stats['diag_norm_std'])
        sem_diag = std_diag / n_sqrt
        line, = ax.plot(epochs, mean_diag, marker='', linestyle='-', label='RMS of Diagonal Elements', color='green')
        ax.fill_between(epochs, mean_diag - sem_diag, mean_diag + sem_diag, color=line.get_color(), alpha=0.2)

        # --- Plot Off-Diagonal RMS ---
        mean_offdiag = np.array(stats['offdiag_norm_mean'])
        std_offdiag = np.array(stats['offdiag_norm_std'])
        sem_offdiag = std_offdiag / n_sqrt
        line, = ax.plot(epochs, mean_offdiag, marker='', linestyle='--', label='RMS of Off-Diagonal Elements', color='purple')
        ax.fill_between(epochs, mean_offdiag - sem_offdiag, mean_offdiag + sem_offdiag, color=line.get_color(), alpha=0.2)

        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_xscale('log')
        # --- MODIFIED LABEL ---
        ax.set_ylabel('Root Mean Square (RMS) Value (log scale)')
        ax.set_title('Evolution of Kernel Diagonal vs. Off-Diagonal Element Magnitudes (RMS)')
        # --- END MODIFICATION ---
        ax.grid(True, which="both", linestyle=':')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/kernel_diag_vs_offdiag_rms.png", dpi=300) # Changed filename for clarity
        plt.close()

    
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

    # --- Paste these two new methods anywhere inside the class ---

    def _compute_hermite_evolution(self, epoch):
        """
        Computes the Hermite coefficients c_l for all models in the ensemble
        up to L_max and stores the mean, std dev, and the full data array.
        """
        # Matrix to store coefficients for all runs at this epoch: (n_ensemble, L_max + 1)
        all_coeffs = torch.zeros(self.n_ensemble, self.L_max + 1, device=self.device)

        for i, model in enumerate(self.models):
            model.eval()
            a, W1 = model.a.detach(), model.W1.detach()
            for l in range(self.L_max + 1):
                all_coeffs[i, l] = self._hermite_coeff_centered(a, W1, self.X_test, l)

        # Calculate mean and std across the ensemble
        mean_coeffs = torch.mean(all_coeffs, dim=0).cpu().numpy()
        std_coeffs = torch.std(all_coeffs, dim=0).cpu().numpy()

        # Store the results
        self.hermite_evolution_stats['epochs'].append(epoch)
        self.hermite_evolution_stats['coeffs_mean'].append(mean_coeffs)
        self.hermite_evolution_stats['coeffs_std'].append(std_coeffs)
        # --- ADD THIS LINE TO STORE DATA FOR THE NEW PLOT ---
        self.hermite_evolution_stats['coeffs_all_models'].append(all_coeffs.cpu().numpy())
        
    def plot_hermite_evolution_heatmap(self):
        """
        Plots the evolution of the Hermite coefficients c_l as a heatmap.
        Epochs are on the y-axis, Hermite order l is on the x-axis.
        """
        print("Plotting Hermite coefficient evolution heatmap...")
        stats = self.hermite_evolution_stats
        if not stats['epochs']:
            print("Hermite evolution data not available.")
            return

        # Stack the list of mean coefficient arrays into a 2D matrix for plotting
        heatmap_data = np.stack(stats['coeffs_mean'], axis=0)

        fig, ax = plt.subplots(figsize=(14, 8))
        
        # --- FIX START ---
        # Create a SymLogNorm object and pass it to the norm argument
        img = ax.imshow(
            heatmap_data,
            aspect='auto',
            cmap='bwr', # Blue-White-Red is great for values centered at zero
            norm=SymLogNorm(linthresh=1e-6), # CORRECTED LINE
            origin='lower',
            extent=[-0.5, self.L_max + 0.5, stats['epochs'][0], stats['epochs'][-1]]
        )
        # --- FIX END ---
        
        # Add vertical lines for theoretically important channels
        ax.axvline(x=self.k, color='red', linestyle='--', label=f'$k={self.k}$')
        ax.axvline(x=2 * self.k, color='magenta', linestyle='--', label=f'$2k={2*self.k}$')

        ax.set_xlabel('Hermite Order $\ell$')
        ax.set_ylabel('Epoch')
        ax.set_title('Evolution of Hermite Coefficients $\langle f - \\langle f \\rangle, \\tilde{H}_\\ell \\rangle$')
        fig.colorbar(img, ax=ax, label='Coefficient Value')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/hermite_evolution_heatmap.png", dpi=300)
        plt.close()

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


    # --- ADD THIS NEW PLOTTING METHOD ---
    def plot_individual_hermite_evolution_heatmaps(self):
        """
        Plots the evolution of Hermite coefficients for each model as a separate
        heatmap, arranged vertically in a single figure.
        """
        print("Plotting individual Hermite coefficient evolution heatmaps...")
        stats = self.hermite_evolution_stats
        if not stats['coeffs_all_models']:
            print("Individual Hermite evolution data not available.")
            return

        # Stack the data: shape becomes (n_models, n_epochs, n_coeffs)
        full_evolution_data = np.stack(stats['coeffs_all_models'], axis=1)
        
        epochs = stats['epochs']
        num_models = self.n_ensemble

        fig, axs = plt.subplots(
            num_models, 1, 
            figsize=(14, 3 * num_models), 
            sharex=True, 
            squeeze=False # Ensures axs is always 2D
        )
        axs = axs.flatten() # Flatten to 1D array for easy iteration

        # Define the normalization once for all subplots
        norm = SymLogNorm(linthresh=1e-6)

        for i in range(num_models):
            ax = axs[i]
            heatmap_data = full_evolution_data[i, :, :]
            
            img = ax.imshow(
                heatmap_data.T, # Transpose to have epochs on x-axis, coeffs on y-axis
                aspect='auto',
                cmap='bwr',
                norm=norm,
                origin='lower',
                extent=[epochs[0], epochs[-1], -0.5, self.L_max + 0.5]
            )
            
            ax.set_title(f'Model {i}')
            ax.set_ylabel('Hermite Order $\ell$')
            ax.axhline(y=self.k, color='red', linestyle='--', lw=1, label=f'$k={self.k}$')
            fig.colorbar(img, ax=ax)
        
        axs[-1].set_xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/hermite_evolution_individual_heatmaps.png", dpi=300)
        plt.close()


        
        # ------------------------------------------------------------------
    # Analytic kernel eigen-value  K_ℓ(q_u,q_v)
    # ------------------------------------------------------------------
    def _K_theory(self, ell: int, q_u: float, q_v: float) -> float:
        """
        σ_a   – set to 1 because you initialise  a_i  with Var = 1/M₁
        d_ℓ   – Hermite coefficient of ReLU   (only ℓ ≡ 1 (mod 2) non-zero)
        For the numeric comparison any constant prefactor cancels,
        so we keep   σ_a² d_ℓ² = 1  below.
        """
        return math.factorial(ell) * q_u**(ell/2.0)        #  σ_a² d_ℓ² = 1
                                                        #  Irrelevant q_v part
    # ------------------------------------------------------------------
    def calculate_and_print_g_theory(self, max_ell: int = 12, gamma: float = 1.0):
        """
        Prints the large-N stiffness  g_ℓ  =
            N^{(1-γ)/2}  /  √K_ℓ(q_u,q_v)
        with   N = self.M1.   For NTK  γ=0 , for MF  γ=1 .
        """
        if not self.macro_observables['epochs']:
            print("Cannot compute g_theory – train the ensemble first."); return

        qu = self.macro_observables['qu_mean'][-1]      # ensemble mean at final epoch
        qv = self.macro_observables['qv_mean'][-1]
        N  = float(self.M1)

        print("\nTheoretical large-N stiffness  g_ℓ  (final macroscopic state)")
        g_dict = {}
        for ell in range(1, max_ell + 1):
            K_ell = self._K_theory(ell, qu, qv)
            g     = N**((1.0 - gamma)/2.0) / math.sqrt(K_ell)
            g_dict[ell] = g
            print(f"  ℓ = {ell:<2d} :  g = {g: .3e}")
        print("-" * 72)

        # (optional) write to JSON
        import json, os
        with open(os.path.join(self.save_dir, "g_theory_values.json"), "w") as fp:
            json.dump(g_dict, fp, indent=2)


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


    # ------------------------------------------------------------------
    def plot_all_coeff_evolution_ribbon(self):
        """
        Mean ± SEM for every ℓ ≤ L_max with a Viridis colour gradient.
        """
        print("Plotting all Hermite coefficients with ribbons (viridis)…")
        stats = self.hermite_evolution_stats
        if not stats['epochs']:
            print("Hermite data not available.");  return

        epochs   = np.array(stats['epochs'])
        coeffs   = np.stack(stats['coeffs_mean'], axis=0)
        coeffsSD = np.stack(stats['coeffs_std'],  axis=0)
        n_runs   = float(self.n_ensemble)

        viridis = plt.get_cmap('viridis')
        L_max   = coeffs.shape[1] - 1

        plt.figure(figsize=(12, 8))
        for ell in range(L_max + 1):
            col   = viridis(ell / L_max)
            mean  = coeffs[:, ell]
            sem   = coeffsSD[:, ell] / np.sqrt(n_runs)

            plt.plot(epochs, mean, lw=1.2, color=col, label=f'ℓ={ell}')
            plt.fill_between(epochs, mean - sem, mean + sem,
                            color=col, alpha=0.18)

        plt.xscale('log')
        plt.yscale('symlog', linthresh=1e-6)
        plt.xlabel('Epoch');  plt.ylabel(r'Ensemble mean  $c_\ell$')
        plt.title('Hermite coefficient evolution (mean ± SEM)')
        plt.grid(True, ls=':')
        # show legend only for the first few labels to reduce clutter
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:10], labels[:10], ncol=5, fontsize=8,
                loc='upper left', frameon=False)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/hermite_coeffs_all_ribbon.png", dpi=300)
        plt.close()
# ------------------------------------------------------------------


    def plot_raw_projection_evolution(self):
        """
        Plots the evolution of the raw Hermite projections p_l as a line plot.
        - X-axis: Epoch (log scale)
        - Y-axis: Projection value (symlog scale)
        - Lines are colored by Hermite order l.
        - A mean and standard deviation ribbon is plotted over all orders.
        """
        print("Plotting raw Hermite projection evolution as a line plot...")
        stats = self.raw_projection_stats
        if not stats['epochs']:
            print("Raw projection data not available.")
            return

        # Prepare data: shape (num_epochs, L_max + 1)
        heatmap_data = np.stack(stats['projections_mean'], axis=0)
        epochs = stats['epochs']
        l_max = heatmap_data.shape[1] - 1

        fig, ax = plt.subplots(figsize=(14, 8))
        colors = cm.get_cmap('viridis', l_max + 1)

        # Plot the evolution for each Hermite order l
        for l in range(l_max + 1):
            # We plot the absolute value to better visualize magnitude on a log scale
            ax.plot(epochs, np.abs(heatmap_data[:, l]), color=colors(l), label=f'$\ell={l}$' if l <= 10 else None, alpha=0.7)

        # Calculate and plot the mean and std dev over all l's for each epoch
        mean_over_l = np.mean(heatmap_data, axis=1)
        std_over_l = np.std(heatmap_data, axis=1)

        # Plot the mean line
        ax.plot(epochs, np.abs(mean_over_l), color='red', linestyle='--', linewidth=2, label='Mean over all $\ell$')
        
        # Plot the standard deviation ribbon around the mean
        ax.fill_between(
            epochs,
            np.abs(mean_over_l) - std_over_l,
            np.abs(mean_over_l) + std_over_l,
            color='red',
            alpha=0.2,
            label='Std Dev over all $\ell$'
        )

        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log') # Use log scale to see wide range of magnitudes
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Absolute Projection Value $|p_\ell|$ (log scale)')
        ax.set_title('Evolution of Raw Hermite Projections $|p_\ell| = |\langle f - \\langle f \\rangle, \\tilde{H}_\\ell \\rangle|$')
        ax.grid(True, which="both", linestyle=':')
        
        # Create a clean legend
        handles, labels = ax.get_legend_handles_labels()
        # Display legend for individual lines (l=0 to 10) and the mean/std dev
        display_handles = [h for h, l in zip(handles, labels) if not l.startswith('$\ell=') or int(l.split('=')[1][:-1]) <= 10]
        ax.legend(handles=display_handles, ncol=2)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/raw_projection_evolution_lines.png", dpi=300)
        plt.close()


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
    

    # ------------------------------------------------------------
# one global helper, outside the class
# ------------------------------------------------------------
 
    
    # ------------------------------------------------------------
# inside EnsembleParityExperiment
# ------------------------------------------------------------
    def hermite_coeff_relevant(self, model, n):
        """
        ⟨ f(x) - ⟨f⟩ ,  H̃_n(z_S(x)) ⟩   with   z_S = Σ_{j≤k} x_j / √k
        """
        # 1.  generate a *Gaussian* probe set
        X = torch.randn( 20000, self.d, device=self.device )   # N(0,1)
        zS = collective_coordinate(X, self.k)                  # (N,)

        # 2.  forward pass of the network
        with torch.no_grad():
            f = model(X).squeeze()
        f = f - f.mean()

        # 3.  Hermite polynomial on the *same* z_S
        Hn = EnsembleParityExperiment._orthonormal_hermite(n, zS)

        return (f * Hn).mean().item()
        # shape (n_samples,)


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


# --- ADD THESE NEW FUNCTIONS OUTSIDE THE CLASS ---

def collect_coeffs(model, X, L_max=14):
    """
    Collects the absolute value of the centered Hermite coefficients for a single model.
    """
    a, W = model.a.detach(), model.W1.detach()
    zs  = W @ X.T                       # shape (M1, n_test)
    f   = (a[:,None] * torch.relu(zs)).sum(0)/np.sqrt(len(a))
    f  -= f.mean()                      # Centre the function
    
    coeffs = []
    for l in range(L_max + 1):
        # Call the static method from the class
        H_l = EnsembleParityExperiment._orthonormal_hermite(l, zs).mean(0)
        # Note: we take the absolute value of the projection for this specific diagnostic
        coeffs.append(torch.mean(torch.abs(f * H_l)).item())
        
    return np.array(coeffs)

def plot_hermite_coefficient_errorbar(experiment):
    """
    Plots the mean of the final Hermite coefficients across the ensemble, 
    with error bars representing the standard error of the mean (SEM).
    This avoids the issue of sign-flipping between seeds.
    """
    print("Plotting final Hermite coefficients with SEM error bars...")
    stats = experiment.hermite_evolution_stats
    if not stats['epochs']:
        print("Hermite data not available.")
        return

    # Get coefficients from the final epoch
    final_coeffs_mean = stats['coeffs_mean'][-1]
    final_coeffs_std = stats['coeffs_std'][-1]
    n_ensemble = experiment.n_ensemble
    L_max = experiment.L_max

    plt.figure(figsize=(12, 7))
    plt.errorbar(
        x=np.arange(L_max + 1),
        y=final_coeffs_mean,
        yerr=final_coeffs_std / np.sqrt(n_ensemble),
        fmt='o-',
        capsize=5,
        label='Ensemble Mean ± SEM'
    )
    
    plt.axvline(x=experiment.k, color='red', linestyle='--', label=f'Relevant Channel (k={experiment.k})')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.yscale('symlog', linthresh=1e-5) # Use 'symlog' to handle values near zero
    plt.title('Final Hermite Coefficients (Ensemble Average)')
    plt.xlabel('Hermite Order $\ell$')
    plt.ylabel(r'Mean Coefficient $\langle c_\ell \rangle$')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{experiment.save_dir}/hermite_coeffs_final_errorbar.png", dpi=300)
    plt.close()

def run_d_scaling_diagnostic(base_config):
    """
    Runs experiments for various dimensions 'd' to check the scaling
    of irrelevant Hermite coefficients, comparing the empirical slope
    to the theoretical prediction of -l/2.
    """
    print("\n" + "="*80)
    print("Running d-scaling diagnostic for irrelevant channels...")
    
    dims = [50, 70,90, 105, 120]
    # Check a few irrelevant even channels beyond the relevant one
    suspect_channels = [l for l in (10, 12, 14) if l != base_config.get('k')]
                      
    for l in suspect_channels:
        amps = []
        print(f"\n--- Testing channel ℓ={l} ---")
        for d in dims:
            print(f"  Training for d={d}...")
            cfg = dict(base_config)
            cfg['d'] = d
            # Use a temporary directory for these short-lived experiments
            cfg['save_dir'] = f"{base_config['save_dir']}_d_scaling_temp"
            
            exp = EnsembleParityExperiment(**cfg)
            # Train just long enough for features to emerge
            exp.train(early_stop_corr=0.95) 
            
            # Collect the mean absolute coefficient for channel l
            c_vals = [collect_coeffs(m, exp.X_test, L_max=l)[l] for m in exp.models]
            c_mean_abs = np.mean(c_vals)
            amps.append(c_mean_abs)
            print(f"    d={d}, |c_{l}| ≈ {c_mean_abs:.6f}")
            
        # Perform a linear fit on the log-log data to find the slope
        # Use abs() to avoid errors if an average amplitude is negative
        log_dims = np.log(dims)
        log_amps = np.log(np.abs(amps))
        
        # Check for non-finite values before fitting
        if not np.all(np.isfinite(log_amps)):
            print(f"Could not calculate slope for ℓ={l} due to non-finite log(amplitude).")
            continue
            
        slope, _ = np.polyfit(log_dims, log_amps, 1)
        print(f"Result for ℓ={l}: Empirical slope = {slope:.2f} (Theory ≈ -{l/2:.1f})")
    print("="*80 + "\n")

def run_hermite_projection_sanity_check():
    """
    Runs a quick numeric cross-check on the Hermite projection logic
    using a synthetic function with a known spectrum, as described in Check 5.
    """
    print("\n" + "="*80)
    print("Running numeric sanity check on Hermite projection logic...")
    
    # Create a synthetic signal with a known Hermite decomposition
    z = torch.randn(200_000) # Use more samples for better precision
    f_known = (EnsembleParityExperiment._orthonormal_hermite(8, z) + 
               0.1 * EnsembleParityExperiment._orthonormal_hermite(2, z))

    print("Testing synthetic function: f(z) = H̃₈(z) + 0.1 * H̃₂(z)")
    print("Expected coefficients: c₀≈0, c₂≈0.1, c₈≈1.0")
    
    # Project the synthetic signal onto different Hermite polynomials
    for ell in (0, 1, 2, 7, 8, 9):
        # The projection formula is E[(f - E[f]) * H̃_l].
        # f_known is already zero-mean by construction, but we subtract for robustness.
        coeff = torch.mean(
            (f_known - f_known.mean()) * EnsembleParityExperiment._orthonormal_hermite(ell, z)
        )
        print(f"  Calculated c_{ell:<2} = {coeff.item():.6f}")
    print("="*80 + "\n")


def collective_coordinate(x, k):
    """
    z_S(x) = (1/sqrt(k)) * sum_{j=1..k} x_j
    """
    return x[:, :k].sum(dim=1) / math.sqrt(k)  
if __name__ == "__main__":
    # --- Base Experiment Configuration ---
    base_config = {
        'd': 100,
        'k': 10,
        'M1': 1024,
        'max_epochs': 250000, # Max epochs per run
        'batch_size': 1024,
        'learning_rate': 0.008,
        'tracker': 1000,
        'n_ensemble': 8,
        'n_parallel': 8,
        'weight_decay_W1': 1e-5,
        'weight_decay_a': 1e-5,
        'top_n_neurons': 100, 
        'kernel_n_samples': 100000,
        'device_id': 0,
        'gamma': 1.0,
        'a_init_scale': 1.0,
        'test_set_size': 20000,
        # Base directory where all run folders will be created
        'base_save_dir': "/home/goring/OnlineSGD/results_ana/2406_100_8_a1_walsh2" 
    }

    # --- Parameters to Iterate Over ---
    train_sizes_to_test = [20000,50000,100000,500000,1000000] 

    run_hermite_projection_sanity_check()

    # --- Main Loop to Run Experiments ---
    for p in train_sizes_to_test:
        print("\n" + "#"*80)
        print(f"# Starting Run for Training Set Size P = {p}")
        print("#"*80 + "\n")

        # Create a copy of the base config for this run
        run_config = base_config.copy()
        
        # Set the training set size for this specific run
        run_config['train_set_size'] = p
        
        # Create a unique save directory for this run
        run_save_dir = os.path.join(base_config['base_save_dir'], f"train_size_{p}")
        run_config['save_dir'] = run_save_dir
        
        # Remove the base dir key as the experiment class doesn't expect it
        del run_config['base_save_dir']

        # --- Run Experiment ---
        experiment = EnsembleParityExperiment(**run_config)
        
        # Train with the new loss-based convergence criteria
        # Stop if train loss < 0.001 for 10,000 consecutive epochs
        experiment.train(loss_threshold=1e-3, patience=2500)
        
        # --- Run Analysis and Plotting for this run ---
        experiment._compute_eigenvector_projections()
        experiment.evaluate_final_performance()
        experiment.plot_all()
        experiment.calculate_and_print_g_theory()
        # The run_and_plot_final_diagnostics() is likely old, I'm commenting it out
        # but you can re-enable it if needed.
        # experiment.run_and_plot_final_diagnostics() 

    print("\nAll experimental runs complete.")