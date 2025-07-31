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

# class TwoLayerReLUNet(torch.nn.Module):
#     def __init__(self, input_dim, hidden1_width, hidden2_width, W1_grad=True, W2_grad=True, a_grad=True, W1_init='random', k=None):
#         super(TwoLayerReLUNet, self).__init__()
#         # Initialize layers
#         if W1_init == 'random':
#             # Standard initialization
#             self.W1 = torch.nn.Parameter(torch.randn(hidden1_width, input_dim) / np.sqrt(input_dim))
#         elif W1_init == 'sparse' and k is not None:
#             # Initialize only the first k columns (relevant features), rest are zero
#             W1_init_tensor = torch.zeros(hidden1_width, input_dim) # Renamed to avoid conflict
#             W1_init_tensor[:, :k] = torch.randn(hidden1_width, k) / np.sqrt(input_dim)
#             self.W1 = torch.nn.Parameter(W1_init_tensor)
#         else:
#             # Fallback to random initialization
#             self.W1 = torch.nn.Parameter(torch.randn(hidden1_width, input_dim) / np.sqrt(input_dim))
        
#         self.W2 = torch.nn.Parameter(torch.randn(hidden2_width, hidden1_width) / np.sqrt(hidden1_width))
#         self.a = torch.nn.Parameter(torch.randn(hidden2_width) / np.sqrt(hidden2_width))
        
#         # Set requires_grad based on flags
#         self.W1.requires_grad = W1_grad
#         self.W2.requires_grad = W2_grad
#         self.a.requires_grad = a_grad
        
#     def forward(self, x):
#         # First hidden layer without bias
#         h1 = torch.relu(torch.matmul(x, self.W1.t()))
#         # Second hidden layer without bias
#         h2 = torch.relu(torch.matmul(h1, self.W2.t()))
#         # Output layer
#         output = torch.matmul(h2, self.a)
#         return output
    
#     def get_first_layer_activations(self, x):
#         return torch.relu(torch.matmul(x, self.W1.t()))
    
#     def get_second_layer_activations(self, x):
#         h1 = torch.relu(torch.matmul(x, self.W1.t()))
#         return torch.relu(torch.matmul(h1, self.W2.t()))
    
#     def get_first_layer_preactivations(self, x):
#         return torch.matmul(x, self.W1.t())
    
#     def get_second_layer_preactivations(self, x, h1=None):
#         if h1 is None:
#             h1 = torch.relu(torch.matmul(x, self.W1.t()))
#         return torch.matmul(h1, self.W2.t())

class TwoLayerReLUNet(torch.nn.Module):
    def __init__(self, input_dim, hidden1_width, hidden2_width, W1_grad=True, W2_grad=True, a_grad=True, W1_init='random', k=None):
        super(TwoLayerReLUNet, self).__init__()
        # Initialize layers
        if W1_init == 'random':
            # Standard initialization
            self.W1 = torch.nn.Parameter(torch.randn(hidden1_width, input_dim) / np.sqrt(input_dim))
        elif W1_init == 'sparse' and k is not None:
            # Initialize only the first k columns (relevant features), rest are zero
            W1_init_val = torch.zeros(hidden1_width, input_dim) # Use a different name
            W1_init_val[:, :k] = torch.randn(hidden1_width, k) / np.sqrt(input_dim)
            self.W1 = torch.nn.Parameter(W1_init_val)
        else:
            # Fallback to random initialization
            self.W1 = torch.nn.Parameter(torch.randn(hidden1_width, input_dim) / np.sqrt(input_dim))
            
        self.W2 = torch.nn.Parameter(torch.randn(hidden2_width, hidden1_width) / np.sqrt(hidden1_width))
        self.a = torch.nn.Parameter(torch.randn(hidden2_width) / np.sqrt(hidden2_width))
        
        # Set requires_grad based on flags
        self.W1.requires_grad = W1_grad
        self.W2.requires_grad = W2_grad
        self.a.requires_grad = a_grad
    
    def activation(self, x):
        return torch.relu(x)  # Square activation function
        
    def forward(self, x):
        # First hidden layer without bias
        h1 = self.activation(torch.matmul(x, self.W1.t()))
        # Second hidden layer without bias
        h2 = self.activation(torch.matmul(h1, self.W2.t()))
        # Output layer
        output = torch.matmul(h2, self.a)
        return output
    
    def get_first_layer_activations(self, x):
        return self.activation(torch.matmul(x, self.W1.t()))
    
    def get_second_layer_activations(self, x):
        h1 = self.activation(torch.matmul(x, self.W1.t()))
        return self.activation(torch.matmul(h1, self.W2.t()))
    
    def get_first_layer_preactivations(self, x):
        return torch.matmul(x, self.W1.t())
    
    def get_second_layer_preactivations(self, x, h1=None):
        if h1 is None:
            h1 = self.activation(torch.matmul(x, self.W1.t()))
        return torch.matmul(h1, self.W2.t())


class ParityNetAnalyzer:
    def __init__(self, d=30, k=6, M1=512, M2=512, learning_rate=0.01,
             batch_size=512, device_id=None, save_dir="parity_analysis",
             tracker=20, top_k_eigen=5, W1_grad=True, W2_grad=True, a_grad=True, W1_init='random',
             kernel_samples=100000):
        """
        Initialize the ParityNetAnalyzer to analyze neural networks learning parity functions
        """
        self.d = d
        self.k = k
        self.M1 = M1
        self.M2 = M2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tracker = tracker
        self.top_k_eigen = top_k_eigen
        self.W1_grad = W1_grad
        self.W2_grad = W2_grad
        self.a_grad = a_grad
        self.W1_init = W1_init
        self.kernel_samples = kernel_samples

        # Set device based on device_id
        if device_id is not None:
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.model = self._create_model()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

        # --- History Storage Dictionaries (Restored to full versions) ---
        self.loss_history = []
        self.correlation_history = []

        self.feature_correlations = {
            'epochs': [],
            'relevant_features': [],
            'irrelevant_features': [],
            'relevant_mean': [],
            'irrelevant_mean': [],
            'relevant_std': [],
            'irrelevant_std': [],
        }

        self.empirical_signal_noise = {
            'epochs': [],
            'empirical_signal': [],
            'empirical_noise': [],
            'empirical_ratio': [],
            'theoretical_signal': [],
            'theoretical_noise': [],
            'theoretical_ratio': [],
        }

        self.gradient_stats = {
            'epochs': [],
            'relevant_grad_mean': [],
            'irrelevant_grad_mean': [],
            'relevant_grad_std': [],
            'irrelevant_grad_std': [],
            'w1_grad_norm': [],
        }

        self.weight_snapshots = {'epochs': [], 'W1': [], 'W2': [], 'a': []}

        self.feature_importance = {'epochs': [], 'relevant_importance': [], 'irrelevant_importance': [], 'ratio': []}

        self.phase_transition = {
            'detected': False, 'epoch': None, 'correlation': None, 'feature_ratio': None,
            'theoretical_epoch': self.get_theoretical_transition_epoch(),
        }

        self.activation_stats = {'epochs': [], 'layer1_percent_active': [], 'layer2_percent_active': []}

        self.gradient_correlations = {
            'epochs': [],
            'rel_rel_corr': [],
            'rel_irrel_corr': [],
            'irrel_irrel_corr': [],
            'mean_rel_rel': [],
            'mean_rel_irrel': [],
            'mean_irrel_irrel': [],
        }

        self.hidden_target_correlations = {'epochs': [], 'layer1_corr_sum': [], 'layer2_corr_sum': []}

        self.gradient_eigen = {
            'epochs': [],
            'W1_eigenvalues': [],
            'W2_eigenvalues': [],
            'W1_eigenvectors': [],
            'W2_eigenvectors': [],
        }

        self.principal_angles = {'epochs': [], 'W1_angles': [], 'W2_angles': []}

        self.neuron_activation_evolution = {'epochs': [], 'layer1_activation_ratios': [], 'layer2_activation_ratios': []}

        self.neuron_target_correlation_evolution = {'epochs': [], 'layer1_neuron_correlations': [], 'layer2_neuron_correlations': []}

        self.macroscopic_quantities = {'epochs': [], 'r_values': [], 'p_values': []}

        self.s_m_squared_evolution = {'epochs': [], 's_squared_values': [], 'm_squared_values': [], 'phi1': [], 'phi2': []}

        # Corrected Condensation & Alignment Stats
        self.condensation_stats = {
            'epochs': [], 'cosine_similarity': [], 'relative_frobenius_norm': [],
        }
        self.second_layer_alignment_stats = {
            'epochs': [], 'cosine_similarity': [], 'phi2_emp': [],
        }

        # --- Other Initializations ---
        self.prior_W1_eigenvectors = None
        self.prior_W2_eigenvectors = None

        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, d)), dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)

        self.X_analysis = torch.tensor(np.random.choice([-1, 1], size=(5000, d)), dtype=torch.float32).to(self.device)
        self.y_analysis = self._target_function(self.X_analysis)

        print(f"ParityNetAnalyzer initialized on {self.device}")
        print(f"Analyzing {k}-parity function in {d} dimensions with ReLU activation")
        print(f"Network: {M1} → {M2} → 1")
        print(f"Batch size: {batch_size}, Tracking metrics every {tracker} epochs")
        print(f"Kernel samples for condensation analysis: {self.kernel_samples}")
        print(f"W1 trainable: {W1_grad}, W2 trainable: {W2_grad}, a trainable: {a_grad}")

    
    def plot_rank_one_condensation(self):
        """
        Plots the corrected diagnostics for the rank-one condensation assumption.
        """
        if not self.condensation_stats['epochs']:
            print("No rank-one condensation data available to plot.")
            return

        epochs = self.condensation_stats['epochs']
        # Use a 2-panel plot since we are skipping the eigenvalue analysis
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # --- Plot 1: Cosine Similarity ---
        ax1 = axs[0]
        ax1.plot(epochs, self.condensation_stats['cosine_similarity'], marker='o', linestyle='-', color='b')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_title('Condensation Diagnostic 1: Alignment of Signal Vector and Top PC')
        ax1.axhline(y=1.0, color='red', linestyle='--', label='Perfect Alignment (1.0)')
        ax1.legend(['cos($\\eta^*$, $u_1$)'])
        ax1.set_ylim(bottom=-0.05, top=1.05)
        if self.phase_transition['detected']:
            ax1.axvline(x=self.phase_transition['epoch'], color='black', linestyle=':', label=f'PT (Epoch {self.phase_transition["epoch"]})')
        ax1.grid(True, alpha=0.5)

        # --- Plot 2: Relative Frobenius Norm of Off-Diagonal Residual ---
        ax2 = axs[1]
        ax2.semilogy(epochs, self.condensation_stats['relative_frobenius_norm'], marker='x', linestyle='-', color='g')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Relative Frobenius Norm (log scale)')
        ax2.set_title('Condensation Diagnostic 2: Off-Diagonal Residual')
        if self.phase_transition['detected']:
            ax2.axvline(x=self.phase_transition['epoch'], color='black', linestyle=':')
        ax2.grid(True, which="both", alpha=0.5)
        ax2.legend(['$||M_1 C C^T - \\tilde{K}^{(1)}||_{F, off} / ||\\tilde{K}^{(1)}||_{F, off}$'])

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/rank_one_condensation_diagnostics_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()

    def compute_second_layer_alignment(self, epoch):
        """
        Tracks the alignment of the second-layer weights with the readout direction.
        """
        self.model.eval()

        # Recompute eta* for consistency, using a large number of samples
        X_kernel = torch.tensor(np.random.choice([-1, 1], size=(self.kernel_samples, self.d)),
                                dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_kernel = self._target_function(X_kernel).float()
            h1 = self.model.get_first_layer_activations(X_kernel)
            eta_star = (y_kernel.unsqueeze(1) * h1).mean(dim=0)
            eta_star /= torch.norm(eta_star) + 1e-12

            # Project W2 onto the signal direction eta* to get m
            W2 = self.model.W2.detach()
            m = W2 @ eta_star

            # Normalize m and the readout weights 'a'
            m_normalized = m / (torch.norm(m) + 1e-12)
            a = self.model.a.detach()
            a_normalized = a / (torch.norm(a) + 1e-12)

            # Compute cosine similarity
            cosine_sim = torch.abs(torch.dot(m_normalized, a_normalized)).item()

            # Compute empirical Phi2
            phi2_emp = torch.mean(m**2).item()

        # Store results
        self.second_layer_alignment_stats['epochs'].append(epoch)
        self.second_layer_alignment_stats['cosine_similarity'].append(cosine_sim)
        self.second_layer_alignment_stats['phi2_emp'].append(phi2_emp)

    def compute_rank_one_condensation(self, epoch):
        """
        Correctly and robustly verifies the rank-1 condensation by:
        1. Using the label-weighted mean activation vector (eta*).
        2. Centering activations before PCA.
        3. Computing the explicit off-diagonal kernel and residual for comparison.
        This version is memory-intensive and assumes a powerful GPU.
        """
        self.model.eval()
        print(f"\nEpoch {epoch}: Computing corrected rank-1 condensation stats...")

        # Use a large number of samples for eta* and PCA
        X_kernel = torch.tensor(np.random.choice([-1, 1], size=(self.kernel_samples, self.d)),
                                dtype=torch.float32).to(self.device)
        
        # These calculations can be done on the full sample set
        with torch.no_grad():
            y_kernel = self._target_function(X_kernel).float()
            h1 = self.model.get_first_layer_activations(X_kernel)

            # 1. Build the *label-weighted* mean direction (eta*)
            eta_star = (y_kernel.unsqueeze(1) * h1).mean(dim=0)
            eta_star /= torch.norm(eta_star) + 1e-12

            # 2. Centre the activations before PCA
            h1_c = h1 - h1.mean(dim=0, keepdim=True)
            try:
                u1 = torch.linalg.svd(h1_c, full_matrices=False).Vh[0]
                cosine_sim = torch.abs(torch.dot(eta_star, u1)).item()
            except torch.linalg.LinAlgError:
                print(f"Warning: SVD failed for condensation analysis at epoch {epoch}. Setting cosine to 0.")
                cosine_sim = 0.0
                
        # 3. Compare the *off-diagonal* part of the kernel. This is MEMORY INTENSIVE.
        try:
            with torch.no_grad():
                # Empirical kernel K_tilde
                K_tilde = (h1 @ h1.T) / self.M1
                
                # Rank-1 approximation CC
                C = h1 @ eta_star
                CC = torch.outer(C, C)
                
                residual = 1/self.M1 * CC - K_tilde

                # Zero out the diagonals in-place
                K_tilde.diagonal().zero_()
                residual.diagonal().zero_()
                
                # Compute norms and eigenvalues
                norm_K_tilde_offdiag = torch.norm(K_tilde, 'fro')
                rel_frob_norm = torch.norm(residual, 'fro') / (norm_K_tilde_offdiag + 1e-12)
                
                # Note: Eigendecomposition of the full kernel is very slow.
                # We will skip it for now to keep the training loop fast, as the other two
                # metrics (cosine similarity and Frobenius norm) are the most critical.
                # If needed, this can be enabled for a final, detailed analysis run.
                residual_lambda_max = torch.nan
                kernel_lambda_max = torch.nan
                
                rel_frob_norm = rel_frob_norm.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n--- WARNING: CUDA Out of Memory ---")
            print(f"Failed to allocate the ({self.kernel_samples}, {self.kernel_samples}) kernel matrix.")
            print(f"Reduce `kernel_samples` or use a machine with more GPU RAM.")
            print(f"Skipping Frobenius norm calculation for this epoch.")
            rel_frob_norm = torch.nan
            residual_lambda_max = torch.nan
            kernel_lambda_max = torch.nan
                
        # --- Store all results ---
        self.condensation_stats['epochs'].append(epoch)
        self.condensation_stats['cosine_similarity'].append(cosine_sim)
        self.condensation_stats['relative_frobenius_norm'].append(rel_frob_norm)
        # The keys for lambda_max are added here, even if they are NaN, to prevent KeyErrors
        self.condensation_stats.setdefault('residual_lambda_max', []).append(residual_lambda_max)
        self.condensation_stats.setdefault('kernel_lambda_max', []).append(kernel_lambda_max)

        print(f"Corrected Condensation -> Cos(η*, PC1): {cosine_sim:.4f}, Off-Diag Rel Frob Norm: {rel_frob_norm:.4e}")


    def _create_model(self):
        """Create a two-layer network with the specified parameters"""
        model = TwoLayerReLUNet( # Class name is still TwoLayerReLUNet but uses x^2 activation
            self.d, 
            self.M1, 
            self.M2, 
            W1_grad=self.W1_grad, 
            W2_grad=self.W2_grad,
            a_grad=self.a_grad,
            W1_init=self.W1_init, 
            k=self.k
        ).to(self.device)
        return model
    
    def _target_function(self, x):
        """Compute the k-sparse parity function on the first k inputs"""
        return torch.prod(x[:, :self.k], dim=1)
    
    def get_theoretical_transition_epoch(self):
        """
        Compute the theoretical epoch for phase transition based on the formula:
        T_k = Θ(d^(k/2)/η)
        
        Using a more precise formula: (d^(k-1)/2)^2 / B where B is batch size
        """
        # The constant factor is approximate and would need to be calibrated
        factor = 0.1 
        theoretical_epoch = factor * (self.d ** (self.k/2)) / self.learning_rate
        return int(theoretical_epoch)

    def compute_feature_correlations(self, epoch, batch_size=None):
        """
        Compute correlations C_l between each feature and the output
        C_l = E[(h_θ(x) - y) · x_l]
        
        Args:
            epoch: Current epoch number
            batch_size: Number of samples to use (None for all X_analysis)
        """
        self.model.eval()
        
        if batch_size is None or batch_size > self.X_analysis.shape[0]:
            X = self.X_analysis
            y_true = self.y_analysis
        else:
            # Take a random subset
            indices = torch.randperm(self.X_analysis.shape[0])[:batch_size]
            X = self.X_analysis[indices]
            y_true = self.y_analysis[indices]
        
        # Compute model predictions
        with torch.no_grad():
            y_pred = self.model(X)
            error = y_pred - y_true
        
        # Compute correlation for each feature
        feature_corrs = []
        for l in range(self.d):
            # Compute C_l = E[(h_θ(x) - y) · x_l]
            corr = torch.mean(error * X[:, l]).item()
            feature_corrs.append(corr)
        
        # Separate correlations for relevant vs irrelevant features
        relevant_corrs = feature_corrs[:self.k]
        irrelevant_corrs = feature_corrs[self.k:]
        
        # Compute statistics
        relevant_mean = np.mean(np.abs(relevant_corrs)) if relevant_corrs else 0.0
        irrelevant_mean = np.mean(np.abs(irrelevant_corrs)) if irrelevant_corrs else 0.0
        relevant_std = np.std(np.abs(relevant_corrs)) if relevant_corrs else 0.0
        irrelevant_std = np.std(np.abs(irrelevant_corrs)) if irrelevant_corrs else 0.0
        
        # Store the correlations
        self.feature_correlations['epochs'].append(epoch)
        self.feature_correlations['relevant_features'].append(relevant_corrs)
        self.feature_correlations['irrelevant_features'].append(irrelevant_corrs)
        self.feature_correlations['relevant_mean'].append(relevant_mean)
        self.feature_correlations['irrelevant_mean'].append(irrelevant_mean)
        self.feature_correlations['relevant_std'].append(relevant_std)
        self.feature_correlations['irrelevant_std'].append(irrelevant_std)
        
        return relevant_mean, irrelevant_mean

    def compute_macroscopic_quantities(self, epoch, num_samples=5000):
        """
        Compute macroscopic quantities r_i and p_j at the current epoch.
        
        Args:
            epoch: Current epoch
            num_samples: Number of random samples to use for estimating p_j
        """
        # Get weights
        W1 = self.model.W1.detach() # Keep on device for now
        W2 = self.model.W2.detach() # Keep on device for now
        
        # Compute r_i for each neuron in the first layer
        # r_i(t) = (1/k) * sum_{l=1}^k W^{(1)}_{il}(t)
        if self.k > 0:
            r_values_tensor = torch.mean(W1[:, :self.k], axis=1)
        else:
            r_values_tensor = torch.zeros(self.M1, device=self.device)
        r_values = r_values_tensor.cpu().numpy()
        
        # Use self.X_analysis for consistency, or generate random samples
        if num_samples >= self.X_analysis.shape[0]:
            X = self.X_analysis
        else:
             X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                             dtype=torch.float32).to(self.device)
        
        # Compute activations of first layer neurons
        with torch.no_grad():
            h1 = self.model.get_first_layer_activations(X) # Shape (num_samples, M1)
        
        # Compute p_j for each neuron in the second layer
        # p_j(t) = (1/M1) * sum_{i=1}^{M1} W^{(2)}_{ji}(t) * E_x[σ(u_i(x;t))]
        mean_activations = torch.mean(h1, dim=0)  # E_x[σ(u_i(x;t))], Shape (M1)
        
        # p_values_tensor = torch.matmul(W2, mean_activations) / self.M1 # This is sum, not mean over W2 entries
        # Corrected p_j: p_j = E_i [ W2_ji * E_x[h1_i] ]
        p_values_tensor = torch.mean(W2 * mean_activations.unsqueeze(0), dim=1) # mean_activations needs to be (1, M1) for broadcasting
        p_values = p_values_tensor.cpu().numpy()
        
        # Store values
        self.macroscopic_quantities['epochs'].append(epoch)
        self.macroscopic_quantities['r_values'].append(r_values)
        self.macroscopic_quantities['p_values'].append(p_values)
        
        return r_values, p_values

    def compute_s_m_squared(self, epoch):
        """
        Compute s_j^2 for the first layer and m_a^2 for the second layer,
        as well as their means phi1 and phi2.

        s_j = W_j^(1) . e, where e is a vector proportional to (1,1,..,1,0,0..0) for relevant features, normalized by 1/sqrt(k).
        phi1 = (1/M1) * sum_j s_j^2.

        m_a = W_a^(2) . eta, where eta = (1/n) * sum_mu y_mu * h_mu^(1) / ||h_mu^(1)||.
        phi2 = (1/M2) * sum_a m_a^2.

        Args:
            epoch: Current epoch number.
        """
        self.model.eval()
        W1 = self.model.W1.detach()  # Shape (M1, d)
        W2 = self.model.W2.detach()  # Shape (M2, M1)

        # --- Compute s_j^2 and phi1 ---
        e_vec = torch.zeros(self.d, device=self.device, dtype=W1.dtype)
        if self.k > 0: # Avoid division by zero if k=0 or sqrt(k)=0
            e_vec[:self.k] = 1.0 / np.sqrt(self.k) 
        
        s_values = torch.matmul(W1, e_vec)  # Shape (M1)
        s_squared_array = (s_values**2).cpu().numpy()
        phi1_val = np.mean(s_squared_array) if s_squared_array.size > 0 else 0.0

        # --- Compute m_a^2 and phi2 ---
        X_eta = self.X_analysis # Use pre-defined analysis samples
        y_eta = self.y_analysis # And their targets
        
        with torch.no_grad():
            h1_activations = self.model.get_first_layer_activations(X_eta)  # Shape (num_samples_eta, M1)
        
        y_eta_tensor = y_eta.unsqueeze(1).to(self.device, dtype=h1_activations.dtype) # Shape (num_samples_eta, 1)
        
        h1_norms = torch.norm(h1_activations, p=2, dim=1, keepdim=True)
        # Avoid division by zero for norms
        safe_h1_norms = h1_norms + 1e-12 # Add a small epsilon
        normalized_h1 = h1_activations / safe_h1_norms
        
        eta_contributions = y_eta_tensor * normalized_h1  # Element-wise multiply, broadcasts y_eta_tensor
        eta_vec = torch.mean(eta_contributions, dim=0)  # Shape (M1)
        
        m_values = torch.matmul(W2, eta_vec)  # Shape (M2)
        m_squared_array = (m_values**2).cpu().numpy()
        phi2_val = np.mean(m_squared_array) if m_squared_array.size > 0 else 0.0

        # Store results
        self.s_m_squared_evolution['epochs'].append(epoch)
        self.s_m_squared_evolution['s_squared_values'].append(s_squared_array)
        self.s_m_squared_evolution['m_squared_values'].append(m_squared_array)
        self.s_m_squared_evolution['phi1'].append(phi1_val)
        self.s_m_squared_evolution['phi2'].append(phi2_val)

        return s_squared_array, m_squared_array, phi1_val, phi2_val

    def compute_signal_noise(self, epoch, empirical_gradients=None):
        """
        Compute both empirical and theoretical signal and noise terms
        
        Args:
            epoch: Current epoch number
            empirical_gradients: Dictionary with gradient statistics (if None, use stored values)
        """
        # Theoretical signal and noise calculations
        eta = self.learning_rate
        T = epoch + 1  # Add 1 to avoid T=0
        B = self.batch_size
        
        # Theoretical signal strength for relevant features
        theoretical_signal = eta * T * (self.d ** (-(self.k-1)/2)) if self.k > 0 else eta * T # Handle k=0 case
        
        # Theoretical noise level
        theoretical_noise = eta * np.sqrt(T / B) if B > 0 else eta * np.sqrt(T) # Handle B=0 case
        
        # Theoretical ratio
        theoretical_ratio = theoretical_signal / max(theoretical_noise, 1e-10)
        
        # Empirical calculations
        if empirical_gradients is not None:
            # Use the provided gradient statistics
            empirical_signal = empirical_gradients['relevant_grad_mean']  
            empirical_noise = empirical_gradients['irrelevant_grad_std'] # Using std of irrelevant as noise proxy
        elif len(self.gradient_stats['epochs']) > 0 and self.gradient_stats['epochs'][-1] == epoch:
            # Use the latest stored gradient statistics
            empirical_signal = self.gradient_stats['relevant_grad_mean'][-1]
            empirical_noise = self.gradient_stats['irrelevant_grad_std'][-1]
        else:
            # Default fallback (should be avoided by ensuring gradients are computed)
            empirical_signal = 0.0 
            empirical_noise = 0.0
        
        # Empirical ratio
        empirical_ratio = empirical_signal / max(empirical_noise, 1e-10)
        
        # Store values
        self.empirical_signal_noise['epochs'].append(epoch)
        self.empirical_signal_noise['empirical_signal'].append(empirical_signal)
        self.empirical_signal_noise['empirical_noise'].append(empirical_noise)
        self.empirical_signal_noise['empirical_ratio'].append(empirical_ratio)
        self.empirical_signal_noise['theoretical_signal'].append(theoretical_signal)
        self.empirical_signal_noise['theoretical_noise'].append(theoretical_noise)
        self.empirical_signal_noise['theoretical_ratio'].append(theoretical_ratio)
        
        return {
            'empirical_signal': empirical_signal,
            'empirical_noise': empirical_noise,
            'empirical_ratio': empirical_ratio,
            'theoretical_signal': theoretical_signal,
            'theoretical_noise': theoretical_noise,
            'theoretical_ratio': theoretical_ratio
        }

    def compute_feature_importance(self, epoch):
        """
        Calculate and store the relative importance of relevant vs irrelevant features
        """
        W1 = self.model.W1.detach().cpu().numpy()
        
        # Average absolute weight for relevant features (first k)
        if self.k > 0:
            relevant_importance = np.mean(np.abs(W1[:, :self.k])) if W1[:, :self.k].size > 0 else 0.0
        else:
            relevant_importance = 0.0

        # Average absolute weight for irrelevant features
        if self.d > self.k:
            irrelevant_importance = np.mean(np.abs(W1[:, self.k:])) if W1[:, self.k:].size > 0 else 0.0
        else: # No irrelevant features
            irrelevant_importance = 0.0

        # Compute ratio of relevant to irrelevant feature importance
        ratio = relevant_importance / max(irrelevant_importance, 1e-10)
        
        # Store values
        self.feature_importance['epochs'].append(epoch)
        self.feature_importance['relevant_importance'].append(relevant_importance)
        self.feature_importance['irrelevant_importance'].append(irrelevant_importance)
        self.feature_importance['ratio'].append(ratio)
        
        return relevant_importance, irrelevant_importance, ratio

    def take_weight_snapshot(self, epoch):
        """
        Take a snapshot of weight matrices at the current epoch
        """
        self.weight_snapshots['epochs'].append(epoch)
        self.weight_snapshots['W1'].append(self.model.W1.detach().cpu().numpy())
        self.weight_snapshots['W2'].append(self.model.W2.detach().cpu().numpy())
        self.weight_snapshots['a'].append(self.model.a.detach().cpu().numpy())

    def compute_gradient_statistics(self, epoch):
        """
        Compute and store gradient statistics for W1 
        to empirically measure signal and noise
        """
        # Only compute if W1 is trainable
        if not self.W1_grad:
            # Return default values
            return {
                'relevant_grad_mean': 0.0,
                'irrelevant_grad_mean': 0.0,
                'relevant_grad_std': 0.0,
                'irrelevant_grad_std': 0.0,
                'w1_grad_norm': 0.0,
                'W1_grads': np.zeros((self.M1, self.d)),
                'W2_grads': np.zeros((self.M2, self.M1)) if self.W2_grad else None 
            }
            
        # Generate a batch of random samples
        X = torch.tensor(np.random.choice([-1, 1], size=(self.batch_size, self.d)), 
                           dtype=torch.float32).to(self.device)
        y = self._target_function(X)
        
        # Forward pass
        self.model.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        
        # Backward pass
        loss.backward()
        
        # Get gradients for W1
        W1_grads_tensor = self.model.W1.grad if self.model.W1.grad is not None else torch.zeros_like(self.model.W1)
        W1_grads = W1_grads_tensor.detach().cpu().numpy()
        
        W2_grads_tensor = self.model.W2.grad if self.W2_grad and self.model.W2.grad is not None else torch.zeros_like(self.model.W2)
        W2_grads = W2_grads_tensor.detach().cpu().numpy() if self.W2_grad else np.zeros((self.M2, self.M1))
        
        # Calculate gradient statistics for relevant features
        if self.k > 0:
            relevant_grads = W1_grads[:, :self.k]
            relevant_grad_mean = np.mean(np.abs(relevant_grads)) if relevant_grads.size > 0 else 0.0
            relevant_grad_std = np.std(relevant_grads) if relevant_grads.size > 0 else 0.0 # Std of actual grads, not abs
        else:
            relevant_grads = np.array([])
            relevant_grad_mean = 0.0
            relevant_grad_std = 0.0

        if self.d > self.k :
            irrelevant_grads = W1_grads[:, self.k:]
            irrelevant_grad_mean = np.mean(np.abs(irrelevant_grads)) if irrelevant_grads.size > 0 else 0.0
            irrelevant_grad_std = np.std(irrelevant_grads) if irrelevant_grads.size > 0 else 0.0 # Std of actual grads
        else:
            irrelevant_grads = np.array([])
            irrelevant_grad_mean = 0.0
            irrelevant_grad_std = 0.0
        
        # Compute overall gradient norm
        w1_grad_norm = np.linalg.norm(W1_grads) if W1_grads.size > 0 else 0.0
        
        # Store gradient statistics
        self.gradient_stats['epochs'].append(epoch)
        self.gradient_stats['relevant_grad_mean'].append(relevant_grad_mean)
        self.gradient_stats['irrelevant_grad_mean'].append(irrelevant_grad_mean)
        self.gradient_stats['relevant_grad_std'].append(relevant_grad_std)
        self.gradient_stats['irrelevant_grad_std'].append(irrelevant_grad_std)
        self.gradient_stats['w1_grad_norm'].append(w1_grad_norm)
        
        # Return gradient statistics for signal-noise calculation
        return {
            'relevant_grad_mean': relevant_grad_mean,
            'irrelevant_grad_mean': irrelevant_grad_mean,
            'relevant_grad_std': relevant_grad_std, # This is std of grads
            'irrelevant_grad_std': irrelevant_grad_std, # This is std of grads (used as noise proxy)
            'w1_grad_norm': w1_grad_norm,
            'W1_grads': W1_grads,
            'W2_grads': W2_grads
        }

    def compute_activation_statistics(self, epoch, num_samples=10000):
        """
        Compute statistics on x^2 activations: what percentage of neurons have non-zero output
        in each layer when given random inputs, and track per-neuron activation ratios.
        
        Args:
            epoch: Current epoch
            num_samples: Number of random input vectors to sample
        """
        self.model.eval()
        
        # Generate random inputs
        X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                           dtype=torch.float32).to(self.device)
        
        # Compute activations
        with torch.no_grad():
            # Layer 1 activations
            h1 = self.model.get_first_layer_activations(X)
            # Layer 2 activations
            h2 = self.model.get_second_layer_activations(X)
            
            # For x^2 activation, "active" means > 0 (i.e., pre-activation was non-zero)
            layer1_percent_active = torch.mean((h1 > 1e-9).float()).item() * 100 # Using a small threshold
            layer2_percent_active = torch.mean((h2 > 1e-9).float()).item() * 100
            
            # Calculate per-neuron activation ratios (fraction of samples for which neuron > threshold)
            layer1_activation_ratios = torch.mean((h1 > 1e-9).float(), dim=0).cpu().numpy()
            layer2_activation_ratios = torch.mean((h2 > 1e-9).float(), dim=0).cpu().numpy()
        
        # Store overall activation percentages
        self.activation_stats['epochs'].append(epoch)
        self.activation_stats['layer1_percent_active'].append(layer1_percent_active)
        self.activation_stats['layer2_percent_active'].append(layer2_percent_active)
        
        # Store per-neuron activation ratios
        self.neuron_activation_evolution['epochs'].append(epoch)
        self.neuron_activation_evolution['layer1_activation_ratios'].append(layer1_activation_ratios)
        self.neuron_activation_evolution['layer2_activation_ratios'].append(layer2_activation_ratios)
        
        return layer1_percent_active, layer2_percent_active

    def track_individual_neuron_target_correlations(self, epoch, num_samples=5000):
        """
        Track the correlation ρᵢ = E_x[hᵢ(x)·f*(x)] between each individual neuron's 
        activation and the target function across training epochs.
        
        Args:
            epoch: Current epoch
            num_samples: Number of samples to use for correlation calculation
        """
        self.model.eval()
        
        # Use self.X_analysis for consistency
        if num_samples >= self.X_analysis.shape[0]:
            X = self.X_analysis
            y_target = self.y_analysis
        else:
            X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                               dtype=torch.float32).to(self.device)
            y_target = self._target_function(X) # Compute target function values
        y_target_float = y_target.float() # Ensure float for correlation

        # Compute hidden activations
        with torch.no_grad():
            # Layer 1 activations
            h1 = self.model.get_first_layer_activations(X)
            # Layer 2 activations
            h2 = self.model.get_second_layer_activations(X)
            
            # Compute correlation for each neuron in layer 1
            layer1_correlations = np.zeros(self.M1)
            for i in range(self.M1):
                # Compute ρᵢ = E_x[hᵢ(x)·f*(x)]
                # Pearson correlation: Cov(X,Y) / (std(X)std(Y))
                # Here, just expectation of product as per formula E[h_i(x) * f_S(x)]
                neuron_corr = torch.mean(h1[:, i] * y_target_float).item()
                layer1_correlations[i] = neuron_corr
            
            # Compute correlation for each neuron in layer 2
            layer2_correlations = np.zeros(self.M2)
            for i in range(self.M2):
                neuron_corr = torch.mean(h2[:, i] * y_target_float).item()
                layer2_correlations[i] = neuron_corr
        
        # Store results
        self.neuron_target_correlation_evolution['epochs'].append(epoch)
        self.neuron_target_correlation_evolution['layer1_neuron_correlations'].append(layer1_correlations)
        self.neuron_target_correlation_evolution['layer2_neuron_correlations'].append(layer2_correlations)
        
        return layer1_correlations, layer2_correlations

    def compute_gradient_correlations(self, epoch, W1_grads=None, num_samples=2000):
        """
        Compute correlations between gradients for different features as described:
        ρj,j' = E[∇W1,ijL⋅∇W1,ij'L] / sqrt(E[(∇W1,ijL)^2]⋅E[(∇W1,ij'L)^2])
        Approximated by sample correlation of gradient vectors for features.
        
        Args:
            epoch: Current epoch
            W1_grads: Pre-computed gradients (if None, will compute)
            num_samples: Number of samples to use for computing new gradients (if needed)
        """
        # Skip if W1 is not trainable
        if not self.W1_grad:
            self.gradient_correlations['epochs'].append(epoch)
            self.gradient_correlations['rel_rel_corr'].append(np.zeros((self.k, self.k)))
            self.gradient_correlations['rel_irrel_corr'].append(np.zeros((self.k, self.d - self.k)))
            self.gradient_correlations['irrel_irrel_corr'].append(np.zeros((self.d - self.k, self.d - self.k)))
            self.gradient_correlations['mean_rel_rel'].append(0.0)
            self.gradient_correlations['mean_rel_irrel'].append(0.0)
            self.gradient_correlations['mean_irrel_irrel'].append(0.0)
            
            return {
                'mean_rel_rel': 0.0,
                'mean_rel_irrel': 0.0,
                'mean_irrel_irrel': 0.0
            }
            
        if W1_grads is None:
            # We need to compute the gradients
            grad_stats_dict = self.compute_gradient_statistics(epoch) # This computes grads
            W1_grads = grad_stats_dict['W1_grads']
            if W1_grads is None: # Should not happen if W1_grad is True
                 W1_grads = np.zeros((self.M1, self.d))

        # Extract relevant/irrelevant feature gradients
        rel_grads = W1_grads[:, :self.k]  # M1 x k
        irrel_grads = W1_grads[:, self.k:]  # M1 x (d-k)
        
        # Initialize correlation matrices
        rel_rel_corr = np.zeros((self.k, self.k))
        rel_irrel_corr = np.zeros((self.k, self.d - self.k))
        irrel_irrel_corr = np.zeros((self.d - self.k, self.d - self.k))
        
        # Compute correlations for relevant-relevant pairs
        if self.k > 1:
            for j1 in range(self.k):
                for j2 in range(j1 + 1, self.k):
                    grad_j1 = rel_grads[:, j1] 
                    grad_j2 = rel_grads[:, j2] 
                    if np.std(grad_j1) > 1e-9 and np.std(grad_j2) > 1e-9: # Avoid issues with zero variance
                        corr = np.corrcoef(grad_j1, grad_j2)[0, 1]
                    else:
                        corr = 0.0
                    if np.isnan(corr): corr = 0.0
                    rel_rel_corr[j1, j2] = corr
                    rel_rel_corr[j2, j1] = corr
            np.fill_diagonal(rel_rel_corr, 1.0)

        # Compute correlations for relevant-irrelevant pairs
        if self.k > 0 and self.d - self.k > 0:
            for j1 in range(self.k):
                for j2 in range(self.d - self.k):
                    grad_j1 = rel_grads[:, j1]
                    grad_j2 = irrel_grads[:, j2]
                    if np.std(grad_j1) > 1e-9 and np.std(grad_j2) > 1e-9:
                        corr = np.corrcoef(grad_j1, grad_j2)[0, 1]
                    else:
                        corr = 0.0
                    if np.isnan(corr): corr = 0.0
                    rel_irrel_corr[j1, j2] = corr
        
        # Compute correlations for irrelevant-irrelevant pairs
        if self.d - self.k > 1:
            for j1 in range(self.d - self.k):
                for j2 in range(j1 + 1, self.d - self.k):
                    grad_j1 = irrel_grads[:, j1]
                    grad_j2 = irrel_grads[:, j2]
                    if np.std(grad_j1) > 1e-9 and np.std(grad_j2) > 1e-9:
                        corr = np.corrcoef(grad_j1, grad_j2)[0, 1]
                    else:
                        corr = 0.0
                    if np.isnan(corr): corr = 0.0
                    irrel_irrel_corr[j1, j2] = corr
                    irrel_irrel_corr[j2, j1] = corr
            np.fill_diagonal(irrel_irrel_corr, 1.0)
        
        # Compute mean correlations for each type (excluding self-correlations along diagonals)
        mean_rel_rel = np.mean(np.abs(rel_rel_corr[~np.eye(rel_rel_corr.shape[0], dtype=bool)])) if self.k > 1 else 0.0
        mean_rel_irrel = np.mean(np.abs(rel_irrel_corr)) if (self.k > 0 and self.d - self.k > 0) else 0.0
        mean_irrel_irrel = np.mean(np.abs(irrel_irrel_corr[~np.eye(irrel_irrel_corr.shape[0], dtype=bool)])) if self.d - self.k > 1 else 0.0
        
        # Store results
        self.gradient_correlations['epochs'].append(epoch)
        self.gradient_correlations['rel_rel_corr'].append(rel_rel_corr)
        self.gradient_correlations['rel_irrel_corr'].append(rel_irrel_corr)
        self.gradient_correlations['irrel_irrel_corr'].append(irrel_irrel_corr)
        self.gradient_correlations['mean_rel_rel'].append(mean_rel_rel)
        self.gradient_correlations['mean_rel_irrel'].append(mean_rel_irrel)
        self.gradient_correlations['mean_irrel_irrel'].append(mean_irrel_irrel)
        
        return {
            'mean_rel_rel': mean_rel_rel,
            'mean_rel_irrel': mean_rel_irrel,
            'mean_irrel_irrel': mean_irrel_irrel
        }

    def compute_hidden_target_correlations(self, epoch, num_samples=5000):
        """
        Calculate and track ∑(i=1 to M1)|E[h_i(x)⋅f_S(x)]| where:
        - h_i is the hidden activation (using x^2)
        - f_S is the k-parity function
        
        Args:
            epoch: Current epoch
            num_samples: Number of samples to use
        """
        self.model.eval()
        
        # Use self.X_analysis for consistency
        if num_samples >= self.X_analysis.shape[0]:
            X = self.X_analysis
            y_target = self.y_analysis
        else:
            X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                               dtype=torch.float32).to(self.device)
            y_target = self._target_function(X) # Compute target function values
        y_target_float = y_target.float()


        # Compute hidden activations
        with torch.no_grad():
            # Layer 1 activations
            h1 = self.model.get_first_layer_activations(X)
            # Layer 2 activations
            h2 = self.model.get_second_layer_activations(X)
            
            # Compute correlation for each neuron in layer 1
            layer1_corr_sum = 0.0
            for i in range(self.M1):
                # Compute E[h_i(x)⋅f_S(x)]
                neuron_corr = torch.mean(h1[:, i] * y_target_float).item()
                layer1_corr_sum += abs(neuron_corr)
            
            # Compute correlation for each neuron in layer 2
            layer2_corr_sum = 0.0
            for i in range(self.M2):
                # Compute E[h_i(x)⋅f_S(x)]
                neuron_corr = torch.mean(h2[:, i] * y_target_float).item()
                layer2_corr_sum += abs(neuron_corr)
        
        # Store results
        self.hidden_target_correlations['epochs'].append(epoch)
        self.hidden_target_correlations['layer1_corr_sum'].append(layer1_corr_sum)
        self.hidden_target_correlations['layer2_corr_sum'].append(layer2_corr_sum)
        
        return layer1_corr_sum, layer2_corr_sum

    def compute_gradient_eigen_statistics(self, epoch, W1_grads=None, W2_grads=None, num_samples=2000):
        """
        Compute and track the top k eigenvalues/vectors of the gradient matrices
        
        Args:
            epoch: Current epoch
            W1_grads: Pre-computed W1 gradients (if None, will compute)
            W2_grads: Pre-computed W2 gradients (if None, will compute)
            num_samples: Number of samples to use for computing new gradients (if needed)
        """
        # Skip computation if both layers are frozen
        if not self.W1_grad and not self.W2_grad:
            self.gradient_eigen['epochs'].append(epoch)
            self.gradient_eigen['W1_eigenvalues'].append(np.zeros(self.top_k_eigen))
            self.gradient_eigen['W2_eigenvalues'].append(np.zeros(self.top_k_eigen))
            self.gradient_eigen['W1_eigenvectors'].append(np.zeros((self.d, self.top_k_eigen)))
            self.gradient_eigen['W2_eigenvectors'].append(np.zeros((self.M1, self.top_k_eigen)))
            
            self.principal_angles['epochs'].append(epoch) # Also update this for consistency
            self.principal_angles['W1_angles'].append(np.zeros(self.top_k_eigen))
            self.principal_angles['W2_angles'].append(np.zeros(self.top_k_eigen))

            return {
                'W1_eigenvalues': np.zeros(self.top_k_eigen),
                'W2_eigenvalues': np.zeros(self.top_k_eigen),
                'W1_angles': None,
                'W2_angles': None
            }
            
        if W1_grads is None or (self.W2_grad and W2_grads is None) : # Only need W2_grads if W2 is trainable
            grad_stats_dict = self.compute_gradient_statistics(epoch) #This gets fresh grads
            W1_grads = grad_stats_dict['W1_grads']
            if self.W2_grad:
                W2_grads = grad_stats_dict['W2_grads']
            else: # W2 not trainable, use zeros
                W2_grads = np.zeros((self.M2, self.M1))

        # Ensure W1_grads and W2_grads are not None before proceeding
        if W1_grads is None: W1_grads = np.zeros((self.M1, self.d))
        if W2_grads is None and self.W2_grad: W2_grads = np.zeros((self.M2, self.M1))


        # For W1 (M1 x d), the gradient covariance is (d x d)
        W1_cov = np.dot(W1_grads.T, W1_grads) if self.W1_grad and W1_grads.size > 0 else np.zeros((self.d, self.d))
        
        # For W2 (M2 x M1), the gradient covariance is (M1 x M1)
        W2_cov = np.dot(W2_grads.T, W2_grads) if self.W2_grad and W2_grads.size > 0 else np.zeros((self.M1, self.M1))
        
        # Compute eigenvalues and eigenvectors
        # Ensure k is not larger than the matrix dimensions
        k_w1 = min(self.top_k_eigen, W1_cov.shape[0])
        k_w2 = min(self.top_k_eigen, W2_cov.shape[0])

        # For W1 gradient covariance
        if self.W1_grad and W1_grads.size > 0:
            W1_eigenvalues_all, W1_eigenvectors_all = np.linalg.eigh(W1_cov)
            W1_indices = np.argsort(W1_eigenvalues_all)[::-1] # Sort descending
            W1_eigenvalues = W1_eigenvalues_all[W1_indices[:k_w1]]
            W1_eigenvectors = W1_eigenvectors_all[:, W1_indices[:k_w1]]
        else:
            W1_eigenvalues = np.zeros(k_w1)
            W1_eigenvectors = np.zeros((self.d, k_w1))
        
        # For W2 gradient covariance
        if self.W2_grad and W2_grads.size > 0:
            W2_eigenvalues_all, W2_eigenvectors_all = np.linalg.eigh(W2_cov)
            W2_indices = np.argsort(W2_eigenvalues_all)[::-1] # Sort descending
            W2_eigenvalues = W2_eigenvalues_all[W2_indices[:k_w2]]
            W2_eigenvectors = W2_eigenvectors_all[:, W2_indices[:k_w2]]
        else:
            W2_eigenvalues = np.zeros(k_w2)
            W2_eigenvectors = np.zeros((self.M1, k_w2))
        
        # Pad with zeros if k_w1/k_w2 < self.top_k_eigen
        if k_w1 < self.top_k_eigen:
            W1_eigenvalues = np.pad(W1_eigenvalues, (0, self.top_k_eigen - k_w1), 'constant')
            W1_eigenvectors = np.pad(W1_eigenvectors, ((0,0), (0, self.top_k_eigen - k_w1)), 'constant')
        if k_w2 < self.top_k_eigen:
            W2_eigenvalues = np.pad(W2_eigenvalues, (0, self.top_k_eigen - k_w2), 'constant')
            W2_eigenvectors = np.pad(W2_eigenvectors, ((0,0), (0, self.top_k_eigen - k_w2)), 'constant')


        # Compute principal angles if we have previous eigenvectors
        W1_angles = None
        W2_angles = None
        
        if self.prior_W1_eigenvectors is not None and self.W1_grad and W1_eigenvectors.shape[1] > 0 and self.prior_W1_eigenvectors.shape[1] > 0:
             # Ensure subspaces have the same number of columns (min of current and prior k)
            num_cols_w1 = min(W1_eigenvectors.shape[1], self.prior_W1_eigenvectors.shape[1])
            if num_cols_w1 > 0:
                try:
                    W1_angles = subspace_angles(self.prior_W1_eigenvectors[:, :num_cols_w1], W1_eigenvectors[:, :num_cols_w1])
                except ValueError as e:
                    print(f"Warning: ValueError computing W1 subspace angles: {e}")
                    W1_angles = np.zeros(num_cols_w1) # Or handle as appropriate
            else: # if num_cols is 0
                 W1_angles = np.array([]) # Empty array if no common dimension
        
        if self.prior_W2_eigenvectors is not None and self.W2_grad and W2_eigenvectors.shape[1] > 0 and self.prior_W2_eigenvectors.shape[1] > 0:
            num_cols_w2 = min(W2_eigenvectors.shape[1], self.prior_W2_eigenvectors.shape[1])
            if num_cols_w2 > 0:
                try:
                    W2_angles = subspace_angles(self.prior_W2_eigenvectors[:, :num_cols_w2], W2_eigenvectors[:, :num_cols_w2])
                except ValueError as e:
                    print(f"Warning: ValueError computing W2 subspace angles: {e}")
                    W2_angles = np.zeros(num_cols_w2)
            else: # if num_cols is 0
                 W2_angles = np.array([]) # Empty array if no common dimension

        # Pad angles array to top_k_eigen length for consistent storage
        W1_angles_padded = np.zeros(self.top_k_eigen)
        if W1_angles is not None and len(W1_angles) > 0:
            W1_angles_padded[:len(W1_angles)] = W1_angles
            
        W2_angles_padded = np.zeros(self.top_k_eigen)
        if W2_angles is not None and len(W2_angles) > 0:
            W2_angles_padded[:len(W2_angles)] = W2_angles

        if W1_angles is not None or W2_angles is not None or not self.principal_angles['epochs'] : # Store even if angles are None to keep epochs aligned
            self.principal_angles['epochs'].append(epoch)
            self.principal_angles['W1_angles'].append(W1_angles_padded)
            self.principal_angles['W2_angles'].append(W2_angles_padded)
        
        # Update prior eigenvectors for next computation
        if self.W1_grad and W1_eigenvectors.shape[1] > 0: self.prior_W1_eigenvectors = W1_eigenvectors
        if self.W2_grad and W2_eigenvectors.shape[1] > 0: self.prior_W2_eigenvectors = W2_eigenvectors
        
        # Store results
        self.gradient_eigen['epochs'].append(epoch)
        self.gradient_eigen['W1_eigenvalues'].append(W1_eigenvalues)
        self.gradient_eigen['W2_eigenvalues'].append(W2_eigenvalues)
        self.gradient_eigen['W1_eigenvectors'].append(W1_eigenvectors)
        self.gradient_eigen['W2_eigenvectors'].append(W2_eigenvectors)
        
        return {
            'W1_eigenvalues': W1_eigenvalues,
            'W2_eigenvalues': W2_eigenvalues,
            'W1_angles': W1_angles_padded if W1_angles is not None else None, # Return padded for consistency if computed
            'W2_angles': W2_angles_padded if W2_angles is not None else None
        }

    def train(self, n_epochs=10000, early_stop_corr=0.9999):
        """
        Train the network and track relevant metrics
        
        Args:
            n_epochs: Number of epochs to train
            early_stop_corr: Correlation threshold for early stopping
        
        Returns:
            final_correlation: Final correlation achieved
            stopping_epoch: Epoch at which training stopped
        """
        print(f"Starting training for {n_epochs} epochs...")
        
        # Take initial snapshot and metrics
        self.take_weight_snapshot(0)
        self.compute_feature_importance(0)
        self.compute_feature_correlations(0)
        self.compute_second_layer_alignment(0)
        #self.compute_rank_one_condensation(0)
        
        # Compute initial gradient statistics
        grad_stats = self.compute_gradient_statistics(0)
        self.compute_signal_noise(0, grad_stats)
        
        # Compute new metrics at epoch 0
        self.compute_activation_statistics(0)
        self.compute_gradient_correlations(0, grad_stats['W1_grads'])
        self.compute_hidden_target_correlations(0)
        self.compute_gradient_eigen_statistics(0, grad_stats['W1_grads'], grad_stats['W2_grads'])
        self.track_individual_neuron_target_correlations(0)
        self.compute_macroscopic_quantities(0)
        self.compute_s_m_squared(0)
        self.compute_rank_one_condensation(0) # ADDED: Compute initial condensation stats
        
        # For timing
        start_time = time.time()
        
        for epoch in tqdm(range(1, n_epochs + 1), desc="Training Progress"):
            # Training step
            self.model.train()
            
            # Generate random batch
            X = torch.tensor(np.random.choice([-1, 1], size=(self.batch_size, self.d)),
                            dtype=torch.float32).to(self.device)
            y = self._target_function(X)
            
            # Forward pass
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            self.loss_history.append(loss.item())
            
            # Evaluation metrics at intervals defined by tracker
            # Also compute on last epoch
            if epoch % self.tracker == 0 or epoch == n_epochs:
                self.model.eval()
                with torch.no_grad():
                    preds = self.model(self.X_test)
                    test_loss = self.criterion(preds, self.y_test).item() # Use criterion for consistency
                    
                    # Check if preds has variance before calculating correlation
                    preds_squeezed = preds.squeeze()
                    if preds_squeezed.ndim == 0: # Handle scalar prediction case if batch size is 1 for test
                        preds_squeezed = preds_squeezed.unsqueeze(0)

                    if torch.var(preds_squeezed) == 0 or torch.isnan(torch.var(preds_squeezed)) or len(preds_squeezed) <=1 :
                        correlation = 0.0
                    else:
                        try:
                            # Ensure y_test is also 1D for corrcoef
                            y_test_squeezed = self.y_test.squeeze()
                            if y_test_squeezed.ndim == 0:
                                y_test_squeezed = y_test_squeezed.unsqueeze(0)
                            
                            correlation_matrix = torch.corrcoef(torch.stack([preds_squeezed, y_test_squeezed]))
                            correlation = correlation_matrix[0, 1].item()
                            if torch.isnan(torch.tensor(correlation)):
                                correlation = 0.0
                        except Exception as e:
                            preds_np = preds_squeezed.cpu().numpy()
                            targets_np = self.y_test.cpu().numpy()
                            if np.var(preds_np) > 1e-9 and np.var(targets_np) > 1e-9 and len(preds_np) > 1:
                                correlation = np.corrcoef(preds_np, targets_np)[0, 1]
                                if np.isnan(correlation): correlation = 0.0
                            else:
                                correlation = 0.0
                
                self.correlation_history.append((epoch, correlation))
                
                # Check for phase transition based on correlation
                if not self.phase_transition['detected'] and correlation > 0.99: # Threshold for phase transition
                    self.phase_transition['detected'] = True
                    self.phase_transition['epoch'] = epoch
                    self.phase_transition['correlation'] = correlation
                    
                    # Record feature importance ratio at transition
                    if self.feature_importance['epochs']:
                        closest_idx = min(range(len(self.feature_importance['epochs'])),
                                        key=lambda i: abs(self.feature_importance['epochs'][i] - epoch))
                        self.phase_transition['feature_ratio'] = self.feature_importance['ratio'][closest_idx]
                    
                    print(f"Phase transition detected at epoch {epoch} with correlation {correlation:.4f}")
                
                # Compute original metrics
                self.compute_feature_importance(epoch)
                self.compute_feature_correlations(epoch)
                
                # Compute gradient statistics to get empirical signal and noise
                grad_stats = self.compute_gradient_statistics(epoch)
                self.compute_signal_noise(epoch, grad_stats)
                
                # Take weight snapshot
                self.take_weight_snapshot(epoch)
                
                # Compute new metrics
                self.compute_activation_statistics(epoch)
                self.compute_gradient_correlations(epoch, grad_stats['W1_grads'])
                self.compute_hidden_target_correlations(epoch)
                self.compute_gradient_eigen_statistics(epoch, grad_stats['W1_grads'], grad_stats['W2_grads'])
                self.track_individual_neuron_target_correlations(epoch)
                self.compute_macroscopic_quantities(epoch)
                self.compute_s_m_squared(epoch)
                self.compute_rank_one_condensation(epoch)
                self.compute_second_layer_alignment(epoch)

                
                # Print progress
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}: MSE={test_loss:.6f}, Correlation={correlation:.4f}, Time={elapsed:.1f}s")
                
                # Early stopping
                if correlation > early_stop_corr:
                    print(f"Early stopping at epoch {epoch} with correlation {correlation:.4f}")
                    return correlation, epoch
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test)
            final_loss = self.criterion(preds, self.y_test).item()
            
            preds_squeezed = preds.squeeze()
            if preds_squeezed.ndim == 0: preds_squeezed = preds_squeezed.unsqueeze(0)
            y_test_squeezed = self.y_test.squeeze()
            if y_test_squeezed.ndim == 0: y_test_squeezed = y_test_squeezed.unsqueeze(0)

            if torch.var(preds_squeezed) > 1e-9 and torch.var(y_test_squeezed) > 1e-9 and len(preds_squeezed)>1:
                final_correlation_matrix = torch.corrcoef(torch.stack([preds_squeezed, y_test_squeezed]))
                final_correlation = final_correlation_matrix[0, 1].item()
                if torch.isnan(torch.tensor(final_correlation)): final_correlation = 0.0
            else:
                final_correlation = 0.0

        print("Training completed!")
        print(f"Final MSE: {final_loss:.6f}")
        print(f"Final correlation: {final_correlation:.4f}")
        
        return final_correlation, epoch if 'epoch' in locals() else n_epochs
    
    # ... [All existing plot methods: plot_macroscopic_quantities_evolution, ..., plot_neuron_activation_trajectories] ...
    # ... [All existing analysis methods: analyze_neuron_activations, ..., _plot_detailed_neuron_connections] ...
    # Make sure they are here in the actual class

    def plot_macroscopic_quantities_evolution(self):
        """
        Plot the evolution of macroscopic quantities r_i and p_j using
        a heatmap visualization similar to neuron_activation_evolution.
        """
        if not self.macroscopic_quantities['epochs']:
            print("No macroscopic quantities data available")
            return
        
        # Extract data
        epochs = self.macroscopic_quantities['epochs']
        r_values_list = self.macroscopic_quantities['r_values']
        p_values_list = self.macroscopic_quantities['p_values']
        
        if not r_values_list or not p_values_list:
            print("Empty r_values or p_values in macroscopic_quantities.")
            return

        r_data = np.array(r_values_list)
        p_data = np.array(p_values_list)

        # Create figure with two subplots - one for r_i and one for p_j
        fig, axs = plt.subplots(2, 1, figsize=(15, 12))
        
        # Setup for colormaps
        cmap = plt.cm.RdBu_r # Diverging colormap centered at 0
        
        # Determine max absolute value for symmetric colormap
        max_abs_r = max(np.max(np.abs(r_data)) if r_data.size > 0 else 0.001, 0.001)
        max_abs_p = max(np.max(np.abs(p_data)) if p_data.size > 0 else 0.001, 0.001)
        
        # Plot r_i evolution (Layer 1)
        ax1 = axs[0]
        im1 = ax1.imshow(r_data, aspect='auto', cmap=cmap, 
                         vmin=-max_abs_r, vmax=max_abs_r,
                         extent=[0, self.M1, epochs[-1], epochs[0]])  # x=neurons, y=epochs
        
        ax1.set_title(f'First Layer Macroscopic Quantity $r_i$ Evolution (d={self.d}, k={self.k})')
        ax1.set_xlabel('Neuron Index in Layer 1 ($i$)')
        ax1.set_ylabel('Epoch')
        
        if self.phase_transition['detected']:
            ax1.axhline(y=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7,
                        label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax1.legend()
        
        fig.colorbar(im1, ax=ax1, label='$r_i = \\frac{1}{k}\\sum_{l=1}^k W^{(1)}_{il}(t)$')
        
        # Plot p_j evolution (Layer 2)
        ax2 = axs[1]
        im2 = ax2.imshow(p_data, aspect='auto', cmap=cmap,
                         vmin=-max_abs_p, vmax=max_abs_p,
                         extent=[0, self.M2, epochs[-1], epochs[0]])  # x=neurons, y=epochs
        
        ax2.set_title(f'Second Layer Macroscopic Quantity $p_j$ Evolution (d={self.d}, k={self.k})')
        ax2.set_xlabel('Neuron Index in Layer 2 ($j$)')
        ax2.set_ylabel('Epoch')
        
        if self.phase_transition['detected']:
            ax2.axhline(y=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7,
                        label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax2.legend()
        
        fig.colorbar(im2, ax=ax2, label='$p_j = E_i [W^{(2)}_{ji} \\cdot E_x[\\sigma(u_i(x))] ]$') # Updated label
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/macroscopic_quantities_evolution_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_second_layer_alignment(self):
        """
        Plots the diagnostics for the second-layer alignment over epochs.
        """
        if not self.second_layer_alignment_stats['epochs']:
            print("No second-layer alignment data available to plot.")
            return

        epochs = self.second_layer_alignment_stats['epochs']
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # Plot 1: Cosine Similarity
        ax1 = axs[0]
        ax1.plot(epochs, self.second_layer_alignment_stats['cosine_similarity'], marker='o', linestyle='-', color='purple')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_title(f'Second-Layer Alignment: Cosine(m, a) (d={self.d}, k={self.k})')
        ax1.axhline(y=1.0, color='red', linestyle='--', label='Perfect Alignment (1.0)')
        ax1.set_ylim(bottom=-0.05, top=1.05)
        if self.phase_transition['detected']:
            ax1.axvline(x=self.phase_transition['epoch'], color='black', linestyle=':', label=f'PT (Epoch {self.phase_transition["epoch"]})')
        ax1.grid(True, alpha=0.5)
        ax1.legend()

        # Plot 2: Empirical Phi2
        ax2 = axs[1]
        ax2.semilogy(epochs, self.second_layer_alignment_stats['phi2_emp'], marker='x', linestyle='-', color='teal')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value (log scale)')
        ax2.set_title('Empirical $\\Phi_2 = E[m^2]$ where $m = W^{(2)} \\eta^*$')
        if self.phase_transition['detected']:
            ax2.axvline(x=self.phase_transition['epoch'], color='black', linestyle=':')
        ax2.grid(True, which="both", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/second_layer_alignment_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()






    def plot_s_m_squared_evolution(self):
        """
        Plot the evolution of s_j^2 and m_a^2 using heatmaps.
        s_j^2 = (W_j^(1) . e)^2
        m_a^2 = (W_a^(2) . eta)^2
        """
        if not self.s_m_squared_evolution['epochs']:
            print("No s_j^2/m_a^2 evolution data available")
            return

        epochs = self.s_m_squared_evolution['epochs']
        s_sq_values_list = self.s_m_squared_evolution['s_squared_values'] 
        m_sq_values_list = self.s_m_squared_evolution['m_squared_values']

        if not s_sq_values_list or not m_sq_values_list:
            print("Empty s_squared_values or m_squared_values lists.")
            return

        s_sq_data = np.array(s_sq_values_list) # Shape (num_tracked_epochs, M1)
        m_sq_data = np.array(m_sq_values_list) # Shape (num_tracked_epochs, M2)


        fig, axs = plt.subplots(2, 1, figsize=(15, 12))

        # --- Plot s_j^2 evolution (Layer 1 related) ---
        ax1 = axs[0]
        vmax_s = max(np.max(s_sq_data) if s_sq_data.size > 0 else 0.001, 0.001)
        
        cmap_s = plt.cm.viridis 
        norm_s = Normalize(vmin=0, vmax=vmax_s)

        im1 = ax1.imshow(s_sq_data, aspect='auto', cmap=cmap_s, norm=norm_s,
                         extent=[0, self.M1, epochs[-1], epochs[0]]) 
        
        e_def = "e \\propto (1_k, 0_{d-k}) / \\sqrt{k}" 
        ax1.set_title(f'First Layer $s_j^2 = (W_j^{{(1)}} \\cdot e)^2$ Evolution ({e_def}) (d={self.d}, k={self.k})')
        ax1.set_xlabel('Neuron Index in Layer 1 ($j$)')
        ax1.set_ylabel('Epoch')

        if self.phase_transition['detected']:
            ax1.axhline(y=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7,
                        label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax1.legend()
        
        fig.colorbar(im1, ax=ax1, label='$s_j^2$ Value')

        # --- Plot m_a^2 evolution (Layer 2 related) ---
        ax2 = axs[1]
        vmax_m = max(np.max(m_sq_data) if m_sq_data.size > 0 else 0.001, 0.001)

        cmap_m = plt.cm.viridis
        norm_m = Normalize(vmin=0, vmax=vmax_m)

        im2 = ax2.imshow(m_sq_data, aspect='auto', cmap=cmap_m, norm=norm_m,
                         extent=[0, self.M2, epochs[-1], epochs[0]])

        eta_def = "\\eta = E_x[y \\cdot h^{(1)}(x) / ||h^{(1)}(x)||]"
        ax2.set_title(f'Second Layer $m_a^2 = (W_a^{{(2)}} \\cdot \\eta)^2$ Evolution ($\\eta$: {eta_def}) (d={self.d}, k={self.k})')
        ax2.set_xlabel('Neuron Index in Layer 2 ($a$)')
        ax2.set_ylabel('Epoch')

        if self.phase_transition['detected']:
            ax2.axhline(y=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7,
                        label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax2.legend()
            
        fig.colorbar(im2, ax=ax2, label='$m_a^2$ Value')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/s_m_squared_evolution_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()

    def plot_phi_values_evolution(self):
        """
        Plot the evolution of phi1 and phi2 (means of s_j^2 and m_a^2).
        phi1 = E_j[s_j^2]
        phi2 = E_a[m_a^2]
        """
        if not self.s_m_squared_evolution['epochs'] or \
           not self.s_m_squared_evolution['phi1'] or \
           not self.s_m_squared_evolution['phi2']:
            print("No phi1/phi2 evolution data available or lists are empty")
            return

        epochs = self.s_m_squared_evolution['epochs']
        phi1_values = self.s_m_squared_evolution['phi1']
        phi2_values = self.s_m_squared_evolution['phi2']

        plt.figure(figsize=(12, 8))
        
        plt.plot(epochs, phi1_values, marker='o', linestyle='-', label='$\\phi_1 = E_j[s_j^2]$ (Layer 1 based)')
        plt.plot(epochs, phi2_values, marker='x', linestyle='--', label='$\\phi_2 = E_a[m_a^2]$ (Layer 2 based)')

        if self.phase_transition['detected']:
            pt_epoch = self.phase_transition['epoch']
            plt.axvline(x=pt_epoch, color='red', linestyle=':', alpha=0.9,
                        label=f'Phase Transition (Epoch {pt_epoch})')
            
            # Mark values at phase transition
            try: # Ensure epoch list is not empty for np.argmin
                if epochs:
                    idx = np.argmin(np.abs(np.array(epochs) - pt_epoch))
                    plt.scatter([epochs[idx]], [phi1_values[idx]], s=100, color='blue', marker='*', zorder=5, label=f'$\\phi_1$ at PT')
                    plt.scatter([epochs[idx]], [phi2_values[idx]], s=100, color='orange', marker='*', zorder=5, label=f'$\\phi_2$ at PT')
            except ValueError:
                 print("Warning: Could not find phase transition epoch in phi_values plot.")


        plt.xlabel('Epoch')
        plt.yscale('log')  # Linear scale for phi values
        plt.ylabel('Value of $\\phi_1, \\phi_2$')
        plt.title(f'Evolution of Collective Fields $\\phi_1$ and $\\phi_2$ (d={self.d}, k={self.k})')
        plt.legend()
        plt.grid(True, alpha=0.5)
        # Consider if yscale log is always appropriate. Values can be zero.
        # plt.yscale('log') 
        # Using a small positive floor for log scale if values can be zero or negative
        # For s_j^2 and m_a^2, values are non-negative. Means phi1, phi2 also non-negative.
        # A linear scale might be better initially, or symlog if values are very small.
        # If using log, ensure values are positive or handle non-positive appropriately.
        # For now, let's use linear, can be changed to log if needed.
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/phi_values_evolution_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()

    # ... [The rest of your existing plot and analysis methods] ...
    # Ensure plot_macroscopic_quantities_trajectories, plot_neuron_target_correlation_evolution, etc. are present
    # The placeholder methods are just to indicate where they would go.
    # You should have the full definitions of these methods as in your original provided code.

    def plot_macroscopic_quantities_trajectories(self):
        if not self.macroscopic_quantities['epochs']:
            print("No macroscopic quantities data available")
            return
        epochs = self.macroscopic_quantities['epochs']
        r_values_all = np.array(self.macroscopic_quantities['r_values'])
        p_values_all = np.array(self.macroscopic_quantities['p_values'])

        if r_values_all.size == 0 or p_values_all.size == 0:
            print("Empty r_values or p_values arrays.")
            return

        fig1, ax1 = plt.subplots(figsize=(15, 8))
        max_abs_r = np.max(np.abs(r_values_all), axis=0) if r_values_all.shape[0] > 0 else np.array([])
        significant_r_neurons = np.argsort(max_abs_r)[-20:] if max_abs_r.size > 0 else []
        cmap_r = plt.cm.viridis
        for idx, i in enumerate(significant_r_neurons):
            r_trajectory = r_values_all[:, i]
            color_val = max_abs_r[i] / np.max(max_abs_r) if np.max(max_abs_r) > 0 else 0
            color = cmap_r(color_val)
            ax1.plot(epochs, r_trajectory, '-', color=color, linewidth=2, label=f'Neuron {i}' if idx < 10 else None)
        if self.phase_transition['detected']:
            ax1.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7, label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title(f'Layer 1 Macroscopic Quantity $r_i$ Trajectories (d={self.d}, k={self.k})')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('$r_i$'); ax1.grid(True, alpha=0.3); ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/r_trajectories_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(15, 8))
        max_abs_p = np.max(np.abs(p_values_all), axis=0) if p_values_all.shape[0] > 0 else np.array([])
        significant_p_neurons = np.argsort(max_abs_p)[-20:] if max_abs_p.size > 0 else []
        cmap_p = plt.cm.viridis
        for idx, j_idx in enumerate(significant_p_neurons): # Renamed j to j_idx
            p_trajectory = p_values_all[:, j_idx]
            color_val = max_abs_p[j_idx] / np.max(max_abs_p) if np.max(max_abs_p) > 0 else 0
            color = cmap_p(color_val)
            ax2.plot(epochs, p_trajectory, '-', color=color, linewidth=2, label=f'Neuron {j_idx}' if idx < 10 else None)
        if self.phase_transition['detected']:
            ax2.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7, label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title(f'Layer 2 Macroscopic Quantity $p_j$ Trajectories (d={self.d}, k={self.k})')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('$p_j$'); ax2.grid(True, alpha=0.3); ax2.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/p_trajectories_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(15, 8))
        mean_abs_r = np.mean(np.abs(r_values_all), axis=1) if r_values_all.size > 0 else np.array([0]*len(epochs))
        mean_abs_p = np.mean(np.abs(p_values_all), axis=1) if p_values_all.size > 0 else np.array([0]*len(epochs))
        ax3.plot(epochs, mean_abs_r, '-', color='blue', linewidth=2, label='Mean $|r_i|$')
        ax3.plot(epochs, mean_abs_p, '-', color='green', linewidth=2, label='Mean $|p_j|$')
        if self.phase_transition['detected']:
            ax3.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7, label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
        ax3.set_title(f'Mean Absolute Macroscopic Quantities (d={self.d}, k={self.k})')
        ax3.set_xlabel('Epoch'); ax3.set_ylabel('Mean Absolute Value'); ax3.grid(True, alpha=0.3); ax3.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/mean_macroscopic_quantities_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig3)

    def plot_neuron_target_correlation_evolution(self):
        if not self.neuron_target_correlation_evolution['epochs']:
            print("No neuron-target correlation evolution data available")
            return
        epochs = self.neuron_target_correlation_evolution['epochs']
        layer1_corrs_list = self.neuron_target_correlation_evolution['layer1_neuron_correlations']
        layer2_corrs_list = self.neuron_target_correlation_evolution['layer2_neuron_correlations']

        if not layer1_corrs_list or not layer2_corrs_list: return
        layer1_data = np.array(layer1_corrs_list)
        layer2_data = np.array(layer2_corrs_list)

        fig, axs = plt.subplots(2, 1, figsize=(15, 12))
        vmax_abs = max(np.max(np.abs(layer1_data)) if layer1_data.size > 0 else 0, 
                       np.max(np.abs(layer2_data)) if layer2_data.size > 0 else 0, 0.001)
        cmap = plt.cm.RdBu_r; norm = Normalize(vmin=-vmax_abs, vmax=vmax_abs)

        ax1 = axs[0]
        im1 = ax1.imshow(layer1_data, aspect='auto', cmap=cmap, norm=norm, extent=[0, self.M1, epochs[-1], epochs[0]])
        ax1.set_title(f'First Layer Neuron-Target Correlation Evolution (d={self.d}, k={self.k})')
        ax1.set_xlabel('Neuron Index'); ax1.set_ylabel('Epoch')
        if self.phase_transition['detected']:
            ax1.axhline(y=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7, label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax1.legend()
        fig.colorbar(im1, ax=ax1, label='Correlation with Target $\\rho_i$')

        ax2 = axs[1]
        im2 = ax2.imshow(layer2_data, aspect='auto', cmap=cmap, norm=norm, extent=[0, self.M2, epochs[-1], epochs[0]])
        ax2.set_title(f'Second Layer Neuron-Target Correlation Evolution (d={self.d}, k={self.k})')
        ax2.set_xlabel('Neuron Index'); ax2.set_ylabel('Epoch')
        if self.phase_transition['detected']:
            ax2.axhline(y=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7, label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax2.legend()
        fig.colorbar(im2, ax=ax2, label='Correlation with Target $\\rho_i$')
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/neuron_target_correlation_evolution_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_neuron_target_correlation_trajectories(self):
        if not self.neuron_target_correlation_evolution['epochs']: return
        epochs = np.array(self.neuron_target_correlation_evolution['epochs'])
        layer1_corrs_all = np.array(self.neuron_target_correlation_evolution['layer1_neuron_correlations'])
        layer2_corrs_all = np.array(self.neuron_target_correlation_evolution['layer2_neuron_correlations'])

        for layer_num, layer_corrs, M_layer in [(1, layer1_corrs_all, self.M1), (2, layer2_corrs_all, self.M2)]:
            if layer_corrs.size == 0: continue
            max_abs_corrs = np.max(np.abs(layer_corrs), axis=0) if layer_corrs.shape[0] > 0 else np.array([])
            strong_neurons = np.argsort(max_abs_corrs)[-20:] if max_abs_corrs.size > 0 else []
            
            fig, ax = plt.subplots(figsize=(15, 8))
            cmap_pos = plt.cm.Reds; cmap_neg = plt.cm.Blues
            for neuron_idx in strong_neurons:
                corr_trajectory = layer_corrs[:, neuron_idx]
                final_corr = corr_trajectory[-1] if len(corr_trajectory)>0 else 0
                color = cmap_pos(min(abs(final_corr) * 1.5, 0.9)) if final_corr >= 0 else cmap_neg(min(abs(final_corr) * 1.5, 0.9))
                linestyle = '-' if final_corr >=0 else '--'
                alpha_val = min(0.7 + abs(final_corr) * 0.3, 1.0)
                ax.plot(epochs, corr_trajectory, linestyle=linestyle, color=color, alpha=alpha_val, linewidth=2, label=f'Neuron {neuron_idx}: {final_corr:.3f}')
            if self.phase_transition['detected']:
                ax.axvline(x=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7, label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_title(f'Layer {layer_num} Neuron-Target Correlation Trajectories (d={self.d}, k={self.k})')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Correlation with Target $\\rho_i$'); ax.grid(True, alpha=0.3)
            if len(strong_neurons) <= 10: ax.legend(loc='upper left', bbox_to_anchor=(1,1))
            plt.tight_layout(); plt.savefig(f"{self.save_dir}/layer{layer_num}_neuron_target_correlation_trajectories_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig)

            fig_abs, ax_abs = plt.subplots(figsize=(15, 8))
            for neuron_idx in strong_neurons:
                abs_corr_trajectory = np.abs(layer_corrs[:, neuron_idx])
                max_corr = np.max(abs_corr_trajectory) if len(abs_corr_trajectory) > 0 else 0
                color = plt.cm.viridis(min(max_corr * 1.5, 0.9))
                ax_abs.plot(epochs, abs_corr_trajectory, '-', color=color, alpha=min(0.7 + max_corr * 0.3, 1.0), linewidth=2, label=f'Neuron {neuron_idx}: {max_corr:.3f}')
            if self.phase_transition['detected']:
                ax_abs.axvline(x=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7, label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax_abs.set_title(f'Layer {layer_num} Absolute Neuron-Target Correlation Trajectories (d={self.d}, k={self.k})')
            ax_abs.set_xlabel('Epoch'); ax_abs.set_ylabel('Absolute Correlation with Target $|\\rho_i|$'); ax_abs.grid(True, alpha=0.3)
            if len(strong_neurons) <= 10: ax_abs.legend(loc='upper left', bbox_to_anchor=(1,1))
            plt.tight_layout(); plt.savefig(f"{self.save_dir}/layer{layer_num}_abs_neuron_target_correlation_trajectories_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig_abs)

    def plot_neuron_target_correlation_distribution(self):
        if not self.neuron_target_correlation_evolution['epochs']: return
        epochs = np.array(self.neuron_target_correlation_evolution['epochs'])
        layer1_corrs_all = np.array(self.neuron_target_correlation_evolution['layer1_neuron_correlations'])
        layer2_corrs_all = np.array(self.neuron_target_correlation_evolution['layer2_neuron_correlations'])

        if len(epochs) == 0: return
        selected_indices = []
        epoch_labels = []
        if len(epochs) >=3:
            if self.phase_transition['detected']:
                pt_idx = np.argmin(np.abs(epochs - self.phase_transition['epoch']))
                selected_indices = sorted(list(set([0, pt_idx, len(epochs)-1]))) # Ensure unique and sorted indices
                epoch_labels = ["Initial" if i==0 else f"Phase Transition (Epoch {epochs[pt_idx]})" if i == pt_idx else "Final" for i in selected_indices]
                # Fix labels if pt_idx is 0 or len(epochs)-1
                temp_labels = []
                if 0 in selected_indices: temp_labels.append("Initial")
                if self.phase_transition['detected'] and pt_idx in selected_indices and pt_idx != 0 and pt_idx != len(epochs)-1 : temp_labels.append(f"PT (Epoch {epochs[pt_idx]})")
                if len(epochs)-1 in selected_indices and len(epochs)-1 !=0 and (not self.phase_transition['detected'] or len(epochs)-1 != pt_idx) : temp_labels.append("Final")
                if len(temp_labels) != len(selected_indices): # Fallback for complex overlaps
                    epoch_labels = [f"Epoch {epochs[i]}" for i in selected_indices]

            else:
                selected_indices = [0, len(epochs)//2, len(epochs)-1]
                epoch_labels = ["Initial", f"Middle (Epoch {epochs[len(epochs)//2]})", "Final"]
        elif len(epochs) > 0:
            selected_indices = list(range(len(epochs)))
            epoch_labels = [f"Epoch {epochs[e]}" for e in selected_indices]
        else: return # No epochs to plot

        if not selected_indices: return

        fig, axs = plt.subplots(2, len(selected_indices), figsize=(5*len(selected_indices), 10), squeeze=False)
        
        for i, (layer_corrs, layer_name) in enumerate([(layer1_corrs_all, "First Layer"), (layer2_corrs_all, "Second Layer")]):
            if layer_corrs.size == 0: continue
            for j, (idx, label) in enumerate(zip(selected_indices, epoch_labels)):
                if idx >= layer_corrs.shape[0]: continue # Should not happen if selected_indices is correct
                ax = axs[i, j]
                corrs_at_epoch = layer_corrs[idx]
                ax.hist(corrs_at_epoch, bins=30, alpha=0.7, color='skyblue' if i==0 else 'lightgreen')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                mean_corr = np.mean(corrs_at_epoch); mean_abs_corr = np.mean(np.abs(corrs_at_epoch)); max_abs_corr = np.max(np.abs(corrs_at_epoch))
                ax.text(0.05, 0.95, f"Mean: {mean_corr:.3f}\nMean |ρ|: {mean_abs_corr:.3f}\nMax |ρ|: {max_abs_corr:.3f}",
                        transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.set_title(f"{layer_name}: {label}")
                if i == 1: ax.set_xlabel("Correlation with Target $\\rho_i$")
                if j == 0: ax.set_ylabel("Number of Neurons")
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/neuron_target_correlation_distribution_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_weight_matrices(self):
        if len(self.weight_snapshots['epochs']) < 1: # Changed from < 2 to allow single snapshot plot
            print("Not enough weight snapshots for visualization")
            return
        
        epochs_to_show_indices = [0] # Initial
        if len(self.weight_snapshots['epochs']) > 1:
            if self.phase_transition['detected']:
                pt_epoch = self.phase_transition['epoch']
                closest_idx_to_pt = min(range(len(self.weight_snapshots['epochs'])), 
                                     key=lambda i: abs(self.weight_snapshots['epochs'][i] - pt_epoch))
                if closest_idx_to_pt not in epochs_to_show_indices:
                    epochs_to_show_indices.append(closest_idx_to_pt)
            
            final_idx = len(self.weight_snapshots['epochs']) - 1
            if final_idx not in epochs_to_show_indices:
                 epochs_to_show_indices.append(final_idx)
        epochs_to_show_indices = sorted(list(set(epochs_to_show_indices)))


        titles_map = {0: 'Initial Weights'}
        if len(self.weight_snapshots['epochs']) -1 not in titles_map and len(self.weight_snapshots['epochs']) -1 in epochs_to_show_indices :
            titles_map[len(self.weight_snapshots['epochs']) -1] = 'Final Weights'
        if self.phase_transition['detected']:
             pt_snapshot_idx = min(range(len(self.weight_snapshots['epochs'])), key=lambda i: abs(self.weight_snapshots['epochs'][i] - self.phase_transition['epoch']))
             if pt_snapshot_idx in epochs_to_show_indices and pt_snapshot_idx not in titles_map : titles_map[pt_snapshot_idx] = 'Phase Transition Weights'
        
        num_plots = len(epochs_to_show_indices)
        if num_plots == 0: return

        fig_w1, axs_w1 = plt.subplots(num_plots, 1, figsize=(16, 5 * num_plots), squeeze=False)
        for i, snapshot_idx in enumerate(epochs_to_show_indices):
            epoch = self.weight_snapshots['epochs'][snapshot_idx]
            W1 = self.weight_snapshots['W1'][snapshot_idx]
            ax = axs_w1[i,0]
            im = ax.imshow(W1, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im, ax=ax)
            title = titles_map.get(snapshot_idx, f'Weights at Epoch {epoch}')
            ax.set_title(f"{title} - First Layer (W1)")
            ax.set_xlabel("Input Feature"); ax.set_ylabel("Neuron Index")
            if self.k > 0 and self.k < self.d : ax.axvline(x=self.k-0.5, color='green', linestyle='--', alpha=0.7, label='Relevant/Irrelevant Boundary')
            # if self.phase_transition['detected']: ax.axhline(y=self.M1/2, color='red', linestyle='--', alpha=0.7, label=f'Detected Transition (Epoch {self.phase_transition["epoch"]})') # This line might be confusing here
            ax.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/W1_matrices_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig_w1)

        fig_w2, axs_w2 = plt.subplots(num_plots, 1, figsize=(16, 5 * num_plots), squeeze=False)
        for i, snapshot_idx in enumerate(epochs_to_show_indices):
            epoch = self.weight_snapshots['epochs'][snapshot_idx]
            W2_data = self.weight_snapshots['W2'][snapshot_idx] # Renamed W2 to W2_data
            ax = axs_w2[i,0]
            im = ax.imshow(W2_data, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im, ax=ax)
            title = titles_map.get(snapshot_idx, f'Weights at Epoch {epoch}')
            ax.set_title(f"{title} - Second Layer (W2)")
            ax.set_xlabel("Layer 1 Neuron"); ax.set_ylabel("Layer 2 Neuron")
            # if self.phase_transition['detected']: ax.axhline(y=self.M2/2, color='red', linestyle='--', alpha=0.7, label=f'Detected Transition (Epoch {self.phase_transition["epoch"]})')
            ax.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/W2_matrices_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig_w2)

        fig_a, axs_a = plt.subplots(num_plots, 1, figsize=(16, 5 * num_plots), squeeze=False)
        for i, snapshot_idx in enumerate(epochs_to_show_indices):
            epoch = self.weight_snapshots['epochs'][snapshot_idx]
            a_weights = self.weight_snapshots['a'][snapshot_idx] # Renamed a to a_weights
            ax = axs_a[i,0]
            ax.bar(range(len(a_weights)), a_weights)
            title = titles_map.get(snapshot_idx, f'Weights at Epoch {epoch}')
            ax.set_title(f"{title} - Readout Layer Weights (a)")
            ax.set_xlabel("Layer 2 Neuron"); ax.set_ylabel("Weight Value")
            # if self.phase_transition['detected']: ax.axvline(x=self.M2/2, color='red', linestyle='--', alpha=0.7, label=f'Detected Transition (Epoch {self.phase_transition["epoch"]})')
            ax.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/a_weights_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig_a)

    def plot_feature_importance_evolution(self):
        if not self.feature_importance['epochs']: return
        plt.figure(figsize=(12, 8))
        plt.semilogy(self.feature_importance['epochs'], self.feature_importance['ratio'], marker='o', linestyle='-', linewidth=2, label='Feature Importance Ratio')
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', label=f'Detected Transition (Epoch {detected_epoch})')
            if self.phase_transition['feature_ratio'] is not None:
                plt.axhline(y=self.phase_transition['feature_ratio'], color='red', linestyle=':', alpha=0.7, label=f'Ratio at Transition = {self.phase_transition["feature_ratio"]:.2f}')
        plt.xlabel('Epoch'); plt.ylabel('Relevant/Irrelevant Feature Importance Ratio (log scale)')
        plt.title(f'Evolution of Feature Importance Ratio (d={self.d}, k={self.k})'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/feature_importance_evolution_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_feature_correlations(self):
        if not self.feature_correlations['epochs']: return
        plt.figure(figsize=(12, 8))
        plt.semilogy(self.feature_correlations['epochs'], self.feature_correlations['relevant_mean'], marker='o', linestyle='-', label='Relevant Features Mean $|C_l|$')
        plt.semilogy(self.feature_correlations['epochs'], self.feature_correlations['irrelevant_mean'], marker='x', linestyle='--', label='Irrelevant Features Mean $|C_l|$')
        
        epochs_arr = np.array(self.feature_correlations['epochs'])
        if len(epochs_arr[epochs_arr > 0]) > 0 and self.k > 0 :
             scaling = self.d ** (-(self.k-1)/2) if self.k > 0 else 1.0 # Handle k=0
             plt.semilogy(epochs_arr[epochs_arr > 0], scaling * np.ones_like(epochs_arr[epochs_arr > 0]), linestyle='-.', color='green', label=f'$d^{{-(k-1)/2}} = {scaling:.1e}$')

        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', label=f'Detected Transition (Epoch {detected_epoch})')
        plt.xlabel('Epoch'); plt.ylabel('Mean Absolute Correlation $|C_l|$ (log scale)')
        plt.title(f'Feature Correlations $C_l$ Evolution (d={self.d}, k={self.k})'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/feature_correlations_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_signal_noise_ratio(self):
        if not self.empirical_signal_noise['epochs']: return
        plt.figure(figsize=(12, 8))
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['empirical_signal'], marker='o', linestyle='-', label='Empirical Signal (Rel Grad Mean)')
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['empirical_noise'], marker='x', linestyle='-', label='Empirical Noise (Irrel Grad Std)')
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['theoretical_signal'], marker='^', linestyle='--', alpha=0.7, label='Theoretical Signal: $\\eta T d^{-(k-1)/2}$')
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['theoretical_noise'], marker='v', linestyle='--', alpha=0.7, label='Theoretical Noise: $\\eta \\sqrt{T/B}$')
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', label=f'Detected Transition (Epoch {detected_epoch})')
        plt.xlabel('Epoch'); plt.ylabel('Magnitude (log scale)')
        plt.title(f'Signal vs Noise Evolution (d={self.d}, k={self.k}, B={self.batch_size})'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/signal_noise_magnitudes_d{self.d}_k{self.k}.png", dpi=300); plt.close()

        plt.figure(figsize=(12, 8))
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['empirical_ratio'], marker='o', linestyle='-', linewidth=2, label='Empirical Signal/Noise Ratio')
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['theoretical_ratio'], marker='x', linestyle='--', linewidth=2, alpha=0.7, label='Theoretical Signal/Noise Ratio')
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Ratio = 1')
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', label=f'Detected Transition (Epoch {detected_epoch})')
        plt.xlabel('Epoch'); plt.ylabel('Signal/Noise Ratio (log scale)')
        plt.title(f'Signal to Noise Ratio (d={self.d}, k={self.k}, B={self.batch_size})'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/signal_noise_ratio_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_gradient_statistics(self):
        if not self.gradient_stats['epochs'] or not self.W1_grad:
            print("No gradient statistics recorded or W1 not trainable")
            return
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['relevant_grad_mean'], marker='o', linestyle='-', label='Relevant Features Mean Grad Mag')
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['irrelevant_grad_mean'], marker='x', linestyle='--', label='Irrelevant Features Mean Grad Mag')
        if self.phase_transition['detected']: plt.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', label=f'PT (Epoch {self.phase_transition["epoch"]})')
        plt.xlabel('Epoch'); plt.ylabel('Mean Gradient Magnitude (log)'); plt.title('Gradient Magnitude Evolution'); plt.grid(True); plt.legend()

        plt.subplot(2, 2, 2)
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['relevant_grad_std'], marker='o', linestyle='-', label='Relevant Features Grad Std')
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['irrelevant_grad_std'], marker='x', linestyle='--', label='Irrelevant Features Grad Std')
        if self.phase_transition['detected']: plt.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', label=f'PT (Epoch {self.phase_transition["epoch"]})')
        plt.xlabel('Epoch'); plt.ylabel('Gradient Std Dev (log)'); plt.title('Gradient Variability Evolution'); plt.grid(True); plt.legend()

        plt.subplot(2, 2, 3)
        relevant_means = np.array(self.gradient_stats['relevant_grad_mean'])
        irrelevant_means = np.array(self.gradient_stats['irrelevant_grad_mean'])
        ratio = relevant_means / np.maximum(irrelevant_means, 1e-10) # Avoid division by zero
        plt.semilogy(self.gradient_stats['epochs'], ratio, marker='o', linestyle='-', label='Relevant/Irrelevant Ratio')
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7)
        if self.phase_transition['detected']: plt.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', label=f'PT (Epoch {self.phase_transition["epoch"]})')
        plt.xlabel('Epoch'); plt.ylabel('Ratio (log)'); plt.title('Ratio of Relevant to Irrelevant Gradient Magnitudes'); plt.grid(True); plt.legend()

        plt.subplot(2, 2, 4)
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['w1_grad_norm'], marker='o', linestyle='-', label='W1 Gradient Norm')
        if self.phase_transition['detected']: plt.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', label=f'PT (Epoch {self.phase_transition["epoch"]})')
        plt.xlabel('Epoch'); plt.ylabel('Gradient Norm (log)'); plt.title('Overall W1 Gradient Norm Evolution'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/gradient_statistics_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_training_progress(self):
        if not self.loss_history or not self.correlation_history:
            print("No training history recorded")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.semilogy(range(1, len(self.loss_history) + 1), self.loss_history, label="Training Loss") # Epochs start from 1 for loss history
        ax1.set_ylabel('Loss (log scale)'); ax1.set_title(f'Training Progress (d={self.d}, k={self.k})'); ax1.grid(True)
        if self.phase_transition['detected']: ax1.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', label=f'PT (Epoch {self.phase_transition["epoch"]})')
        ax1.legend()

        corr_epochs, corr_values = zip(*self.correlation_history)
        ax2.plot(corr_epochs, corr_values, marker='o', label="Test Correlation")
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Correlation'); ax2.grid(True)
        if self.phase_transition['detected']: ax2.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--') # Label already in ax1 legend
        for threshold in [0.5, 0.9]: ax2.axhline(y=threshold, color=f'C{int(threshold*10)}', linestyle=':', alpha=0.7, label=f'Correlation = {threshold}')
        ax2.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/training_progress_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_activation_statistics(self): # For x^2 activation
        if not self.activation_stats['epochs']: return
        plt.figure(figsize=(12, 8))
        plt.plot(self.activation_stats['epochs'], self.activation_stats['layer1_percent_active'], marker='o', linestyle='-', label='Layer 1 (% Non-Zero Output)')
        plt.plot(self.activation_stats['epochs'], self.activation_stats['layer2_percent_active'], marker='x', linestyle='--', label='Layer 2 (% Non-Zero Output)')
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', label=f'PT (Epoch {detected_epoch})')
            idx = np.argmin(np.abs(np.array(self.activation_stats['epochs']) - detected_epoch))
            if idx < len(self.activation_stats['layer1_percent_active']): # Check index bounds
                 l1_at_pt = self.activation_stats['layer1_percent_active'][idx]
                 l2_at_pt = self.activation_stats['layer2_percent_active'][idx]
                 plt.axhline(y=l1_at_pt, color='C0', linestyle=':', alpha=0.7, label=f'L1 at PT: {l1_at_pt:.2f}%')
                 plt.axhline(y=l2_at_pt, color='C1', linestyle=':', alpha=0.7, label=f'L2 at PT: {l2_at_pt:.2f}%')
        plt.xlabel('Epoch'); plt.ylabel('Percentage of Neurons with Non-Zero Output (%)')
        plt.title(f'$x^2$ Activation Non-Zero Output Percentage (d={self.d}, k={self.k})'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/activation_output_stats_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_gradient_correlations(self):
        if not self.gradient_correlations['epochs'] or not self.W1_grad: return
        plt.figure(figsize=(12, 8))
        plt.plot(self.gradient_correlations['epochs'], self.gradient_correlations['mean_rel_rel'], marker='o', linestyle='-', label='Mean |Corr(Rel, Rel)|')
        plt.plot(self.gradient_correlations['epochs'], self.gradient_correlations['mean_rel_irrel'], marker='x', linestyle='--', label='Mean |Corr(Rel, Irrel)|')
        plt.plot(self.gradient_correlations['epochs'], self.gradient_correlations['mean_irrel_irrel'], marker='^', linestyle='-.', label='Mean |Corr(Irrel, Irrel)|')
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', label=f'PT (Epoch {detected_epoch})')
            idx = np.argmin(np.abs(np.array(self.gradient_correlations['epochs']) - detected_epoch))
            if idx < len(self.gradient_correlations['mean_rel_rel']): # Check index bounds
                rr_at_pt = self.gradient_correlations['mean_rel_rel'][idx]
                ri_at_pt = self.gradient_correlations['mean_rel_irrel'][idx]
                plt.axhline(y=rr_at_pt, color='C0', linestyle=':', alpha=0.7, label=f'Rel-Rel at PT: {rr_at_pt:.4f}')
                plt.axhline(y=ri_at_pt, color='C1', linestyle=':', alpha=0.7, label=f'Rel-Irrel at PT: {ri_at_pt:.4f}')
        plt.xlabel('Epoch'); plt.ylabel('Mean Absolute Gradient Correlation')
        plt.title(f'Gradient Correlation Statistics (d={self.d}, k={self.k})'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/gradient_correlations_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_hidden_target_correlations(self): # For x^2 activation
        if not self.hidden_target_correlations['epochs']: return
        plt.figure(figsize=(12, 8))
        plt.plot(self.hidden_target_correlations['epochs'], self.hidden_target_correlations['layer1_corr_sum'], marker='o', linestyle='-', label='Layer 1 Sum $|E[h_i(x)f_S(x)]|$')
        plt.plot(self.hidden_target_correlations['epochs'], self.hidden_target_correlations['layer2_corr_sum'], marker='x', linestyle='--', label='Layer 2 Sum $|E[h_i(x)f_S(x)]|$')
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', label=f'PT (Epoch {detected_epoch})')
            idx = np.argmin(np.abs(np.array(self.hidden_target_correlations['epochs']) - detected_epoch))
            if idx < len(self.hidden_target_correlations['layer1_corr_sum']): # Check index bounds
                 l1_at_pt = self.hidden_target_correlations['layer1_corr_sum'][idx]
                 l2_at_pt = self.hidden_target_correlations['layer2_corr_sum'][idx]
                 plt.axhline(y=l1_at_pt, color='C0', linestyle=':', alpha=0.7, label=f'L1 at PT: {l1_at_pt:.4f}')
                 plt.axhline(y=l2_at_pt, color='C1', linestyle=':', alpha=0.7, label=f'L2 at PT: {l2_at_pt:.4f}')
        plt.xlabel('Epoch'); plt.ylabel('Sum of $|E[h_i(x)\\cdot f_S(x)]|$')
        plt.title(f'Hidden Unit ($x^2$ Act) Correlation with Target (d={self.d}, k={self.k})'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/hidden_target_correlations_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_gradient_eigenvalues(self):
        if not self.gradient_eigen['epochs'] or (not self.W1_grad and not self.W2_grad): return
        
        if self.W1_grad and self.gradient_eigen['W1_eigenvalues'] and len(self.gradient_eigen['W1_eigenvalues'][0]) > 0 :
            plt.figure(figsize=(12, 8))
            epochs = self.gradient_eigen['epochs']
            w1_eigenvalues_list = self.gradient_eigen['W1_eigenvalues']
            num_eigen_to_plot = min(self.top_k_eigen, len(w1_eigenvalues_list[0]))
            for i in range(num_eigen_to_plot):
                values = [evals[i] if i < len(evals) else 0 for evals in w1_eigenvalues_list] # Handle variable length
                plt.semilogy(epochs, values, marker='o', linestyle='-', label=f'Eigenvalue {i+1}')
            if self.phase_transition['detected']: plt.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', label=f'PT (Epoch {self.phase_transition["epoch"]})')
            plt.xlabel('Epoch'); plt.ylabel('Eigenvalue (log scale)'); plt.title(f'Top W1 Gradient Eigenvalues (d={self.d}, k={self.k})'); plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(f"{self.save_dir}/w1_gradient_eigenvalues_d{self.d}_k{self.k}.png", dpi=300); plt.close()

        if self.W2_grad and self.gradient_eigen['W2_eigenvalues'] and len(self.gradient_eigen['W2_eigenvalues'][0]) > 0:
            plt.figure(figsize=(12, 8))
            epochs = self.gradient_eigen['epochs']
            w2_eigenvalues_list = self.gradient_eigen['W2_eigenvalues']
            num_eigen_to_plot = min(self.top_k_eigen, len(w2_eigenvalues_list[0]))
            for i in range(num_eigen_to_plot):
                values = [evals[i] if i < len(evals) else 0 for evals in w2_eigenvalues_list]
                plt.semilogy(epochs, values, marker='o', linestyle='-', label=f'Eigenvalue {i+1}')
            if self.phase_transition['detected']: plt.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', label=f'PT (Epoch {self.phase_transition["epoch"]})')
            plt.xlabel('Epoch'); plt.ylabel('Eigenvalue (log scale)'); plt.title(f'Top W2 Gradient Eigenvalues (d={self.d}, k={self.k})'); plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(f"{self.save_dir}/w2_gradient_eigenvalues_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_principal_angles(self):
        if not self.principal_angles['epochs'] or len(self.principal_angles['epochs']) < 1: return # Allow plotting even with 1 epoch (no angles then)

        if self.W1_grad and self.principal_angles['W1_angles'] and len(self.principal_angles['W1_angles']) > 0 and len(self.principal_angles['W1_angles'][0]) > 0:
            plt.figure(figsize=(12, 8))
            epochs = self.principal_angles['epochs']
            w1_angles_list = self.principal_angles['W1_angles']
            mean_angles = [np.mean(angles) if len(angles)>0 else 0 for angles in w1_angles_list]
            plt.plot(epochs, mean_angles, marker='o', linestyle='-', label='Mean Principal Angle')
            num_angles_to_plot = min(self.top_k_eigen, len(w1_angles_list[0])) # Use actual number of angles stored
            if num_angles_to_plot <= 5: # Plot individual angles if few
                for i in range(num_angles_to_plot):
                    values = [angles[i] if i < len(angles) else 0 for angles in w1_angles_list]
                    plt.plot(epochs, values, marker='.', linestyle=':', alpha=0.5, label=f'Angle {i+1}')
            if self.phase_transition['detected']: plt.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', label=f'PT (Epoch {self.phase_transition["epoch"]})')
            plt.xlabel('Epoch'); plt.ylabel('Principal Angle (radians)'); plt.title(f'W1 Gradient Subspace Principal Angles (d={self.d}, k={self.k})'); plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(f"{self.save_dir}/w1_principal_angles_d{self.d}_k{self.k}.png", dpi=300); plt.close()

        if self.W2_grad and self.principal_angles['W2_angles'] and len(self.principal_angles['W2_angles']) > 0 and len(self.principal_angles['W2_angles'][0]) > 0:
            plt.figure(figsize=(12, 8))
            epochs = self.principal_angles['epochs']
            w2_angles_list = self.principal_angles['W2_angles']
            mean_angles = [np.mean(angles) if len(angles)>0 else 0 for angles in w2_angles_list]
            plt.plot(epochs, mean_angles, marker='o', linestyle='-', label='Mean Principal Angle')
            num_angles_to_plot = min(self.top_k_eigen, len(w2_angles_list[0]))
            if num_angles_to_plot <= 5:
                for i in range(num_angles_to_plot):
                    values = [angles[i] if i < len(angles) else 0 for angles in w2_angles_list]
                    plt.plot(epochs, values, marker='.', linestyle=':', alpha=0.5, label=f'Angle {i+1}')
            if self.phase_transition['detected']: plt.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', label=f'PT (Epoch {self.phase_transition["epoch"]})')
            plt.xlabel('Epoch'); plt.ylabel('Principal Angle (radians)'); plt.title(f'W2 Gradient Subspace Principal Angles (d={self.d}, k={self.k})'); plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(f"{self.save_dir}/w2_principal_angles_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_neuron_activation_evolution(self): # For x^2 activation
        if not self.neuron_activation_evolution['epochs']: return
        epochs = self.neuron_activation_evolution['epochs']
        layer1_ratios_list = self.neuron_activation_evolution['layer1_activation_ratios']
        layer2_ratios_list = self.neuron_activation_evolution['layer2_activation_ratios']

        if not layer1_ratios_list or not layer2_ratios_list: return
        layer1_data = np.array(layer1_ratios_list)
        layer2_data = np.array(layer2_ratios_list)

        fig, axs = plt.subplots(2, 1, figsize=(15, 12))
        cmap = cm.viridis; norm = Normalize(vmin=0, vmax=1)

        ax1 = axs[0]
        im1 = ax1.imshow(layer1_data, aspect='auto', cmap=cmap, norm=norm, extent=[0, self.M1, epochs[-1], epochs[0]])
        ax1.set_title(f'First Layer Neuron Non-Zero Output Ratio Evolution (d={self.d}, k={self.k})')
        ax1.set_xlabel('Neuron Index'); ax1.set_ylabel('Epoch')
        if self.phase_transition['detected']:
            ax1.axhline(y=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7, label=f'PT (Epoch {self.phase_transition["epoch"]})')
            ax1.legend()
        fig.colorbar(im1, ax=ax1, label='Non-Zero Output Ratio')

        ax2 = axs[1]
        im2 = ax2.imshow(layer2_data, aspect='auto', cmap=cmap, norm=norm, extent=[0, self.M2, epochs[-1], epochs[0]])
        ax2.set_title(f'Second Layer Neuron Non-Zero Output Ratio Evolution (d={self.d}, k={self.k})')
        ax2.set_xlabel('Neuron Index'); ax2.set_ylabel('Epoch')
        if self.phase_transition['detected']:
            ax2.axhline(y=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7, label=f'PT (Epoch {self.phase_transition["epoch"]})')
            ax2.legend()
        fig.colorbar(im2, ax=ax2, label='Non-Zero Output Ratio')
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/neuron_activation_output_evolution_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig)

        # Active neurons trajectory plot (Layer 2)
        if layer2_data.size > 0:
            max_activations = np.max(layer2_data, axis=0) if layer2_data.shape[0] > 0 else np.array([])
            active_neurons = np.where(max_activations > 0.5)[0] if max_activations.size > 0 else []
            if len(active_neurons) > 0:
                fig_active, ax_active = plt.subplots(figsize=(15, 8))
                for neuron_idx in active_neurons:
                    activation_trajectory = layer2_data[:, neuron_idx]
                    max_act = np.max(activation_trajectory) if len(activation_trajectory) > 0 else 0
                    color = cmap(norm(max_act)); alpha_val = max(0.3, max_act)
                    ax_active.plot(epochs, activation_trajectory, '-', color=color, alpha=alpha_val, linewidth=2, label=f'Neuron {neuron_idx}' if len(active_neurons)<=10 else None)
                if self.phase_transition['detected']: ax_active.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7, label=f'PT (Epoch {self.phase_transition["epoch"]})')
                ax_active.axhline(y=0.5, color='black', linestyle=':', alpha=0.7, label='50% Non-Zero Output Ratio')
                ax_active.set_title(f'Active Second Layer Neurons Non-Zero Output Trajectories (d={self.d}, k={self.k})')
                ax_active.set_xlabel('Epoch'); ax_active.set_ylabel('Non-Zero Output Ratio'); ax_active.grid(True, alpha=0.3)
                sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
                fig_active.colorbar(sm, ax=ax_active, label='Maximum Non-Zero Output Ratio')
                if len(active_neurons) <= 10: ax_active.legend(loc='upper left', bbox_to_anchor=(1,1))
                plt.tight_layout(); plt.savefig(f"{self.save_dir}/active_neurons_trajectories_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig_active)

    def plot_neuron_activation_trajectories(self): # For x^2 activation
        if not self.neuron_activation_evolution['epochs']: return
        epochs = np.array(self.neuron_activation_evolution['epochs'])
        layer1_ratios_all = np.array(self.neuron_activation_evolution['layer1_activation_ratios'])
        layer2_ratios_all = np.array(self.neuron_activation_evolution['layer2_activation_ratios'])

        if layer1_ratios_all.size > 0:
            fig1, ax1 = plt.subplots(figsize=(15, 8))
            cmap_layer1 = plt.cm.nipy_spectral
            for i in range(self.M1):
                neuron_trajectory = layer1_ratios_all[:, i]
                color = cmap_layer1(i / max(1, self.M1-1)) # Normalize index, avoid div by zero if M1=1
                activation_range = np.max(neuron_trajectory) - np.min(neuron_trajectory) if len(neuron_trajectory)>0 else 0
                alpha_val = 0.4 if activation_range > 0.2 else 0.1
                ax1.plot(epochs, neuron_trajectory, '-', color=color, alpha=alpha_val, linewidth=1)
            if self.phase_transition['detected']:
                ax1.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7, label=f'PT (Epoch {self.phase_transition["epoch"]})')
                ax1.legend()
            ax1.set_title(f'First Layer Neuron Non-Zero Output Trajectories (d={self.d}, k={self.k})'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Non-Zero Output Ratio'); ax1.grid(True, alpha=0.3)
            sm_layer1 = plt.cm.ScalarMappable(cmap=cmap_layer1); sm_layer1.set_array([])
            fig1.colorbar(sm_layer1, ax=ax1, label=f'Neuron Index (0-{self.M1-1})')
            plt.tight_layout(); plt.savefig(f"{self.save_dir}/layer1_neuron_output_trajectories_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig1)

        if layer2_ratios_all.size > 0:
            fig2, ax2 = plt.subplots(figsize=(15, 8))
            cmap_layer2 = plt.cm.nipy_spectral
            for i in range(self.M2):
                neuron_trajectory = layer2_ratios_all[:, i]
                color = cmap_layer2(i / max(1, self.M2-1))
                activation_range = np.max(neuron_trajectory) - np.min(neuron_trajectory) if len(neuron_trajectory)>0 else 0
                alpha_val = 0.4 if activation_range > 0.2 else 0.1
                ax2.plot(epochs, neuron_trajectory, '-', color=color, alpha=alpha_val, linewidth=1)
            if self.phase_transition['detected']:
                ax2.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7, label=f'PT (Epoch {self.phase_transition["epoch"]})')
                ax2.legend()
            ax2.set_title(f'Second Layer Neuron Non-Zero Output Trajectories (d={self.d}, k={self.k})'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Non-Zero Output Ratio'); ax2.grid(True, alpha=0.3)
            sm_layer2 = plt.cm.ScalarMappable(cmap=cmap_layer2); sm_layer2.set_array([])
            fig2.colorbar(sm_layer2, ax=ax2, label=f'Neuron Index (0-{self.M2-1})')
            plt.tight_layout(); plt.savefig(f"{self.save_dir}/layer2_neuron_output_trajectories_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig2)

    def analyze_neuron_activations(self): # For x^2 activation
        print("Analyzing neuron non-zero outputs with all 2^k relevant feature vectors...")
        self.model.eval()
        if self.k > 16: # Limit k to avoid excessive computation (2^16 = 65536)
            print(f"Warning: k={self.k} is too large for exhaustive 2^k analysis. Skipping.")
            self.neuron_activation_analysis['layer1_activation_ratio'] = np.zeros(self.M1)
            self.neuron_activation_analysis['layer2_activation_ratio'] = np.zeros(self.M2)
            return np.zeros(self.M1), np.zeros(self.M2)

        relevant_patterns = list(itertools.product([-1, 1], repeat=self.k)) # Changed 1 to -1
        num_patterns = len(relevant_patterns)
        num_random_samples_irrelevant = max(1, 200 // num_patterns if num_patterns > 0 else 200) # Number of random irrelevant parts
        
        layer1_activation_counts = np.zeros(self.M1)
        layer2_activation_counts = np.zeros(self.M2)
        total_samples_tested = 0

        with torch.no_grad():
            for relevant_pattern in tqdm(relevant_patterns, desc="Processing 2^k patterns"):
                relevant_part = torch.tensor(relevant_pattern, dtype=torch.float32).to(self.device)
                current_batch_inputs = torch.zeros((num_random_samples_irrelevant, self.d), dtype=torch.float32).to(self.device)
                for i in range(num_random_samples_irrelevant):
                    irrelevant_part = torch.tensor(np.random.choice([-1, 1], size=(self.d - self.k)), dtype=torch.float32).to(self.device)
                    current_batch_inputs[i, :self.k] = relevant_part
                    if self.d - self.k > 0:
                        current_batch_inputs[i, self.k:] = irrelevant_part
                
                h1 = self.model.get_first_layer_activations(current_batch_inputs)
                h2 = self.model.get_second_layer_activations(current_batch_inputs)
                
                layer1_activation_counts += torch.sum((h1 > 1e-9).float(), dim=0).cpu().numpy()
                layer2_activation_counts += torch.sum((h2 > 1e-9).float(), dim=0).cpu().numpy()
                total_samples_tested += num_random_samples_irrelevant
        
        self.neuron_activation_analysis['layer1_activation_ratio'] = layer1_activation_counts / total_samples_tested if total_samples_tested > 0 else np.zeros(self.M1)
        self.neuron_activation_analysis['layer2_activation_ratio'] = layer2_activation_counts / total_samples_tested if total_samples_tested > 0 else np.zeros(self.M2)
        
        self.plot_neuron_activation_ratios() # Plot the results of this analysis
        return self.neuron_activation_analysis['layer1_activation_ratio'], self.neuron_activation_analysis['layer2_activation_ratio']

    def plot_neuron_activation_ratios(self): # For x^2 activation, "ratio" of non-zero output
        if self.neuron_activation_analysis['layer1_activation_ratio'] is None:
            print("No neuron activation analysis (2^k) data available. Run analyze_neuron_activations() first.")
            return
        
        layer1_ratio = self.neuron_activation_analysis['layer1_activation_ratio']
        layer2_ratio = self.neuron_activation_analysis['layer2_activation_ratio']
        cmap = cm.viridis; norm = Normalize(vmin=0, vmax=1)

        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        ax1 = axs[0]
        for i, ratio_val in enumerate(layer1_ratio): # Renamed ratio to ratio_val
            color = cmap(norm(ratio_val)); alpha_val = max(0.3, ratio_val)
            ax1.bar(i, ratio_val, color=color, alpha=alpha_val, width=1.0)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% non-zero output')
        ax1.set_title(f'First Layer Neuron Non-Zero Output Ratios (2^k Analysis) (d={self.d}, k={self.k})')
        ax1.set_xlabel('Neuron Index'); ax1.set_ylabel('Non-Zero Output Ratio'); ax1.set_xlim(-1, self.M1); ax1.set_ylim(0, 1.05); ax1.grid(alpha=0.3)
        sm1 = cm.ScalarMappable(norm=norm, cmap=cmap); sm1.set_array([])
        fig.colorbar(sm1, ax=ax1, label='Non-Zero Output Ratio'); ax1.legend()

        ax2 = axs[1]
        for i, ratio_val in enumerate(layer2_ratio):
            color = cmap(norm(ratio_val)); alpha_val = max(0.3, ratio_val)
            ax2.bar(i, ratio_val, color=color, alpha=alpha_val, width=1.0)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% non-zero output')
        ax2.set_title(f'Second Layer Neuron Non-Zero Output Ratios (2^k Analysis) (d={self.d}, k={self.k})')
        ax2.set_xlabel('Neuron Index'); ax2.set_ylabel('Non-Zero Output Ratio'); ax2.set_xlim(-1, self.M2); ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.3)
        sm2 = cm.ScalarMappable(norm=norm, cmap=cmap); sm2.set_array([])
        fig.colorbar(sm2, ax=ax2, label='Non-Zero Output Ratio'); ax2.legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/neuron_activation_ratios_2k_d{self.d}_k{self.k}.png", dpi=300); plt.close()

        # Histogram plots
        fig_hist, axs_hist = plt.subplots(2, 1, figsize=(15,10))
        axs_hist[0].hist(layer1_ratio, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axs_hist[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='50% non-zero output')
        axs_hist[0].set_title(f'Distribution of First Layer Neuron Non-Zero Output Ratios (2^k Analysis)'); axs_hist[0].set_xlabel('Non-Zero Output Ratio'); axs_hist[0].set_ylabel('Number of Neurons'); axs_hist[0].grid(alpha=0.3); axs_hist[0].legend()
        
        axs_hist[1].hist(layer2_ratio, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        axs_hist[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='50% non-zero output')
        axs_hist[1].set_title(f'Distribution of Second Layer Neuron Non-Zero Output Ratios (2^k Analysis)'); axs_hist[1].set_xlabel('Non-Zero Output Ratio'); axs_hist[1].set_ylabel('Number of Neurons'); axs_hist[1].grid(alpha=0.3); axs_hist[1].legend()
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/neuron_activation_distribution_2k_d{self.d}_k{self.k}.png", dpi=300); plt.close(fig_hist)


    def visualize_second_layer_patterns(self, activation_threshold=0.01): # For x^2 activation
        if self.neuron_activation_analysis['layer2_activation_ratio'] is None:
            print("Running neuron activation (2^k) analysis first...")
            self.analyze_neuron_activations() # This uses 2^k patterns
        
        layer2_activation_ratio = self.neuron_activation_analysis['layer2_activation_ratio']
        if layer2_activation_ratio is None: # Still none after trying to analyze
             print("Failed to get layer 2 activation ratios. Skipping visualize_second_layer_patterns.")
             return {}

        active_neurons_indices = np.where(layer2_activation_ratio > activation_threshold)[0]
        if len(active_neurons_indices) == 0:
            print(f"No active neurons found with threshold {activation_threshold} from 2^k analysis. Try lowering the threshold.")
            return {}
        
        print(f"Found {len(active_neurons_indices)} active neurons in the second layer from 2^k analysis")
        self.second_layer_patterns['active_neurons'] = active_neurons_indices
        
        W1_np = self.model.W1.detach().cpu().numpy()
        W2_np = self.model.W2.detach().cpu().numpy()
        a_np = self.model.a.detach().cpu().numpy()
        
        num_to_plot = min(10, len(active_neurons_indices))
        if num_to_plot == 0: return self.second_layer_patterns

        fig, axs = plt.subplots(num_to_plot, 1, figsize=(15, 4 * num_to_plot), squeeze=False) # Ensure axs is always 2D array-like
            
        for i, neuron_idx in enumerate(active_neurons_indices[:num_to_plot]):
            ax = axs[i, 0] 
            w2_neuron_weights = W2_np[neuron_idx, :]
            output_weight = a_np[neuron_idx]
            
            strongest_l1_indices = np.argsort(np.abs(w2_neuron_weights))[-5:]
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            colors = ['red' if k_idx in strongest_l1_indices else 'lightgray' for k_idx in range(self.M1)]
            ax.bar(range(self.M1), w2_neuron_weights, color=colors, alpha=0.7)
            
            for l1_idx in strongest_l1_indices:
                w1_l1neuron_weights = W1_np[l1_idx, :]
                top_input_features = np.argsort(np.abs(w1_l1neuron_weights))[-3:]
                relevant_count = sum(1 for f_idx in top_input_features if f_idx < self.k)
                feature_text = f"L1 Neuron {l1_idx}: {relevant_count}/{len(top_input_features)} rel features" if relevant_count > 0 else f"L1 Neuron {l1_idx}"
                ax.annotate(feature_text, xy=(l1_idx, w2_neuron_weights[l1_idx]), xytext=(0,10), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='blue')
            
            ax.set_title(f'L2 Neuron {neuron_idx} (Out Weight: {output_weight:.3f}, Non-Zero Ratio (2^k): {layer2_activation_ratio[neuron_idx]:.3f})')
            ax.set_xlabel('First Layer Neuron Index'); ax.set_ylabel('Weight Value'); ax.grid(alpha=0.3)
            
            self.second_layer_patterns['input_patterns'][neuron_idx] = {
                'output_weight': output_weight, 'activation_ratio': layer2_activation_ratio[neuron_idx],
                'strongest_connections_from_l1': strongest_l1_indices, 'w2_weights_to_l1': w2_neuron_weights
            }
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/second_layer_patterns_d{self.d}_k{self.k}.png", dpi=300); plt.close()
        self._plot_detailed_neuron_connections(active_neurons_indices[:min(5, len(active_neurons_indices))])
        return self.second_layer_patterns

    def _plot_detailed_neuron_connections(self, neurons_to_analyze): # Renamed arg
        if len(neurons_to_analyze) == 0:
            print("No neurons to analyze in _plot_detailed_neuron_connections")
            return
        
        W1_np = self.model.W1.detach().cpu().numpy()
        W2_np = self.model.W2.detach().cpu().numpy()
        
        for neuron_idx in neurons_to_analyze:
            fig, axs = plt.subplots(2, 1, figsize=(15, 10))
            w2_neuron_weights = W2_np[neuron_idx, :]
            strongest_l1_indices = np.argsort(np.abs(w2_neuron_weights))[-10:] # Top 10 L1 connections
            
            connection_matrix = np.zeros((len(strongest_l1_indices), self.d))
            for i, l1_idx in enumerate(strongest_l1_indices):
                l2_conn_weight = w2_neuron_weights[l1_idx]
                l1_input_weights = W1_np[l1_idx, :]
                connection_matrix[i, :] = l1_input_weights * l2_conn_weight
            
            ax1 = axs[0]
            im = ax1.imshow(connection_matrix, cmap='RdBu_r', aspect='auto')
            fig.colorbar(im, ax=ax1, label='Effective Weight (L1-L2 * L0-L1)')
            ax1.set_title(f'Layer 2 Neuron {neuron_idx} - Detailed Connection Patterns')
            ax1.set_xlabel('Input Feature Index'); ax1.set_ylabel('Connected Layer 1 Neuron Index (Strongest)')
            if self.k > 0 and self.k < self.d: ax1.axvline(x=self.k-0.5, color='green', linestyle='--', alpha=0.7, label=f'Relevant Features (0-{self.k-1})')
            ax1.set_yticks(range(len(strongest_l1_indices))); ax1.set_yticklabels(strongest_l1_indices); ax1.legend()
            
            ax2 = axs[1]
            ax2.bar(range(self.M1), w2_neuron_weights, alpha=0.7)
            ax2.set_title(f'Layer 2 Neuron {neuron_idx} - Weights to All Layer 1 Neurons')
            ax2.set_xlabel('Layer 1 Neuron Index'); ax2.set_ylabel('Weight Value'); ax2.grid(alpha=0.3)
            for l1_idx in strongest_l1_indices: ax2.axvline(x=l1_idx, color='red', linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(f"{self.save_dir}/neuron_{neuron_idx}_detailed_connections_d{self.d}_k{self.k}.png", dpi=300); plt.close()

    def plot_macroscopic_quantities_distribution(self):
        if not self.macroscopic_quantities['epochs']: return
        epochs = np.array(self.macroscopic_quantities['epochs'])
        r_values_all = np.array(self.macroscopic_quantities['r_values'])
        p_values_all = np.array(self.macroscopic_quantities['p_values'])

        if len(epochs) == 0: return
        # Simplified selection of epochs for plotting distributions
        num_epochs_tracked = len(epochs)
        if num_epochs_tracked >= 3:
            indices = [0, num_epochs_tracked // 2, num_epochs_tracked - 1]
            labels = ["Initial", f"Middle (Epoch {epochs[indices[1]]})", f"Final (Epoch {epochs[indices[2]]})"]
            if self.phase_transition['detected']:
                pt_idx = np.argmin(np.abs(epochs - self.phase_transition['epoch']))
                if pt_idx not in indices: # Add PT if not already there, try to keep 3-4 plots
                    if len(indices) < 4 :
                        indices.append(pt_idx)
                        labels.append(f"PT (Epoch {epochs[pt_idx]})")
                    else: # replace middle if we have too many
                        indices[1] = pt_idx
                        labels[1] = f"PT (Epoch {epochs[pt_idx]})"
                indices = sorted(list(set(indices))) # Keep unique and sorted
                # Relabel based on final indices
                temp_labels = []
                for k_idx, current_idx_val in enumerate(indices): # Renamed idx to current_idx_val
                    if current_idx_val == 0 : temp_labels.append("Initial")
                    elif self.phase_transition['detected'] and current_idx_val == pt_idx : temp_labels.append(f"PT (Epoch {epochs[pt_idx]})")
                    elif current_idx_val == num_epochs_tracked -1 : temp_labels.append(f"Final (Epoch {epochs[current_idx_val]})")
                    else: temp_labels.append(f"Epoch {epochs[current_idx_val]}") # Fallback
                labels = temp_labels

        elif num_epochs_tracked > 0:
            indices = list(range(num_epochs_tracked))
            labels = [f"Epoch {epochs[i]}" for i in indices]
        else: return
        
        if not indices : return


        fig, axs = plt.subplots(2, len(indices), figsize=(5*len(indices), 10), squeeze=False)
        for i, (values_all_epochs, quantity_name) in enumerate([(r_values_all, "$r_i$ (First Layer)"), (p_values_all, "$p_j$ (Second Layer)")]):
            if values_all_epochs.size == 0: continue
            for j, (epoch_idx_val, label_str) in enumerate(zip(indices, labels)): # Renamed idx to epoch_idx_val, label to label_str
                if epoch_idx_val >= values_all_epochs.shape[0]: continue
                ax = axs[i, j]
                vals_at_epoch = values_all_epochs[epoch_idx_val]
                ax.hist(vals_at_epoch, bins=30, alpha=0.7, color='skyblue' if i==0 else 'lightgreen')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                mean_val = np.mean(vals_at_epoch); mean_abs_val = np.mean(np.abs(vals_at_epoch)); max_abs_val = np.max(np.abs(vals_at_epoch))
                ax.text(0.05, 0.95, f"Mean: {mean_val:.3f}\nMean |val|: {mean_abs_val:.3f}\nMax |val|: {max_abs_val:.3f}",
                        transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.set_title(f"{quantity_name}: {label_str}")
                if i == 1: ax.set_xlabel("Value")
                if j == 0: ax.set_ylabel("Number of Neurons")
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/macroscopic_quantities_distribution_d{self.d}_k{self.k}.png", dpi=300); plt.close()


    

    def plot_all(self):
        """
        Generate all plots in one go
        """
        print("Generating all plots...")
        
        # Training Progress & High-Level Metrics
        self.plot_training_progress()
        self.plot_feature_importance_evolution()
        self.plot_feature_correlations()
        self.plot_signal_noise_ratio()
        
        # Weight & Gradient Analysis
        self.plot_weight_matrices()
        self.plot_gradient_statistics()
        self.plot_gradient_correlations()
        self.plot_gradient_eigenvalues()
        self.plot_principal_angles()

        # Corrected Condensation & Alignment Analysis (New)
        print("Plotting corrected condensation and alignment diagnostics...")
        self.plot_rank_one_condensation()
        self.plot_second_layer_alignment()
        
        # Activation & Neuron-Level Analysis
        self.plot_activation_statistics()
        self.plot_hidden_target_correlations()
        
        # Neuron Activation Evolution (Time-based)
        if self.neuron_activation_evolution['epochs']:
            print("Plotting neuron activation evolution over time...")
            self.plot_neuron_activation_evolution()
            self.plot_neuron_activation_trajectories()

        # Neuron-Target Correlation Evolution (Time-based)
        if self.neuron_target_correlation_evolution['epochs']:
            print("Plotting neuron-target correlation evolution over time...")
            self.plot_neuron_target_correlation_evolution()
            self.plot_neuron_target_correlation_trajectories()
            self.plot_neuron_target_correlation_distribution()
            
        # Macroscopic Quantities (r_i, p_j) and Collective Fields (phi_1, phi_2)
        if self.macroscopic_quantities['epochs']:
            print("Plotting macroscopic quantities (r_i, p_j) evolution...")
            self.plot_macroscopic_quantities_evolution()
            self.plot_macroscopic_quantities_trajectories()
            self.plot_macroscopic_quantities_distribution()

        if self.s_m_squared_evolution['epochs']:
            print("Plotting collective fields (s_j^2, m_a^2, phi_1, phi_2) evolution...")
            self.plot_s_m_squared_evolution()
            self.plot_phi_values_evolution()
                
        # Final-State Neuron & Circuit Analysis
        print("Running and plotting final-state neuron activation analysis (2^k)...")
        self.analyze_neuron_activations()  # This method calls its own plot
        
        print("Visualizing final-state second layer patterns...")
        self.visualize_second_layer_patterns() # This method calls its own plot
        
        print("All plots generated!")

    def save_results(self):
        """
        Save all collected data to CSV files for further analysis
        """
        results_dir = os.path.join(self.save_dir, f"d{self.d}_k{self.k}_results_x2act") # Indicate x^2 activation
        os.makedirs(results_dir, exist_ok=True)
        
        # Save feature importance data
        if self.feature_importance['epochs']:
            fi_df = pd.DataFrame(self.feature_importance)
            fi_df.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)
        
        # Save feature correlation data
        if self.feature_correlations['epochs']:
            fc_summary_df = pd.DataFrame({
                'epoch': self.feature_correlations['epochs'],
                'relevant_mean': self.feature_correlations['relevant_mean'],
                'irrelevant_mean': self.feature_correlations['irrelevant_mean'],
                'relevant_std': self.feature_correlations['relevant_std'],
                'irrelevant_std': self.feature_correlations['irrelevant_std']
            })
            fc_summary_df.to_csv(os.path.join(results_dir, "feature_correlations_summary.csv"), index=False)

        if self.empirical_signal_noise['epochs']:
            sn_df = pd.DataFrame(self.empirical_signal_noise)
            sn_df.to_csv(os.path.join(results_dir, "signal_noise.csv"), index=False)
        
        if self.gradient_stats['epochs']:
            grad_df = pd.DataFrame(self.gradient_stats)
            grad_df.to_csv(os.path.join(results_dir, "gradient_statistics.csv"), index=False)
        
        if self.correlation_history:
            corr_df = pd.DataFrame(self.correlation_history, columns=['epoch', 'correlation'])
            corr_df.to_csv(os.path.join(results_dir, "correlation_history.csv"), index=False)
        
        pt_data = self.phase_transition.copy() # Make a copy to add other params
        pt_data.update({ 'd': self.d, 'k': self.k, 'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate, 'W1_grad': self.W1_grad,
                        'W2_grad': self.W2_grad, 'a_grad': self.a_grad, 'W1_init': self.W1_init})
        pd.DataFrame([pt_data]).to_csv(os.path.join(results_dir, "phase_transition_and_params.csv"), index=False)
        
        if self.activation_stats['epochs']:
            act_df = pd.DataFrame(self.activation_stats)
            act_df.to_csv(os.path.join(results_dir, "activation_statistics.csv"), index=False)

        if self.gradient_correlations['epochs']:
            gc_summary_data = {key: self.gradient_correlations[key] for key in ['epochs', 'mean_rel_rel', 'mean_rel_irrel', 'mean_irrel_irrel']}
            gc_summary_df = pd.DataFrame(gc_summary_data)
            gc_summary_df.to_csv(os.path.join(results_dir, "gradient_correlations_summary.csv"), index=False)
        
        if self.hidden_target_correlations['epochs']:
            htc_df = pd.DataFrame(self.hidden_target_correlations)
            htc_df.to_csv(os.path.join(results_dir, "hidden_target_correlations.csv"), index=False)

        if self.gradient_eigen['epochs']:
            if self.gradient_eigen['W1_eigenvalues'] and len(self.gradient_eigen['W1_eigenvalues'][0]) > 0:
                w1_eigen_df = pd.DataFrame(
                    np.array(self.gradient_eigen['W1_eigenvalues']),
                    index=self.gradient_eigen['epochs'],
                    columns=[f'W1_eigenval_{i+1}' for i in range(len(self.gradient_eigen['W1_eigenvalues'][0]))]
                )
                w1_eigen_df.index.name = 'epoch'
                w1_eigen_df.to_csv(os.path.join(results_dir, "w1_eigenvalues.csv"))
            if self.gradient_eigen['W2_eigenvalues'] and len(self.gradient_eigen['W2_eigenvalues'][0]) > 0:
                w2_eigen_df = pd.DataFrame(
                    np.array(self.gradient_eigen['W2_eigenvalues']),
                    index=self.gradient_eigen['epochs'],
                    columns=[f'W2_eigenval_{i+1}' for i in range(len(self.gradient_eigen['W2_eigenvalues'][0]))]
                )
                w2_eigen_df.index.name = 'epoch'
                w2_eigen_df.to_csv(os.path.join(results_dir, "w2_eigenvalues.csv"))

        if self.principal_angles['epochs']:
            if self.principal_angles['W1_angles'] and len(self.principal_angles['W1_angles'][0]) > 0:
                w1_angle_df = pd.DataFrame(
                    np.array(self.principal_angles['W1_angles']),
                    index=self.principal_angles['epochs'],
                    columns=[f'W1_angle_{i+1}' for i in range(len(self.principal_angles['W1_angles'][0]))]
                )
                w1_angle_df.index.name = 'epoch'
                w1_angle_df.to_csv(os.path.join(results_dir, "w1_principal_angles.csv"))

            if self.principal_angles['W2_angles'] and len(self.principal_angles['W2_angles'][0]) > 0:
                w2_angle_df = pd.DataFrame(
                    np.array(self.principal_angles['W2_angles']),
                    index=self.principal_angles['epochs'],
                    columns=[f'W2_angle_{i+1}' for i in range(len(self.principal_angles['W2_angles'][0]))]
                )
                w2_angle_df.index.name = 'epoch'
                w2_angle_df.to_csv(os.path.join(results_dir, "w2_principal_angles.csv"))

        if self.neuron_activation_analysis['layer1_activation_ratio'] is not None:
            pd.DataFrame({'neuron_idx': range(self.M1), 'activation_ratio_2k': self.neuron_activation_analysis['layer1_activation_ratio']}).to_csv(os.path.join(results_dir, "layer1_activation_ratios_2k.csv"), index=False)
            pd.DataFrame({'neuron_idx': range(self.M2), 'activation_ratio_2k': self.neuron_activation_analysis['layer2_activation_ratio']}).to_csv(os.path.join(results_dir, "layer2_activation_ratios_2k.csv"), index=False)

        if self.neuron_activation_evolution['epochs']:
            pd.DataFrame(np.array(self.neuron_activation_evolution['layer1_activation_ratios']), index=self.neuron_activation_evolution['epochs'], columns=[f"neuron_{i}" for i in range(self.M1)]).to_csv(os.path.join(results_dir, "layer1_activation_evolution.csv"))
            pd.DataFrame(np.array(self.neuron_activation_evolution['layer2_activation_ratios']), index=self.neuron_activation_evolution['epochs'], columns=[f"neuron_{i}" for i in range(self.M2)]).to_csv(os.path.join(results_dir, "layer2_activation_evolution.csv"))
        
        if self.neuron_target_correlation_evolution['epochs']:
            pd.DataFrame(np.array(self.neuron_target_correlation_evolution['layer1_neuron_correlations']), index=self.neuron_target_correlation_evolution['epochs'], columns=[f"neuron_{i}" for i in range(self.M1)]).to_csv(os.path.join(results_dir, "layer1_target_correlation_evolution.csv"))
            pd.DataFrame(np.array(self.neuron_target_correlation_evolution['layer2_neuron_correlations']), index=self.neuron_target_correlation_evolution['epochs'], columns=[f"neuron_{i}" for i in range(self.M2)]).to_csv(os.path.join(results_dir, "layer2_target_correlation_evolution.csv"))

        if self.macroscopic_quantities['epochs']:
            pd.DataFrame(np.array(self.macroscopic_quantities['r_values']), index=self.macroscopic_quantities['epochs'], columns=[f"r_{i}" for i in range(self.M1)]).to_csv(os.path.join(results_dir, "r_values_evolution.csv"))
            pd.DataFrame(np.array(self.macroscopic_quantities['p_values']), index=self.macroscopic_quantities['epochs'], columns=[f"p_{j}" for j in range(self.M2)]).to_csv(os.path.join(results_dir, "p_values_evolution.csv"))
        
        if self.s_m_squared_evolution['epochs']:
            phi_df = pd.DataFrame({
                'epoch': self.s_m_squared_evolution['epochs'],
                'phi1': self.s_m_squared_evolution['phi1'],
                'phi2': self.s_m_squared_evolution['phi2']
            })
            phi_df.to_csv(os.path.join(results_dir, "phi_values.csv"), index=False)

            if self.s_m_squared_evolution['s_squared_values']:
                s_sq_df = pd.DataFrame(
                    np.array(self.s_m_squared_evolution['s_squared_values']),
                    index=self.s_m_squared_evolution['epochs'],
                    columns=[f"s_sq_neuron_{i}" for i in range(self.M1)]
                )
                s_sq_df.index.name = 'epoch'
                s_sq_df.to_csv(os.path.join(results_dir, "s_squared_evolution.csv"))

            if self.s_m_squared_evolution['m_squared_values']:
                m_sq_df = pd.DataFrame(
                    np.array(self.s_m_squared_evolution['m_squared_values']),
                    index=self.s_m_squared_evolution['epochs'],
                    columns=[f"m_sq_neuron_{i}" for i in range(self.M2)]
                )
                m_sq_df.index.name = 'epoch'
                m_sq_df.to_csv(os.path.join(results_dir, "m_squared_evolution.csv"))

        # Save condensation stats
        if self.condensation_stats['epochs']:
            condensation_df = pd.DataFrame(self.condensation_stats)
            condensation_df.to_csv(os.path.join(results_dir, "condensation_stats.csv"), index=False)

        print(f"Results saved to {results_dir}")
        return results_dir



def run_macroscopic_quantities_comparison(d=30, k=6, M1=512, M2=512, learning_rates=[0.01, 0.001], 
                                          batch_sizes=[512], n_epochs=5000, device_id=None,
                                          save_dir="macroscopic_analysis", tracker=100,
                                          a_grad=True):
    os.makedirs(save_dir, exist_ok=True)
    comparison_dir_name = f"d{d}_k{k}_comparison_a{a_grad}_x2act"
    comparison_dir = os.path.join(save_dir, comparison_dir_name)
    os.makedirs(comparison_dir, exist_ok=True)
    
    results = {}
    for lr_val in learning_rates: # Renamed lr to lr_val
        for bs_val in batch_sizes: # Renamed bs to bs_val
            run_name = f"lr{lr_val}_bs{bs_val}"
            run_specific_dir = os.path.join(comparison_dir, run_name)
            
            print(f"\n{'='*80}")
            print(f"Running experiment with learning_rate={lr_val}, batch_size={bs_val}, readout trainable={a_grad}")
            print(f"{'='*80}\n")
            
            analyzer = ParityNetAnalyzer(
                d=d, k=k, M1=M1, M2=M2, learning_rate=lr_val, batch_size=bs_val,
                device_id=device_id, save_dir=run_specific_dir, tracker=tracker, top_k_eigen=5,
                W1_grad=True, W2_grad=True, a_grad=a_grad, W1_init='random'
            )
            
            final_corr, final_epoch = analyzer.train(n_epochs=n_epochs)
            analyzer.plot_all()
            analyzer.save_results()
            
            results[run_name] = analyzer
            print(f"Run {run_name} completed!")
            print(f"Final correlation: {final_corr:.4f} at epoch {final_epoch}")
            if analyzer.phase_transition['detected']:
                detected_epoch = analyzer.phase_transition['epoch']
                theoretical_epoch = analyzer.phase_transition['theoretical_epoch']
                print(f"Phase transition detected at epoch {detected_epoch}")
                if theoretical_epoch > 0 : print(f"Theoretical prediction: {theoretical_epoch}, Ratio: {detected_epoch/theoretical_epoch:.2f}")
                else: print(f"Theoretical prediction: {theoretical_epoch} (cannot compute ratio)")

            else:
                print("No phase transition detected")
    
    create_comparative_macroscopic_plots(results, comparison_dir, d, k)
    return results

def create_comparative_macroscopic_plots(results, comp_save_dir, d, k): # Renamed save_dir to comp_save_dir
    plots_dir = os.path.join(comp_save_dir, "comparative_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    for run_name, analyzer in results.items():
        if not analyzer.macroscopic_quantities['epochs'] or not analyzer.macroscopic_quantities['r_values']: continue
        epochs = analyzer.macroscopic_quantities['epochs']
        r_values_all = np.array(analyzer.macroscopic_quantities['r_values'])
        if r_values_all.size == 0 : continue
        mean_abs_r = [np.mean(np.abs(r_vals)) if len(r_vals)>0 else 0 for r_vals in r_values_all] # r_vals is 1D array per epoch
        plt.plot(epochs, mean_abs_r, '-', linewidth=2, label=f'{run_name} - Mean $|r_i|$')
        if analyzer.phase_transition['detected']:
            pt_epoch = analyzer.phase_transition['epoch']
            closest_idx = min(range(len(epochs)), key=lambda i: abs(epochs[i] - pt_epoch))
            if closest_idx < len(mean_abs_r):
                 plt.scatter([pt_epoch], [mean_abs_r[closest_idx]], marker='*', s=150, label=f'{run_name} - PT (Epoch {pt_epoch})', zorder=5)
    plt.xlabel('Epoch'); plt.ylabel('Mean Absolute $r_i$ Value'); plt.title(f'Comparison of Mean $|r_i|$ Values (d={d}, k={k})'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparative_mean_r_values.png"), dpi=300); plt.close()

    plt.figure(figsize=(15, 8))
    for run_name, analyzer in results.items():
        if not analyzer.macroscopic_quantities['epochs'] or not analyzer.macroscopic_quantities['p_values']: continue
        epochs = analyzer.macroscopic_quantities['epochs']
        p_values_all = np.array(analyzer.macroscopic_quantities['p_values'])
        if p_values_all.size == 0 : continue
        mean_abs_p = [np.mean(np.abs(p_vals)) if len(p_vals)>0 else 0 for p_vals in p_values_all]
        plt.plot(epochs, mean_abs_p, '-', linewidth=2, label=f'{run_name} - Mean $|p_j|$')
        if analyzer.phase_transition['detected']:
            pt_epoch = analyzer.phase_transition['epoch']
            closest_idx = min(range(len(epochs)), key=lambda i: abs(epochs[i] - pt_epoch))
            if closest_idx < len(mean_abs_p):
                plt.scatter([pt_epoch], [mean_abs_p[closest_idx]], marker='*', s=150, label=f'{run_name} - PT (Epoch {pt_epoch})', zorder=5)
    plt.xlabel('Epoch'); plt.ylabel('Mean Absolute $p_j$ Value'); plt.title(f'Comparison of Mean $|p_j|$ Values (d={d}, k={k})'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparative_mean_p_values.png"), dpi=300); plt.close()

    plt.figure(figsize=(max(12, 2*len(results)), 6)) # Adjust width for more run labels
    x_pos = np.arange(len(results)); detected_pts = []; theoretical_pts = []; run_labels = []
    for run_name, analyzer in results.items():
        run_labels.append(run_name)
        detected_pts.append(analyzer.phase_transition['epoch'] if analyzer.phase_transition['detected'] else 0)
        theoretical_pts.append(analyzer.phase_transition['theoretical_epoch'])
    bar_width = 0.35
    plt.bar(x_pos - bar_width/2, detected_pts, bar_width, label='Detected Phase Transition', color='blue', alpha=0.7)
    plt.bar(x_pos + bar_width/2, theoretical_pts, bar_width, label='Theoretical Prediction', color='red', alpha=0.7)
    plt.xlabel('Run Configuration'); plt.ylabel('Epoch'); plt.title(f'Phase Transition Epochs Across Runs (d={d}, k={k})')
    plt.xticks(x_pos, run_labels, rotation=45, ha="right"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase_transition_comparison.png"), dpi=300); plt.close()

    summary_data = []
    for run_name, analyzer in results.items():
        detected_epoch_val = analyzer.phase_transition['epoch'] if analyzer.phase_transition['detected'] else "Not detected"
        theoretical_epoch_val = analyzer.phase_transition['theoretical_epoch']
        final_corr_val = analyzer.correlation_history[-1][1] if analyzer.correlation_history else "N/A"
        mean_abs_r_at_pt_val, mean_abs_p_at_pt_val = "N/A", "N/A"
        if analyzer.phase_transition['detected'] and analyzer.macroscopic_quantities['epochs']:
            epochs_macro = analyzer.macroscopic_quantities['epochs']
            if epochs_macro: # Ensure list is not empty
                closest_idx_macro = min(range(len(epochs_macro)), key=lambda i: abs(epochs_macro[i] - analyzer.phase_transition['epoch']))
                if closest_idx_macro < len(analyzer.macroscopic_quantities['r_values']):
                     r_vals_at_pt = analyzer.macroscopic_quantities['r_values'][closest_idx_macro]
                     mean_abs_r_at_pt_val = np.mean(np.abs(r_vals_at_pt)) if len(r_vals_at_pt)>0 else "N/A"
                if closest_idx_macro < len(analyzer.macroscopic_quantities['p_values']):
                     p_vals_at_pt = analyzer.macroscopic_quantities['p_values'][closest_idx_macro]
                     mean_abs_p_at_pt_val = np.mean(np.abs(p_vals_at_pt)) if len(p_vals_at_pt)>0 else "N/A"
        summary_data.append({'run_name': run_name, 'detected_epoch': detected_epoch_val, 'theoretical_epoch': theoretical_epoch_val,
                             'final_correlation': final_corr_val, 'mean_abs_r_at_pt': mean_abs_r_at_pt_val, 'mean_abs_p_at_pt': mean_abs_p_at_pt_val})
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(plots_dir, "run_summary.csv"), index=False)
    print(f"Comparative plots and summary saved to {plots_dir}")
    print("Summary of results:"); print(summary_df)


def run_parity_experiment(d=30, k=6, M1=512, M2=512, learning_rate=0.01, batch_size=512,
                          n_epochs=10000, device_id=None, save_dir="parity_analysis",
                          tracker=20, top_k_eigen=5, W1_grad=True, W2_grad=True, a_grad=True, W1_init='random',
                          kernel_samples=10000):
    """
    Run a single experiment analyzing a neural network learning a k-sparse parity function
    """
    run_name_parts = [
        f"d{d}", f"k{k}", f"M1_{M1}", f"M2_{M2}", f"bs{batch_size}", f"lr{learning_rate}",
        f"W1{W1_grad}", f"W2{W2_grad}", f"a{a_grad}", f"init{W1_init}", "x2act" # Indicate x^2 activation
    ]
    run_specific_dir = os.path.join(save_dir, "_".join(run_name_parts))
    os.makedirs(run_specific_dir, exist_ok=True)
    
    analyzer = ParityNetAnalyzer(
        d=d, k=k, M1=M1, M2=M2, learning_rate=learning_rate, batch_size=batch_size,
        device_id=device_id, save_dir=run_specific_dir, tracker=tracker, top_k_eigen=top_k_eigen,
        W1_grad=W1_grad, W2_grad=W2_grad, a_grad=a_grad, W1_init=W1_init, kernel_samples=kernel_samples
    )
    
    final_corr, final_epoch = analyzer.train(n_epochs=n_epochs)
    analyzer.plot_all()
    analyzer.save_results()
    
    print(f"Experiment completed for: {run_specific_dir}")
    print(f"Final correlation: {final_corr:.4f} at epoch {final_epoch}")
    if analyzer.phase_transition['detected']:
        detected_epoch = analyzer.phase_transition['epoch']
        theoretical_epoch = analyzer.phase_transition['theoretical_epoch']
        print(f"Phase transition detected at epoch {detected_epoch}")
        if theoretical_epoch > 0 : print(f"Theoretical prediction: {theoretical_epoch}, Ratio: {detected_epoch/theoretical_epoch:.2f}")
        else: print(f"Theoretical prediction: {theoretical_epoch} (cannot compute ratio)")

    else:
        print("No phase transition detected")
    
    return analyzer


if __name__ == "__main__":
    # Base parameters for experiments
    d = 20
    k = 3
    M1 = 1024
    M2 = 1024
    batch_size = 2000
    learning_rate = 0.0001
    n_epochs = 800
    tracker = 100
    top_k_eigen = 5
    kernel_samples = 20000 # Samples for kernel estimation. WARNING: High values are memory intensive.
    base_save_dir = "parity_analysis_experiments_456_0606_3"
    
    # Check for available GPUs
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_id = 0 if n_gpus > 0 else None
    
    # Print common configurations
    print("Common configurations for all experiments:")
    print(f"d={d}, k={k}, M1={M1}, M2={M2}, batch_size={batch_size}")
    print(f"learning_rate={learning_rate}, n_epochs={n_epochs}, tracker={tracker}")
    print(f"kernel_samples={kernel_samples}")
    print(f"Using device: {'cuda:'+str(device_id) if device_id is not None and torch.cuda.is_available() else 'cpu'}")
    print(f"Activation function: x^2")
    print("=" * 80)
    
    # Experiment 1: Default - All layers trainable, random initialization, x^2 activation
    print("\nRunning Experiment 1: Default (x^2 activation)")
    analyzer1 = run_parity_experiment(
        d=d, k=k, M1=M1, M2=M2, batch_size=batch_size, learning_rate=learning_rate,
        n_epochs=n_epochs, device_id=device_id, save_dir=f"{base_save_dir}/exp1_default_x2",
        tracker=tracker, top_k_eigen=top_k_eigen,
        W1_grad=True, W2_grad=True, a_grad=True, W1_init='random',
        kernel_samples=kernel_samples
    )
    print("Experiment 1 completed.")
    print("=" * 80)

    # Example for macroscopic quantities comparison (can be run separately)
    # print("\nRunning Macroscopic Quantities Comparison (x^2 activation)")
    # results_macro_comp = run_macroscopic_quantities_comparison(
    #     d=d, k=k, M1=M1, M2=M2, 
    #     learning_rates=[0.01, 0.005], # Test a couple of LRs
    #     batch_sizes=[batch_size], 
    #     n_epochs=n_epochs // 2, # Shorter epochs for comparison runs
    #     device_id=device_id,
    #     save_dir=f"{base_save_dir}/macro_comp_x2", 
    #     tracker=tracker,
    #     a_grad=True # Test with trainable readout
    # )
    # print("Macroscopic quantities comparison completed.")
    # print("=" * 80)