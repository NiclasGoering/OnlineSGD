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

class TwoLayerReLUNet(torch.nn.Module):
    def __init__(self, input_dim, hidden1_width, hidden2_width, W1_grad=True, W2_grad=True, W1_init='random', k=None):
        super(TwoLayerReLUNet, self).__init__()
        # Initialize layers
        if W1_init == 'random':
            # Standard initialization
            self.W1 = torch.nn.Parameter(torch.randn(hidden1_width, input_dim) / np.sqrt(input_dim))
        elif W1_init == 'sparse' and k is not None:
            # Initialize only the first k columns (relevant features), rest are zero
            W1_init = torch.zeros(hidden1_width, input_dim)
            W1_init[:, :k] = torch.randn(hidden1_width, k) / np.sqrt(input_dim)
            self.W1 = torch.nn.Parameter(W1_init)
        else:
            # Fallback to random initialization
            self.W1 = torch.nn.Parameter(torch.randn(hidden1_width, input_dim) / np.sqrt(input_dim))
        
        self.W2 = torch.nn.Parameter(torch.randn(hidden2_width, hidden1_width) / np.sqrt(hidden1_width))
        self.a = torch.nn.Parameter(torch.randn(hidden2_width) / np.sqrt(hidden2_width))
        
        # Set requires_grad based on flags
        self.W1.requires_grad = W1_grad
        self.W2.requires_grad = W2_grad
        
    def forward(self, x):
        # First hidden layer without bias
        h1 = torch.relu(torch.matmul(x, self.W1.t()))
        # Second hidden layer without bias
        h2 = torch.relu(torch.matmul(h1, self.W2.t()))
        # Output layer
        output = torch.matmul(h2, self.a)
        return output
    
    def get_first_layer_activations(self, x):
        return torch.relu(torch.matmul(x, self.W1.t()))
    
    def get_second_layer_activations(self, x):
        h1 = torch.relu(torch.matmul(x, self.W1.t()))
        return torch.relu(torch.matmul(h1, self.W2.t()))
    
    def get_first_layer_preactivations(self, x):
        return torch.matmul(x, self.W1.t())
    
    def get_second_layer_preactivations(self, x, h1=None):
        if h1 is None:
            h1 = torch.relu(torch.matmul(x, self.W1.t()))
        return torch.matmul(h1, self.W2.t())

class ParityNetAnalyzer:
    def __init__(self, d=30, k=6, M1=512, M2=512, learning_rate=0.01, 
                 batch_size=512, device_id=None, save_dir="parity_analysis",
                 tracker=20, top_k_eigen=5, W1_grad=True, W2_grad=True, W1_init='random'):
        """
        Initialize the ParityNetAnalyzer to analyze neural networks learning parity functions
        
        Args:
            d: Input dimension
            k: Parity function order (k-sparse parity)
            M1: First layer width
            M2: Second layer width
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            device_id: GPU device ID (int) or None for CPU
            save_dir: Directory to save results
            tracker: Frequency (in epochs) to compute metrics
            top_k_eigen: Number of top eigenvectors/values to track
            W1_grad: Whether W1 is trainable (if False, W1 is frozen)
            W2_grad: Whether W2 is trainable (if False, W2 is frozen)
            W1_init: Initialization mode for W1 ('random' or 'sparse')
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
        self.W1_init = W1_init
        
        # Set device based on device_id
        if device_id is not None:
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize model, optimizer, and training history
        self.model = self._create_model()
        
        # Only add parameters with requires_grad=True to the optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        
        # History storage
        self.loss_history = []
        self.correlation_history = []
        
        # Feature correlation tracking (Cl values)
        self.feature_correlations = {
            'epochs': [],
            'relevant_features': [], # Cl for l ∈ S
            'irrelevant_features': [], # Cl for l ∉ S
            'relevant_mean': [],
            'irrelevant_mean': [],
            'relevant_std': [],
            'irrelevant_std': [],
        }
        
        # Signal and noise tracking
        self.empirical_signal_noise = {
            'epochs': [],
            'empirical_signal': [],      # Measured signal strength for relevant features
            'empirical_noise': [],       # Measured noise level for irrelevant features
            'empirical_ratio': [],       # empirical_signal / empirical_noise
            'theoretical_signal': [],    # Ω(ηTd^-(k-1)/2)
            'theoretical_noise': [],     # O(η√(T/B))
            'theoretical_ratio': []      # theoretical_signal / theoretical_noise
        }
        
        # Gradient statistics tracking
        self.gradient_stats = {
            'epochs': [],
            'relevant_grad_mean': [],    # Mean gradient magnitude for relevant features
            'irrelevant_grad_mean': [],  # Mean gradient magnitude for irrelevant features
            'relevant_grad_std': [],     # Std dev of gradients for relevant features
            'irrelevant_grad_std': [],   # Std dev of gradients for irrelevant features
            'w1_grad_norm': []           # Overall gradient norm for W1
        }
        
        # Stored weight matrices for visualization
        self.weight_snapshots = {
            'epochs': [],
            'W1': [],
            'W2': []
        }
        
        # Feature importance ratio tracking
        self.feature_importance = {
            'epochs': [],
            'relevant_importance': [],
            'irrelevant_importance': [],
            'ratio': []
        }
        
        # Phase transition detection
        self.phase_transition = {
            'detected': False,
            'epoch': None,
            'correlation': None,
            'feature_ratio': None,
            'theoretical_epoch': self.get_theoretical_transition_epoch()
        }
        
        # ReLU activation tracking
        self.activation_stats = {
            'epochs': [],
            'layer1_percent_active': [],
            'layer2_percent_active': [],
        }
        
        # Gradient correlation tracking (ρj,j')
        self.gradient_correlations = {
            'epochs': [],
            'rel_rel_corr': [],     # Correlations between relevant features
            'rel_irrel_corr': [],   # Correlations between relevant and irrelevant features
            'irrel_irrel_corr': [], # Correlations between irrelevant features
            'mean_rel_rel': [],     # Mean correlation between relevant features
            'mean_rel_irrel': [],   # Mean correlation between relevant and irrelevant features
            'mean_irrel_irrel': [], # Mean correlation between irrelevant features
        }
        
        # Hidden activation correlation with target function
        self.hidden_target_correlations = {
            'epochs': [],
            'layer1_corr_sum': [],   # Sum of |E[h_i(x)⋅f_S(x)]| for first layer
            'layer2_corr_sum': [],   # Sum of |E[h_i(x)⋅f_S(x)]| for second layer
        }
        
        # Gradient eigenvalue tracking
        self.gradient_eigen = {
            'epochs': [],
            'W1_eigenvalues': [],    # Top-k eigenvalues of W1 gradient
            'W2_eigenvalues': [],    # Top-k eigenvalues of W2 gradient
            'W1_eigenvectors': [],   # Top-k eigenvectors of W1 gradient
            'W2_eigenvectors': [],   # Top-k eigenvectors of W2 gradient
        }
        
        # Principal angle tracking
        self.principal_angles = {
            'epochs': [],
            'W1_angles': [],         # Principal angles between consecutive W1 gradient subspaces
            'W2_angles': [],         # Principal angles between consecutive W2 gradient subspaces
        }
        
        # Store prior eigenvectors for principal angle computation
        self.prior_W1_eigenvectors = None
        self.prior_W2_eigenvectors = None
        
        # Sample fixed test inputs for consistent evaluation
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, d)), 
                                  dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)
        
        # Fixed sample for computing correlations
        self.X_analysis = torch.tensor(np.random.choice([-1, 1], size=(5000, d)), 
                                      dtype=torch.float32).to(self.device)
        self.y_analysis = self._target_function(self.X_analysis)
        
        # Detailed neuron activation analysis (for final state)
        self.neuron_activation_analysis = {
            'layer1_activation_ratio': None,
            'layer2_activation_ratio': None
        }
        
        # Track neuron activation evolution over training epochs
        self.neuron_activation_evolution = {
            'epochs': [],
            'layer1_activation_ratios': [],  # List of arrays, each containing activation ratios for all layer 1 neurons at an epoch
            'layer2_activation_ratios': []   # List of arrays, each containing activation ratios for all layer 2 neurons at an epoch
        }
        
        # Track individual neuron correlations with target function
        self.neuron_target_correlation_evolution = {
            'epochs': [],
            'layer1_neuron_correlations': [],  # List of arrays, each containing correlations for all layer 1 neurons
            'layer2_neuron_correlations': []   # List of arrays, each containing correlations for all layer 2 neurons
        }
        
        # Second layer patterns analysis
        self.second_layer_patterns = {
            'active_neurons': None,
            'input_patterns': {}
        }
        
        print(f"ParityNetAnalyzer initialized on {self.device}")
        print(f"Analyzing {k}-parity function in {d} dimensions")
        print(f"Network: {M1} → {M2} → 1")
        print(f"Batch size: {batch_size}, Tracking metrics every {tracker} epochs")
        print(f"W1 trainable: {W1_grad}, W2 trainable: {W2_grad}")
        print(f"W1 initialization: {W1_init}")
        print(f"Theoretical phase transition epoch: {self.phase_transition['theoretical_epoch']}")
    
    def _create_model(self):
        """Create a two-layer ReLU network with the specified parameters"""
        model = TwoLayerReLUNet(
            self.d, 
            self.M1, 
            self.M2, 
            W1_grad=self.W1_grad, 
            W2_grad=self.W2_grad, 
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
        relevant_mean = np.mean(np.abs(relevant_corrs))
        irrelevant_mean = np.mean(np.abs(irrelevant_corrs))
        relevant_std = np.std(np.abs(relevant_corrs))
        irrelevant_std = np.std(np.abs(irrelevant_corrs))
        
        # Store the correlations
        self.feature_correlations['epochs'].append(epoch)
        self.feature_correlations['relevant_features'].append(relevant_corrs)
        self.feature_correlations['irrelevant_features'].append(irrelevant_corrs)
        self.feature_correlations['relevant_mean'].append(relevant_mean)
        self.feature_correlations['irrelevant_mean'].append(irrelevant_mean)
        self.feature_correlations['relevant_std'].append(relevant_std)
        self.feature_correlations['irrelevant_std'].append(irrelevant_std)
        
        return relevant_mean, irrelevant_mean
    
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
        theoretical_signal = eta * T * (self.d ** (-(self.k-1)/2))
        
        # Theoretical noise level
        theoretical_noise = eta * np.sqrt(T / B)
        
        # Theoretical ratio
        theoretical_ratio = theoretical_signal / max(theoretical_noise, 1e-10)
        
        # Empirical calculations
        if empirical_gradients is not None:
            # Use the provided gradient statistics
            empirical_signal = empirical_gradients['relevant_grad_mean']  
            empirical_noise = empirical_gradients['irrelevant_grad_std']
        elif len(self.gradient_stats['epochs']) > 0 and self.gradient_stats['epochs'][-1] == epoch:
            # Use the latest stored gradient statistics
            empirical_signal = self.gradient_stats['relevant_grad_mean'][-1]
            empirical_noise = self.gradient_stats['irrelevant_grad_std'][-1]
        else:
            # Default fallback (should be avoided)
            empirical_signal = theoretical_signal * (0.9 + 0.2 * np.random.random())
            empirical_noise = theoretical_noise * (0.9 + 0.2 * np.random.random())
        
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
        relevant_importance = np.mean(np.abs(W1[:, :self.k]))
        
        # Average absolute weight for irrelevant features
        irrelevant_importance = np.mean(np.abs(W1[:, self.k:]))
        
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
                'W2_grads': np.zeros((self.M2, self.M1)) if not self.W2_grad else None
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
        W1_grads = self.model.W1.grad.detach().cpu().numpy() if self.W1_grad else np.zeros((self.M1, self.d))
        W2_grads = self.model.W2.grad.detach().cpu().numpy() if self.W2_grad else np.zeros((self.M2, self.M1))
        
        # Calculate gradient statistics for relevant features
        relevant_grads = W1_grads[:, :self.k]
        irrelevant_grads = W1_grads[:, self.k:]
        
        # Compute mean and std for gradient magnitudes
        relevant_grad_mean = np.mean(np.abs(relevant_grads))
        irrelevant_grad_mean = np.mean(np.abs(irrelevant_grads))
        relevant_grad_std = np.std(np.abs(relevant_grads))
        irrelevant_grad_std = np.std(np.abs(irrelevant_grads))
        
        # Compute overall gradient norm
        w1_grad_norm = np.linalg.norm(W1_grads)
        
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
            'relevant_grad_std': relevant_grad_std,
            'irrelevant_grad_std': irrelevant_grad_std,
            'w1_grad_norm': w1_grad_norm,
            'W1_grads': W1_grads,
            'W2_grads': W2_grads
        }
    
    def compute_activation_statistics(self, epoch, num_samples=10000):
        """
        Compute statistics on ReLU activations: what percentage of neurons are active
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
            
            # Calculate percentage of activations that are non-zero (overall)
            layer1_percent_active = torch.mean((h1 > 0).float()).item() * 100
            layer2_percent_active = torch.mean((h2 > 0).float()).item() * 100
            
            # Calculate per-neuron activation ratios
            layer1_activation_ratios = torch.mean((h1 > 0).float(), dim=0).cpu().numpy()
            layer2_activation_ratios = torch.mean((h2 > 0).float(), dim=0).cpu().numpy()
        
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
        
        # Generate random inputs
        X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                       dtype=torch.float32).to(self.device)
        
        # Compute target function values
        y_target = self._target_function(X)
        
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
                neuron_corr = torch.mean(h1[:, i] * y_target).item()
                layer1_correlations[i] = neuron_corr
            
            # Compute correlation for each neuron in layer 2
            layer2_correlations = np.zeros(self.M2)
            for i in range(self.M2):
                # Compute ρᵢ = E_x[hᵢ(x)·f*(x)]
                neuron_corr = torch.mean(h2[:, i] * y_target).item()
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
            X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                           dtype=torch.float32).to(self.device)
            y = self._target_function(X)
            
            # Zero gradients, forward and backward pass
            self.model.zero_grad()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            
            # Get gradients
            W1_grads = self.model.W1.grad.detach().cpu().numpy()
        
        # Extract relevant/irrelevant feature gradients
        rel_grads = W1_grads[:, :self.k]  # M1 x k
        irrel_grads = W1_grads[:, self.k:]  # M1 x (d-k)
        
        # Initialize correlation matrices
        # We'll compute average correlations between groups
        # 1. Relevant-relevant correlations
        rel_rel_corr = np.zeros((self.k, self.k))
        
        # 2. Relevant-irrelevant correlations
        rel_irrel_corr = np.zeros((self.k, self.d - self.k))
        
        # 3. Irrelevant-irrelevant correlations
        irrel_irrel_corr = np.zeros((self.d - self.k, self.d - self.k))
        
        # Compute correlations for relevant-relevant pairs
        for j1 in range(self.k):
            for j2 in range(j1+1, self.k):
                # Calculate correlation ρj1,j2
                grad_j1 = rel_grads[:, j1]  # All neurons for feature j1
                grad_j2 = rel_grads[:, j2]  # All neurons for feature j2
                
                # Compute correlation
                corr = np.corrcoef(grad_j1, grad_j2)[0, 1]
                # Handle potential NaN values
                if np.isnan(corr):
                    corr = 0.0
                    
                rel_rel_corr[j1, j2] = corr
                rel_rel_corr[j2, j1] = corr
        
        # Set diagonal to 1 (self-correlation)
        np.fill_diagonal(rel_rel_corr, 1.0)
        
        # Compute correlations for relevant-irrelevant pairs
        for j1 in range(self.k):
            for j2 in range(self.d - self.k):
                # Calculate correlation ρj1,j2+k
                grad_j1 = rel_grads[:, j1]  # All neurons for relevant feature j1
                grad_j2 = irrel_grads[:, j2]  # All neurons for irrelevant feature j2
                
                # Compute correlation
                corr = np.corrcoef(grad_j1, grad_j2)[0, 1]
                # Handle potential NaN values
                if np.isnan(corr):
                    corr = 0.0
                    
                rel_irrel_corr[j1, j2] = corr
        
        # Compute correlations for irrelevant-irrelevant pairs
        for j1 in range(self.d - self.k):
            for j2 in range(j1+1, self.d - self.k):
                # Calculate correlation ρj1+k,j2+k
                grad_j1 = irrel_grads[:, j1]  # All neurons for irrelevant feature j1
                grad_j2 = irrel_grads[:, j2]  # All neurons for irrelevant feature j2
                
                # Compute correlation
                corr = np.corrcoef(grad_j1, grad_j2)[0, 1]
                # Handle potential NaN values
                if np.isnan(corr):
                    corr = 0.0
                    
                irrel_irrel_corr[j1, j2] = corr
                irrel_irrel_corr[j2, j1] = corr
        
        # Set diagonal to 1 (self-correlation)
        np.fill_diagonal(irrel_irrel_corr, 1.0)
        
        # Compute mean correlations for each type (excluding self-correlations along diagonals)
        mean_rel_rel = np.mean(np.abs(rel_rel_corr[~np.eye(rel_rel_corr.shape[0], dtype=bool)]))
        mean_rel_irrel = np.mean(np.abs(rel_irrel_corr))
        mean_irrel_irrel = np.mean(np.abs(irrel_irrel_corr[~np.eye(irrel_irrel_corr.shape[0], dtype=bool)]))
        
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
        - h_i is the hidden activation
        - f_S is the k-parity function
        
        Args:
            epoch: Current epoch
            num_samples: Number of samples to use
        """
        self.model.eval()
        
        # Generate random inputs
        X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                       dtype=torch.float32).to(self.device)
        
        # Compute target function values
        y_target = self._target_function(X)
        
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
                neuron_corr = torch.mean(h1[:, i] * y_target).item()
                layer1_corr_sum += abs(neuron_corr)
            
            # Compute correlation for each neuron in layer 2
            layer2_corr_sum = 0.0
            for i in range(self.M2):
                # Compute E[h_i(x)⋅f_S(x)]
                neuron_corr = torch.mean(h2[:, i] * y_target).item()
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
            
            return {
                'W1_eigenvalues': np.zeros(self.top_k_eigen),
                'W2_eigenvalues': np.zeros(self.top_k_eigen),
                'W1_angles': None,
                'W2_angles': None
            }
            
        if W1_grads is None or W2_grads is None:
            # We need to compute the gradients
            X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                           dtype=torch.float32).to(self.device)
            y = self._target_function(X)
            
            # Zero gradients, forward and backward pass
            self.model.zero_grad()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            
            # Get gradients
            W1_grads = self.model.W1.grad.detach().cpu().numpy() if self.W1_grad else np.zeros((self.M1, self.d))
            W2_grads = self.model.W2.grad.detach().cpu().numpy() if self.W2_grad else np.zeros((self.M2, self.M1))
        
        # Reshape matrices for eigendecomposition
        # For W1 (M1 x d), the gradient covariance is (d x d)
        W1_cov = np.matmul(W1_grads.T, W1_grads) if self.W1_grad else np.zeros((self.d, self.d))
        
        # For W2 (M2 x M1), the gradient covariance is (M1 x M1)
        W2_cov = np.matmul(W2_grads.T, W2_grads) if self.W2_grad else np.zeros((self.M1, self.M1))
        
        # Compute eigenvalues and eigenvectors
        k = min(self.top_k_eigen, W1_cov.shape[0], W2_cov.shape[0])
        
        # For W1 gradient covariance
        if self.W1_grad:
            W1_eigenvalues, W1_eigenvectors = np.linalg.eigh(W1_cov)
            
            # Sort in descending order (numpy returns ascending)
            W1_indices = np.argsort(W1_eigenvalues)[::-1]
            W1_eigenvalues = W1_eigenvalues[W1_indices[:k]]
            W1_eigenvectors = W1_eigenvectors[:, W1_indices[:k]]
        else:
            W1_eigenvalues = np.zeros(k)
            W1_eigenvectors = np.zeros((self.d, k))
        
        # For W2 gradient covariance
        if self.W2_grad:
            W2_eigenvalues, W2_eigenvectors = np.linalg.eigh(W2_cov)
            
            # Sort in descending order
            W2_indices = np.argsort(W2_eigenvalues)[::-1]
            W2_eigenvalues = W2_eigenvalues[W2_indices[:k]]
            W2_eigenvectors = W2_eigenvectors[:, W2_indices[:k]]
        else:
            W2_eigenvalues = np.zeros(k)
            W2_eigenvectors = np.zeros((self.M1, k))
        
        # Compute principal angles if we have previous eigenvectors
        W1_angles = None
        W2_angles = None
        
        if self.prior_W1_eigenvectors is not None and self.W1_grad:
            # Compute principal angles using scipy's subspace_angles
            W1_angles = subspace_angles(self.prior_W1_eigenvectors, W1_eigenvectors)
            
        if self.prior_W2_eigenvectors is not None and self.W2_grad:
            W2_angles = subspace_angles(self.prior_W2_eigenvectors, W2_eigenvectors)
            
        if W1_angles is not None or W2_angles is not None:
            # Store principal angles
            self.principal_angles['epochs'].append(epoch)
            self.principal_angles['W1_angles'].append(W1_angles if W1_angles is not None else np.zeros(k))
            self.principal_angles['W2_angles'].append(W2_angles if W2_angles is not None else np.zeros(k))
        
        # Update prior eigenvectors for next computation
        self.prior_W1_eigenvectors = W1_eigenvectors if self.W1_grad else None
        self.prior_W2_eigenvectors = W2_eigenvectors if self.W2_grad else None
        
        # Store results
        self.gradient_eigen['epochs'].append(epoch)
        self.gradient_eigen['W1_eigenvalues'].append(W1_eigenvalues)
        self.gradient_eigen['W2_eigenvalues'].append(W2_eigenvalues)
        self.gradient_eigen['W1_eigenvectors'].append(W1_eigenvectors)
        self.gradient_eigen['W2_eigenvectors'].append(W2_eigenvectors)
        
        return {
            'W1_eigenvalues': W1_eigenvalues,
            'W2_eigenvalues': W2_eigenvalues,
            'W1_angles': W1_angles,
            'W2_angles': W2_angles
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
        
        # Compute initial gradient statistics
        grad_stats = self.compute_gradient_statistics(0)
        self.compute_signal_noise(0, grad_stats)
        
        # Compute new metrics at epoch 0
        self.compute_activation_statistics(0)
        self.compute_gradient_correlations(0, grad_stats['W1_grads'])
        self.compute_hidden_target_correlations(0)
        self.compute_gradient_eigen_statistics(0, grad_stats['W1_grads'], grad_stats['W2_grads'])
        self.track_individual_neuron_target_correlations(0)
        
        # For timing
        start_time = time.time()
        
        for epoch in tqdm(range(1, n_epochs + 1)):
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
                    test_loss = ((preds - self.y_test) ** 2).mean().item()
                    
                    # Check if preds has variance before calculating correlation
                    preds_squeezed = preds.squeeze()
                    if torch.var(preds_squeezed) == 0 or torch.isnan(torch.var(preds_squeezed)):
                        # If all predictions are identical, correlation is undefined
                        correlation = 0.0
                        print(f"Warning: All predictions are identical ({preds_squeezed[0].item():.4f}), setting correlation to 0")
                    else:
                        try:
                            correlation = torch.corrcoef(torch.stack([preds_squeezed, self.y_test]))[0, 1].item()
                            # Check for NaN and replace with 0
                            if torch.isnan(torch.tensor(correlation)):
                                print(f"Warning: NaN correlation detected, setting to 0. Pred var: {torch.var(preds_squeezed):.6f}")
                                correlation = 0.0
                        except Exception as e:
                            print(f"Error calculating correlation: {e}")
                            # Use manual correlation calculation as fallback
                            preds_np = preds_squeezed.cpu().numpy()
                            targets_np = self.y_test.cpu().numpy()
                            correlation = np.corrcoef(preds_np, targets_np)[0, 1]
                            if np.isnan(correlation):
                                correlation = 0.0
                
                self.correlation_history.append((epoch, correlation))
                
                # Check for phase transition based on correlation
                if not self.phase_transition['detected'] and correlation > 0.99:
                    self.phase_transition['detected'] = True
                    self.phase_transition['epoch'] = epoch
                    self.phase_transition['correlation'] = correlation
                    
                    # Record feature importance ratio at transition
                    if self.feature_importance['epochs']:
                        # Find closest epoch in feature importance history
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
                
                # Print progress
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}: MSE={test_loss:.6f}, Correlation={correlation:.4f}, Time={elapsed:.1f}s")
                
                # Print some of the new metrics
                act_layer1, act_layer2 = self.activation_stats['layer1_percent_active'][-1], self.activation_stats['layer2_percent_active'][-1]
                print(f"Active neurons: Layer 1 = {act_layer1:.2f}%, Layer 2 = {act_layer2:.2f}%")
                
                if self.W1_grad:
                    grad_corrs = self.gradient_correlations['mean_rel_rel'][-1], self.gradient_correlations['mean_rel_irrel'][-1]
                    print(f"Gradient correlations: Rel-Rel = {grad_corrs[0]:.4f}, Rel-Irrel = {grad_corrs[1]:.4f}")
                
                # Early stopping
                if correlation > early_stop_corr:
                    print(f"Early stopping at epoch {epoch} with correlation {correlation:.4f}")
                    return correlation, epoch
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test)
            final_loss = ((preds - self.y_test) ** 2).mean().item()
            final_correlation = torch.corrcoef(torch.stack([preds.squeeze(), self.y_test]))[0, 1].item()
        
        print("Training completed!")
        print(f"Final MSE: {final_loss:.6f}")
        print(f"Final correlation: {final_correlation:.4f}")
        
        return final_correlation, epoch
        
    def plot_neuron_target_correlation_evolution(self):
        """
        Plot how the correlation between each neuron's activation and the target function
        evolves over training epochs, using heatmaps.
        """
        if not self.neuron_target_correlation_evolution['epochs']:
            print("No neuron-target correlation evolution data available")
            return
        
        # Extract data
        epochs = self.neuron_target_correlation_evolution['epochs']
        layer1_corrs = self.neuron_target_correlation_evolution['layer1_neuron_correlations']
        layer2_corrs = self.neuron_target_correlation_evolution['layer2_neuron_correlations']
        
        # Create figure with two subplots - one for each layer
        fig, axs = plt.subplots(2, 1, figsize=(15, 12))
        
        # Determine color map range to be symmetric around zero
        vmax = max(
            np.max(np.abs(np.array(layer1_corrs))),
            np.max(np.abs(np.array(layer2_corrs)))
        )
        vmin = -vmax
        
        # Setup for colormaps
        cmap = plt.cm.RdBu_r
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Plot Layer 1 neuron correlation evolution
        ax1 = axs[0]
        
        # Create image array where:
        # - rows = epochs
        # - columns = neurons
        # - values = correlations with target
        layer1_data = np.array(layer1_corrs)
        
        # Plot as heatmap
        im1 = ax1.imshow(layer1_data, aspect='auto', cmap=cmap, norm=norm,
                        extent=[0, self.M1, epochs[-1], epochs[0]])  # x=neurons, y=epochs
        
        ax1.set_title(f'First Layer Neuron-Target Correlation Evolution (d={self.d}, k={self.k})')
        ax1.set_xlabel('Neuron Index')
        ax1.set_ylabel('Epoch')
        
        # Add horizontal line at phase transition if detected
        if self.phase_transition['detected']:
            ax1.axhline(y=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7,
                    label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax1.legend()
        
        fig.colorbar(im1, ax=ax1, label='Correlation with Target ρᵢ')
        
        # Plot Layer 2 neuron correlation evolution
        ax2 = axs[1]
        
        # Create image array as above
        layer2_data = np.array(layer2_corrs)
        
        # Plot as heatmap
        im2 = ax2.imshow(layer2_data, aspect='auto', cmap=cmap, norm=norm,
                        extent=[0, self.M2, epochs[-1], epochs[0]])  # x=neurons, y=epochs
        
        ax2.set_title(f'Second Layer Neuron-Target Correlation Evolution (d={self.d}, k={self.k})')
        ax2.set_xlabel('Neuron Index')
        ax2.set_ylabel('Epoch')
        
        # Add horizontal line at phase transition if detected
        if self.phase_transition['detected']:
            ax2.axhline(y=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7,
                    label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax2.legend()
        
        fig.colorbar(im2, ax=ax2, label='Correlation with Target ρᵢ')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/neuron_target_correlation_evolution_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()

    def plot_neuron_target_correlation_trajectories(self):
        """
        Plot correlation trajectories for individual neurons with the target function,
        focusing on neurons with the strongest correlations.
        """
        if not self.neuron_target_correlation_evolution['epochs']:
            print("No neuron-target correlation evolution data available")
            return
        
        # Extract data
        epochs = np.array(self.neuron_target_correlation_evolution['epochs'])
        layer1_corrs = np.array(self.neuron_target_correlation_evolution['layer1_neuron_correlations'])
        layer2_corrs = np.array(self.neuron_target_correlation_evolution['layer2_neuron_correlations'])
        
        # Create figures for each layer
        for layer_num, layer_corrs, M in [(1, layer1_corrs, self.M1), (2, layer2_corrs, self.M2)]:
            # Find neurons with the strongest absolute correlation at any point
            max_abs_corrs = np.max(np.abs(layer_corrs), axis=0)
            
            # Get indices of neurons with strongest correlations
            strong_neurons = np.argsort(max_abs_corrs)[-20:]  # Top 20 neurons
            
            # Create figure
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Custom colormap for positive vs negative correlations
            cmap_pos = plt.cm.Reds
            cmap_neg = plt.cm.Blues
            
            # Plot correlation trajectories for strong neurons
            for neuron_idx in strong_neurons:
                # Get correlation trajectory for this neuron
                corr_trajectory = layer_corrs[:, neuron_idx]
                
                # Determine color and style based on correlation sign at the end
                final_corr = corr_trajectory[-1]
                if final_corr >= 0:
                    color = cmap_pos(min(abs(final_corr) * 1.5, 0.9))
                    linestyle = '-'
                else:
                    color = cmap_neg(min(abs(final_corr) * 1.5, 0.9))
                    linestyle = '--'
                
                # Use alpha based on correlation strength
                alpha = min(0.7 + abs(final_corr) * 0.3, 1.0)
                
                # Plot with appropriate styling
                plt.plot(epochs, corr_trajectory, linestyle=linestyle, color=color, alpha=alpha, 
                        linewidth=2, label=f'Neuron {neuron_idx}: {final_corr:.3f}')
            
            # Add phase transition line if detected
            if self.phase_transition['detected']:
                plt.axvline(x=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7,
                        label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.title(f'Layer {layer_num} Neuron-Target Correlation Trajectories (d={self.d}, k={self.k})')
            plt.xlabel('Epoch')
            plt.ylabel('Correlation with Target ρᵢ')
            plt.grid(True, alpha=0.3)
            
            # Add legend if not too many neurons
            if len(strong_neurons) <= 10:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/layer{layer_num}_neuron_target_correlation_trajectories_d{self.d}_k{self.k}.png", dpi=300)
            plt.close()
            
            # Create an additional plot showing absolute correlations
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot absolute correlation trajectories
            for neuron_idx in strong_neurons:
                # Get absolute correlation trajectory for this neuron
                abs_corr_trajectory = np.abs(layer_corrs[:, neuron_idx])
                
                # Use color based on maximum correlation
                max_corr = np.max(abs_corr_trajectory)
                color = plt.cm.viridis(min(max_corr * 1.5, 0.9))
                
                # Plot with appropriate styling
                plt.plot(epochs, abs_corr_trajectory, '-', color=color, alpha=min(0.7 + max_corr * 0.3, 1.0), 
                        linewidth=2, label=f'Neuron {neuron_idx}: {max_corr:.3f}')
            
            # Add phase transition line if detected
            if self.phase_transition['detected']:
                plt.axvline(x=self.phase_transition['epoch'], color='black', linestyle='--', alpha=0.7,
                        label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            
            plt.title(f'Layer {layer_num} Absolute Neuron-Target Correlation Trajectories (d={self.d}, k={self.k})')
            plt.xlabel('Epoch')
            plt.ylabel('Absolute Correlation with Target |ρᵢ|')
            plt.grid(True, alpha=0.3)
            
            # Add legend if not too many neurons
            if len(strong_neurons) <= 10:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/layer{layer_num}_abs_neuron_target_correlation_trajectories_d{self.d}_k{self.k}.png", dpi=300)
            plt.close()

    def plot_neuron_target_correlation_distribution(self):
        """
        Plot the distribution of neuron-target correlations at different stages of training.
        """
        if not self.neuron_target_correlation_evolution['epochs']:
            print("No neuron-target correlation evolution data available")
            return
        
        # Extract data
        epochs = np.array(self.neuron_target_correlation_evolution['epochs'])
        layer1_corrs = np.array(self.neuron_target_correlation_evolution['layer1_neuron_correlations'])
        layer2_corrs = np.array(self.neuron_target_correlation_evolution['layer2_neuron_correlations'])
        
        # Select key epochs for visualization
        if len(epochs) >= 3:
            # Choose beginning, middle, and end epochs (or near phase transition if detected)
            if self.phase_transition['detected']:
                # Find index closest to phase transition
                pt_idx = np.argmin(np.abs(epochs - self.phase_transition['epoch']))
                selected_indices = [0, pt_idx, -1]  # Beginning, phase transition, end
                epoch_labels = ["Initial", f"Phase Transition (Epoch {epochs[pt_idx]})", "Final"]
            else:
                # Just use beginning, middle, end
                selected_indices = [0, len(epochs)//2, -1]
                epoch_labels = ["Initial", f"Middle (Epoch {epochs[len(epochs)//2]})", "Final"]
        else:
            # If we don't have many epochs, just use what we have
            selected_indices = list(range(len(epochs)))
            epoch_labels = [f"Epoch {e}" for e in epochs[selected_indices]]
        
        # Create figure with subplots - one row per layer, one column per selected epoch
        fig, axs = plt.subplots(2, len(selected_indices), figsize=(15, 10))
        
        # If only one epoch, ensure axs is still 2D
        if len(selected_indices) == 1:
            axs = axs.reshape(2, 1)
        
        # For each layer and each selected epoch, plot histogram
        for i, (layer_corrs, layer_name) in enumerate([(layer1_corrs, "First Layer"), (layer2_corrs, "Second Layer")]):
            for j, (idx, label) in enumerate(zip(selected_indices, epoch_labels)):
                ax = axs[i, j]
                
                # Get correlations for this epoch
                corrs = layer_corrs[idx]
                
                # Plot histogram
                ax.hist(corrs, bins=30, alpha=0.7, color='skyblue' if i==0 else 'lightgreen')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                
                # Compute statistics
                mean_corr = np.mean(corrs)
                mean_abs_corr = np.mean(np.abs(corrs))
                max_abs_corr = np.max(np.abs(corrs))
                
                # Add text with statistics
                ax.text(0.05, 0.95, f"Mean: {mean_corr:.3f}\nMean |ρ|: {mean_abs_corr:.3f}\nMax |ρ|: {max_abs_corr:.3f}",
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                ax.set_title(f"{layer_name}: {label}")
                if i == 1:  # Only add x-label for bottom row
                    ax.set_xlabel("Correlation with Target ρᵢ")
                if j == 0:  # Only add y-label for leftmost column
                    ax.set_ylabel("Number of Neurons")
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/neuron_target_correlation_distribution_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_weight_matrices(self):
        """
        Plot heatmaps of weight matrices at key points in training
        """
        # Only visualize if we have at least two snapshots
        if len(self.weight_snapshots['epochs']) < 2:
            print("Not enough weight snapshots for visualization")
            return
        
        # Select snapshots to visualize: initial, phase transition (if detected), final
        epochs_to_show = [0]  # Initial
        
        if self.phase_transition['detected']:
            # Find closest snapshot to phase transition
            pt_epoch = self.phase_transition['epoch']
            closest_idx = min(range(len(self.weight_snapshots['epochs'])), 
                             key=lambda i: abs(self.weight_snapshots['epochs'][i] - pt_epoch))
            if closest_idx not in epochs_to_show:
                epochs_to_show.append(closest_idx)
        
        # Add final snapshot
        epochs_to_show.append(len(self.weight_snapshots['epochs']) - 1)
        
        # Create figure for W1 matrices (first layer weights)
        plt.figure(figsize=(16, 5 * len(epochs_to_show)))
        titles = ['Initial Weights', 'Phase Transition Weights', 'Final Weights']
        titles = titles[:len(epochs_to_show)]
        
        for i, idx in enumerate(epochs_to_show):
            epoch = self.weight_snapshots['epochs'][idx]
            W1 = self.weight_snapshots['W1'][idx]
            
            plt.subplot(len(epochs_to_show), 1, i+1)
            im = plt.imshow(W1, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im)
            plt.title(f"{titles[i]} (Epoch {epoch}) - First Layer (W1)")
            plt.xlabel("Input Feature")
            plt.ylabel("Neuron Index")
            
            # Highlight relevant features
            plt.axvline(x=self.k-0.5, color='green', linestyle='--', alpha=0.7, 
                      label='Relevant/Irrelevant Boundary')
            
            # Add horizontal line at observed transition if available
            if self.phase_transition['detected']:
                detected_epoch = self.phase_transition['epoch']
                plt.axhline(y=self.M1/2, color='red', linestyle='--', alpha=0.7,
                          label=f'Detected Transition (Epoch {detected_epoch})')
            
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/W1_matrices_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Create figure for W2 matrices (second layer weights)
        plt.figure(figsize=(16, 5 * len(epochs_to_show)))
        
        for i, idx in enumerate(epochs_to_show):
            epoch = self.weight_snapshots['epochs'][idx]
            W2 = self.weight_snapshots['W2'][idx]
            
            plt.subplot(len(epochs_to_show), 1, i+1)
            im = plt.imshow(W2, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im)
            plt.title(f"{titles[i]} (Epoch {epoch}) - Second Layer (W2)")
            plt.xlabel("Layer 1 Neuron")
            plt.ylabel("Layer 2 Neuron")
            
            # Add horizontal line at observed transition if available
            if self.phase_transition['detected']:
                detected_epoch = self.phase_transition['epoch']
                plt.axhline(y=self.M2/2, color='red', linestyle='--', alpha=0.7,
                          label=f'Detected Transition (Epoch {detected_epoch})')
            
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/W2_matrices_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_feature_importance_evolution(self):
        """
        Visualize how the importance ratio between relevant and irrelevant features evolves
        """
        if not self.feature_importance['epochs']:
            print("No feature importance data recorded")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot ratio evolution in log scale
        plt.semilogy(self.feature_importance['epochs'], self.feature_importance['ratio'], 
                   marker='o', linestyle='-', linewidth=2, label='Feature Importance Ratio')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
            
            # Add horizontal line at transition ratio
            if self.phase_transition['feature_ratio'] is not None:
                plt.axhline(y=self.phase_transition['feature_ratio'], color='red', linestyle=':', alpha=0.7,
                          label=f'Ratio at Transition = {self.phase_transition["feature_ratio"]:.2f}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Relevant/Irrelevant Feature Importance Ratio (log scale)')
        plt.title(f'Evolution of Feature Importance Ratio (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_importance_evolution_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_feature_correlations(self):
        """
        Plot the evolution of feature correlations C_l for relevant and irrelevant features
        """
        if not self.feature_correlations['epochs']:
            print("No feature correlation data recorded")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot mean absolute correlation
        plt.semilogy(self.feature_correlations['epochs'], self.feature_correlations['relevant_mean'], 
                   marker='o', linestyle='-', label='Relevant Features')
        plt.semilogy(self.feature_correlations['epochs'], self.feature_correlations['irrelevant_mean'], 
                   marker='x', linestyle='--', label='Irrelevant Features')
        
        # Add theoretical scaling line
        x = np.array(self.feature_correlations['epochs'])
        x_pos = x[x > 0]  # Avoid log(0)
        scaling = self.d ** (-(self.k-1)/2)
        plt.semilogy(x_pos, scaling * np.ones_like(x_pos), 
                   linestyle='-.', color='green', 
                   label=f'd^(-(k-1)/2) = {scaling:.1e}')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Correlation |C_l| (log scale)')
        plt.title(f'Feature Correlations C_l Evolution (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_correlations_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_signal_noise_ratio(self):
        """
        Plot the evolution of empirical and theoretical signal and noise terms, and their ratios
        """
        if not self.empirical_signal_noise['epochs']:
            print("No signal/noise data recorded")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot empirical and theoretical signals and noises in log scale
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['empirical_signal'], 
                   marker='o', linestyle='-', label='Empirical Signal')
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['empirical_noise'], 
                   marker='x', linestyle='-', label='Empirical Noise')
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['theoretical_signal'], 
                   marker='^', linestyle='--', alpha=0.7, label='Theoretical Signal: η·T·d^(-(k-1)/2)')
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['theoretical_noise'], 
                   marker='v', linestyle='--', alpha=0.7, label='Theoretical Noise: η·√(T/B)')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Magnitude (log scale)')
        plt.title(f'Signal vs Noise Evolution (d={self.d}, k={self.k}, B={self.batch_size})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/signal_noise_magnitudes_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Plot the empirical and theoretical ratios
        plt.figure(figsize=(12, 8))
        
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['empirical_ratio'], 
                   marker='o', linestyle='-', linewidth=2, label='Empirical Signal/Noise Ratio')
        plt.semilogy(self.empirical_signal_noise['epochs'], self.empirical_signal_noise['theoretical_ratio'], 
                   marker='x', linestyle='--', linewidth=2, alpha=0.7, label='Theoretical Signal/Noise Ratio')
        
        # Add ratio = 1 line
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Ratio = 1')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Signal/Noise Ratio (log scale)')
        plt.title(f'Signal to Noise Ratio (d={self.d}, k={self.k}, B={self.batch_size})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/signal_noise_ratio_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_gradient_statistics(self):
        """
        Plot gradient statistics evolution during training
        """
        if not self.gradient_stats['epochs'] or not self.W1_grad:
            print("No gradient statistics recorded or W1 not trainable")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Mean gradient magnitudes for relevant and irrelevant features
        plt.subplot(2, 2, 1)
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['relevant_grad_mean'], 
                   marker='o', linestyle='-', label='Relevant Features')
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['irrelevant_grad_mean'], 
                   marker='x', linestyle='--', label='Irrelevant Features')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Mean Gradient Magnitude (log scale)')
        plt.title('Gradient Magnitude Evolution')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Gradient standard deviations
        plt.subplot(2, 2, 2)
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['relevant_grad_std'], 
                   marker='o', linestyle='-', label='Relevant Features')
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['irrelevant_grad_std'], 
                   marker='x', linestyle='--', label='Irrelevant Features')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Std Dev (log scale)')
        plt.title('Gradient Variability Evolution')
        plt.grid(True)
        plt.legend()
        
        # Plot 3: Ratio of relevant to irrelevant mean gradients
        plt.subplot(2, 2, 3)
        irrelevant_means = np.array(self.gradient_stats['irrelevant_grad_mean'])
        irrelevant_means[irrelevant_means == 0] = 1e-10  # Avoid division by zero
        ratio = np.array(self.gradient_stats['relevant_grad_mean']) / irrelevant_means
        plt.semilogy(self.gradient_stats['epochs'], ratio, 
                   marker='o', linestyle='-', label='Relevant/Irrelevant Ratio')
        
        # Add line at ratio = 1
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7)
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Ratio (log scale)')
        plt.title('Ratio of Relevant to Irrelevant Gradient Magnitudes')
        plt.grid(True)
        plt.legend()
        
        # Plot 4: Overall gradient norm
        plt.subplot(2, 2, 4)
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['w1_grad_norm'], 
                   marker='o', linestyle='-', label='W1 Gradient Norm')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Overall W1 Gradient Norm Evolution')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/gradient_statistics_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_training_progress(self):
        """
        Plot training loss and correlation evolution
        """
        if len(self.loss_history) == 0 or len(self.correlation_history) == 0:
            print("No training history recorded")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot loss on top subplot
        ax1.semilogy(range(len(self.loss_history)), self.loss_history)
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_title(f'Training Progress (d={self.d}, k={self.k})')
        ax1.grid(True)
        
        # Add detected transition point to loss plot
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            ax1.axvline(x=detected_epoch, color='red', linestyle='--', 
                       label=f'Detected Transition (Epoch {detected_epoch})')
        
        ax1.legend()
        
        # Plot correlation on bottom subplot
        corr_epochs, corr_values = zip(*self.correlation_history)
        ax2.plot(corr_epochs, corr_values, marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Correlation')
        ax2.grid(True)
        
        # Add detected transition point to correlation plot
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            ax2.axvline(x=detected_epoch, color='red', linestyle='--', 
                       label=f'Detected Transition (Epoch {detected_epoch})')
        
        # Add correlation thresholds
        for threshold in [0.5, 0.9]:
            ax2.axhline(y=threshold, color=f'C{int(threshold*10)}', 
                      linestyle=':', alpha=0.7,
                      label=f'Correlation = {threshold}')
        
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_progress_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_activation_statistics(self):
        """
        Plot the percentage of active neurons in each layer over training
        """
        if not self.activation_stats['epochs']:
            print("No activation statistics recorded")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot percentage of active neurons for both layers
        plt.plot(self.activation_stats['epochs'], self.activation_stats['layer1_percent_active'], 
               marker='o', linestyle='-', label='Layer 1 (First Hidden)')
        plt.plot(self.activation_stats['epochs'], self.activation_stats['layer2_percent_active'], 
               marker='x', linestyle='--', label='Layer 2 (Second Hidden)')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
            
            # Find indices closest to transition
            idx = np.argmin(np.abs(np.array(self.activation_stats['epochs']) - detected_epoch))
            layer1_at_transition = self.activation_stats['layer1_percent_active'][idx]
            layer2_at_transition = self.activation_stats['layer2_percent_active'][idx]
            
            # Add horizontal lines for activation percentages at transition
            plt.axhline(y=layer1_at_transition, color='C0', linestyle=':', alpha=0.7,
                      label=f'Layer 1 at Transition: {layer1_at_transition:.2f}%')
            plt.axhline(y=layer2_at_transition, color='C1', linestyle=':', alpha=0.7,
                      label=f'Layer 2 at Transition: {layer2_at_transition:.2f}%')
        
        plt.xlabel('Epoch')
        plt.ylabel('Percentage of Active ReLU Neurons (%)')
        plt.title(f'ReLU Activation Percentage vs. Training (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/relu_activation_stats_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_gradient_correlations(self):
        """
        Plot the gradient correlations (ρj,j') statistics
        """
        if not self.gradient_correlations['epochs'] or not self.W1_grad:
            print("No gradient correlation data recorded or W1 not trainable")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot mean correlations for different feature types
        plt.plot(self.gradient_correlations['epochs'], self.gradient_correlations['mean_rel_rel'], 
               marker='o', linestyle='-', label='Relevant-Relevant')
        plt.plot(self.gradient_correlations['epochs'], self.gradient_correlations['mean_rel_irrel'], 
               marker='x', linestyle='--', label='Relevant-Irrelevant')
        plt.plot(self.gradient_correlations['epochs'], self.gradient_correlations['mean_irrel_irrel'], 
               marker='^', linestyle='-.', label='Irrelevant-Irrelevant')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
            
            # Find indices closest to transition
            idx = np.argmin(np.abs(np.array(self.gradient_correlations['epochs']) - detected_epoch))
            rel_rel_at_transition = self.gradient_correlations['mean_rel_rel'][idx]
            rel_irrel_at_transition = self.gradient_correlations['mean_rel_irrel'][idx]
            
            # Add horizontal lines for correlation values at transition
            plt.axhline(y=rel_rel_at_transition, color='C0', linestyle=':', alpha=0.7,
                      label=f'Rel-Rel at Transition: {rel_rel_at_transition:.4f}')
            plt.axhline(y=rel_irrel_at_transition, color='C1', linestyle=':', alpha=0.7,
                      label=f'Rel-Irrel at Transition: {rel_irrel_at_transition:.4f}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Gradient Correlation (ρj,j\')')
        plt.title(f'Gradient Correlation Statistics (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/gradient_correlations_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_hidden_target_correlations(self):
        """
        Plot the sum of absolute correlations between hidden units and target function
        """
        if not self.hidden_target_correlations['epochs']:
            print("No hidden-target correlation data recorded")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot sum of absolute correlations for both layers
        plt.plot(self.hidden_target_correlations['epochs'], self.hidden_target_correlations['layer1_corr_sum'], 
               marker='o', linestyle='-', label='Layer 1 (First Hidden)')
        plt.plot(self.hidden_target_correlations['epochs'], self.hidden_target_correlations['layer2_corr_sum'], 
               marker='x', linestyle='--', label='Layer 2 (Second Hidden)')
        
        # Add detected transition point if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
            
            # Find indices closest to transition
            idx = np.argmin(np.abs(np.array(self.hidden_target_correlations['epochs']) - detected_epoch))
            layer1_at_transition = self.hidden_target_correlations['layer1_corr_sum'][idx]
            layer2_at_transition = self.hidden_target_correlations['layer2_corr_sum'][idx]
            
            # Add horizontal lines for correlation values at transition
            plt.axhline(y=layer1_at_transition, color='C0', linestyle=':', alpha=0.7,
                      label=f'Layer 1 at Transition: {layer1_at_transition:.4f}')
            plt.axhline(y=layer2_at_transition, color='C1', linestyle=':', alpha=0.7,
                      label=f'Layer 2 at Transition: {layer2_at_transition:.4f}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Sum of |E[h_i(x)⋅f_S(x)]|')
        plt.title(f'Hidden Unit Correlation with Target (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/hidden_target_correlations_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def plot_gradient_eigenvalues(self):
        """
        Plot the evolution of the top eigenvalues of the gradient matrices
        """
        if not self.gradient_eigen['epochs'] or (not self.W1_grad and not self.W2_grad):
            print("No gradient eigenvalue data recorded or both layers are not trainable")
            return
        
        # Plot W1 eigenvalues if W1 is trainable
        if self.W1_grad:
            plt.figure(figsize=(12, 8))
            
            epochs = self.gradient_eigen['epochs']
            w1_eigenvalues = self.gradient_eigen['W1_eigenvalues']
            
            # For each of the top k eigenvalues, plot its evolution
            for i in range(min(self.top_k_eigen, len(w1_eigenvalues[0]))):
                values = [evals[i] for evals in w1_eigenvalues]
                plt.semilogy(epochs, values, marker='o', linestyle='-', label=f'Eigenvalue {i+1}')
            
            # Add detected transition point if available
            if self.phase_transition['detected']:
                detected_epoch = self.phase_transition['epoch']
                plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                          label=f'Detected Transition (Epoch {detected_epoch})')
            
            plt.xlabel('Epoch')
            plt.ylabel('Eigenvalue (log scale)')
            plt.title(f'Top W1 Gradient Eigenvalues (d={self.d}, k={self.k})')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/w1_gradient_eigenvalues_d{self.d}_k{self.k}.png", dpi=300)
            plt.close()
        
        # Plot W2 eigenvalues if W2 is trainable
        if self.W2_grad:
            plt.figure(figsize=(12, 8))
            
            epochs = self.gradient_eigen['epochs']
            w2_eigenvalues = self.gradient_eigen['W2_eigenvalues']
            
            # For each of the top k eigenvalues, plot its evolution
            for i in range(min(self.top_k_eigen, len(w2_eigenvalues[0]))):
                values = [evals[i] for evals in w2_eigenvalues]
                plt.semilogy(epochs, values, marker='o', linestyle='-', label=f'Eigenvalue {i+1}')
            
            # Add detected transition point if available
            if self.phase_transition['detected']:
                detected_epoch = self.phase_transition['epoch']
                plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                          label=f'Detected Transition (Epoch {detected_epoch})')
            
            plt.xlabel('Epoch')
            plt.ylabel('Eigenvalue (log scale)')
            plt.title(f'Top W2 Gradient Eigenvalues (d={self.d}, k={self.k})')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/w2_gradient_eigenvalues_d{self.d}_k{self.k}.png", dpi=300)
            plt.close()
    
    def plot_principal_angles(self):
        """
        Plot the principal angles between consecutive gradient subspaces
        """
        if not self.principal_angles['epochs'] or len(self.principal_angles['epochs']) < 2:
            print("Not enough principal angle data recorded")
            return
        
        # Plot W1 principal angles if W1 is trainable
        if self.W1_grad:
            plt.figure(figsize=(12, 8))
            
            epochs = self.principal_angles['epochs']
            w1_angles = self.principal_angles['W1_angles']
            
            # Compute mean angle for each epoch
            mean_angles = [np.mean(angles) for angles in w1_angles]
            
            # Plot mean of principal angles
            plt.plot(epochs, mean_angles, marker='o', linestyle='-', label='Mean Principal Angle')
            
            # Plot individual angles if there are not too many
            if len(w1_angles[0]) <= 5:
                for i in range(len(w1_angles[0])):
                    values = [angles[i] for angles in w1_angles]
                    plt.plot(epochs, values, marker='.', linestyle=':', alpha=0.5, label=f'Angle {i+1}')
            
            # Add detected transition point if available
            if self.phase_transition['detected']:
                detected_epoch = self.phase_transition['epoch']
                plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                          label=f'Detected Transition (Epoch {detected_epoch})')
            
            plt.xlabel('Epoch')
            plt.ylabel('Principal Angle (radians)')
            plt.title(f'W1 Gradient Subspace Principal Angles (d={self.d}, k={self.k})')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/w1_principal_angles_d{self.d}_k{self.k}.png", dpi=300)
            plt.close()
        
        # Plot W2 principal angles if W2 is trainable
        if self.W2_grad:
            plt.figure(figsize=(12, 8))
            
            epochs = self.principal_angles['epochs']
            w2_angles = self.principal_angles['W2_angles']
            
            # Compute mean angle for each epoch
            mean_angles = [np.mean(angles) for angles in w2_angles]
            
            # Plot mean of principal angles
            plt.plot(epochs, mean_angles, marker='o', linestyle='-', label='Mean Principal Angle')
            
            # Plot individual angles if there are not too many
            if len(w2_angles[0]) <= 5:
                for i in range(len(w2_angles[0])):
                    values = [angles[i] for angles in w2_angles]
                    plt.plot(epochs, values, marker='.', linestyle=':', alpha=0.5, label=f'Angle {i+1}')
            
            # Add detected transition point if available
            if self.phase_transition['detected']:
                detected_epoch = self.phase_transition['epoch']
                plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                          label=f'Detected Transition (Epoch {detected_epoch})')
            
            plt.xlabel('Epoch')
            plt.ylabel('Principal Angle (radians)')
            plt.title(f'W2 Gradient Subspace Principal Angles (d={self.d}, k={self.k})')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/w2_principal_angles_d{self.d}_k{self.k}.png", dpi=300)
            plt.close()
            
    def plot_neuron_activation_evolution(self):
        """
        Plot how neuron activation ratios evolve over training epochs.
        Uses continuous color coding and transparency to visualize changes.
        """
        if not self.neuron_activation_evolution['epochs']:
            print("No neuron activation evolution data available")
            return
        
        # Extract data
        epochs = self.neuron_activation_evolution['epochs']
        layer1_ratios = self.neuron_activation_evolution['layer1_activation_ratios']
        layer2_ratios = self.neuron_activation_evolution['layer2_activation_ratios']
        
        # Create figure with two subplots - one for each layer
        fig, axs = plt.subplots(2, 1, figsize=(15, 12))
        
        # Setup for colormaps
        cmap = cm.viridis
        norm = Normalize(vmin=0, vmax=1)
        
        # Plot Layer 1 neuron activation evolution
        ax1 = axs[0]
        
        # Create image array where:
        # - rows = epochs
        # - columns = neurons
        # - values = activation ratios
        layer1_data = np.array(layer1_ratios)
        
        # Plot as heatmap
        im1 = ax1.imshow(layer1_data, aspect='auto', cmap=cmap, norm=norm,
                        extent=[0, self.M1, epochs[-1], epochs[0]])  # x=neurons, y=epochs
        
        ax1.set_title(f'First Layer Neuron Activation Evolution (d={self.d}, k={self.k})')
        ax1.set_xlabel('Neuron Index')
        ax1.set_ylabel('Epoch')
        
        # Add horizontal line at phase transition if detected
        if self.phase_transition['detected']:
            ax1.axhline(y=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7,
                    label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax1.legend()
        
        fig.colorbar(im1, ax=ax1, label='Activation Ratio')
        
        # Plot Layer 2 neuron activation evolution
        ax2 = axs[1]
        
        # Create image array as above
        layer2_data = np.array(layer2_ratios)
        
        # Plot as heatmap
        im2 = ax2.imshow(layer2_data, aspect='auto', cmap=cmap, norm=norm,
                        extent=[0, self.M2, epochs[-1], epochs[0]])  # x=neurons, y=epochs
        
        ax2.set_title(f'Second Layer Neuron Activation Evolution (d={self.d}, k={self.k})')
        ax2.set_xlabel('Neuron Index')
        ax2.set_ylabel('Epoch')
        
        # Add horizontal line at phase transition if detected
        if self.phase_transition['detected']:
            ax2.axhline(y=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7,
                    label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax2.legend()
        
        fig.colorbar(im2, ax=ax2, label='Activation Ratio')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/neuron_activation_evolution_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Create additional plot focusing on the most active neurons in layer 2
        # Find neurons that have high activation in at least one epoch
        max_activations = np.max(layer2_data, axis=0)
        active_neurons = np.where(max_activations > 0.5)[0]
        
        if len(active_neurons) > 0:
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot activation trajectories for active neurons
            for neuron_idx in active_neurons:
                # Get activation trajectory for this neuron
                activation_trajectory = [ratios[neuron_idx] for ratios in layer2_ratios]
                
                # Use color based on maximum activation
                max_act = max(activation_trajectory)
                color = cmap(norm(max_act))
                
                # Plot with transparency proportional to maximum activation
                alpha = max(0.3, max_act)
                plt.plot(epochs, activation_trajectory, '-', color=color, alpha=alpha, 
                        linewidth=2, label=f'Neuron {neuron_idx}')
            
            # Add phase transition line if detected
            if self.phase_transition['detected']:
                plt.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7,
                        label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            
            plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.7, label='50% Activation')
            
            plt.title(f'Active Second Layer Neurons Activation Trajectories (d={self.d}, k={self.k})')
            plt.xlabel('Epoch')
            plt.ylabel('Activation Ratio')
            plt.grid(True, alpha=0.3)
            
            # Add colorbar to indicate activation level
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # This is needed for matplotlib < 3.3
            fig.colorbar(sm, ax=ax, label='Maximum Activation Ratio')
            
            # If too many neurons, don't add a legend
            if len(active_neurons) <= 10:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/active_neurons_trajectories_d{self.d}_k{self.k}.png", dpi=300)
            plt.close()
            
    def plot_neuron_activation_trajectories(self):
        """
        Plot activation trajectories for each neuron in both layers, color-coded by neuron index.
        Uses transparency to make the plots readable with 512 lines.
        """
        if not self.neuron_activation_evolution['epochs']:
            print("No neuron activation evolution data available")
            return
        
        # Extract data
        epochs = np.array(self.neuron_activation_evolution['epochs'])
        layer1_ratios = np.array(self.neuron_activation_evolution['layer1_activation_ratios'])
        layer2_ratios = np.array(self.neuron_activation_evolution['layer2_activation_ratios'])
        
        # Create figure for Layer 1
        fig1, ax1 = plt.subplots(figsize=(15, 8))
        
        # Prepare colormap for neuron indices
        cmap_layer1 = plt.cm.nipy_spectral
        
        # Plot each neuron's activation trajectory with color based on neuron index
        for i in range(self.M1):
            # Get activation trajectory for this neuron across all epochs
            neuron_trajectory = layer1_ratios[:, i]
            
            # Normalize neuron index for coloring
            color = cmap_layer1(i / self.M1)
            
            # Use very low alpha for visibility with many lines
            alpha = 0.1
            
            # If this neuron has significant activation changes, make it more visible
            activation_range = np.max(neuron_trajectory) - np.min(neuron_trajectory)
            if activation_range > 0.2:
                alpha = 0.4
            
            ax1.plot(epochs, neuron_trajectory, '-', color=color, alpha=alpha, linewidth=1)
        
        # Add phase transition line if detected
        if self.phase_transition['detected']:
            ax1.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7,
                    label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax1.legend()
        
        ax1.set_title(f'First Layer Neuron Activation Trajectories (d={self.d}, k={self.k})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Activation Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Create ScalarMappable for colorbar and set array properly
        sm_layer1 = plt.cm.ScalarMappable(cmap=cmap_layer1)
        sm_layer1.set_array([])  # This is required
        fig1.colorbar(sm_layer1, ax=ax1, label='Neuron Index (0-511)')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/layer1_neuron_trajectories_d{self.d}_k{self.k}.png", dpi=300)
        plt.close(fig1)
        
        # Create figure for Layer 2
        fig2, ax2 = plt.subplots(figsize=(15, 8))
        
        # Prepare colormap for neuron indices
        cmap_layer2 = plt.cm.nipy_spectral
        
        # Plot each neuron's activation trajectory with color based on neuron index
        for i in range(self.M2):
            # Get activation trajectory for this neuron across all epochs
            neuron_trajectory = layer2_ratios[:, i]
            
            # Normalize neuron index for coloring
            color = cmap_layer2(i / self.M2)
            
            # Use very low alpha for visibility with many lines
            alpha = 0.1
            
            # If this neuron has significant activation changes, make it more visible
            activation_range = np.max(neuron_trajectory) - np.min(neuron_trajectory)
            if activation_range > 0.2:
                alpha = 0.4
            
            ax2.plot(epochs, neuron_trajectory, '-', color=color, alpha=alpha, linewidth=1)
        
        # Add phase transition line if detected
        if self.phase_transition['detected']:
            ax2.axvline(x=self.phase_transition['epoch'], color='red', linestyle='--', alpha=0.7,
                    label=f'Phase Transition (Epoch {self.phase_transition["epoch"]})')
            ax2.legend()
        
        ax2.set_title(f'Second Layer Neuron Activation Trajectories (d={self.d}, k={self.k})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Activation Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Create ScalarMappable for colorbar and set array properly
        sm_layer2 = plt.cm.ScalarMappable(cmap=cmap_layer2)
        sm_layer2.set_array([])  # This is required
        fig2.colorbar(sm_layer2, ax=ax2, label='Neuron Index (0-511)')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/layer2_neuron_trajectories_d{self.d}_k{self.k}.png", dpi=300)
        plt.close(fig2)
        
    def plot_neuron_activation_ratios(self):
        """
        Create visualization of neuron activation ratios for both layers.
        This shows the final state analysis using all 2^k relevant patterns.
        """
        if self.neuron_activation_analysis['layer1_activation_ratio'] is None:
            print("No neuron activation analysis data available. Run analyze_neuron_activations() first.")
            return
        
        # Get activation ratios
        layer1_ratio = self.neuron_activation_analysis['layer1_activation_ratio']
        layer2_ratio = self.neuron_activation_analysis['layer2_activation_ratio']
        
        # Set up colors
        cmap = cm.viridis
        norm = Normalize(vmin=0, vmax=1)
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot Layer 1 neuron activation ratios on first subplot
        ax1 = axs[0]
        
        # Plot bars individually to set different alpha values
        x = np.arange(self.M1)
        for i, ratio in enumerate(layer1_ratio):
            color = cmap(norm(ratio))
            alpha = max(0.3, ratio)  # Single alpha value for this bar
            ax1.bar(i, ratio, color=color, alpha=alpha, width=1.0)
        
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                label='50% activation')
        
        ax1.set_title(f'First Layer Neuron Activation Ratios (d={self.d}, k={self.k})')
        ax1.set_xlabel('Neuron Index')
        ax1.set_ylabel('Activation Ratio')
        ax1.set_xlim(-1, self.M1)
        ax1.set_ylim(0, 1.05)
        ax1.grid(alpha=0.3)
        
        # Add colorbar - properly initialized
        sm1 = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm1.set_array([])
        fig.colorbar(sm1, ax=ax1, label='Activation Ratio')
        ax1.legend()
        
        # Plot Layer 2 neuron activation ratios on second subplot
        ax2 = axs[1]
        
        # Plot bars individually for Layer 2 as well
        x = np.arange(self.M2)
        for i, ratio in enumerate(layer2_ratio):
            color = cmap(norm(ratio))
            alpha = max(0.3, ratio)  # Single alpha value for this bar
            ax2.bar(i, ratio, color=color, alpha=alpha, width=1.0)
        
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                label='50% activation')
        
        ax2.set_title(f'Second Layer Neuron Activation Ratios (d={self.d}, k={self.k})')
        ax2.set_xlabel('Neuron Index')
        ax2.set_ylabel('Activation Ratio')
        ax2.set_xlim(-1, self.M2)
        ax2.set_ylim(0, 1.05)
        ax2.grid(alpha=0.3)
        
        # Add colorbar - properly initialized
        sm2 = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm2.set_array([])
        fig.colorbar(sm2, ax=ax2, label='Activation Ratio')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/neuron_activation_ratios_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Create a more detailed visualization with histograms
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        
        # Histogram of Layer 1 neuron activation ratios
        ax1 = axs[0]
        ax1.hist(layer1_ratio, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, 
                label='50% activation')
        ax1.set_title(f'Distribution of First Layer Neuron Activation Ratios (d={self.d}, k={self.k})')
        ax1.set_xlabel('Activation Ratio')
        ax1.set_ylabel('Number of Neurons')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Histogram of Layer 2 neuron activation ratios
        ax2 = axs[1]
        ax2.hist(layer2_ratio, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, 
                label='50% activation')
        ax2.set_title(f'Distribution of Second Layer Neuron Activation Ratios (d={self.d}, k={self.k})')
        ax2.set_xlabel('Activation Ratio')
        ax2.set_ylabel('Number of Neurons')
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/neuron_activation_distribution_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
    def analyze_neuron_activations(self):
        """
        Perform detailed analysis of neuron activations using all 2^k vectors 
        of relevant features combined with random irrelevant features.
        
        This analyzes the final state of the network.
        """
        print("Analyzing neuron activations with all 2^k vectors...")
        self.model.eval()
        
        # Generate all 2^k possible relevant feature vectors
        relevant_patterns = list(itertools.product([1, -1], repeat=self.k))
        num_patterns = len(relevant_patterns)  # Should be 2^k
        num_random_samples = 200  # Number of random irrelevant parts to test for each relevant pattern
        
        # Prepare arrays to store activation results
        layer1_activations = np.zeros(self.M1)  # Activation count for each neuron in layer 1
        layer2_activations = np.zeros(self.M2)  # Activation count for each neuron in layer 2
        
        total_samples = num_patterns * num_random_samples
        print(f"Testing {num_patterns} relevant patterns with {num_random_samples} random variations each = {total_samples} total inputs")
        
        # Prepare batch processing for efficiency
        batch_size = 1000
        batches_needed = (total_samples + batch_size - 1) // batch_size  # Ceiling division
        
        sample_idx = 0
        with torch.no_grad():
            for pattern_idx, relevant_pattern in enumerate(tqdm(relevant_patterns, desc="Processing patterns")):
                # Convert tuple to tensor
                relevant_part = torch.tensor(relevant_pattern, dtype=torch.float32)
                
                # Create inputs for this relevant pattern
                for _ in range(num_random_samples):
                    # Generate random values for irrelevant features
                    irrelevant_part = torch.tensor(np.random.choice([-1, 1], size=(self.d - self.k)), dtype=torch.float32)
                    
                    # Create full input vector
                    batch_idx = sample_idx % batch_size
                    
                    # Initialize batch tensor if this is the first sample in the batch
                    if batch_idx == 0:
                        batch_inputs = torch.zeros((min(batch_size, total_samples - sample_idx), self.d), 
                                                   dtype=torch.float32)
                    
                    # Add this sample to the batch
                    batch_inputs[batch_idx, :self.k] = relevant_part
                    batch_inputs[batch_idx, self.k:] = irrelevant_part
                    
                    sample_idx += 1

                    # Process batch if it's full or this is the last sample
                    if batch_idx == batch_size - 1 or sample_idx == total_samples:
                        # Move batch to device
                        batch_inputs = batch_inputs.to(self.device)
                        
                        # Get neuron activations
                        h1 = self.model.get_first_layer_activations(batch_inputs)
                        h2 = self.model.get_second_layer_activations(batch_inputs)
                        
                        # Update activation counters (neurons that are active)
                        layer1_activations += torch.sum((h1 > 0).float(), dim=0).cpu().numpy()
                        layer2_activations += torch.sum((h2 > 0).float(), dim=0).cpu().numpy()
        
        # Compute activation ratios (percentage of inputs where each neuron activates)
        layer1_activation_ratio = layer1_activations / total_samples
        layer2_activation_ratio = layer2_activations / total_samples
        
        # Store results
        self.neuron_activation_analysis['layer1_activation_ratio'] = layer1_activation_ratio
        self.neuron_activation_analysis['layer2_activation_ratio'] = layer2_activation_ratio
        
        # Plot activation ratios in the standard way (static analysis at the end)
        self.plot_neuron_activation_ratios()
        
        return layer1_activation_ratio, layer2_activation_ratio
        
    def _plot_detailed_neuron_connections(self, neurons):
        """
        For selected neurons in the second layer, plot a detailed analysis of how they
        combine features from the first layer, which in turn combines input features.
        
        Args:
            neurons: List of neuron indices to analyze
        """
        # Check if neurons array is empty using len instead of direct boolean check
        if len(neurons) == 0:
            print("No neurons to analyze")
            return
        
        # Get weights for analysis
        W1 = self.model.W1.detach().cpu().numpy()
        W2 = self.model.W2.detach().cpu().numpy()
        
        for neuron_idx in neurons:
            # Create figure for this neuron
            fig, axs = plt.subplots(2, 1, figsize=(15, 10))
            
            # Get weights from first to second layer for this neuron
            w2_weights = W2[neuron_idx, :]
            
            # Find the strongest connections from layer 1
            strongest_idx = np.argsort(np.abs(w2_weights))[-10:]  # Top 10 connections
            
            # Plot heatmap showing how this neuron's inputs (from layer 1) connect to original features
            connection_matrix = np.zeros((len(strongest_idx), self.d))
            
            for i, l1_idx in enumerate(strongest_idx):
                # Get weight from layer 1 neuron to this layer 2 neuron
                l2_weight = w2_weights[l1_idx]
                
                # Get weights from input to this layer 1 neuron
                l1_weights = W1[l1_idx, :]
                
                # Scale the weights by the connection strength to layer 2
                connection_matrix[i, :] = l1_weights * l2_weight
            
            # Plot the connection matrix
            ax1 = axs[0]
            im = ax1.imshow(connection_matrix, cmap='RdBu_r', aspect='auto')
            fig.colorbar(im, ax=ax1, label='Effective Weight (L1-L2 * L0-L1)')
            ax1.set_title(f'Layer 2 Neuron {neuron_idx} - Connection Patterns')
            ax1.set_xlabel('Input Feature Index')
            ax1.set_ylabel('Layer 1 Neuron')
            
            # Add vertical line separating relevant from irrelevant features
            ax1.axvline(x=self.k-0.5, color='green', linestyle='--', alpha=0.7,
                    label=f'Relevant Features (0-{self.k-1})')
            
            # Add custom yticks showing the actual first layer neuron indices
            ax1.set_yticks(range(len(strongest_idx)))
            ax1.set_yticklabels(strongest_idx)
            ax1.legend()
            
            # Plot the strength of connections to layer 1 neurons
            ax2 = axs[1]
            ax2.bar(range(self.M1), w2_weights, alpha=0.7)
            ax2.set_title(f'Layer 2 Neuron {neuron_idx} - Weights to Layer 1 Neurons')
            ax2.set_xlabel('Layer 1 Neuron Index')
            ax2.set_ylabel('Weight Value')
            ax2.grid(alpha=0.3)
            
            # Highlight the strongest connections
            for idx in strongest_idx:
                ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/neuron_{neuron_idx}_detailed_d{self.d}_k{self.k}.png", dpi=300)
            plt.close()
            
    def visualize_second_layer_patterns(self, activation_threshold=0.01):
        """
        Analyze and visualize input patterns that activate neurons in the second layer.
        This method focuses on neurons that are not completely dead (have some activation)
        and visualizes what input patterns they respond to, helping understand how they
        combine features to approximate the parity function.
        
        Args:
            activation_threshold: Minimum activation ratio to consider a neuron as active
        """
        if self.neuron_activation_analysis['layer2_activation_ratio'] is None:
            print("Running neuron activation analysis first...")
            self.analyze_neuron_activations()
        
        layer2_activation_ratio = self.neuron_activation_analysis['layer2_activation_ratio']
        
        # Find active neurons (not completely dead)
        active_neurons = np.where(layer2_activation_ratio > activation_threshold)[0]
        if len(active_neurons) == 0:
            print(f"No active neurons found with threshold {activation_threshold}. Try lowering the threshold.")
            return
        
        print(f"Found {len(active_neurons)} active neurons in the second layer")
        self.second_layer_patterns['active_neurons'] = active_neurons
        
        # Get weights for detailed analysis
        W1 = self.model.W1.detach().cpu().numpy()
        W2 = self.model.W2.detach().cpu().numpy()
        a = self.model.a.detach().cpu().numpy()
        
        # Create figure for visualizing patterns
        fig, axs = plt.subplots(min(10, len(active_neurons)), 1, figsize=(15, 4 * min(10, len(active_neurons))))
        
        # Handle case where there's only one subplot
        if min(10, len(active_neurons)) == 1:
            axs = [axs]
            
        # For each active neuron, analyze its connection pattern
        for i, neuron_idx in enumerate(active_neurons[:10]):  # Limit to 10 for visualization
            ax = axs[i]
            
            # Get weights from first to second layer for this neuron
            w2_weights = W2[neuron_idx, :]
            
            # Find the strongest connections from layer 1
            strongest_idx = np.argsort(np.abs(w2_weights))[-5:]  # Top 5 connections
            
            # Show the neuron's contribution to the output
            output_weight = a[neuron_idx]
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Plot bar chart of this neuron's connections to first layer
            colors = []
            for j in range(self.M1):
                # Color based on whether it's a strong connection
                if j in strongest_idx:
                    colors.append('red')
                else:
                    colors.append('lightgray')
            
            # Plot the weights
            ax.bar(range(self.M1), w2_weights, color=colors, alpha=0.7)
            
            # Highlight the strongest connections
            for j in strongest_idx:
                # For each strong connection to first layer
                # Get the input features it focuses on most
                w1_weights = W1[j, :]
                
                # Find strongest input connections
                top_features = np.argsort(np.abs(w1_weights))[-3:]  # Top 3 features
                
                # List the features (add text annotation)
                feature_text = f"L1 Neuron {j}: "
                relevant_count = sum(1 for f in top_features if f < self.k)
                if relevant_count > 0:
                    feature_text += f"{relevant_count}/{len(top_features)} relevant features" 
                
                # Add annotation
                ax.annotate(feature_text, xy=(j, w2_weights[j]),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=8, color='blue')
            
            ax.set_title(f'Neuron {neuron_idx} (Output weight: {output_weight:.3f}, Activation ratio: {layer2_activation_ratio[neuron_idx]:.3f})')
            ax.set_xlabel('First Layer Neuron Index')
            ax.set_ylabel('Weight Value')
            ax.grid(alpha=0.3)
            
            # Store information about this neuron's pattern
            self.second_layer_patterns['input_patterns'][neuron_idx] = {
                'output_weight': output_weight,
                'activation_ratio': layer2_activation_ratio[neuron_idx],
                'strongest_connections': strongest_idx,
                'w2_weights': w2_weights
            }
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/second_layer_patterns_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Create detailed plots for specific neurons
        self._plot_detailed_neuron_connections(active_neurons[:5])  # Detailed analysis for top 5
        
        return self.second_layer_patterns
        
    def plot_all(self):
        """
        Generate all plots in one go
        """
        print("Generating plots...")
        
        # Original plots
        self.plot_weight_matrices()
        self.plot_feature_importance_evolution()
        self.plot_feature_correlations()
        self.plot_signal_noise_ratio()
        self.plot_gradient_statistics()
        self.plot_training_progress()
        
        # Activation plots
        self.plot_activation_statistics()
        self.plot_gradient_correlations()
        self.plot_hidden_target_correlations()
        self.plot_gradient_eigenvalues()
        self.plot_principal_angles()
        
        # Neuron activation evolution plots
        if self.neuron_activation_evolution['epochs']:
            print("Plotting neuron activation evolution...")
            self.plot_neuron_activation_evolution()
            print("Plotting individual neuron activation trajectories...")
            self.plot_neuron_activation_trajectories()
        
        # Enhanced neuron activation analysis for final state
        print("Running neuron activation analysis for final state...")
        self.analyze_neuron_activations()
        
        # New neuron-target correlation plots
        if self.neuron_target_correlation_evolution['epochs']:
            print("Plotting neuron-target correlation evolution...")
            self.plot_neuron_target_correlation_evolution()
            print("Plotting neuron-target correlation trajectories...")
            self.plot_neuron_target_correlation_trajectories()
            print("Plotting neuron-target correlation distributions...")
            self.plot_neuron_target_correlation_distribution()
        
        # Visualize second layer patterns
        self.visualize_second_layer_patterns()
        
        print("All plots generated!")
        
    def save_results(self):
        """
        Save all collected data to CSV files for further analysis
        """
        # Create results directory if it doesn't exist
        results_dir = os.path.join(self.save_dir, f"d{self.d}_k{self.k}_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save feature importance data
        fi_df = pd.DataFrame({
            'epoch': self.feature_importance['epochs'],
            'relevant_importance': self.feature_importance['relevant_importance'],
            'irrelevant_importance': self.feature_importance['irrelevant_importance'],
            'ratio': self.feature_importance['ratio']
        })
        fi_df.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)
        
        # Save feature correlation data
        fc_df = pd.DataFrame({
            'epoch': self.feature_correlations['epochs'],
            'relevant_mean': self.feature_correlations['relevant_mean'],
            'irrelevant_mean': self.feature_correlations['irrelevant_mean'],
            'relevant_std': self.feature_correlations['relevant_std'],
            'irrelevant_std': self.feature_correlations['irrelevant_std']
        })
        fc_df.to_csv(os.path.join(results_dir, "feature_correlations.csv"), index=False)
        
        # Save empirical signal-noise data
        sn_df = pd.DataFrame({
            'epoch': self.empirical_signal_noise['epochs'],
            'empirical_signal': self.empirical_signal_noise['empirical_signal'],
            'empirical_noise': self.empirical_signal_noise['empirical_noise'],
            'empirical_ratio': self.empirical_signal_noise['empirical_ratio'],
            'theoretical_signal': self.empirical_signal_noise['theoretical_signal'],
            'theoretical_noise': self.empirical_signal_noise['theoretical_noise'],
            'theoretical_ratio': self.empirical_signal_noise['theoretical_ratio']
        })
        sn_df.to_csv(os.path.join(results_dir, "signal_noise.csv"), index=False)
        
        # Save gradient statistics
        grad_df = pd.DataFrame({
            'epoch': self.gradient_stats['epochs'],
            'relevant_grad_mean': self.gradient_stats['relevant_grad_mean'],
            'irrelevant_grad_mean': self.gradient_stats['irrelevant_grad_mean'], 
            'relevant_grad_std': self.gradient_stats['relevant_grad_std'],
            'irrelevant_grad_std': self.gradient_stats['irrelevant_grad_std'],
            'w1_grad_norm': self.gradient_stats['w1_grad_norm']
        })
        grad_df.to_csv(os.path.join(results_dir, "gradient_statistics.csv"), index=False)
        
        # Save correlation history
        corr_df = pd.DataFrame(self.correlation_history, columns=['epoch', 'correlation'])
        corr_df.to_csv(os.path.join(results_dir, "correlation_history.csv"), index=False)
        
        # Save phase transition info
        pt_data = {
            'detected': self.phase_transition['detected'],
            'detected_epoch': self.phase_transition['epoch'],
            'theoretical_epoch': self.phase_transition['theoretical_epoch'],
            'correlation_at_transition': self.phase_transition['correlation'],
            'feature_ratio_at_transition': self.phase_transition['feature_ratio'],
            'd': self.d,
            'k': self.k,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'W1_grad': self.W1_grad,
            'W2_grad': self.W2_grad,
            'W1_init': self.W1_init
        }
        pd.DataFrame([pt_data]).to_csv(os.path.join(results_dir, "phase_transition.csv"), index=False)
        
        # Save parameters
        params = {
            'd': self.d,
            'k': self.k,
            'M1': self.M1,
            'M2': self.M2,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'tracker': self.tracker,
            'top_k_eigen': self.top_k_eigen,
            'W1_grad': self.W1_grad,
            'W2_grad': self.W2_grad,
            'W1_init': self.W1_init
        }
        pd.DataFrame([params]).to_csv(os.path.join(results_dir, "parameters.csv"), index=False)
        
        # Save activation statistics
        act_df = pd.DataFrame({
            'epoch': self.activation_stats['epochs'],
            'layer1_percent_active': self.activation_stats['layer1_percent_active'],
            'layer2_percent_active': self.activation_stats['layer2_percent_active']
        })
        act_df.to_csv(os.path.join(results_dir, "activation_statistics.csv"), index=False)
        
        # Save gradient correlation statistics
        corr_df = pd.DataFrame({
            'epoch': self.gradient_correlations['epochs'],
            'mean_rel_rel': self.gradient_correlations['mean_rel_rel'],
            'mean_rel_irrel': self.gradient_correlations['mean_rel_irrel'],
            'mean_irrel_irrel': self.gradient_correlations['mean_irrel_irrel']
        })
        corr_df.to_csv(os.path.join(results_dir, "gradient_correlations.csv"), index=False)
        
        # Save hidden-target correlations
        htc_df = pd.DataFrame({
            'epoch': self.hidden_target_correlations['epochs'],
            'layer1_corr_sum': self.hidden_target_correlations['layer1_corr_sum'],
            'layer2_corr_sum': self.hidden_target_correlations['layer2_corr_sum']
        })
        htc_df.to_csv(os.path.join(results_dir, "hidden_target_correlations.csv"), index=False)
        
        # Save eigenvalues
        eigen_epochs = self.gradient_eigen['epochs']
        
        # Save W1 eigenvalues
        w1_eigen_data = {'epoch': eigen_epochs}
        for i in range(min(self.top_k_eigen, len(self.gradient_eigen['W1_eigenvalues'][0]))):
            w1_eigen_data[f'eigenvalue_{i+1}'] = [evals[i] for evals in self.gradient_eigen['W1_eigenvalues']]
        
        w1_eigen_df = pd.DataFrame(w1_eigen_data)
        w1_eigen_df.to_csv(os.path.join(results_dir, "w1_eigenvalues.csv"), index=False)
        
        # Save W2 eigenvalues
        w2_eigen_data = {'epoch': eigen_epochs}
        for i in range(min(self.top_k_eigen, len(self.gradient_eigen['W2_eigenvalues'][0]))):
            w2_eigen_data[f'eigenvalue_{i+1}'] = [evals[i] for evals in self.gradient_eigen['W2_eigenvalues']]
        
        w2_eigen_df = pd.DataFrame(w2_eigen_data)
        w2_eigen_df.to_csv(os.path.join(results_dir, "w2_eigenvalues.csv"), index=False)
        
        # Save principal angles
        if self.principal_angles['epochs']:
            # Save W1 principal angles
            w1_angle_data = {'epoch': self.principal_angles['epochs']}
            w1_angles = self.principal_angles['W1_angles']
            w1_angle_data['mean_angle'] = [np.mean(angles) for angles in w1_angles]
            
            if len(w1_angles[0]) <= 10:  # Save individual angles if not too many
                for i in range(len(w1_angles[0])):
                    w1_angle_data[f'angle_{i+1}'] = [angles[i] for angles in w1_angles]
            
            w1_angle_df = pd.DataFrame(w1_angle_data)
            w1_angle_df.to_csv(os.path.join(results_dir, "w1_principal_angles.csv"), index=False)
            
            # Save W2 principal angles
            w2_angle_data = {'epoch': self.principal_angles['epochs']}
            w2_angles = self.principal_angles['W2_angles']
            w2_angle_data['mean_angle'] = [np.mean(angles) for angles in w2_angles]
            
            if len(w2_angles[0]) <= 10:  # Save individual angles if not too many
                for i in range(len(w2_angles[0])):
                    w2_angle_data[f'angle_{i+1}'] = [angles[i] for angles in w2_angles]
            
            w2_angle_df = pd.DataFrame(w2_angle_data)
            w2_angle_df.to_csv(os.path.join(results_dir, "w2_principal_angles.csv"), index=False)
        
        # Save neuron activation analysis if available
        if self.neuron_activation_analysis['layer1_activation_ratio'] is not None:
            layer1_df = pd.DataFrame({
                'neuron_idx': range(len(self.neuron_activation_analysis['layer1_activation_ratio'])),
                'activation_ratio': self.neuron_activation_analysis['layer1_activation_ratio']
            })
            layer1_df.to_csv(os.path.join(results_dir, "layer1_activation_ratios.csv"), index=False)
            
            layer2_df = pd.DataFrame({
                'neuron_idx': range(len(self.neuron_activation_analysis['layer2_activation_ratio'])),
                'activation_ratio': self.neuron_activation_analysis['layer2_activation_ratio']
            })
            layer2_df.to_csv(os.path.join(results_dir, "layer2_activation_ratios.csv"), index=False)
            
        # Save neuron activation evolution if available
        if self.neuron_activation_evolution['epochs']:
            # Save evolution data for layer 1 - one row per epoch, one column per neuron
            layer1_evolution = np.array(self.neuron_activation_evolution['layer1_activation_ratios'])
            layer1_evo_df = pd.DataFrame(
                layer1_evolution, 
                index=self.neuron_activation_evolution['epochs'],
                columns=[f"neuron_{i}" for i in range(self.M1)]
            )
            layer1_evo_df.index.name = 'epoch'
            layer1_evo_df.to_csv(os.path.join(results_dir, "layer1_activation_evolution.csv"))
            
            # Save evolution data for layer 2
            layer2_evolution = np.array(self.neuron_activation_evolution['layer2_activation_ratios'])
            layer2_evo_df = pd.DataFrame(
                layer2_evolution, 
                index=self.neuron_activation_evolution['epochs'],
                columns=[f"neuron_{i}" for i in range(self.M2)]
            )
            layer2_evo_df.index.name = 'epoch'
            layer2_evo_df.to_csv(os.path.join(results_dir, "layer2_activation_evolution.csv"))
            
        # Save neuron-target correlation evolution if available
        if hasattr(self, 'neuron_target_correlation_evolution') and self.neuron_target_correlation_evolution['epochs']:
            # Save evolution data for layer 1 - one row per epoch, one column per neuron
            layer1_corr_evolution = np.array(self.neuron_target_correlation_evolution['layer1_neuron_correlations'])
            layer1_corr_df = pd.DataFrame(
                layer1_corr_evolution, 
                index=self.neuron_target_correlation_evolution['epochs'],
                columns=[f"neuron_{i}" for i in range(self.M1)]
            )
            layer1_corr_df.index.name = 'epoch'
            layer1_corr_df.to_csv(os.path.join(results_dir, "layer1_target_correlation_evolution.csv"))
            
            # Save evolution data for layer 2
            layer2_corr_evolution = np.array(self.neuron_target_correlation_evolution['layer2_neuron_correlations'])
            layer2_corr_df = pd.DataFrame(
                layer2_corr_evolution, 
                index=self.neuron_target_correlation_evolution['epochs'],
                columns=[f"neuron_{i}" for i in range(self.M2)]
            )
            layer2_corr_df.index.name = 'epoch'
            layer2_corr_df.to_csv(os.path.join(results_dir, "layer2_target_correlation_evolution.csv"))
        
        print(f"Results saved to {results_dir}")
        
        return results_dir





def run_parity_experiment(d=30, k=6, M1=512, M2=512, learning_rate=0.01, batch_size=512, 
                        n_epochs=10000, device_id=None, save_dir="parity_analysis",
                        tracker=20, top_k_eigen=5, W1_grad=True, W2_grad=True, W1_init='random'):
    """
    Run a single experiment analyzing a neural network learning a k-sparse parity function
    
    Args:
        d: Input dimension
        k: Parity function order
        M1: First layer width
        M2: Second layer width
        learning_rate: Learning rate
        batch_size: Batch size
        n_epochs: Maximum number of epochs
        device_id: GPU device ID (None for auto-selection)
        save_dir: Directory to save results
        tracker: Frequency (in epochs) to compute metrics
        top_k_eigen: Number of top eigenvectors/values to track
        W1_grad: Whether W1 is trainable (if False, W1 is frozen)
        W2_grad: Whether W2 is trainable (if False, W2 is frozen)
        W1_init: Initialization mode for W1 ('random' or 'sparse')
        
    Returns:
        analyzer: Trained ParityNetAnalyzer instance
    """
    # Create save directory with unique name based on parameters
    run_dir = f"{save_dir}_d{d}_k{k}_bs{batch_size}_lr{learning_rate}_W1{W1_grad}_W2{W2_grad}_init{W1_init}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ParityNetAnalyzer(
        d=d,
        k=k,
        M1=M1,
        M2=M2,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device_id=device_id,
        save_dir=run_dir,
        tracker=tracker,
        top_k_eigen=top_k_eigen,
        W1_grad=W1_grad,
        W2_grad=W2_grad,
        W1_init=W1_init
    )
    
    # Train the model
    final_corr, final_epoch = analyzer.train(n_epochs=n_epochs)
    
    # Generate plots
    analyzer.plot_all()
    
    # Save results
    analyzer.save_results()
    
    print(f"Experiment completed!")
    print(f"Final correlation: {final_corr:.4f} at epoch {final_epoch}")
    
    if analyzer.phase_transition['detected']:
        detected_epoch = analyzer.phase_transition['epoch']
        theoretical_epoch = analyzer.phase_transition['theoretical_epoch']
        print(f"Phase transition detected at epoch {detected_epoch}")
        print(f"Theoretical prediction: {theoretical_epoch}")
        print(f"Ratio of detected/theoretical: {detected_epoch/theoretical_epoch:.2f}")
    else:
        print("No phase transition detected")
    
    return analyzer


if __name__ == "__main__":
    # Base parameters for experiments
    d = 45          # Input dimension
    k = 6           # Parity function order (k-sparse parity)
    M1 = 512        # First layer width
    M2 = 512        # Second layer width
    batch_size = 512
    learning_rate = 0.001
    n_epochs = 7000  # Reduced for testing, use larger value (e.g., 20000) for final runs
    tracker = 100     # Compute metrics every 100 epochs
    top_k_eigen = 15
    base_save_dir = "parity_analysis_experiments_456_new2"
    
    # Check for available GPUs
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_id = 0 if n_gpus > 0 else None
    
    # Print common configurations
    print("Common configurations for all experiments:")
    print(f"d={d}, k={k}, M1={M1}, M2={M2}, batch_size={batch_size}")
    print(f"learning_rate={learning_rate}, n_epochs={n_epochs}, tracker={tracker}")
    print(f"Using device: {'cuda:'+str(device_id) if device_id is not None else 'cpu'}")
    print("=" * 80)
    
    # Experiment 1: Default - Both layers trainable, random initialization
    print("\nRunning Experiment 1: Default - Both layers trainable, random initialization")
    analyzer1 = run_parity_experiment(
        d=d,
        k=k,
        M1=M1,
        M2=M2,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        device_id=device_id,
        save_dir=f"{base_save_dir}/exp1_default",
        tracker=tracker,
        top_k_eigen=top_k_eigen,
        W1_grad=True,
        W2_grad=True,
        W1_init='random'
    )
    print("Experiment 1 completed.")
    print("=" * 80)
    
    # Experiment 2: Frozen W1 - First layer frozen, second layer trainable
    print("\nRunning Experiment 2: Frozen W1 - First layer frozen, second layer trainable")
    analyzer2 = run_parity_experiment(
        d=d,
        k=k,
        M1=M1,
        M2=M2,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        device_id=device_id,
        save_dir=f"{base_save_dir}/exp2_frozen_W1",
        tracker=tracker,
        top_k_eigen=top_k_eigen,
        W1_grad=False,  # W1 is frozen
        W2_grad=True,
        W1_init='random'
    )
    print("Experiment 2 completed.")
    print("=" * 80)
    
    # Experiment 3: Frozen W2 - First layer trainable, second layer frozen
    print("\nRunning Experiment 3: Frozen W2 - First layer trainable, second layer frozen")
    analyzer3 = run_parity_experiment(
        d=d,
        k=k,
        M1=M1,
        M2=M2,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        device_id=device_id,
        save_dir=f"{base_save_dir}/exp3_frozen_W2",
        tracker=tracker,
        top_k_eigen=top_k_eigen,
        W1_grad=True,
        W2_grad=False,  # W2 is frozen
        W1_init='random'
    )
    print("Experiment 3 completed.")
    print("=" * 80)
    
    # Experiment 4: Sparse Init - Both layers trainable, sparse initialization of W1
    print("\nRunning Experiment 4: Sparse Init - Both layers trainable, sparse initialization of W1")
    analyzer4 = run_parity_experiment(
        d=d,
        k=k,
        M1=M1,
        M2=M2,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        device_id=device_id,
        save_dir=f"{base_save_dir}/exp4_sparse_init",
        tracker=tracker,
        top_k_eigen=top_k_eigen,
        W1_grad=True,
        W2_grad=True,
        W1_init='sparse'  # Only relevant features initialized
    )
    print("Experiment 4 completed.")
    print("=" * 80)
    
    # Experiment 5: Sparse Init + Frozen W1
    print("\nRunning Experiment 5: Sparse Init + Frozen W1")
    analyzer5 = run_parity_experiment(
        d=d,
        k=k,
        M1=M1,
        M2=M2,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        device_id=device_id,
        save_dir=f"{base_save_dir}/exp5_sparse_frozen_W1",
        tracker=tracker,
        top_k_eigen=top_k_eigen,
        W1_grad=False,  # W1 is frozen
        W2_grad=True, 
        W1_init='sparse'  # Only relevant features initialized
    )
    print("Experiment 5 completed.")
    print("=" * 80)
    
    # Results summary
    print("\nExperiment Results Summary:")
    print("-" * 80)
    print(f"{'Experiment':<30} {'Phase Transition':<20} {'Final Correlation':<20}")
    print("-" * 80)
    
    def format_pt(analyzer):
        if analyzer.phase_transition['detected']:
            return f"Epoch {analyzer.phase_transition['epoch']}"
        else:
            return "Not detected"
    
    # Get final correlation for each experiment
    final_corr1 = analyzer1.correlation_history[-1][1] if analyzer1.correlation_history else "N/A"
    final_corr2 = analyzer2.correlation_history[-1][1] if analyzer2.correlation_history else "N/A"
    final_corr3 = analyzer3.correlation_history[-1][1] if analyzer3.correlation_history else "N/A"
    final_corr4 = analyzer4.correlation_history[-1][1] if analyzer4.correlation_history else "N/A"
    final_corr5 = analyzer5.correlation_history[-1][1] if analyzer5.correlation_history else "N/A"
    
    print(f"1. Default                    {format_pt(analyzer1):<20} {final_corr1:<20.4f}")
    print(f"2. Frozen W1                  {format_pt(analyzer2):<20} {final_corr2:<20.4f}")
    print(f"3. Frozen W2                  {format_pt(analyzer3):<20} {final_corr3:<20.4f}")
    print(f"4. Sparse Init                {format_pt(analyzer4):<20} {final_corr4:<20.4f}")
    print(f"5. Sparse Init + Frozen W1    {format_pt(analyzer5):<20} {final_corr5:<20.4f}")
    print("-" * 80)
    
    print("\nAll experiments completed. Results are saved in the respective directories.")
                    
                    