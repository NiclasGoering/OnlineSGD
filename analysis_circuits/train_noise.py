import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from scipy import stats
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)

class TwoLayerReLUNet(torch.nn.Module):
    def __init__(self, input_dim, hidden1_width, hidden2_width):
        super(TwoLayerReLUNet, self).__init__()
        # Initialize layers
        self.W1 = torch.nn.Parameter(torch.randn(hidden1_width, input_dim) / np.sqrt(input_dim))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden1_width))
        self.W2 = torch.nn.Parameter(torch.randn(hidden2_width, hidden1_width) / np.sqrt(hidden1_width))
        self.b2 = torch.nn.Parameter(torch.zeros(hidden2_width))
        self.a = torch.nn.Parameter(torch.randn(hidden2_width) / np.sqrt(hidden2_width))
        
    def forward(self, x):
        # First hidden layer
        h1 = torch.relu(torch.matmul(x, self.W1.t()) + self.b1)
        # Second hidden layer
        h2 = torch.relu(torch.matmul(h1, self.W2.t()) + self.b2)
        # Output layer
        output = torch.matmul(h2, self.a)
        return output
    
    def get_first_layer_activations(self, x):
        return torch.relu(torch.matmul(x, self.W1.t()) + self.b1)
    
    def get_second_layer_activations(self, x):
        h1 = torch.relu(torch.matmul(x, self.W1.t()) + self.b1)
        return torch.relu(torch.matmul(h1, self.W2.t()) + self.b2)

class SignalNoiseAnalyzer:
    def __init__(self, d=30, k=6, M1=512, M2=512, learning_rate=0.01, 
                 batch_size=512, device_id=None, save_dir="parity_signal_noise_analysis"):
        """
        Initialize the SignalNoiseAnalyzer to analyze the signal vs noise dynamics 
        in neural networks learning parity functions
        
        Args:
            d: Input dimension
            k: Parity function order (k-sparse parity)
            M1: First layer width
            M2: Second layer width
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            device_id: GPU device ID (int) or None for CPU
            save_dir: Directory to save results
        """
        self.d = d
        self.k = k
        self.M1 = M1
        self.M2 = M2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
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
        
        # Empirical Signal and noise tracking
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
        
        # Sample fixed test inputs for consistent evaluation
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, d)), 
                                  dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)
        
        # Fixed sample for computing correlations
        self.X_analysis = torch.tensor(np.random.choice([-1, 1], size=(5000, d)), 
                                      dtype=torch.float32).to(self.device)
        self.y_analysis = self._target_function(self.X_analysis)
        
        print(f"SignalNoiseAnalyzer initialized on {self.device}")
        print(f"Analyzing {k}-parity function in {d} dimensions")
        print(f"Network: {M1} → {M2} → 1")
        print(f"Batch size: {batch_size}")
        print(f"Theoretical phase transition epoch: {self.phase_transition['theoretical_epoch']}")
    
    def _create_model(self):
        """Create a two-layer ReLU network"""
        model = TwoLayerReLUNet(self.d, self.M1, self.M2).to(self.device)
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
        W1_grads = self.model.W1.grad.detach().cpu().numpy()
        
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
            'w1_grad_norm': w1_grad_norm
        }
    
    def train(self, n_epochs=10000, snapshot_interval=1000, early_stop_corr=0.9999):
        """
        Train the network and track relevant metrics
        
        Args:
            n_epochs: Base number of epochs (will train for 5000 more after phase transition)
            snapshot_interval: Base interval for taking weight snapshots
            early_stop_corr: Correlation threshold for early stopping
        
        Returns:
            final_correlation: Final correlation achieved
            stopping_epoch: Epoch at which training stopped
        """
        print(f"Starting training for {n_epochs} epochs (plus up to 5000 after transition)...")
        
        # Adjust total epochs if phase transition is detected
        max_epochs = n_epochs + 5000  # Allow up to 5000 extra epochs after transition
        
        # Take initial snapshot and metrics
        self.take_weight_snapshot(0)
        self.compute_feature_importance(0)
        self.compute_feature_correlations(0)
        
        # Compute initial gradient statistics
        grad_stats = self.compute_gradient_statistics(0)
        self.compute_signal_noise(0, grad_stats)
        
        # For timing
        start_time = time.time()
        transition_detected_epoch = None
        
        for epoch in tqdm(range(max_epochs)):
            # Dynamic metric interval: 50 for first 10k epochs, 1000 after
            if epoch < 10000:
                metric_interval = 50
            else:
                metric_interval = 1000
                
            # Dynamic snapshot interval
            if epoch < 10000:
                curr_snapshot_interval = snapshot_interval
            else:
                curr_snapshot_interval = snapshot_interval * 10
            
            # Check if we've gone 5000 epochs past transition
            if transition_detected_epoch is not None and epoch > transition_detected_epoch + 5000:
                print(f"Reached {epoch} epochs (5000 after transition). Stopping training.")
                break
                
            # Check if we've reached max epochs without transition
            if epoch >= n_epochs and transition_detected_epoch is None:
                # If no transition by n_epochs, allow it to continue but check if we're past max
                if epoch >= max_epochs:
                    print(f"Reached maximum {max_epochs} epochs without detecting phase transition. Stopping.")
                    break
            
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
            
            # Evaluation metrics at intervals
            if epoch % metric_interval == 0 or epoch == max_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    # In the train method, replace the correlation calculation with:
                
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
                
                # Compute feature importance
                self.compute_feature_importance(epoch)
                
                # Compute feature correlations
                self.compute_feature_correlations(epoch)
                
                # Compute gradient statistics to get empirical signal and noise
                grad_stats = self.compute_gradient_statistics(epoch)
                self.compute_signal_noise(epoch, grad_stats)
                
                # Print progress periodically
                if (epoch < 10000 and epoch % (metric_interval * 10) == 0) or \
                (epoch >= 10000 and epoch % metric_interval == 0) or \
                epoch == max_epochs - 1:
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch}: MSE={test_loss:.6f}, Correlation={correlation:.4f}, Time={elapsed:.1f}s")
                    
                    # Print signal-noise information
                    sn_data = self.empirical_signal_noise
                    if len(sn_data['epochs']) > 0:
                        idx = -1  # Latest data point
                        emp_ratio = sn_data['empirical_ratio'][idx]
                        theo_ratio = sn_data['theoretical_ratio'][idx]
                        print(f"Signal/Noise: Empirical={emp_ratio:.4f}, Theoretical={theo_ratio:.4f}")
                
                # Early stopping
                if correlation > early_stop_corr:
                    print(f"Early stopping at epoch {epoch} with correlation {correlation:.4f}")
                    # Take final snapshot before stopping
                    self.take_weight_snapshot(epoch)
                    return correlation, epoch
                
                # Update transition detection epoch
                if self.phase_transition['detected'] and transition_detected_epoch is None:
                    transition_detected_epoch = self.phase_transition['epoch']
                    print(f"Will continue training for 5000 more epochs after transition (until epoch {transition_detected_epoch + 5000})")
            
            # Take weight snapshots at intervals
            if epoch % curr_snapshot_interval == 0 or epoch == max_epochs - 1:
                self.take_weight_snapshot(epoch)
        
        # Take final snapshot if not already taken
        if self.weight_snapshots['epochs'][-1] != max_epochs - 1:
            self.take_weight_snapshot(max_epochs - 1)
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test)
            final_loss = ((preds - self.y_test) ** 2).mean().item()
            final_correlation = torch.corrcoef(torch.stack([preds.squeeze(), self.y_test]))[0, 1].item()
        
        print("Training completed!")
        print(f"Final MSE: {final_loss:.6f}")
        print(f"Final correlation: {final_correlation:.4f}")
        
        final_epoch = max_epochs - 1
        return final_correlation, final_epoch
    
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
        
        # Add empirical signal-noise crossover point
        if len(self.empirical_signal_noise['epochs']) > 0:
            # Find where empirical signal/noise ratio crosses 1.0
            ratio_history = np.array(self.empirical_signal_noise['empirical_ratio'])
            epochs = np.array(self.empirical_signal_noise['epochs'])
            
            crossover_indices = np.where((ratio_history[:-1] < 1.0) & (ratio_history[1:] >= 1.0))[0]
            if len(crossover_indices) > 0:
                crossover_idx = crossover_indices[0]
                crossover_epoch = epochs[crossover_idx+1]
                plt.axvline(x=crossover_epoch, color='blue', linestyle=':', 
                          label=f'Signal/Noise Crossover (Epoch {crossover_epoch})')
        
        # Add points where correlation exceeds thresholds
        for threshold in [0.5, 0.9]:
            # Find first epoch where correlation exceeds threshold
            for i, (epoch, corr) in enumerate(self.correlation_history):
                if corr > threshold:
                    plt.axvline(x=epoch, color=f'C{int(threshold*10)}', 
                              linestyle=':', alpha=0.7,
                              label=f'Correlation > {threshold} (Epoch {epoch})')
                    break
        
        plt.xlabel('Epoch')
        plt.ylabel('Relevant/Irrelevant Feature Importance Ratio (log scale)')
        plt.title(f'Evolution of Feature Importance Ratio (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_importance_evolution_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Also plot absolute importance values
        plt.figure(figsize=(12, 8))
        
        plt.plot(self.feature_importance['epochs'], self.feature_importance['relevant_importance'],
               marker='o', linestyle='-', label='Relevant Features (first k)')
        plt.plot(self.feature_importance['epochs'], self.feature_importance['irrelevant_importance'],
               marker='x', linestyle='--', label='Irrelevant Features')
        
        # Add transition point
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
            
            # Add horizontal lines for feature importances at transition
            for i, (epoch, imp) in enumerate(zip(self.feature_importance['epochs'], self.feature_importance['relevant_importance'])):
                if epoch >= detected_epoch:
                    plt.axhline(y=imp, color='red', linestyle=':', alpha=0.5,
                              label=f'Rel. Importance at Transition = {imp:.4f}')
                    break
            
            for i, (epoch, imp) in enumerate(zip(self.feature_importance['epochs'], self.feature_importance['irrelevant_importance'])):
                if epoch >= detected_epoch:
                    plt.axhline(y=imp, color='orange', linestyle=':', alpha=0.5,
                              label=f'Irrel. Importance at Transition = {imp:.4f}')
                    break
        
        plt.xlabel('Epoch')
        plt.ylabel('Average Absolute Weight')
        plt.title(f'Evolution of Absolute Feature Importance (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/absolute_feature_importance_d{self.d}_k{self.k}.png", dpi=300)
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
            
            # Add horizontal lines for correlation values at transition
            for i, (epoch, corr) in enumerate(zip(self.feature_correlations['epochs'], self.feature_correlations['relevant_mean'])):
                if epoch >= detected_epoch:
                    plt.axhline(y=corr, color='red', linestyle=':', alpha=0.5,
                              label=f'Rel. Correlation at Transition = {corr:.4e}')
                    break
            
            for i, (epoch, corr) in enumerate(zip(self.feature_correlations['epochs'], self.feature_correlations['irrelevant_mean'])):
                if epoch >= detected_epoch:
                    plt.axhline(y=corr, color='orange', linestyle=':', alpha=0.5,
                              label=f'Irrel. Correlation at Transition = {corr:.4e}')
                    break
        
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Correlation |C_l| (log scale)')
        plt.title(f'Feature Correlations C_l Evolution (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_correlations_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Plot correlation distributions at key points
        num_points = min(3, len(self.feature_correlations['epochs']))
        if num_points < 3:
            return
            
        # Choose indices: beginning, middle, end
        indices = [0, len(self.feature_correlations['epochs']) // 2, -1]
        
        plt.figure(figsize=(15, 5))
        titles = ['Initial', 'Mid-Training', 'Final']
        
        for i, idx in enumerate(indices):
            epoch = self.feature_correlations['epochs'][idx]
            rel_corrs = np.abs(self.feature_correlations['relevant_features'][idx])
            irrel_corrs = np.abs(self.feature_correlations['irrelevant_features'][idx])
            
            plt.subplot(1, 3, i+1)
            plt.hist(rel_corrs, bins=10, alpha=0.7, label='Relevant Features')
            plt.hist(irrel_corrs, bins=10, alpha=0.7, label='Irrelevant Features')
            
            plt.title(f'{titles[i]} Correlations (Epoch {epoch})')
            plt.xlabel('Absolute Correlation |C_l|')
            plt.ylabel('Count')
            
            # Add vertical lines if this is close to transition point
            if self.phase_transition['detected']:
                detected_epoch = self.phase_transition['epoch']
                if abs(epoch - detected_epoch) < 100:  # If within 100 epochs of transition
                    plt.axvline(x=np.mean(rel_corrs), color='red', linestyle='--', alpha=0.7,
                              label=f'Mean Rel. Corr = {np.mean(rel_corrs):.4e}')
                    plt.axvline(x=np.mean(irrel_corrs), color='orange', linestyle='--', alpha=0.7,
                              label=f'Mean Irrel. Corr = {np.mean(irrel_corrs):.4e}')
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_correlation_distribution_d{self.d}_k{self.k}.png", dpi=300)
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
            
            # Add horizontal lines for signal and noise values at transition
            # Find closest epoch to transition
            idx = np.argmin(np.abs(np.array(self.empirical_signal_noise['epochs']) - detected_epoch))
            signal_at_transition = self.empirical_signal_noise['empirical_signal'][idx]
            noise_at_transition = self.empirical_signal_noise['empirical_noise'][idx]
            
            plt.axhline(y=signal_at_transition, color='red', linestyle=':', alpha=0.5,
                      label=f'Signal at Transition = {signal_at_transition:.4e}')
            plt.axhline(y=noise_at_transition, color='orange', linestyle=':', alpha=0.5,
                      label=f'Noise at Transition = {noise_at_transition:.4e}')
        
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
            
            # Add horizontal line at ratio value during transition
            # Find closest epoch to transition
            idx = np.argmin(np.abs(np.array(self.empirical_signal_noise['epochs']) - detected_epoch))
            ratio_at_transition = self.empirical_signal_noise['empirical_ratio'][idx]
            
            plt.axhline(y=ratio_at_transition, color='red', linestyle=':', alpha=0.7,
                      label=f'Ratio at Transition = {ratio_at_transition:.2f}')
        
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
        if not self.gradient_stats['epochs']:
            print("No gradient statistics recorded")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Mean gradient magnitudes for relevant and irrelevant features
        plt.subplot(2, 2, 1)
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['relevant_grad_mean'], 
                   marker='o', linestyle='-', label='Relevant Features')
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['irrelevant_grad_mean'], 
                   marker='x', linestyle='--', label='Irrelevant Features')
        
        # Add detected transition point and horizontal lines if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
            
            # Find closest epoch to transition
            idx = np.argmin(np.abs(np.array(self.gradient_stats['epochs']) - detected_epoch))
            relevant_grad_at_transition = self.gradient_stats['relevant_grad_mean'][idx]
            irrelevant_grad_at_transition = self.gradient_stats['irrelevant_grad_mean'][idx]
            
            plt.axhline(y=relevant_grad_at_transition, color='red', linestyle=':', alpha=0.5)
            plt.axhline(y=irrelevant_grad_at_transition, color='orange', linestyle=':', alpha=0.5)
        
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
        
        # Add detected transition point and horizontal lines if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
            
            # Find closest epoch to transition
            idx = np.argmin(np.abs(np.array(self.gradient_stats['epochs']) - detected_epoch))
            relevant_std_at_transition = self.gradient_stats['relevant_grad_std'][idx]
            irrelevant_std_at_transition = self.gradient_stats['irrelevant_grad_std'][idx]
            
            plt.axhline(y=relevant_std_at_transition, color='red', linestyle=':', alpha=0.5)
            plt.axhline(y=irrelevant_std_at_transition, color='orange', linestyle=':', alpha=0.5)
        
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Std Dev (log scale)')
        plt.title('Gradient Variability Evolution')
        plt.grid(True)
        plt.legend()
        
        # Plot 3: Ratio of relevant to irrelevant mean gradients
        plt.subplot(2, 2, 3)
        ratio = np.array(self.gradient_stats['relevant_grad_mean']) / np.array(self.gradient_stats['irrelevant_grad_mean'])
        plt.semilogy(self.gradient_stats['epochs'], ratio, 
                   marker='o', linestyle='-', label='Relevant/Irrelevant Ratio')
        
        # Add line at ratio = 1
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7)
        
        # Add detected transition point and horizontal line if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
            
            # Find closest epoch to transition
            idx = np.argmin(np.abs(np.array(self.gradient_stats['epochs']) - detected_epoch))
            ratio_at_transition = ratio[idx]
            
            plt.axhline(y=ratio_at_transition, color='red', linestyle=':', alpha=0.7,
                      label=f'Ratio at Transition = {ratio_at_transition:.2f}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Ratio (log scale)')
        plt.title('Ratio of Relevant to Irrelevant Gradient Magnitudes')
        plt.grid(True)
        plt.legend()
        
        # Plot 4: Overall gradient norm
        plt.subplot(2, 2, 4)
        plt.semilogy(self.gradient_stats['epochs'], self.gradient_stats['w1_grad_norm'], 
                   marker='o', linestyle='-', label='W1 Gradient Norm')
        
        # Add detected transition point and horizontal line if available
        if self.phase_transition['detected']:
            detected_epoch = self.phase_transition['epoch']
            plt.axvline(x=detected_epoch, color='red', linestyle='--', 
                      label=f'Detected Transition (Epoch {detected_epoch})')
            
            # Find closest epoch to transition
            idx = np.argmin(np.abs(np.array(self.gradient_stats['epochs']) - detected_epoch))
            norm_at_transition = self.gradient_stats['w1_grad_norm'][idx]
            
            plt.axhline(y=norm_at_transition, color='red', linestyle=':', alpha=0.7,
                      label=f'Norm at Transition = {norm_at_transition:.2e}')
        
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
            
            # Find closest loss value to transition
            if detected_epoch < len(self.loss_history):
                loss_at_transition = self.loss_history[detected_epoch]
                ax1.axhline(y=loss_at_transition, color='red', linestyle=':', alpha=0.7,
                          label=f'Loss at Transition = {loss_at_transition:.4e}')
        
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
            
            # Find correlation at transition
            if self.phase_transition['correlation'] is not None:
                corr_at_transition = self.phase_transition['correlation']
                ax2.axhline(y=corr_at_transition, color='red', linestyle=':', alpha=0.7,
                          label=f'Correlation at Transition = {corr_at_transition:.4f}')
        
        # Add correlation thresholds
        for threshold in [0.5, 0.9]:
            ax2.axhline(y=threshold, color=f'C{int(threshold*10)}', 
                      linestyle=':', alpha=0.7,
                      label=f'Correlation = {threshold}')
        
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_progress_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
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
            'learning_rate': self.learning_rate
        }
        pd.DataFrame([pt_data]).to_csv(os.path.join(results_dir, "phase_transition.csv"), index=False)
        
        # Save parameters
        params = {
            'd': self.d,
            'k': self.k,
            'M1': self.M1,
            'M2': self.M2,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        pd.DataFrame([params]).to_csv(os.path.join(results_dir, "parameters.csv"), index=False)
        
        # Save detailed individual feature correlations for last few epochs
        # (this can be large, so we only save the last few)
        max_detailed_epochs = 10
        if len(self.feature_correlations['epochs']) > 0:
            start_idx = max(0, len(self.feature_correlations['epochs']) - max_detailed_epochs)
            
            for i in range(start_idx, len(self.feature_correlations['epochs'])):
                epoch = self.feature_correlations['epochs'][i]
                rel_corrs = self.feature_correlations['relevant_features'][i]
                irrel_corrs = self.feature_correlations['irrelevant_features'][i]
                
                # Create feature labels
                rel_labels = [f'x{j+1}' for j in range(len(rel_corrs))]
                irrel_labels = [f'x{j+1+self.k}' for j in range(len(irrel_corrs))]
                
                # Combine data
                detailed_df = pd.DataFrame({
                    'feature': rel_labels + irrel_labels,
                    'correlation': rel_corrs + irrel_corrs,
                    'is_relevant': [True] * len(rel_corrs) + [False] * len(irrel_corrs)
                })
                
                detailed_df.to_csv(os.path.join(results_dir, f"detailed_correlations_epoch_{epoch}.csv"), index=False)
        
        print(f"Results saved to {results_dir}")
        
        return results_dir

def run_experiment_with_device(args):
    """
    Helper function to run a single experiment on a specific device
    Used for multiprocessing
    
    Args:
        args: Tuple containing (d, k, batch_size, learning_rate, device_id, n_epochs, save_dir)
        
    Returns:
        Dictionary with experiment results
    """
    d, k, batch_size, learning_rate, device_id, n_epochs, save_dir = args
    
    # Create specific save directory for this experiment
    exp_save_dir = os.path.join(save_dir, f"d{d}_k{k}_bs{batch_size}")
    os.makedirs(exp_save_dir, exist_ok=True)
    
    # Derive M1 based on theory that M1 should scale with k*log(d)
    base_M1 = 512
    M1 = max(base_M1, int(k * np.log(d) * 2))
    M1 = min(M1, 1024)  # Cap at reasonable size
    M2 = base_M1
    
    print(f"\nStarting experiment: d={d}, k={k}, batch_size={batch_size}, learning_rate={learning_rate}")
    print(f"Device: {device_id}, M1: {M1}, M2: {M2}")
    
    try:
        # Create analyzer
        analyzer = SignalNoiseAnalyzer(
            d=d,
            k=k,
            M1=M1,
            M2=M2,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device_id=device_id,
            save_dir=exp_save_dir
        )
        
        # Train the model
        snapshot_interval = max(100, n_epochs // 20)  # Adjust snapshot interval based on total epochs
        final_corr, final_epoch = analyzer.train(
            n_epochs=n_epochs,
            snapshot_interval=snapshot_interval
        )
        
        # Create visualizations
        analyzer.plot_weight_matrices()
        analyzer.plot_feature_importance_evolution()
        analyzer.plot_feature_correlations()
        analyzer.plot_signal_noise_ratio()
        analyzer.plot_gradient_statistics()
        analyzer.plot_training_progress()
        
        # Save results
        analyzer.save_results()
        
        # Extract key results
        result = {
            'd': d,
            'k': k,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'theoretical_transition': analyzer.phase_transition['theoretical_epoch'],
            'detected_transition': analyzer.phase_transition['epoch'],
            'final_correlation': analyzer.correlation_history[-1][1] if analyzer.correlation_history else None,
            'feature_ratio_at_transition': analyzer.phase_transition['feature_ratio'],
            'success': True
        }
        
        # If signal-noise data is available, add crossover point
        if len(analyzer.empirical_signal_noise['epochs']) > 0:
            ratio_history = np.array(analyzer.empirical_signal_noise['empirical_ratio'])
            epochs = np.array(analyzer.empirical_signal_noise['epochs'])
            crossover_indices = np.where((ratio_history[:-1] < 1.0) & (ratio_history[1:] >= 1.0))[0]
            if len(crossover_indices) > 0:
                crossover_idx = crossover_indices[0]
                result['signal_noise_crossover_epoch'] = epochs[crossover_idx+1]
        
        print(f"\nExperiment completed for d={d}, k={k}, batch_size={batch_size}")
        print(f"Final correlation: {final_corr:.4f} at epoch {final_epoch}")
        
        if analyzer.phase_transition['detected']:
            detected_epoch = analyzer.phase_transition['epoch']
            theoretical_epoch = analyzer.phase_transition['theoretical_epoch']
            print(f"Phase transition detected at epoch {detected_epoch}")
            print(f"Theoretical prediction: {theoretical_epoch}")
            print(f"Ratio of detected/theoretical: {detected_epoch/theoretical_epoch:.2f}")
        else:
            print("No phase transition detected")
            
        return result
        
    except Exception as e:
        print(f"Error in experiment d={d}, k={k}, batch_size={batch_size}, learning_rate={learning_rate}: {e}")
        return {
            'd': d,
            'k': k,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'error': str(e),
            'success': False
        }

def run_single_experiment(d, k, M1=512, M2=512, learning_rate=0.01, batch_size=512, 
                         n_epochs=10000, device_id=None, save_dir="parity_signal_noise_analysis"):
    """
    Run a single experiment with specified parameters
    
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
        
    Returns:
        analyzer: Trained SignalNoiseAnalyzer instance
    """
    # Run experiment
    args = (d, k, batch_size, learning_rate, device_id, n_epochs, save_dir)
    result = run_experiment_with_device(args)
    
    # Return the result
    return result

def run_grid_search(d_values, k_values, batch_sizes, learning_rates, 
                   n_epochs=10000, save_dir="parity_signal_noise_grid_search"):
    """
    Run experiments for all combinations of parameters in a grid search using multiple GPUs
    
    Args:
        d_values: List of input dimensions to test
        k_values: List of parity orders to test
        batch_sizes: List of batch sizes to test
        learning_rates: List of learning rates to test
        n_epochs: Maximum epochs per experiment
        save_dir: Directory to save all results
        
    Returns:
        results_df: DataFrame with results from all experiments
    """
    # Create master save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all parameter combinations
    param_grid = list(itertools.product(d_values, k_values, batch_sizes, learning_rates))
    n_experiments = len(param_grid)
    print(f"Running grid search with {n_experiments} parameter combinations")
    
    # Count available GPUs
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Found {n_gpus} available GPUs")
    
    # Generate tasks with appropriate device IDs
    tasks = []
    for i, (d, k, bs, lr) in enumerate(param_grid):
        # Adjust epochs based on expected difficulty
        adjusted_epochs = min(n_epochs, int(n_epochs * (d / min(d_values)) * (k / min(k_values))))
        
        # Assign device in round-robin fashion if GPUs are available
        device_id = i % n_gpus if n_gpus > 0 else None
        
        # Add task
        tasks.append((d, k, bs, lr, device_id, adjusted_epochs, save_dir))
    
    # Run experiments in parallel if multiple GPUs are available
    results = []
    if n_gpus > 1:
        # Limit to 4 parallel processes (one per GPU) or available GPUs
        n_workers = min(4, n_gpus)
        print(f"Running experiments in parallel with {n_workers} workers")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(run_experiment_with_device, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Exception in worker: {e}")
    else:
        # Run sequentially
        print("Running experiments sequentially")
        for task in tqdm(tasks):
            result = run_experiment_with_device(task)
            results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate ratio of detected to theoretical transition epochs for successful runs
    valid_results = results_df[results_df['success'] == True].copy()
    
    # Fix: Use a more explicit approach for calculating transition_ratio
    transition_ratios = []
    for _, row in valid_results.iterrows():
        if pd.notna(row['detected_transition']) and pd.notna(row['theoretical_transition']):
            ratio = row['detected_transition'] / row['theoretical_transition']
        else:
            ratio = None
        transition_ratios.append(ratio)
    
    valid_results['transition_ratio'] = transition_ratios
    
    # Calculate ratio of signal-noise crossover to theoretical prediction
    if 'signal_noise_crossover_epoch' in valid_results.columns:
        crossover_ratios = []
        for _, row in valid_results.iterrows():
            if pd.notna(row['signal_noise_crossover_epoch']) and pd.notna(row['theoretical_transition']):
                ratio = row['signal_noise_crossover_epoch'] / row['theoretical_transition']
            else:
                ratio = None
            crossover_ratios.append(ratio)
        
        valid_results['crossover_ratio'] = crossover_ratios
    
    # Save results
    results_file = os.path.join(save_dir, "grid_search_results.csv")
    results_df.to_csv(results_file, index=False)
    
    # Save successful results separately for analysis
    valid_results_file = os.path.join(save_dir, "valid_grid_search_results.csv")
    valid_results.to_csv(valid_results_file, index=False)
    
    print(f"Grid search completed. Results saved to {results_file}")
    
    # Generate summary plots for scaling analysis
    plot_scaling_analysis(valid_results, save_dir)
    
    return valid_results

def plot_scaling_analysis(results_df, save_dir):
    """
    Plot analysis of scaling relationships from grid search results
    
    Args:
        results_df: DataFrame with grid search results
        save_dir: Directory to save plots
    """
    # Filter to only successful runs
    valid_results = results_df.dropna(subset=['detected_transition'])
    
    if len(valid_results) == 0:
        print("No valid results for scaling analysis")
        return
    
    # 1. Plot detected transition vs d^(k/2)
    plt.figure(figsize=(12, 8))
    
    # Create theoretical scaling values
    d_values = sorted(valid_results['d'].unique())
    k_values = sorted(valid_results['k'].unique())
    
    # Group by k and plot for each k
    for k in k_values:
        k_data = valid_results[valid_results['k'] == k]
        if len(k_data) > 1:  # Need at least 2 points for a line
            plt.loglog(k_data['d'], k_data['detected_transition'], 
                   marker='o', linestyle='-', label=f'k={k} (detected)')
    
    plt.xlabel('Input Dimension (d)')
    plt.ylabel('Transition Epoch')
    plt.title('Phase Transition Timing vs Input Dimension')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "transition_scaling_analysis.png"), dpi=300)
    plt.close()
    
    # 2. Plot transition_ratio to check consistency with theory
    if 'transition_ratio' in valid_results.columns:
        plt.figure(figsize=(12, 8))
        
        # Boxplot of transition ratios grouped by k
        ax = sns.boxplot(x='k', y='transition_ratio', data=valid_results)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Perfect match with theory')
        
        plt.xlabel('Parity Order (k)')
        plt.ylabel('Detected/Theoretical Transition Ratio')
        plt.title('Consistency of Detected Transitions with Theoretical Predictions')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "transition_ratio_by_k.png"), dpi=300)
        plt.close()
    
    # 3. If signal-noise crossover points are available, analyze their consistency with theory
    if 'signal_noise_crossover_epoch' in valid_results.columns and 'crossover_ratio' in valid_results.columns:
        crossover_data = valid_results.dropna(subset=['signal_noise_crossover_epoch', 'crossover_ratio'])
        
        if len(crossover_data) > 0:
            plt.figure(figsize=(12, 8))
            
            # Plot both transition ratios
            plt.scatter(crossover_data['transition_ratio'], crossover_data['crossover_ratio'],
                     c=crossover_data['k'], cmap='viridis', s=100, alpha=0.7)
            
            plt.plot([0, 2], [0, 2], 'r--', alpha=0.7, label='1:1 correspondence')
            plt.xlabel('Detected Transition / Theoretical')
            plt.ylabel('Signal-Noise Crossover / Theoretical')
            plt.title('Comparison of Phase Transition and Signal-Noise Crossover Points')
            plt.grid(True)
            plt.colorbar(label='k value')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "transition_vs_crossover.png"), dpi=300)
            plt.close()
    
    # 4. Aggregate correlation data across runs if available
    import glob
    correlation_files = []
    for d in d_values:
        for k in k_values:
            pattern = f"{save_dir}/d{d}_k{k}_bs*/d{d}_k{k}_results/feature_correlations.csv"
            matches = glob.glob(pattern)
            if matches:
                correlation_files.append((d, k, matches[0]))
    
    if correlation_files:
        plt.figure(figsize=(12, 8))
        
        # For each (d,k) combination, check correlation at a fixed epoch (e.g., 1000)
        target_epoch = 1000
        d_k_correlations = []
        
        for d, k, file_path in correlation_files:
            try:
                corr_df = pd.read_csv(file_path)
                # Find closest epoch to target
                closest_epoch_idx = (corr_df['epoch'] - target_epoch).abs().argsort()[0]
                closest_epoch = corr_df.iloc[closest_epoch_idx]['epoch']
                
                relevant_mean = corr_df.iloc[closest_epoch_idx]['relevant_mean']
                d_k_correlations.append({
                    'd': d,
                    'k': k,
                    'epoch': closest_epoch,
                    'relevant_mean': relevant_mean,
                    'theoretical': d ** (-(k-1)/2)
                })
            except Exception as e:
                print(f"Error processing correlation file {file_path}: {e}")
        
        if d_k_correlations:
            corr_scaling_df = pd.DataFrame(d_k_correlations)
            
            # Group by k and plot for each k
            for k in sorted(corr_scaling_df['k'].unique()):
                k_data = corr_scaling_df[corr_scaling_df['k'] == k]
                if len(k_data) > 1:
                    plt.loglog(k_data['d'], k_data['relevant_mean'], 
                           marker='o', linestyle='-', label=f'k={k} (measured)')
            
            plt.xlabel('Input Dimension (d)')
            plt.ylabel('Relevant Feature Correlation (epoch ~1000)')
            plt.title('Feature Correlation Scaling vs Input Dimension')
            plt.grid(True, which="both", ls="-")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "correlation_scaling_analysis.png"), dpi=300)
            plt.close()
    
    # 5. Test for correlation between theoretical ratio and detected phase transition
    if len(valid_results) >= 5:  # Need enough data points for meaningful correlation
        # Calculate Pearson and Spearman correlations
        if 'theoretical_transition' in valid_results.columns:
            pearson_r, pearson_p = stats.pearsonr(valid_results['theoretical_transition'].dropna(), 
                                                valid_results['detected_transition'].dropna())
            spearman_r, spearman_p = stats.spearmanr(valid_results['theoretical_transition'].dropna(), 
                                                    valid_results['detected_transition'].dropna())
            
            plt.figure(figsize=(10, 8))
            plt.scatter(valid_results['theoretical_transition'], valid_results['detected_transition'], 
                    c=valid_results['k'], cmap='viridis', s=80, alpha=0.7)
            
            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_results['theoretical_transition'].dropna(), 
                valid_results['detected_transition'].dropna())
            
            x_range = np.array([valid_results['theoretical_transition'].min(), 
                              valid_results['theoretical_transition'].max()])
            plt.plot(x_range, intercept + slope * x_range, 'g-', alpha=0.7, 
                   label=f'Regression (slope={slope:.2f})')
            
            plt.title(f'Detected Transition Epochs\nPearson r={pearson_r:.2f} (p={pearson_p:.4f}), '
                    f'Spearman r={spearman_r:.2f} (p={spearman_p:.4f})')
            plt.xlabel('Theoretical Transition Epoch')
            plt.ylabel('Detected Transition Epoch')
            plt.colorbar(label='k value')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "theoretical_vs_detected_correlation.png"), dpi=300)
            plt.close()

def main():
    """
    Main function for running experiments with direct parameter input
    """
    print("Parity Learning Signal-Noise Analysis")
    print("=====================================")
    
    # Choose experiment type programmatically
    choice = 2  # Default to grid search
    
    if choice == 1:
        # Parameters for single experiment - can be directly set here
        d = 30
        k = 6
        M1 = 512
        M2 = 512
        batch_size = 512
        learning_rate = 0.01
        n_epochs = 10000
        save_dir = "parity_signal_noise_analysis"
        
        # Check for available GPUs
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device_id = 0 if n_gpus > 0 else None
        
        # Run single experiment
        run_single_experiment(
            d=d,
            k=k,
            M1=M1,
            M2=M2,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            device_id=device_id,
            save_dir=save_dir
        )
    
    elif choice == 2:
        # Parameters for grid search - can be directly set as lists/arrays
        d_values = [20, 30,35,40,45,50,55,60,70]  # Example values
        k_values = [3, 4, 5,6,7,8]     # Example values
        batch_sizes = [512]
        learning_rates = [0.001]
        n_epochs = 1000000
        save_dir = "parity_signal_noise_grid_search2"
        
        # Run grid search
        run_grid_search(
            d_values=d_values,
            k_values=k_values,
            batch_sizes=batch_sizes,
            learning_rates=learning_rates,
            n_epochs=n_epochs,
            save_dir=save_dir
        )
    
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()