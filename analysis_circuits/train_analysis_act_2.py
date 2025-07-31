import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from matplotlib.gridspec import GridSpec
import networkx as nx
import time
import itertools
from IPython.display import clear_output

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
    
    def get_layerwise_forward(self, x):
        """Get activations from each layer for analysis"""
        h1_preact = torch.matmul(x, self.W1.t()) + self.b1
        h1 = torch.relu(h1_preact)
        h2_preact = torch.matmul(h1, self.W2.t()) + self.b2
        h2 = torch.relu(h2_preact)
        output = torch.matmul(h2, self.a)
        
        return {
            'h1_preact': h1_preact,
            'h1': h1,
            'h2_preact': h2_preact,
            'h2': h2,
            'output': output
        }


def count_nonzero_params(model):
    """Count the number of non-zero parameters in a model"""
    count = 0
    
    # First layer weights
    count += torch.count_nonzero(model.W1).item()
    count += torch.count_nonzero(model.b1).item()
    
    # Second layer weights
    count += torch.count_nonzero(model.W2).item()
    count += torch.count_nonzero(model.b2).item()
    
    # Output weights
    count += torch.count_nonzero(model.a).item()
    
    return count


class ParityNetworkAnalyzer:
    def __init__(self, d=30, k=6, M1=128, M2=128, learning_rate=0.01, 
                 batch_size=512, device=None, save_dir="parity_analysis_results"):
        """
        Initialize the ParityNetworkAnalyzer to analyze neural networks learning parity functions
        
        Args:
            d: Input dimension
            k: Parity function order (k-sparse parity)
            M1: First layer width
            M2: Second layer width
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            device: Torch device (cuda or cpu)
            save_dir: Directory to save results
        """
        self.d = d
        self.k = k
        self.M1 = M1
        self.M2 = M2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.neuron_correlations = {
            'epochs': [],
            'layer1': [],
            'layer2': []
        }
        
        # Stored weight matrices for visualization
        self.weight_snapshots = {
            'epochs': [],
            'W1': [],
            'W2': [],
            'a': []
        }
        
        # Feature importance ratio tracking
        self.feature_importance = {
            'epochs': [],
            'relevant_importance': [],
            'irrelevant_importance': [],
            'ratio': []
        }
        
        # Gradient information
        self.gradient_snapshots = {
            'epochs': [],
            'W1_grad_norm': [],
            'W2_grad_norm': [],
            'a_grad_norm': [],
            'W1_grad_relevant': [],
            'W1_grad_irrelevant': []
        }
        
        # Sample fixed test inputs for consistent evaluation
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, d)), 
                                  dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)
        
        # Create smaller sample for detailed analysis (to avoid memory issues)
        self.X_analysis = torch.tensor(np.random.choice([-1, 1], size=(500, d)), 
                                      dtype=torch.float32).to(self.device)
        self.y_analysis = self._target_function(self.X_analysis)
        
        print(f"ParityNetworkAnalyzer initialized on {self.device}")
        print(f"Analyzing {k}-parity function in {d} dimensions")
        print(f"Network: {M1} → {M2} → 1")
    
    def _create_model(self):
        """Create a two-layer ReLU network"""
        model = TwoLayerReLUNet(self.d, self.M1, self.M2).to(self.device)
        return model
    
    def _target_function(self, x):
        """Compute the k-sparse parity function on the first k inputs"""
        return torch.prod(x[:, :self.k], dim=1)
    
    def take_weight_snapshot(self, epoch):
        """
        Take a snapshot of all weight matrices at the current epoch
        """
        self.weight_snapshots['epochs'].append(epoch)
        self.weight_snapshots['W1'].append(self.model.W1.detach().cpu().numpy())
        self.weight_snapshots['W2'].append(self.model.W2.detach().cpu().numpy())
        self.weight_snapshots['a'].append(self.model.a.detach().cpu().numpy())
    
    def compute_neuron_correlations(self, epoch):
        """
        Compute correlation of each neuron's activation with the target function
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get neuron activations
            h1 = self.model.get_first_layer_activations(self.X_analysis)
            h2 = self.model.get_second_layer_activations(self.X_analysis)
            
            # Get target values
            y_true = self.y_analysis
            
            # Compute correlations for each neuron
            layer1_corrs = []
            for j in range(self.M1):
                corr = torch.corrcoef(torch.stack([h1[:, j], y_true]))[0, 1].item()
                layer1_corrs.append(corr)
            
            layer2_corrs = []
            for j in range(self.M2):
                corr = torch.corrcoef(torch.stack([h2[:, j], y_true]))[0, 1].item()
                layer2_corrs.append(corr)
            
            # Store correlations
            self.neuron_correlations['epochs'].append(epoch)
            self.neuron_correlations['layer1'].append(layer1_corrs)
            self.neuron_correlations['layer2'].append(layer2_corrs)
    
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
    
    def compute_gradient_info(self, epoch, X_batch, y_batch):
        """
        Compute and store gradient norms and other gradient metrics
        """
        # Forward pass
        outputs = self.model(X_batch)
        loss = self.criterion(outputs, y_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norms
        W1_grad_norm = torch.norm(self.model.W1.grad).item()
        W2_grad_norm = torch.norm(self.model.W2.grad).item()
        a_grad_norm = torch.norm(self.model.a.grad).item()
        
        # Compute gradient for relevant vs irrelevant features
        W1_grad = self.model.W1.grad.detach().cpu().numpy()
        W1_grad_relevant = np.mean(np.abs(W1_grad[:, :self.k]))
        W1_grad_irrelevant = np.mean(np.abs(W1_grad[:, self.k:]))
        
        # Store values
        self.gradient_snapshots['epochs'].append(epoch)
        self.gradient_snapshots['W1_grad_norm'].append(W1_grad_norm)
        self.gradient_snapshots['W2_grad_norm'].append(W2_grad_norm)
        self.gradient_snapshots['a_grad_norm'].append(a_grad_norm)
        self.gradient_snapshots['W1_grad_relevant'].append(W1_grad_relevant)
        self.gradient_snapshots['W1_grad_irrelevant'].append(W1_grad_irrelevant)
    
    def train(self, n_epochs=10000, snapshot_interval=1000, metric_interval=100, early_stop_corr=0.99999):
        """
        Train the network and track relevant metrics
        
        Args:
            n_epochs: Maximum number of epochs
            snapshot_interval: Interval for taking weight snapshots
            metric_interval: Interval for computing neuron correlations
            early_stop_corr: Correlation threshold for early stopping
        
        Returns:
            final_correlation: Final correlation achieved
            stopping_epoch: Epoch at which training stopped
        """
        print(f"Starting training for {n_epochs} epochs...")
        
        # Take initial snapshot
        self.take_weight_snapshot(0)
        self.compute_neuron_correlations(0)
        self.compute_feature_importance(0)
        
        # For timing
        start_time = time.time()
        
        for epoch in tqdm(range(n_epochs)):
            # Training step
            self.model.train()
            
            # Generate random batch
            X = torch.tensor(np.random.choice([-1, 1], size=(self.batch_size, self.d)), 
                           dtype=torch.float32).to(self.device)
            y = self._target_function(X)
            
            # Gradient info at metric intervals
            if epoch % metric_interval == 0 or epoch == n_epochs - 1:
                self.compute_gradient_info(epoch, X, y)
            
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
            if epoch % metric_interval == 0 or epoch == n_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    preds = self.model(self.X_test)
                    test_loss = ((preds - self.y_test) ** 2).mean().item()
                    correlation = torch.corrcoef(torch.stack([preds.squeeze(), self.y_test]))[0, 1].item()
                
                self.correlation_history.append((epoch, correlation))
                
                # Compute neuron correlations
                self.compute_neuron_correlations(epoch)
                
                # Compute feature importance
                self.compute_feature_importance(epoch)
                
                # Print progress
                if epoch % (metric_interval * 10) == 0 or epoch == n_epochs - 1:
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch}: MSE={test_loss:.6f}, Correlation={correlation:.4f}, Time={elapsed:.1f}s")
                
                # Early stopping
                if correlation > early_stop_corr:
                    print(f"Early stopping at epoch {epoch} with correlation {correlation:.4f}")
                    # Take final snapshot before stopping
                    self.take_weight_snapshot(epoch)
                    return correlation, epoch
            
            # Take weight snapshots at intervals
            if epoch % snapshot_interval == 0 or epoch == n_epochs - 1:
                self.take_weight_snapshot(epoch)
        
        # Take final snapshot if not already taken
        if self.weight_snapshots['epochs'][-1] != n_epochs - 1:
            self.take_weight_snapshot(n_epochs - 1)
            self.compute_neuron_correlations(n_epochs - 1)
            self.compute_feature_importance(n_epochs - 1)
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test)
            final_loss = ((preds - self.y_test) ** 2).mean().item()
            final_correlation = torch.corrcoef(torch.stack([preds.squeeze(), self.y_test]))[0, 1].item()
        
        print("Training completed!")
        print(f"Final MSE: {final_loss:.6f}")
        print(f"Final correlation: {final_correlation:.4f}")
        
        return final_correlation, n_epochs - 1
    
    def visualize_feature_importance_evolution(self):
        """
        Visualize how the importance ratio between relevant and irrelevant features evolves
        """
        if not self.feature_importance['epochs']:
            print("No feature importance data recorded")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot ratio evolution
        plt.semilogy(self.feature_importance['epochs'], self.feature_importance['ratio'], 
                   marker='o', linestyle='-', linewidth=2)
        
        # Add points where correlation exceeds thresholds
        for threshold in [0.5, 0.9, 0.95]:
            # Find first epoch where correlation exceeds threshold
            for i, (epoch, corr) in enumerate(self.correlation_history):
                if corr > threshold:
                    plt.axvline(x=epoch, color=f'C{int(threshold*10)}', 
                              linestyle='--', alpha=0.7,
                              label=f'Correlation > {threshold}')
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
        
        plt.xlabel('Epoch')
        plt.ylabel('Average Absolute Weight')
        plt.title(f'Evolution of Absolute Feature Importance (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/absolute_feature_importance_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def visualize_weight_matrices(self):
        """
        Plot full heatmaps of weight matrices at different stages of training
        """
        # Only visualize if we have at least two snapshots
        if len(self.weight_snapshots['epochs']) < 2:
            print("Not enough weight snapshots for visualization")
            return
        
        # Select snapshots to visualize: initial, mid-training, final
        n_snapshots = len(self.weight_snapshots['epochs'])
        snapshot_indices = [0]  # Initial
        
        # Add pre-learning snapshot if available
        if n_snapshots > 3:
            # Find the last snapshot before correlation > 0.5
            pre_learning_idx = 0
            for i, (epoch, corr) in enumerate(self.correlation_history):
                if corr > 0.5:
                    pre_learning_idx = i - 1 if i > 0 else 0
                    break
            
            # Find closest snapshot to this epoch
            pre_learning_epoch = self.correlation_history[pre_learning_idx][0]
            closest_idx = 0
            min_diff = float('inf')
            for i, epoch in enumerate(self.weight_snapshots['epochs']):
                diff = abs(epoch - pre_learning_epoch)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            if closest_idx not in snapshot_indices:
                snapshot_indices.append(closest_idx)
        
        # Add final snapshot
        snapshot_indices.append(n_snapshots - 1)
        
        # Create figure for W1 matrices (first layer weights)
        plt.figure(figsize=(20, 5 * len(snapshot_indices)))
        titles = ['Initial Weights', 'Pre-Learning Weights', 'Final Weights']
        titles = titles[:len(snapshot_indices)]
        
        for i, idx in enumerate(snapshot_indices):
            epoch = self.weight_snapshots['epochs'][idx]
            W1 = self.weight_snapshots['W1'][idx]
            
            plt.subplot(len(snapshot_indices), 1, i+1)
            im = plt.imshow(W1, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im)
            plt.title(f"{titles[i]} (Epoch {epoch}) - First Layer (W1)")
            plt.xlabel("Input Feature")
            plt.ylabel("Neuron Index")
            
            # Highlight relevant features
            plt.axvline(x=self.k-0.5, color='green', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/W1_full_matrices_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Create figure for W2 matrices (second layer weights)
        plt.figure(figsize=(20, 5 * len(snapshot_indices)))
        
        for i, idx in enumerate(snapshot_indices):
            epoch = self.weight_snapshots['epochs'][idx]
            W2 = self.weight_snapshots['W2'][idx]
            
            plt.subplot(len(snapshot_indices), 1, i+1)
            im = plt.imshow(W2, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im)
            plt.title(f"{titles[i]} (Epoch {epoch}) - Second Layer (W2)")
            plt.xlabel("Layer 1 Neuron")
            plt.ylabel("Layer 2 Neuron")
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/W2_full_matrices_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Create figure for output weights (a)
        plt.figure(figsize=(20, 5 * len(snapshot_indices)))
        
        for i, idx in enumerate(snapshot_indices):
            epoch = self.weight_snapshots['epochs'][idx]
            a = self.weight_snapshots['a'][idx]
            
            plt.subplot(len(snapshot_indices), 1, i+1)
            plt.bar(range(len(a)), a)
            plt.title(f"{titles[i]} (Epoch {epoch}) - Output Weights (a)")
            plt.xlabel("Layer 2 Neuron")
            plt.ylabel("Weight Value")
            
            # Highlight top 5 neurons with highest absolute weights
            top_indices = np.argsort(-np.abs(a))[:5]
            for j in top_indices:
                plt.text(j, a[j], f"{j}", ha='center', va='bottom', 
                       color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/output_weights_full_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def visualize_gradient_evolution(self):
        """
        Visualize the evolution of gradient norms and gradients for relevant vs irrelevant features
        """
        if not self.gradient_snapshots['epochs']:
            print("No gradient data recorded")
            return
        
        # Plot gradient norms
        plt.figure(figsize=(12, 8))
        
        plt.semilogy(self.gradient_snapshots['epochs'], self.gradient_snapshots['W1_grad_norm'],
                   marker='o', label='W1 Grad Norm')
        plt.semilogy(self.gradient_snapshots['epochs'], self.gradient_snapshots['W2_grad_norm'],
                   marker='x', label='W2 Grad Norm')
        plt.semilogy(self.gradient_snapshots['epochs'], self.gradient_snapshots['a_grad_norm'],
                   marker='^', label='Output Grad Norm')
        
        # Add correlation milestones
        for threshold in [0.5, 0.9]:
            # Find first epoch where correlation exceeds threshold
            for i, (epoch, corr) in enumerate(self.correlation_history):
                if corr > threshold:
                    plt.axvline(x=epoch, color=f'C{int(threshold*10)}', 
                              linestyle='--', alpha=0.7,
                              label=f'Correlation > {threshold}')
                    break
        
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title(f'Evolution of Gradient Norms (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/gradient_norms_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Plot relevant vs irrelevant feature gradients
        plt.figure(figsize=(12, 8))
        
        plt.semilogy(self.gradient_snapshots['epochs'], self.gradient_snapshots['W1_grad_relevant'],
                   marker='o', label='Relevant Features Gradient')
        plt.semilogy(self.gradient_snapshots['epochs'], self.gradient_snapshots['W1_grad_irrelevant'],
                   marker='x', label='Irrelevant Features Gradient')
        
        # Compute and plot the ratio
        grad_ratios = [r / max(i, 1e-10) for r, i in zip(
            self.gradient_snapshots['W1_grad_relevant'], 
            self.gradient_snapshots['W1_grad_irrelevant']
        )]
        
        plt.semilogy(self.gradient_snapshots['epochs'], grad_ratios,
                   marker='^', label='Relevant/Irrelevant Ratio')
        
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Magnitude (log scale)')
        plt.title(f'Gradient Evolution for Relevant vs Irrelevant Features (d={self.d}, k={self.k})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_gradients_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def analyze_distributed_computation(self):
        """
        Analyze how the network distributes the computation of parity function
        across neurons in both layers
        """
        print("Analyzing distributed computation patterns...")
        
        # Get final weights
        W1 = self.weight_snapshots['W1'][-1]
        W2 = self.weight_snapshots['W2'][-1]
        a = self.weight_snapshots['a'][-1]
        
        # 1. Analyze first layer feature detectors
        # For each neuron, identify which features it responds to most strongly
        neuron_patterns = []
        
        for i in range(self.M1):
            # Get weights for neuron i (only look at first k features)
            weights = W1[i, :self.k]
            
            # Threshold to find features with strong connections
            # We consider both strong positive and negative connections
            threshold = np.percentile(np.abs(weights), 75)  # Top 25% strongest
            
            strong_positive = np.where(weights > threshold)[0]
            strong_negative = np.where(weights < -threshold)[0]
            
            pattern = {
                'neuron_idx': i,
                'strong_positive': strong_positive,
                'strong_negative': strong_negative,
                'net_weight': np.sum(weights)
            }
            
            neuron_patterns.append(pattern)
        
        # Sort patterns by number of strong connections (to find potential XOR-like detectors)
        neuron_patterns.sort(key=lambda p: len(p['strong_positive']) + len(p['strong_negative']), reverse=True)
        
        # 2. Analyze second layer output contribution
        # For each layer 2 neuron, determine how much it contributes to the output
        output_contributions = []
        
        for i in range(self.M2):
            contribution = a[i]
            
            # Find which layer 1 neurons this neuron listens to
            strong_connections = []
            threshold = np.percentile(np.abs(W2[i]), 75)  # Top 25% strongest
            
            for j in range(self.M1):
                if abs(W2[i, j]) > threshold:
                    strong_connections.append({
                        'neuron_idx': j,
                        'weight': W2[i, j]
                    })
            
            strong_connections.sort(key=lambda c: abs(c['weight']), reverse=True)
            
            output_contributions.append({
                'neuron_idx': i,
                'output_weight': contribution,
                'strong_connections': strong_connections[:10]  # Top 10 connections
            })
        
        # Sort by absolute contribution to output
        output_contributions.sort(key=lambda c: abs(c['output_weight']), reverse=True)
        
        # 3. Identify "motifs" or patterns in the computation
        # We'll look for common patterns in how layer 1 neurons connect to inputs
        
        # Group neurons by their top positive and negative connection patterns
        pattern_groups = {}
        
        for pattern in neuron_patterns:
            # Create a simple key based on the strongest positive and negative connections
            # We'll limit to top 3 of each to avoid excessive fragmentation
            pos_key = tuple(sorted(pattern['strong_positive'][:3]))
            neg_key = tuple(sorted(pattern['strong_negative'][:3]))
            key = (pos_key, neg_key)
            
            if key not in pattern_groups:
                pattern_groups[key] = []
            
            pattern_groups[key].append(pattern['neuron_idx'])
        
        # Find the largest pattern groups
        sorted_groups = sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        # 4. Find "detector chains" - paths from input through layer 1 to layer 2 to output
        # Start with top output contributors
        detector_chains = []
        
        for i, contrib in enumerate(output_contributions[:5]):  # Top 5 output neurons
            neuron_idx = contrib['output_weight']
            chains = []
            
            # For each strong connection to layer 1
            for conn in contrib['strong_connections'][:3]:  # Top 3 connections
                l1_idx = conn['neuron_idx']
                l1_pattern = next(p for p in neuron_patterns if p['neuron_idx'] == l1_idx)
                
                # Create a chain linking this path
                chain = {
                    'output_neuron': neuron_idx,
                    'output_weight': contrib['output_weight'],
                    'l2_l1_weight': conn['weight'],
                    'l1_neuron': l1_idx,
                    'l1_pos_features': l1_pattern['strong_positive'],
                    'l1_neg_features': l1_pattern['strong_negative']
                }
                
                chains.append(chain)
            
            detector_chains.append(chains)
        
        # Save the analysis results
        analysis = {
            'neuron_patterns': neuron_patterns[:20],  # Top 20 for brevity
            'output_contributions': output_contributions[:20],
            'pattern_groups': [(key, values) for key, values in sorted_groups[:10]],
            'detector_chains': detector_chains
        }
        
        # Save as a text report
        with open(f"{self.save_dir}/distributed_computation_analysis_d{self.d}_k{self.k}.txt", 'w') as f:
            f.write(f"Distributed Computation Analysis for {self.k}-Parity in {self.d} dimensions\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Top Layer 1 Neuron Patterns:\n")
            f.write("-" * 40 + "\n")
            for i, pattern in enumerate(analysis['neuron_patterns']):
                f.write(f"Neuron {pattern['neuron_idx']}:\n")
                f.write(f"  Positive connections to features: {pattern['strong_positive']}\n")
                f.write(f"  Negative connections to features: {pattern['strong_negative']}\n")
                f.write(f"  Net weight: {pattern['net_weight']:.4f}\n\n")
            
            f.write("\nTop Output Contributing Neurons:\n")
            f.write("-" * 40 + "\n")
            for i, contrib in enumerate(analysis['output_contributions']):
                f.write(f"Layer 2 Neuron {contrib['neuron_idx']}:\n")
                f.write(f"  Output weight: {contrib['output_weight']:.4f}\n")
                f.write("  Top connections to Layer 1:\n")
                for conn in contrib['strong_connections'][:5]:
                    f.write(f"    Neuron {conn['neuron_idx']}: Weight {conn['weight']:.4f}\n")
                f.write("\n")
            
            f.write("\nCommon Pattern Groups:\n")
            f.write("-" * 40 + "\n")
            for i, (key, neurons) in enumerate(analysis['pattern_groups']):
                pos_key, neg_key = key
                f.write(f"Pattern Group {i+1}:\n")
                f.write(f"  Positive connections: {pos_key}\n")
                f.write(f"  Negative connections: {neg_key}\n")
                f.write(f"  Neurons in group: {neurons}\n")
                f.write(f"  Count: {len(neurons)}\n\n")
            
            f.write("\nDetector Chains (Input → Layer 1 → Layer 2 → Output):\n")
            f.write("-" * 40 + "\n")
            for i, chains in enumerate(analysis['detector_chains']):
                f.write(f"Chain Group {i+1}:\n")
                for j, chain in enumerate(chains):
                    f.write(f"  Chain {j+1}:\n")
                    f.write(f"    Output Neuron {chain['output_neuron']} (weight: {chain['output_weight']:.4f})\n")
                    f.write(f"      ↑ weight: {chain['l2_l1_weight']:.4f}\n")
                    f.write(f"    Layer 1 Neuron {chain['l1_neuron']}\n")
                    f.write(f"      ↑ positive from features: {chain['l1_pos_features']}\n")
                    f.write(f"      ↑ negative from features: {chain['l1_neg_features']}\n")
                f.write("\n")
        
        print(f"Analysis saved to {self.save_dir}/distributed_computation_analysis_d{self.d}_k{self.k}.txt")
        
        # Visualize key patterns with a network diagram for the top contributors
        self._visualize_computational_network(W1, W2, a)
        
        return analysis
    
    def _visualize_computational_network(self, W1, W2, a):
        """Create a visualization of the computational network focusing on key neurons"""
        # Find top output neurons
        top_output_neurons = np.argsort(-np.abs(a))[:3]  # Top 3
        
        # Create graph
        G = nx.DiGraph()
        
        # Add input nodes (only relevant features)
        for i in range(self.k):
            G.add_node(f"x{i}", layer=0, pos=(i, 0))
        
        # Threshold for showing connections
        W1_threshold = np.percentile(np.abs(W1[:, :self.k]), 75)  # Top 25% of W1
        W2_threshold = np.percentile(np.abs(W2), 75)  # Top 25% of W2
        
        # Add layer 1 neurons connected to top output neurons
        l1_neurons = set()
        for out_idx in top_output_neurons:
            for l1_idx in range(self.M1):
                if abs(W2[out_idx, l1_idx]) > W2_threshold:
                    l1_neurons.add(l1_idx)
        
        # Limit to 15 layer 1 neurons for visibility
        if len(l1_neurons) > 15:
            # Keep neurons with strongest connections to output
            strength_dict = {}
            for l1_idx in l1_neurons:
                max_strength = max(abs(W2[out_idx, l1_idx]) for out_idx in top_output_neurons)
                strength_dict[l1_idx] = max_strength
            
            l1_neurons = set(sorted(strength_dict.keys(), key=lambda x: strength_dict[x], reverse=True)[:15])
        
        # Add layer 1 neurons to graph
        for l1_idx in l1_neurons:
            G.add_node(f"h1_{l1_idx}", layer=1, pos=(l1_idx % self.k, 1))
            
            # Add connections from inputs to this layer 1 neuron
            for j in range(self.k):
                if abs(W1[l1_idx, j]) > W1_threshold:
                    G.add_edge(f"x{j}", f"h1_{l1_idx}", weight=W1[l1_idx, j])
        
        # Add layer 2 neurons (top contributors to output)
        for l2_idx in top_output_neurons:
            G.add_node(f"h2_{l2_idx}", layer=2, pos=(l2_idx % self.k, 2))
            
            # Add connections from layer 1 to this layer 2 neuron
            for l1_idx in l1_neurons:
                if abs(W2[l2_idx, l1_idx]) > W2_threshold:
                    G.add_edge(f"h1_{l1_idx}", f"h2_{l2_idx}", weight=W2[l2_idx, l1_idx])
        
        # Add output node
        G.add_node("y", layer=3, pos=(self.k//2, 3))
        
        # Connect layer 2 to output
        for l2_idx in top_output_neurons:
            G.add_edge(f"h2_{l2_idx}", "y", weight=a[l2_idx])
        
        # Set up positions for drawing
        pos = nx.get_node_attributes(G, 'pos')
        
        # Adjust positions for better visualization
        for node, position in pos.items():
            x, y = position
            layer = G.nodes[node]['layer']
            
            # Spread out nodes in each layer
            if layer == 1:  # First hidden layer
                x = x / self.k * 10
            elif layer == 2:  # Second hidden layer
                x = (x / self.k * 10) + 0.5  # Offset slightly
            
            pos[node] = (x, y * 3)  # Increase vertical spacing
        
        # Plot
        plt.figure(figsize=(15, 12))
        
        # Draw nodes by layer
        node_colors = ['lightblue', 'lightgreen', 'salmon', 'purple']
        for layer in range(4):
            nodelist = [n for n, d in G.nodes(data=True) if d.get('layer') == layer]
            if nodelist:
                nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=node_colors[layer], 
                                      node_size=500, alpha=0.8)
        
        # Draw edges with colors based on weights
        edges = G.edges(data='weight')
        if edges:
            edge_colors = []
            for u, v, w in edges:
                edge_colors.append(plt.cm.RdBu(0.5 * (1 + w / max(1, abs(w)))))
            
            # Scale edge widths by weight magnitude
            edge_widths = []
            for u, v, w in edges:
                edge_widths.append(1 + 3 * abs(w))
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                                  arrowsize=15, min_source_margin=15, min_target_margin=15)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_weight='bold')
        
        plt.title(f"Key Computational Subnetwork for {self.k}-Parity (d={self.d})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/computational_network_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
    
    def perform_parity_check_analysis(self):
        """
        Analyze how the network computes all possible parity combinations
        for the relevant features to understand the distributed computation
        """
        print("Performing parity check analysis...")
        
        # Generate all possible inputs for the relevant features
        num_configs = 2**self.k
        
        # Create binary configurations (0, 1)
        binary_configs = []
        for i in range(num_configs):
            # Convert to binary and pad with zeros
            binary = format(i, f'0{self.k}b')
            binary_configs.append([int(bit) for bit in binary])
        
        # Convert to inputs (-1, 1)
        input_configs = np.array([[1 if bit == 0 else -1 for bit in config] for config in binary_configs])
        
        # Create full inputs with zeros for irrelevant features
        full_inputs = np.zeros((num_configs, self.d))
        full_inputs[:, :self.k] = input_configs
        
        # Convert to torch tensor
        X = torch.tensor(full_inputs, dtype=torch.float32).to(self.device)
        
        # Compute target values
        y_true = self._target_function(X)
        
        # Get layerwise activations
        self.model.eval()
        with torch.no_grad():
            activations = self.model.get_layerwise_forward(X)
            
            # Get final outputs
            outputs = activations['output']
            
            # Compute accuracy
            predictions = torch.sign(outputs)
            accuracy = (predictions == y_true).float().mean().item()
            
            # Get activations at each layer
            h1 = activations['h1'].cpu().numpy()  # After ReLU
            h1_preact = activations['h1_preact'].cpu().numpy()  # Before ReLU
            h2 = activations['h2'].cpu().numpy()  # After ReLU
        
        # Analyze Layer 1 activations
        # For each input configuration, which neurons activate?
        l1_activation_patterns = []
        
        for i in range(num_configs):
            # Find neurons that activate for this input
            active_neurons = np.where(h1[i] > 0)[0]
            
            # Add to results
            l1_activation_patterns.append({
                'config_idx': i,
                'binary_config': binary_configs[i],
                'input_config': input_configs[i],
                'target': y_true[i].item(),
                'prediction': predictions[i].item(),
                'output': outputs[i].item(),
                'active_neurons': active_neurons,
                'num_active': len(active_neurons)
            })
        
        # Group inputs by their parity
        even_parity_inputs = [p for p in l1_activation_patterns if p['target'] == 1]
        odd_parity_inputs = [p for p in l1_activation_patterns if p['target'] == -1]
        
        # Analyze differences in activation patterns
        even_neurons = set()
        for p in even_parity_inputs:
            even_neurons.update(p['active_neurons'])
        
        odd_neurons = set()
        for p in odd_parity_inputs:
            odd_neurons.update(p['active_neurons'])
        
        # Find neurons that help discriminate between even and odd parity
        even_only = even_neurons - odd_neurons
        odd_only = odd_neurons - even_neurons
        shared = even_neurons.intersection(odd_neurons)
        
        # Calculate average activations for even and odd examples
        even_activations = np.mean([h1[i] for i, p in enumerate(l1_activation_patterns) if p['target'] == 1], axis=0)
        odd_activations = np.mean([h1[i] for i, p in enumerate(l1_activation_patterns) if p['target'] == -1], axis=0)
        
        # Calculate discrimination power for each neuron
        discrimination_power = np.abs(even_activations - odd_activations)
        
        # Find top discriminative neurons
        top_discriminators = np.argsort(-discrimination_power)[:10]
        
        # Analyze patterns for these discriminative neurons
        discriminator_patterns = []
        
        # Get final weights
        W1 = self.model.W1.detach().cpu().numpy()
        W2 = self.model.W2.detach().cpu().numpy()
        a = self.model.a.detach().cpu().numpy()
        
        for neuron_idx in top_discriminators:
            # Get weights from inputs to this neuron
            weights = W1[neuron_idx, :self.k]
            
            # Get weights from this neuron to layer 2
            outgoing = W2[:, neuron_idx]
            
            # Find layer 2 neurons most influenced by this neuron
            top_outgoing = np.argsort(-np.abs(outgoing))[:5]
            
            # Determine if this neuron primarily detects even or odd parity
            mean_even_activation = np.mean([h1[i, neuron_idx] for i, p in enumerate(l1_activation_patterns) if p['target'] == 1])
            mean_odd_activation = np.mean([h1[i, neuron_idx] for i, p in enumerate(l1_activation_patterns) if p['target'] == -1])
            
            primarily_detects = "even" if mean_even_activation > mean_odd_activation else "odd"
            
            # Record pattern
            pattern = {
                'neuron_idx': neuron_idx,
                'weights': weights,
                'top_positive': np.where(weights > np.percentile(weights, 75))[0],
                'top_negative': np.where(weights < np.percentile(weights, 25))[0],
                'discrimination_power': discrimination_power[neuron_idx],
                'primarily_detects': primarily_detects,
                'mean_even_activation': mean_even_activation,
                'mean_odd_activation': mean_odd_activation,
                'top_outgoing': [(idx, outgoing[idx]) for idx in top_outgoing]
            }
            
            discriminator_patterns.append(pattern)
        
        # Save analysis results
        analysis_results = {
            'accuracy': accuracy,
            'num_configurations': num_configs,
            'even_only_neurons': list(even_only),
            'odd_only_neurons': list(odd_only),
            'shared_neurons': list(shared),
            'top_discriminators': top_discriminators.tolist(),
            'discriminator_patterns': discriminator_patterns
        }
        
        # Save as a text report
        with open(f"{self.save_dir}/parity_check_analysis_d{self.d}_k{self.k}.txt", 'w') as f:
            f.write(f"Parity Check Analysis for {self.k}-Parity in {self.d} dimensions\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Overall accuracy on all {num_configs} configurations: {accuracy:.4f}\n\n")
            
            f.write("Layer 1 Neuron Activation Patterns:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Neurons that activate only for even parity: {len(even_only)}\n")
            f.write(f"Neurons that activate only for odd parity: {len(odd_only)}\n")
            f.write(f"Neurons that activate for both: {len(shared)}\n\n")
            
            f.write("Top Discriminative Neurons:\n")
            f.write("-" * 40 + "\n")
            for i, pattern in enumerate(discriminator_patterns):
                f.write(f"Neuron {pattern['neuron_idx']}:\n")
                f.write(f"  Discrimination power: {pattern['discrimination_power']:.4f}\n")
                f.write(f"  Primarily detects: {pattern['primarily_detects']} parity\n")
                f.write(f"  Mean activation for even: {pattern['mean_even_activation']:.4f}\n")
                f.write(f"  Mean activation for odd: {pattern['mean_odd_activation']:.4f}\n")
                f.write(f"  Top positive connections: {pattern['top_positive']}\n")
                f.write(f"  Top negative connections: {pattern['top_negative']}\n")
                f.write("  Top connections to layer 2:\n")
                for l2_idx, weight in pattern['top_outgoing']:
                    f.write(f"    Neuron {l2_idx}: Weight {weight:.4f}\n")
                f.write("\n")
        
        print(f"Parity check analysis saved to {self.save_dir}/parity_check_analysis_d{self.d}_k{self.k}.txt")
        
        # Visualize discriminative power
        plt.figure(figsize=(12, 8))
        
        # Plot discrimination power for all neurons
        plt.stem(range(self.M1), discrimination_power)
        
        # Highlight top discriminators
        plt.plot(top_discriminators, discrimination_power[top_discriminators], 'ro', markersize=8)
        
        plt.xlabel('Layer 1 Neuron Index')
        plt.ylabel('Discrimination Power (|mean_even - mean_odd|)')
        plt.title(f'Neuron Discrimination Power for {self.k}-Parity (d={self.d})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/discrimination_power_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Visualize activation patterns for different parity inputs
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.bar(range(self.M1), even_activations, alpha=0.7, label='Even Parity')
        plt.xlabel('Layer 1 Neuron Index')
        plt.ylabel('Mean Activation')
        plt.title('Mean Activation for Even Parity Inputs')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.bar(range(self.M1), odd_activations, alpha=0.7, color='orange', label='Odd Parity')
        plt.xlabel('Layer 1 Neuron Index')
        plt.ylabel('Mean Activation')
        plt.title('Mean Activation for Odd Parity Inputs')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/activation_patterns_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        return analysis_results

    def run_analysis(self, n_epochs=5000, snapshot_interval=1000, metric_interval=100):
        """
        Run a complete analysis of a neural network learning parity
        
        Args:
            n_epochs: Maximum number of epochs
            snapshot_interval: Interval for taking weight snapshots
            metric_interval: Interval for computing neuron correlations
        """
        print("\n========== PARITY NEURAL NETWORK ANALYSIS ==========")
        
        # 1. Train the network
        print("\n----- Training Network -----")
        final_corr, final_epoch = self.train(n_epochs=n_epochs, snapshot_interval=snapshot_interval, metric_interval=metric_interval)
        
        # 2. Visualize feature importance evolution
        print("\n----- Visualizing Feature Importance Evolution -----")
        self.visualize_feature_importance_evolution()
        
        # 3. Visualize weight matrices
        print("\n----- Visualizing Weight Matrices -----")
        self.visualize_weight_matrices()
        
        # 4. Visualize gradient evolution
        print("\n----- Visualizing Gradient Evolution -----")
        self.visualize_gradient_evolution()
        
        # 5. Analyze distributed computation
        print("\n----- Analyzing Distributed Computation -----")
        computation_analysis = self.analyze_distributed_computation()
        
        # 6. Perform parity check analysis
        print("\n----- Performing Parity Check Analysis -----")
        parity_analysis = self.perform_parity_check_analysis()
        
        print("\n========== ANALYSIS COMPLETE ==========")
        
        return {
            'final_correlation': final_corr,
            'final_epoch': final_epoch,
            'computation_analysis': computation_analysis,
            'parity_analysis': parity_analysis
        }


def grid_search_parity_networks():
    """
    Perform a grid search over different dimensions and parity sizes
    to analyze how these parameters affect learning
    """
    # Parameters for grid search
    dimensions = [20,30,40, 50,60,70]  # Input dimensions
    parity_orders = [3,4,5, 6,7]  # k-sparse parity
    
    # Fixed parameters
    hidden_layer_width = 512
    epochs = 100000
    learning_rate = 0.005
    batch_size = 512
    
    # Create results directory
    results_dir = "parity_grid_search_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Run grid search
    for d in dimensions:
        for k in parity_orders:
            print(f"\n\n{'='*80}")
            print(f"Running analysis for d={d}, k={k}")
            print(f"{'='*80}\n")
            
            # Create analyzer
            save_dir = f"{results_dir}/d{d}_k{k}"
            analyzer = ParityNetworkAnalyzer(
                d=d, 
                k=k, 
                M1=hidden_layer_width, 
                M2=hidden_layer_width,
                learning_rate=learning_rate,
                batch_size=batch_size,
                save_dir=save_dir
            )
            
            # Run analysis
            start_time = time.time()
            analysis_results = analyzer.run_analysis(n_epochs=epochs)
            end_time = time.time()
            
            # Record results
            result = {
                'd': d,
                'k': k,
                'final_correlation': analysis_results['final_correlation'],
                'final_epoch': analysis_results['final_epoch'],
                'training_time': end_time - start_time,
                'save_dir': save_dir
            }
            
            results.append(result)
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{results_dir}/grid_search_results.csv", index=False)
    
    # Create summary visualizations
    visualize_grid_search_results(results, results_dir)
    
    return results


def visualize_grid_search_results(results, results_dir):
    """
    Create visualizations summarizing grid search results
    
    Args:
        results: List of result dictionaries
        results_dir: Directory to save visualizations
    """
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # 1. Plot correlation vs. dimensions for each k
    plt.figure(figsize=(12, 8))
    
    for k in df['k'].unique():
        subset = df[df['k'] == k]
        plt.plot(subset['d'], subset['final_correlation'], marker='o', label=f'k={k}')
    
    plt.xlabel('Input Dimensions (d)')
    plt.ylabel('Final Correlation')
    plt.title('Parity Learning Performance vs. Input Dimension')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/correlation_vs_dimension.png", dpi=300)
    plt.close()
    
    # 2. Plot training time vs. dimensions for each k
    plt.figure(figsize=(12, 8))
    
    for k in df['k'].unique():
        subset = df[df['k'] == k]
        plt.plot(subset['d'], subset['training_time'] / 3600, marker='o', label=f'k={k}')  # Convert to hours
    
    plt.xlabel('Input Dimensions (d)')
    plt.ylabel('Training Time (hours)')
    plt.title('Training Time vs. Input Dimension')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_time_vs_dimension.png", dpi=300)
    plt.close()
    
    # 3. Create a heatmap of final correlation
    heatmap_data = df.pivot(index='k', columns='d', values='final_correlation')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f', vmin=0.5, vmax=1.0)
    plt.xlabel('Input Dimensions (d)')
    plt.ylabel('Parity Order (k)')
    plt.title('Final Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/correlation_heatmap.png", dpi=300)
    plt.close()
    
    # 4. Create a heatmap of final epochs
    heatmap_data = df.pivot(index='k', columns='d', values='final_epoch')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.0f')
    plt.xlabel('Input Dimensions (d)')
    plt.ylabel('Parity Order (k)')
    plt.title('Training Epochs Heatmap')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/epochs_heatmap.png", dpi=300)
    plt.close()


def run_single_parity_analysis(d, k, epochs=10000, M1=128, M2=128):
    """
    Run a single parity network analysis with specified parameters
    
    Args:
        d: Input dimension
        k: Parity order
        epochs: Maximum number of epochs
        M1: First layer width
        M2: Second layer width
    """
    print(f"\nRunning analysis for {k}-Parity in {d} dimensions")
    print(f"Network size: {M1} → {M2} → 1")
    print(f"Maximum epochs: {epochs}")
    
    # Create save directory
    save_dir = f"parity_analysis_d{d}_k{k}"
    
    # Create analyzer
    analyzer = ParityNetworkAnalyzer(
        d=d,
        k=k,
        M1=M1,
        M2=M2,
        learning_rate=0.01,
        batch_size=512,
        save_dir=save_dir
    )
    
    # Run analysis
    analyzer.run_analysis(n_epochs=epochs)
    
    print(f"\nAnalysis complete. Results saved to {save_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze neural networks learning parity functions')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'grid'],
                      help='Run mode: single analysis or grid search')
    
    # Single analysis parameters
    parser.add_argument('--d', type=int, default=30, help='Input dimension')
    parser.add_argument('--k', type=int, default=6, help='Parity function order')
    parser.add_argument('--M1', type=int, default=512, help='First layer width')
    parser.add_argument('--M2', type=int, default=512, help='Second layer width')
    parser.add_argument('--epochs', type=int, default=500000, help='Maximum number of epochs')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_parity_analysis(args.d, args.k, args.epochs, args.M1, args.M2)
    else:
        grid_search_parity_networks()