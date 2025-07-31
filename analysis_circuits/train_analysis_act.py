import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import os
import copy
from matplotlib.gridspec import GridSpec
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    
    def train(self, n_epochs=10000, snapshot_interval=1000, corr_interval=100, early_stop_corr=0.99):
        """
        Train the network and track relevant metrics
        
        Args:
            n_epochs: Maximum number of epochs
            snapshot_interval: Interval for taking weight snapshots
            corr_interval: Interval for computing neuron correlations
            early_stop_corr: Correlation threshold for early stopping
        """
        print(f"Starting training for {n_epochs} epochs...")
        
        # Take initial snapshot
        self.take_weight_snapshot(0)
        self.compute_neuron_correlations(0)
        
        for epoch in tqdm(range(n_epochs)):
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
            if epoch % corr_interval == 0 or epoch == n_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    preds = self.model(self.X_test)
                    test_loss = ((preds - self.y_test) ** 2).mean().item()
                    correlation = torch.corrcoef(torch.stack([preds.squeeze(), self.y_test]))[0, 1].item()
                
                self.correlation_history.append((epoch, correlation))
                
                # Compute neuron correlations
                self.compute_neuron_correlations(epoch)
                
                # Print progress
                print(f"Epoch {epoch}: MSE={test_loss:.6f}, Correlation={correlation:.4f}")
                
                # Early stopping
                #if correlation > early_stop_corr:
                #    print(f"Early stopping at epoch {epoch} with correlation {correlation:.4f}")
                #    # Take final snapshot before stopping
                #    self.take_weight_snapshot(epoch)
                #    break
            
            # Take weight snapshots at intervals
            if epoch % snapshot_interval == 0 or epoch == n_epochs - 1:
                self.take_weight_snapshot(epoch)
        
        # Take final snapshot if not already taken
        if self.weight_snapshots['epochs'][-1] != epoch:
            self.take_weight_snapshot(epoch)
            self.compute_neuron_correlations(epoch)
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test)
            final_loss = ((preds - self.y_test) ** 2).mean().item()
            final_correlation = torch.corrcoef(torch.stack([preds.squeeze(), self.y_test]))[0, 1].item()
        
        print("Training completed!")
        print(f"Final MSE: {final_loss:.6f}")
        print(f"Final correlation: {final_correlation:.4f}")
        
        return final_correlation
    
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
        plt.savefig(f"{self.save_dir}/W1_full_matrices.png", dpi=300)
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
        plt.savefig(f"{self.save_dir}/W2_full_matrices.png", dpi=300)
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
        plt.savefig(f"{self.save_dir}/output_weights_full.png", dpi=300)
        plt.close()
    
    def visualize_neuron_correlations(self):
        """
        Visualize how neuron correlations with the target function evolve over time
        """
        if not self.neuron_correlations['epochs']:
            print("No neuron correlations recorded")
            return
        
        # 1. Plot heatmaps of all neuron correlations over time
        epochs = self.neuron_correlations['epochs']
        layer1_corrs = np.array(self.neuron_correlations['layer1'])
        layer2_corrs = np.array(self.neuron_correlations['layer2'])
        
        # Layer 1 correlations heatmap
        plt.figure(figsize=(15, 10))
        im = plt.imshow(layer1_corrs.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation with Target')
        plt.xlabel('Training Progress (Snapshot Index)')
        plt.ylabel('Layer 1 Neuron Index')
        plt.title('Evolution of Layer 1 Neuron Correlations with Target Function')
        
        # Add epoch labels
        plt.xticks(range(len(epochs)), [f"{e}" for e in epochs], rotation=90)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/layer1_correlation_evolution.png", dpi=300)
        plt.close()
        
        # Layer 2 correlations heatmap
        plt.figure(figsize=(15, 10))
        im = plt.imshow(layer2_corrs.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation with Target')
        plt.xlabel('Training Progress (Snapshot Index)')
        plt.ylabel('Layer 2 Neuron Index')
        plt.title('Evolution of Layer 2 Neuron Correlations with Target Function')
        
        # Add epoch labels
        plt.xticks(range(len(epochs)), [f"{e}" for e in epochs], rotation=90)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/layer2_correlation_evolution.png", dpi=300)
        plt.close()
        
        # 2. Plot trajectories of top correlating neurons
        # Find neurons with highest final correlation (absolute value)
        final_layer1_corrs = np.abs(layer1_corrs[-1])
        final_layer2_corrs = np.abs(layer2_corrs[-1])
        
        top_layer1 = np.argsort(-final_layer1_corrs)[:10]  # Top 10 neurons
        top_layer2 = np.argsort(-final_layer2_corrs)[:10]  # Top 10 neurons
        
        # Plot trajectories for Layer 1
        plt.figure(figsize=(12, 8))
        for neuron_idx in top_layer1:
            corr_trajectory = layer1_corrs[:, neuron_idx]
            plt.plot(epochs, corr_trajectory, marker='o', label=f'Neuron {neuron_idx}')
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Correlation with Target')
        plt.title('Correlation Trajectories for Top Layer 1 Neurons')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/top_layer1_correlation_trajectories.png", dpi=300)
        plt.close()
        
        # Plot trajectories for Layer 2
        plt.figure(figsize=(12, 8))
        for neuron_idx in top_layer2:
            corr_trajectory = layer2_corrs[:, neuron_idx]
            plt.plot(epochs, corr_trajectory, marker='o', label=f'Neuron {neuron_idx}')
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Correlation with Target')
        plt.title('Correlation Trajectories for Top Layer 2 Neurons')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/top_layer2_correlation_trajectories.png", dpi=300)
        plt.close()
        
        # 3. Visualize emergence of correlations - did they exist at initialization?
        # Plot change in absolute correlations from initial to final state
        plt.figure(figsize=(15, 10))
        
        # Layer 1 correlation changes
        plt.subplot(2, 1, 1)
        initial_l1_corrs = np.abs(layer1_corrs[0])
        final_l1_corrs = np.abs(layer1_corrs[-1])
        correlation_growth = final_l1_corrs - initial_l1_corrs
        
        # Sort by final correlation
        sort_idx = np.argsort(-final_l1_corrs)
        plt.bar(range(len(sort_idx)), correlation_growth[sort_idx], alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Neuron Index (Sorted by Final Correlation)')
        plt.ylabel('Increase in Absolute Correlation')
        plt.title('Layer 1: Change in Absolute Correlation from Initial to Final State')
        
        # Layer 2 correlation changes
        plt.subplot(2, 1, 2)
        initial_l2_corrs = np.abs(layer2_corrs[0])
        final_l2_corrs = np.abs(layer2_corrs[-1])
        correlation_growth = final_l2_corrs - initial_l2_corrs
        
        # Sort by final correlation
        sort_idx = np.argsort(-final_l2_corrs)
        plt.bar(range(len(sort_idx)), correlation_growth[sort_idx], alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Neuron Index (Sorted by Final Correlation)')
        plt.ylabel('Increase in Absolute Correlation')
        plt.title('Layer 2: Change in Absolute Correlation from Initial to Final State')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/correlation_emergence.png", dpi=300)
        plt.close()
    
    def analyze_key_output_neuron(self):
        """
        Analyze the subnetwork feeding into the most important output neuron
        """
        # Get final output weights
        final_a = self.weight_snapshots['a'][-1]
        
        # Find the neuron with the largest absolute output weight
        key_neuron_idx = np.argmax(np.abs(final_a))
        key_neuron_weight = final_a[key_neuron_idx]
        
        print(f"Key output neuron identified: Neuron {key_neuron_idx} with weight {key_neuron_weight:.4f}")
        
        # Get the weights connected to this neuron
        final_W2 = self.weight_snapshots['W2'][-1]
        key_connections = final_W2[key_neuron_idx]
        
        # Find the strongest connections to the key neuron
        # abs_connections = np.abs(key_connections)
        top_connection_indices = np.argsort(-np.abs(key_connections))[:10]  # Top 10
        
        print("\nTop connections to key neuron:")
        for idx in top_connection_indices:
            weight = key_connections[idx]
            print(f"Layer 1 Neuron {idx}: Weight = {weight:.4f}")
        
        # Visualize the subnetwork
        final_W1 = self.weight_snapshots['W1'][-1]
        
        # Threshold for showing connections
        w2_threshold = np.percentile(np.abs(key_connections), 90)  # Top 10% of connections
        
        # Create graph
        G = nx.DiGraph()
        
        # Add input nodes (only relevant features)
        for i in range(self.k):
            G.add_node(f"x{i}", layer=0, pos=(i, 0))
        
        # Add layer 1 nodes that have strong connections to the key neuron
        l1_nodes = np.where(np.abs(key_connections) > w2_threshold)[0]
        for idx in l1_nodes:
            G.add_node(f"h1_{idx}", layer=1, pos=(idx % self.k, 1))
            
            # Add connections from inputs to this layer 1 neuron
            # Only show significant weights to reduce clutter
            w1_weights = final_W1[idx, :self.k]  # Only relevant features
            w1_threshold = np.percentile(np.abs(w1_weights), 70)  # Top 30% of connections
            
            for j in range(self.k):
                if abs(w1_weights[j]) > w1_threshold:
                    G.add_edge(f"x{j}", f"h1_{idx}", weight=w1_weights[j])
        
        # Add key output neuron
        G.add_node(f"h2_{key_neuron_idx}", layer=2, pos=(self.k//2, 2))
        
        # Add connections from layer 1 to key neuron
        for idx in l1_nodes:
            weight = key_connections[idx]
            G.add_edge(f"h1_{idx}", f"h2_{key_neuron_idx}", weight=weight)
        
        # Add final output node
        G.add_node("y", layer=3, pos=(self.k//2, 3))
        G.add_edge(f"h2_{key_neuron_idx}", "y", weight=key_neuron_weight)
        
        # Set up positions for drawing
        pos = nx.get_node_attributes(G, 'pos')
        
        # Adjust positions for better visualization
        for node, position in pos.items():
            x, y = position
            layer = G.nodes[node]['layer']
            
            # Spread out nodes in each layer
            if layer == 1:  # First hidden layer
                x = x / self.k * 10
            
            pos[node] = (x, y * 3)  # Increase vertical spacing
        
        # Plot
        plt.figure(figsize=(12, 10))
        
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
        
        plt.title(f"Subnetwork Feeding into Key Neuron {key_neuron_idx}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/key_neuron_subnetwork.png", dpi=300)
        plt.close()
        
        # Also create correlation bar plot for the layer 1 neurons connected to the key neuron
        # Get final neuron correlations
        final_layer1_corrs = self.neuron_correlations['layer1'][-1]
        
        plt.figure(figsize=(12, 6))
        # Sort by connection strength
        sorted_indices = np.argsort(-np.abs(key_connections))
        top_indices = sorted_indices[:20]  # Top 20 for visualization
        
        bars = plt.bar(range(len(top_indices)), 
                      [final_layer1_corrs[idx] for idx in top_indices], 
                      alpha=0.7)
        
        # Color bars based on the sign of the connection
        for i, idx in enumerate(top_indices):
            if key_connections[idx] < 0:
                bars[i].set_color('red')
            else:
                bars[i].set_color('blue')
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xticks(range(len(top_indices)), [f"{idx}" for idx in top_indices], rotation=90)
        plt.xlabel('Layer 1 Neuron Index')
        plt.ylabel('Correlation with Target')
        plt.title(f'Correlations of Layer 1 Neurons Connected to Key Neuron {key_neuron_idx}')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/key_neuron_input_correlations.png", dpi=300)
        plt.close()
        
        return key_neuron_idx
    
    def progressive_masking_analysis(self, key_neuron_idx=None):
        """
        Perform progressive masking analysis to find the minimal computational subnetwork
        
        Args:
            key_neuron_idx: Optional index of key output neuron (if already identified)
        """
        # If key_neuron_idx not provided, find it
        if key_neuron_idx is None:
            final_a = self.weight_snapshots['a'][-1]
            key_neuron_idx = np.argmax(np.abs(final_a))
        
        print(f"Performing progressive masking analysis focused on neuron {key_neuron_idx}...")
        
        # Get final weights
        final_W1 = self.weight_snapshots['W1'][-1]
        final_W2 = self.weight_snapshots['W2'][-1]
        final_a = self.weight_snapshots['a'][-1]
        
        # 1. Test baseline model performance
        self.model.eval()
        with torch.no_grad():
            baseline_preds = self.model(self.X_test)
            baseline_corr = torch.corrcoef(torch.stack([baseline_preds.squeeze(), self.y_test]))[0, 1].item()
        
        print(f"Baseline model correlation: {baseline_corr:.4f}")
        
        # 2. Test with only the key neuron
        key_neuron_model = copy.deepcopy(self.model)
        
        # Zero out all output weights except for the key neuron
        key_neuron_mask = torch.zeros_like(key_neuron_model.a)
        key_neuron_mask[key_neuron_idx] = 1.0
        key_neuron_model.a.data = key_neuron_model.a.data * key_neuron_mask
        
        # Evaluate
        key_neuron_model.eval()
        with torch.no_grad():
            key_only_preds = key_neuron_model(self.X_test)
            key_only_corr = torch.corrcoef(torch.stack([key_only_preds.squeeze(), self.y_test]))[0, 1].item()
        
        print(f"Key neuron only correlation: {key_only_corr:.4f}")
        
        # 3. Progressive pruning of connections to key neuron
        # Get weights connecting to key neuron
        key_connections = final_W2[key_neuron_idx]
        
        # Sort connections by absolute weight
        sorted_indices = np.argsort(-np.abs(key_connections))
        
        # Test model with progressively more connections
        connection_counts = []
        pruned_correlations = []
        
        # Try different numbers of connections to keep
        for keep_count in [1, 2, 3, 5, 10, 20, 50, 100, key_connections.shape[0]]:
            if keep_count > key_connections.shape[0]:
                keep_count = key_connections.shape[0]
                
            # Create pruned model
            pruned_model = copy.deepcopy(self.model)
            
            # Zero out all output weights except for the key neuron
            key_neuron_mask = torch.zeros_like(pruned_model.a)
            key_neuron_mask[key_neuron_idx] = 1.0
            pruned_model.a.data = pruned_model.a.data * key_neuron_mask
            
            # Zero out all second layer connections except top ones to key neuron
            connection_mask = torch.zeros_like(pruned_model.W2)
            keep_indices = sorted_indices[:keep_count]
            connection_mask[key_neuron_idx, keep_indices] = 1.0
            pruned_model.W2.data = pruned_model.W2.data * connection_mask
            
            # Evaluate
            pruned_model.eval()
            with torch.no_grad():
                pruned_preds = pruned_model(self.X_test)
                pruned_corr = torch.corrcoef(torch.stack([pruned_preds.squeeze(), self.y_test]))[0, 1].item()
            
            connection_counts.append(keep_count)
            pruned_correlations.append(pruned_corr)
            
            print(f"Keeping top {keep_count} connections: Correlation = {pruned_corr:.4f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(connection_counts, pruned_correlations, marker='o')
        plt.axhline(y=baseline_corr, color='r', linestyle='--', label=f'Baseline ({baseline_corr:.4f})')
        plt.axhline(y=0.9 * baseline_corr, color='orange', linestyle='--', label='90% of Baseline')
        plt.xscale('log')
        plt.xlabel('Number of Layer 1 Connections to Key Neuron')
        plt.ylabel('Correlation with Target')
        plt.title('Progressive Pruning of Connections to Key Neuron')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/progressive_pruning_key_neuron.png", dpi=300)
        plt.close()
        
        # 4. Find minimal subnetwork - further prune layer 1
        # Identify which layer 1 neurons were kept in the best performing minimal network
        min_viable_count = 0
        for i, corr in enumerate(pruned_correlations):
            if corr > 0.9 * baseline_corr:  # Keep 90% of performance
                min_viable_count = connection_counts[i]
                break
        
        if min_viable_count == 0:
            min_viable_count = connection_counts[-1]
            
        print(f"Minimal viable subnetwork requires {min_viable_count} connections to key neuron")
        
        # Get the layer 1 neurons in this minimal network
        minimal_l1_neurons = sorted_indices[:min_viable_count]
        
        # Now prune connections to these layer 1 neurons
        # For each layer 1 neuron, try to keep only top connections from input
        l1_pruning_results = []
        for l1_idx in minimal_l1_neurons[:min(10, len(minimal_l1_neurons))]:  # Test at most 10 for efficiency
            l1_weights = final_W1[l1_idx, :self.k]  # Only relevant inputs
            l1_sorted = np.argsort(-np.abs(l1_weights))
            
            keep_counts = []
            l1_correlations = []
            
            # Try keeping different numbers of input connections
            for keep_input_count in [1, 2, 3, self.k]:
                if keep_input_count > self.k:
                    keep_input_count = self.k
                
                # Create pruned model
                l1_pruned_model = copy.deepcopy(self.model)
                
                # First apply same pruning as before to focus on key neuron
                key_neuron_mask = torch.zeros_like(l1_pruned_model.a)
                key_neuron_mask[key_neuron_idx] = 1.0
                l1_pruned_model.a.data = l1_pruned_model.a.data * key_neuron_mask
                
                connection_mask = torch.zeros_like(l1_pruned_model.W2)
                connection_mask[key_neuron_idx, minimal_l1_neurons] = 1.0
                l1_pruned_model.W2.data = l1_pruned_model.W2.data * connection_mask
                
                # Now prune input connections to the specific layer 1 neuron
                input_mask = torch.ones_like(l1_pruned_model.W1)
                keep_inputs = l1_sorted[:keep_input_count]
                
                # Zero out connections except those we want to keep
                for j in range(self.k):
                    if j not in keep_inputs:
                        input_mask[l1_idx, j] = 0.0
                
                l1_pruned_model.W1.data = l1_pruned_model.W1.data * input_mask
                
                # Evaluate
                l1_pruned_model.eval()
                with torch.no_grad():
                    l1_pruned_preds = l1_pruned_model(self.X_test)
                    l1_pruned_corr = torch.corrcoef(torch.stack([l1_pruned_preds.squeeze(), self.y_test]))[0, 1].item()
                
                keep_counts.append(keep_input_count)
                l1_correlations.append(l1_pruned_corr)
            
            l1_pruning_results.append({
                'neuron_idx': l1_idx,
                'keep_counts': keep_counts,
                'correlations': l1_correlations
            })
            
            print(f"Layer 1 Neuron {l1_idx} pruning results:")
            for kc, corr in zip(keep_counts, l1_correlations):
                print(f"  Keeping top {kc} inputs: Correlation = {corr:.4f}")
        
        # Create visualization of the minimal computational network
        minimal_model = copy.deepcopy(self.model)
        
        # Apply pruning based on findings
        # First, only keep connections to key output neuron
        a_mask = torch.zeros_like(minimal_model.a)
        a_mask[key_neuron_idx] = 1.0
        minimal_model.a.data = minimal_model.a.data * a_mask
        
        # Only keep top connections from layer 1 to key neuron
        W2_mask = torch.zeros_like(minimal_model.W2)
        W2_mask[key_neuron_idx, minimal_l1_neurons] = 1.0
        minimal_model.W2.data = minimal_model.W2.data * W2_mask
        
        # For each layer 1 neuron, only keep its strongest input connections
        W1_mask = torch.zeros_like(minimal_model.W1)
        for l1_idx in minimal_l1_neurons:
            l1_weights = final_W1[l1_idx, :self.k]
            top_inputs = np.argsort(-np.abs(l1_weights))[:max(2, self.k//2)]  # Keep at least 2 inputs
            W1_mask[l1_idx, top_inputs] = 1.0
        
        minimal_model.W1.data = minimal_model.W1.data * W1_mask
        
        # Evaluate minimal model
        minimal_model.eval()
        with torch.no_grad():
            minimal_preds = minimal_model(self.X_test)
            minimal_corr = torch.corrcoef(torch.stack([minimal_preds.squeeze(), self.y_test]))[0, 1].item()
        
        print(f"Minimal computational network correlation: {minimal_corr:.4f}")
        
        # Visualize the minimal network
        self._visualize_subnetwork(minimal_model, "minimal_computational_network")
        
        # Compare with weight-based pruning
        weight_model = copy.deepcopy(self.model)
        
        # Keep only weights with magnitude above threshold
        # Adjust thresholds to get roughly same sparsity as minimal model
        W1_percentile = 90  # Top 10% of weights
        W2_percentile = 90
        a_percentile = 90
        
        W1_thresh = np.percentile(np.abs(final_W1), W1_percentile)
        W2_thresh = np.percentile(np.abs(final_W2), W2_percentile)
        a_thresh = np.percentile(np.abs(final_a), a_percentile)
        
        W1_weight_mask = torch.zeros_like(weight_model.W1)
        W1_weight_mask[torch.abs(weight_model.W1) > W1_thresh] = 1.0
        weight_model.W1.data = weight_model.W1.data * W1_weight_mask
        
        W2_weight_mask = torch.zeros_like(weight_model.W2)
        W2_weight_mask[torch.abs(weight_model.W2) > W2_thresh] = 1.0
        weight_model.W2.data = weight_model.W2.data * W2_weight_mask
        
        a_weight_mask = torch.zeros_like(weight_model.a)
        a_weight_mask[torch.abs(weight_model.a) > a_thresh] = 1.0
        weight_model.a.data = weight_model.a.data * a_weight_mask
        
        # Evaluate weight-based pruned model
        weight_model.eval()
        with torch.no_grad():
            weight_preds = weight_model(self.X_test)
            weight_corr = torch.corrcoef(torch.stack([weight_preds.squeeze(), self.y_test]))[0, 1].item()
        
        print(f"Weight-based pruning correlation: {weight_corr:.4f}")
        
        # Visualize weight-based pruned network
        self._visualize_subnetwork(weight_model, "weight_based_pruned_network")
        
        # Compare all pruning methods
        methods = ['Full Model', 'Key Neuron Only', 'Minimal Network', 'Weight-Based Pruning']
        correlations = [baseline_corr, key_only_corr, minimal_corr, weight_corr]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, correlations)
        
        # Color bars
        for i, bar in enumerate(bars):
            bar.set_color(['blue', 'green', 'orange', 'red'][i])
        
        plt.axhline(y=0.9 * baseline_corr, color='gray', linestyle='--', label='90% of Baseline')
        plt.ylabel('Correlation with Target')
        plt.title('Comparison of Pruning Methods')
        plt.grid(True, axis='y')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/pruning_methods_comparison.png", dpi=300)
        plt.close()
        
        # Save the parameter counts
        full_params = count_nonzero_params(self.model)
        key_only_params = count_nonzero_params(key_neuron_model)
        minimal_params = count_nonzero_params(minimal_model)
        weight_params = count_nonzero_params(weight_model)
        
        param_counts = [full_params, key_only_params, minimal_params, weight_params]
        print("\nParameter counts:")
        for method, count in zip(methods, param_counts):
            print(f"{method}: {count} non-zero parameters")
        
        # Plot parameter counts
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, param_counts)
        
        # Color bars
        for i, bar in enumerate(bars):
            bar.set_color(['blue', 'green', 'orange', 'red'][i])
        
        plt.yscale('log')
        plt.ylabel('Number of Non-Zero Parameters')
        plt.title('Parameter Count Comparison')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/parameter_count_comparison.png", dpi=300)
        plt.close()
        
        return {
            'baseline_corr': baseline_corr,
            'key_only_corr': key_only_corr,
            'minimal_corr': minimal_corr,
            'weight_corr': weight_corr,
            'full_params': full_params,
            'key_only_params': key_only_params,
            'minimal_params': minimal_params,
            'weight_params': weight_params
        }
    
    def _visualize_subnetwork(self, model, name):
        """
        Visualize a subnetwork model, highlighting only non-zero connections
        
        Args:
            model: The model to visualize
            name: Name for the output file
        """
        # Extract weights
        W1 = model.W1.detach().cpu().numpy()
        W2 = model.W2.detach().cpu().numpy()
        a = model.a.detach().cpu().numpy()
        
        # Create graph
        G = nx.DiGraph()
        
        # Add input nodes (only relevant features)
        for i in range(self.k):
            G.add_node(f"x{i}", layer=0, pos=(i, 0))
        
        # Find active layer 1 neurons (those with any non-zero incoming weights)
        active_l1 = []
        for i in range(self.M1):
            if np.any(W1[i, :self.k] != 0):
                active_l1.append(i)
                G.add_node(f"h1_{i}", layer=1, pos=(i % self.k, 1))
                
                # Add connections from inputs
                for j in range(self.k):
                    if W1[i, j] != 0:
                        G.add_edge(f"x{j}", f"h1_{i}", weight=W1[i, j])
        
        # Find active layer 2 neurons (those with any non-zero incoming weights)
        active_l2 = []
        for i in range(self.M2):
            if np.any(W2[i, active_l1] != 0) and a[i] != 0:
                active_l2.append(i)
                G.add_node(f"h2_{i}", layer=2, pos=(i % self.k, 2))
                
                # Add connections from layer 1
                for j in active_l1:
                    if W2[i, j] != 0:
                        G.add_edge(f"h1_{j}", f"h2_{i}", weight=W2[i, j])
        
        # Add output node
        G.add_node("y", layer=3, pos=(self.k//2, 3))
        
        # Add connections from layer 2 to output
        for i in active_l2:
            if a[i] != 0:
                G.add_edge(f"h2_{i}", "y", weight=a[i])
        
        # Set up positions for drawing
        pos = nx.get_node_attributes(G, 'pos')
        
        # Adjust positions for better visualization
        for node, position in pos.items():
            x, y = position
            layer = G.nodes[node]['layer']
            
            # Spread out nodes in each layer
            if layer == 1:  # First hidden layer
                x = x / self.k * (10 + len(active_l1) / 10)
            elif layer == 2:  # Second hidden layer
                x = x / self.k * (10 + len(active_l2) / 10)
            
            pos[node] = (x, y * 3)  # Increase vertical spacing
        
        # Plot
        plt.figure(figsize=(12, 10))
        
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
        
        # Calculate non-zero parameters
        nonzero_params = count_nonzero_params(model)
        
        plt.title(f"{name.replace('_', ' ').title()} ({nonzero_params} non-zero parameters)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{name}.png", dpi=300)
        plt.close()
    
    def parallel_correlation_analysis(self, n_samples=500):
        """
        Perform correlation analysis using parallel processing for efficiency
        
        Args:
            n_samples: Number of samples to use for correlation analysis
        """
        print("Performing parallel correlation analysis...")
        
        # Generate data
        with torch.no_grad():
            X = torch.tensor(np.random.choice([-1, 1], size=(n_samples, self.d)), 
                           dtype=torch.float32).to(self.device)
            y_true = self._target_function(X)
            
            # Get activations
            h1 = self.model.get_first_layer_activations(X).cpu().numpy()
            h2 = self.model.get_second_layer_activations(X).cpu().numpy()
            y_true = y_true.cpu().numpy()
        
        # Function to compute correlations in parallel
        def compute_correlations(neuron_activations, target):
            return np.corrcoef(neuron_activations, target)[0, 1]
        
        # Compute layer 1 correlations in parallel
        layer1_corrs = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for j in range(self.M1):
                futures.append(executor.submit(compute_correlations, h1[:, j], y_true))
            
            for future in as_completed(futures):
                layer1_corrs.append(future.result())
        
        # Compute layer 2 correlations in parallel
        layer2_corrs = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for j in range(self.M2):
                futures.append(executor.submit(compute_correlations, h2[:, j], y_true))
            
            for future in as_completed(futures):
                layer2_corrs.append(future.result())
        
        # Plot histograms of correlations
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.hist(layer1_corrs, bins=50, alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.xlabel('Correlation with Target')
        plt.ylabel('Count')
        plt.title('Layer 1 Neuron Correlations')
        
        plt.subplot(2, 1, 2)
        plt.hist(layer2_corrs, bins=50, alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.xlabel('Correlation with Target')
        plt.ylabel('Count')
        plt.title('Layer 2 Neuron Correlations')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/parallel_correlation_analysis.png", dpi=300)
        plt.close()
        
        # Identify top correlating neurons
        top_l1_indices = np.argsort(-np.abs(layer1_corrs))[:10]
        top_l2_indices = np.argsort(-np.abs(layer2_corrs))[:10]
        
        print("\nTop correlating layer 1 neurons:")
        for idx in top_l1_indices:
            print(f"Neuron {idx}: Correlation = {layer1_corrs[idx]:.4f}")
        
        print("\nTop correlating layer 2 neurons:")
        for idx in top_l2_indices:
            print(f"Neuron {idx}: Correlation = {layer2_corrs[idx]:.4f}")
        
        return {
            'layer1_corrs': layer1_corrs,
            'layer2_corrs': layer2_corrs,
            'top_l1_indices': top_l1_indices,
            'top_l2_indices': top_l2_indices
        }
    
    def run_analysis(self, n_epochs=5000, snapshot_interval=1000, corr_interval=100):
        """
        Run a complete analysis of a neural network learning parity
        
        Args:
            n_epochs: Maximum number of epochs
            snapshot_interval: Interval for taking weight snapshots
            corr_interval: Interval for computing neuron correlations
        """
        print("\n========== PARITY NEURAL NETWORK ANALYSIS ==========")
        
        # 1. Train the network
        print("\n----- Training Network -----")
        self.train(n_epochs=n_epochs, snapshot_interval=snapshot_interval, corr_interval=corr_interval)
        
        # 2. Visualize weight matrices
        print("\n----- Visualizing Weight Matrices -----")
        self.visualize_weight_matrices()
        
        # 3. Visualize neuron correlations
        print("\n----- Visualizing Neuron Correlations -----")
        self.visualize_neuron_correlations()
        
        # 4. Analyze key output neuron
        print("\n----- Analyzing Key Output Neuron -----")
        key_neuron_idx = self.analyze_key_output_neuron()
        
        # 5. Progressive masking analysis
        print("\n----- Performing Progressive Masking Analysis -----")
        pruning_results = self.progressive_masking_analysis(key_neuron_idx)
        
        # 6. Parallel correlation analysis
        print("\n----- Performing Parallel Correlation Analysis -----")
        correlation_results = self.parallel_correlation_analysis()
        
        print("\n========== ANALYSIS COMPLETE ==========")
        
        return {
            'pruning_results': pruning_results,
            'correlation_results': correlation_results
        }


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


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze neural networks learning parity functions')
    parser.add_argument('--d', type=int, default=45, help='Input dimension')
    parser.add_argument('--k', type=int, default=6, help='Parity function order')
    parser.add_argument('--M1', type=int, default=128, help='First layer width')
    parser.add_argument('--M2', type=int, default=128, help='Second layer width')
    parser.add_argument('--epochs', type=int, default=20000, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='parity_analysis_results_even', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create analyzer with k-parity in d dimensions
    analyzer = ParityNetworkAnalyzer(
        d=args.d, 
        k=args.k, 
        M1=args.M1, 
        M2=args.M2,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    # Run complete analysis
    analyzer.run_analysis(n_epochs=args.epochs)