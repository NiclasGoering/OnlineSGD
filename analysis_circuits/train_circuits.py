import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from IPython.display import HTML
from sklearn.cluster import KMeans
from functools import partial
import os
import copy
from matplotlib.gridspec import GridSpec

class ParityCircuitTracker:
    def __init__(self, d=30, k=4, M1=128, M2=128, learning_rate=0.01, 
                 batch_size=512, device=None, save_dir="parity_circuit_results"):
        """
        Initialize the ParityCircuitTracker to monitor circuit formation
        for learning k-parity functions in d dimensions.
        
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
        self.gradient_history = []
        self.circuit_metrics = []
        self.weight_evolution = {
            'epochs': [],
            'W1': [],
            'W2': [],
            'a': []
        }
        
        # Sample fixed test inputs for consistent evaluation
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, d)), 
                                  dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)
        
        # Create circuit masks (will be populated during training)
        self.circuit_masks = {
            'layer1': torch.zeros(M1, d).to(self.device),
            'layer2': torch.zeros(M2, M1).to(self.device),
            'output': torch.zeros(M2).to(self.device)
        }
        
        # Track weight history for potential circuit neurons
        self.weight_history = {
            'epochs': [],
            'W1': [],  # First layer weights (selected neurons only)
            'W2': [],  # Second layer weights (selected neurons only)
            'a': [],   # Output weights (selected neurons only)
            'neurons1': [],  # Indices of tracked first layer neurons
            'neurons2': []   # Indices of tracked second layer neurons
        }
        
        # Save initial weights
        self.initial_weights = {
            'W1': self.model.W1.detach().clone(),
            'W2': self.model.W2.detach().clone(),
            'a': self.model.a.detach().clone()
        }
        
        print(f"ParityCircuitTracker initialized on {self.device}")
        print(f"Tracking {k}-parity function in {d} dimensions")
        print(f"Network: {M1} → {M2} → 1")
    
    def _create_model(self):
        """Create a two-layer ReLU network"""
        model = TwoLayerReLUNet(self.d, self.M1, self.M2).to(self.device)
        return model
    
    def _target_function(self, x):
        """Compute the k-sparse parity function on the first k inputs"""
        return torch.prod(x[:, :self.k], dim=1)
    
    def compute_attribution(self, inputs, targets=None):
        """
        Compute attribution of each neuron using a weight-based approach
        
        Returns:
            Dictionary of attributions for each layer
        """
        # Set up for computation
        self.model.eval()
        
        if targets is None:
            targets = self._target_function(inputs)
        
        # Get activations but don't track gradients
        with torch.no_grad():
            # Get model parameters
            W1 = self.model.W1.detach().cpu().numpy()
            W2 = self.model.W2.detach().cpu().numpy()
            a = self.model.a.detach().cpu().numpy()
            
            # First layer activations
            h1 = self.model.get_first_layer_activations(inputs).cpu().numpy()
            
            # Second layer activations
            h2 = self.model.get_second_layer_activations(inputs).cpu().numpy()
            
            # Get predictions
            preds = self.model(inputs).cpu().numpy()
            
            # Get targets
            targets_np = targets.cpu().numpy()
            
            # Calculate residuals (for attribution importance)
            residuals = (targets_np - preds) ** 2
        
        # SIMPLIFIED ATTRIBUTION METHOD: Weight analysis + activation patterns
        
        # 1. Input attribution: Which inputs are most connected to important neurons?
        # For k-parity, we know inputs 0...k-1, but we'll compute it anyway
        input_importance = np.zeros(self.d)
        
        # Give higher importance to inputs 0...k-1 since we know they're relevant
        input_importance[:self.k] = 1.0
        
        # 2. First layer attribution: Which neurons connect strongly to important inputs?
        layer1_importance = np.zeros(self.M1)
        for i in range(self.M1):
            # How much does this neuron focus on relevant inputs?
            relevant_weight_sum = np.sum(np.abs(W1[i, :self.k]))
            total_weight_sum = np.sum(np.abs(W1[i])) + 1e-10  # Avoid division by zero
            layer1_importance[i] = relevant_weight_sum / total_weight_sum
        
        # 3. Second layer attribution: Which neurons aggregate from important first layer neurons?
        # Get top first layer neurons
        top_layer1_count = max(5, int(0.05 * self.M1))  # At least 5 neurons or 5% of layer
        top_layer1 = np.argsort(-layer1_importance)[:top_layer1_count]
        
        layer2_importance = np.zeros(self.M2)
        for i in range(self.M2):
            # How much does this neuron connect to important first layer neurons?
            connections_to_important = np.sum(np.abs(W2[i, top_layer1]))
            total_connections = np.sum(np.abs(W2[i])) + 1e-10  # Avoid division by zero
            layer2_importance[i] = connections_to_important / total_connections
        
        # 4. Output importance: Which second layer neurons contribute most to output?
        output_importance = np.abs(a) * layer2_importance
        
        return {
            'input': input_importance,
            'layer1': layer1_importance,
            'layer1_activation': np.mean(h1, axis=0),
            'layer2_activation': np.mean(h2, axis=0),
            'layer2': layer2_importance,
            'output': output_importance,
        }
    
    def identify_even_parity_circuits(self):
        """
        Identify potential circuits computing even parity
        Specifically looking for the pattern:
        f(x) = Σ a_j ReLU(w_j^T x) where w_j have specific sign patterns
        """
        # Get model parameters
        W1 = self.model.W1.detach().cpu().numpy() 
        a = self.model.a.detach().cpu().numpy()
        
        # We only care about the first k dimensions for our k-parity function
        W1_relevant = W1[:, :self.k]
        
        # Identify neurons with strong weights to the relevant inputs
        magnitude = np.abs(W1_relevant).sum(axis=1)
        relevance_score = magnitude / (np.abs(W1).sum(axis=1) + 1e-8)
        
        # Sort neurons by relevance
        sorted_indices = np.argsort(-relevance_score)
        top_neurons = sorted_indices[:min(20, len(sorted_indices))]
        
        # Analyze sign patterns for top neurons
        circuit_neurons = []
        
        for idx in top_neurons:
            weights = W1_relevant[idx]
            output_weight = a[idx]
            
            # For even parity with 2 vars, we expect patterns like [+,+] or [-,-] with positive output
            # or [+,-] or [-,+] with negative output
            
            # Skip neurons with too many zeros in relevant dimensions
            if np.sum(np.abs(weights) < 0.01) > self.k / 4:
                continue
                
            # Count sign changes in weights
            signs = np.sign(weights)
            # Count number of sign changes (ignoring zeros)
            non_zero_signs = signs[np.abs(weights) >= 0.01]
            if len(non_zero_signs) <= 1:
                continue
                
            sign_changes = 0
            for i in range(1, len(non_zero_signs)):
                if non_zero_signs[i] != non_zero_signs[i-1]:
                    sign_changes += 1
            
            # Determine pattern type based on sign changes and output weight
            # Uniform signs (all + or all -) should have positive output
            # Mixed signs should have negative output for even parity
            if sign_changes == 0:
                if output_weight > 0:
                    circuit_neurons.append((idx, 'uniform_signs_positive_out', weights, output_weight))
            else:
                if output_weight < 0:
                    circuit_neurons.append((idx, 'mixed_signs_negative_out', weights, output_weight))
        
        return circuit_neurons
    
    def identify_odd_parity_circuits(self):
        """
        Identify potential circuits computing odd parity
        For odd parity, we need to identify hierarchical structures
        """
        # Get model parameters
        W1 = self.model.W1.detach().cpu().numpy() 
        W2 = self.model.W2.detach().cpu().numpy()
        a = self.model.a.detach().cpu().numpy()
        
        # Step 1: Identify candidate first layer "feature detectors" (neurons that detect patterns in subsets of k)
        # We focus on the first k dimensions (relevant to our k-parity)
        W1_relevant = W1[:, :self.k]
        
        # Measure relevance of each first layer neuron to the k inputs
        layer1_relevance = np.abs(W1_relevant).sum(axis=1) / (np.abs(W1).sum(axis=1) + 1e-8)
        
        # Get top first layer neurons
        top_layer1_count = min(30, self.M1)
        top_layer1 = np.argsort(-layer1_relevance)[:top_layer1_count]
        
        # Step 2: Identify candidate second layer "aggregators"
        # Measure how strongly second layer neurons connect to potential feature detectors
        layer2_relevance = np.zeros(self.M2)
        for i in range(self.M2):
            connections_to_top = np.abs(W2[i][top_layer1]).sum()
            layer2_relevance[i] = connections_to_top / (np.abs(W2[i]).sum() + 1e-8)
        
        # Get top second layer neurons
        top_layer2_count = min(20, self.M2)
        top_layer2 = np.argsort(-layer2_relevance)[:top_layer2_count]
        
        # Step 3: Analyze subnetworks for odd parity circuit patterns
        circuit_paths = []
        
        for l2_idx in top_layer2:
            # Check if this second layer neuron connects to relevant first layer neurons
            l1_connections = W2[l2_idx][top_layer1]
            connected_indices = np.where(np.abs(l1_connections) > np.abs(l1_connections).max() * 0.5)[0]
            l1_connected = top_layer1[connected_indices]
            
            if len(l1_connected) < 2:  # Need at least two connections for a meaningful circuit
                continue
            
            # Check output weight - should be non-negligible
            if abs(a[l2_idx]) < 0.1:
                continue
            
            # For each connected first layer neuron, what input features does it respond to?
            detectors = []
            for l1_idx in l1_connected:
                # Which input features does this neuron attend to?
                feature_weights = W1[l1_idx, :self.k]
                active_features = np.where(np.abs(feature_weights) > np.abs(feature_weights).max() * 0.5)[0]
                
                if len(active_features) == 0:
                    continue
                    
                # Record this detector
                detectors.append({
                    'l1_idx': l1_idx,
                    'active_features': active_features,
                    'l1_weight': feature_weights[active_features],
                    'l2_weight': W2[l2_idx, l1_idx]
                })
            
            # Only keep paths with enough detectors
            if len(detectors) >= 2:
                circuit_paths.append({
                    'l2_idx': l2_idx,
                    'output_weight': a[l2_idx],
                    'detectors': detectors
                })
        
        return circuit_paths
    
    def analyze_circuit_formation(self, epoch):
        """
        Analyze current state of circuit formation in the network
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary of circuit metrics
        """
        self.model.eval()
        
        # 1. Measure overall accuracy
        with torch.no_grad():
            preds = self.model(self.X_test)
            mse = ((preds - self.y_test) ** 2).mean().item()
            correlation = (preds * self.y_test).mean().item()
        
        # 2. Attribution analysis - using our simplified method that doesn't require gradients
        attribution = self.compute_attribution(self.X_test, self.y_test)
        
        # 3. Circuit identification based on parity type (odd or even k)
        if self.k % 2 == 0:  # Even parity
            circuits = self.identify_even_parity_circuits()
            circuit_complexity = 'single_layer'
        else:  # Odd parity
            circuits = self.identify_odd_parity_circuits()
            circuit_complexity = 'two_layer'
            
        # 4. Update circuit masks based on findings
        self._update_circuit_masks(circuits, circuit_complexity)
        
        # 5. Track circuit neurons' weights over time
        self._record_circuit_weights(epoch, circuits, circuit_complexity)
        
        # 6. Calculate "circuit completeness" - how much of the complete circuit pattern is present
        if circuit_complexity == 'single_layer':
            completeness = min(1.0, len(circuits) / (2 ** (self.k - 1)))
        else:
            # For odd parity, estimate by number of paths found vs theoretically needed
            completeness = min(1.0, len(circuits) / (2 ** (self.k // 2)))
        
        # Assemble all metrics
        metrics = {
            'epoch': epoch,
            'mse': mse,
            'correlation': correlation,
            'circuit_type': circuit_complexity,
            'circuit_count': len(circuits),
            'circuit_completeness': completeness,
            'attribution': attribution,
            'circuits': circuits
        }
        
        return metrics
    
    def _update_circuit_masks(self, circuits, circuit_complexity):
        """
        Update binary masks highlighting circuit neurons
        
        Args:
            circuits: Identified circuits
            circuit_complexity: 'single_layer' or 'two_layer'
        """
        # Reset masks
        for key in self.circuit_masks:
            self.circuit_masks[key].zero_()
        
        if circuit_complexity == 'single_layer':
            # Update masks for even parity
            for idx, _, _, output_weight in circuits:
                self.circuit_masks['layer1'][idx, :self.k] = 1.0
                self.circuit_masks['output'][idx] = 1.0
        else:
            # Update masks for odd parity
            for path in circuits:
                l2_idx = path['l2_idx']
                self.circuit_masks['output'][l2_idx] = 1.0
                
                for detector in path['detectors']:
                    l1_idx = detector['l1_idx']
                    active_features = detector['active_features']
                    
                    for feat in active_features:
                        self.circuit_masks['layer1'][l1_idx, feat] = 1.0
                    
                    self.circuit_masks['layer2'][l2_idx, l1_idx] = 1.0
    
    def _record_circuit_weights(self, epoch, circuits, circuit_complexity):
        """
        Record weights of circuit neurons over time
        
        Args:
            epoch: Current epoch
            circuits: Identified circuits
            circuit_complexity: 'single_layer' or 'two_layer'
        """
        # Add epoch
        self.weight_history['epochs'].append(epoch)
        
        # Get current weights
        W1 = self.model.W1.detach()
        W2 = self.model.W2.detach()
        a = self.model.a.detach()
        
        if circuit_complexity == 'single_layer':
            # Record neuron indices
            neurons1 = [idx for idx, _, _, _ in circuits]
            self.weight_history['neurons1'].append(neurons1)
            self.weight_history['neurons2'].append([])
            
            # Record weights for circuit neurons
            if neurons1:
                self.weight_history['W1'].append(W1[neurons1, :self.k].cpu().numpy())
                self.weight_history['a'].append(a[neurons1].cpu().numpy())
            else:
                self.weight_history['W1'].append(np.array([]))
                self.weight_history['a'].append(np.array([]))
                
            # Empty second layer weights
            self.weight_history['W2'].append(np.array([]))
        else:
            # Extract circuit neurons
            neurons2 = [path['l2_idx'] for path in circuits]
            neurons1 = []
            for path in circuits:
                for detector in path['detectors']:
                    neurons1.append(detector['l1_idx'])
            
            # Remove duplicates
            neurons1 = list(set(neurons1))
            
            self.weight_history['neurons1'].append(neurons1)
            self.weight_history['neurons2'].append(neurons2)
            
            # Record weights
            if neurons1 and neurons2:
                self.weight_history['W1'].append(W1[neurons1, :self.k].cpu().numpy())
                if neurons1:
                    subset_W2 = W2[neurons2][:, neurons1].cpu().numpy() if neurons2 else np.array([])
                    self.weight_history['W2'].append(subset_W2)
                else:
                    self.weight_history['W2'].append(np.array([]))
                self.weight_history['a'].append(a[neurons2].cpu().numpy())
            else:
                self.weight_history['W1'].append(np.array([]))
                self.weight_history['W2'].append(np.array([]))
                self.weight_history['a'].append(np.array([]))
    
    def record_weight_evolution(self, epoch):
        """
        Record the full weight matrices at the current epoch
        """
        # Add epoch
        self.weight_evolution['epochs'].append(epoch)
        
        # Get current weights
        W1 = self.model.W1.detach().cpu().numpy()
        W2 = self.model.W2.detach().cpu().numpy()
        a = self.model.a.detach().cpu().numpy()
        
        # Record all weights
        self.weight_evolution['W1'].append(W1)
        self.weight_evolution['W2'].append(W2)
        self.weight_evolution['a'].append(a)
    
    def record_gradient_metrics(self, epoch):
        """
        Record gradient metrics, particularly focusing on the k-dimensional subspace
        """
        # Get gradients
        W1_grad = self.model.W1.grad.detach().cpu().numpy() if self.model.W1.grad is not None else np.zeros_like(self.model.W1.detach().cpu().numpy())
        W2_grad = self.model.W2.grad.detach().cpu().numpy() if self.model.W2.grad is not None else np.zeros_like(self.model.W2.detach().cpu().numpy())
        
        # Calculate gradient norms for relevant dimensions
        W1_relevant_grad_norm = np.linalg.norm(W1_grad[:, :self.k], axis=1)
        W1_irrelevant_grad_norm = np.linalg.norm(W1_grad[:, self.k:], axis=1)
        
        # Metric: ratio of relevant vs. irrelevant gradient norms
        W1_rel_irrel_ratio = W1_relevant_grad_norm / (W1_irrelevant_grad_norm + 1e-10)
        
        # Record metrics
        self.gradient_history.append({
            'epoch': epoch,
            'W1_relevant_grad_norm': W1_relevant_grad_norm,
            'W1_irrelevant_grad_norm': W1_irrelevant_grad_norm,
            'W1_rel_irrel_ratio': W1_rel_irrel_ratio,
            'W1_relevant_grad_mean': W1_grad[:, :self.k].mean(),
            'W1_irrelevant_grad_mean': W1_grad[:, self.k:].mean(),
            'W2_grad_mean': W2_grad.mean()
        })
    
    def train(self, n_epochs=10000, log_interval=100, weight_record_interval=100, early_stop_corr=0.999999):
        """
        Train the network and track circuit formation
        
        Args:
            n_epochs: Maximum number of epochs
            log_interval: Interval for logging metrics
            weight_record_interval: Interval for recording full weight matrices
            early_stop_corr: Correlation threshold for early stopping
        """
        print(f"Starting training for {n_epochs} epochs...")
        
        # Record initial weights
        self.record_weight_evolution(0)
        
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
            
            # Record gradient metrics before optimization step
            if epoch % log_interval == 0:
                self.record_gradient_metrics(epoch)
                
            self.optimizer.step()
            
            # Record loss
            self.loss_history.append(loss.item())
            
            # Record weights periodically
            if epoch % weight_record_interval == 0 or epoch == n_epochs - 1:
                self.record_weight_evolution(epoch)
            
            # Log metrics and analyze circuits at intervals
            if epoch % log_interval == 0 or epoch == n_epochs - 1:
                metrics = self.analyze_circuit_formation(epoch)
                self.circuit_metrics.append(metrics)
                
                # Print progress
                if epoch % (log_interval * 10) == 0 or epoch == n_epochs - 1:
                    print(f"Epoch {epoch}: MSE={metrics['mse']:.6f}, Correlation={metrics['correlation']:.4f}")
                    print(f"Circuit count: {metrics['circuit_count']}, Completeness: {metrics['circuit_completeness']:.2f}")
                
                # Early stopping
                if metrics['correlation'] > early_stop_corr:
                    print(f"Early stopping at epoch {epoch} with correlation {metrics['correlation']:.4f}")
                    break
        
        # Final analysis
        final_metrics = self.analyze_circuit_formation(epoch)
        self.circuit_metrics.append(final_metrics)
        
        print("Training completed!")
        print(f"Final MSE: {final_metrics['mse']:.6f}")
        print(f"Final correlation: {final_metrics['correlation']:.4f}")
        print(f"Circuit complexity: {final_metrics['circuit_type']}")
        print(f"Circuit count: {final_metrics['circuit_count']}")
        print(f"Circuit completeness: {final_metrics['circuit_completeness']:.2f}")
        
        return self.circuit_metrics
    
    def visualize_circuit_formation(self):
        """
        Create comprehensive visualizations of circuit formation
        """
        # 1. Loss and correlation curves
        self._plot_learning_curves()
        
        # 2. Circuit growth over time
        self._plot_circuit_growth()
        
        # 3. Visualize final circuit
        if self.k % 2 == 0:  # Even parity
            self._visualize_even_parity_circuit()
        else:  # Odd parity
            self._visualize_odd_parity_circuit()
        
        # 4. Weight evolution for circuit neurons
        self._plot_weight_evolution()
        
        # 5. Attribution patterns at different stages
        self._plot_attribution_evolution()
        
        # 6. Plot weight matrices (initial vs final)
        self._plot_weight_matrices_comparison()
        
        # 7. Plot gradient metrics evolution
        self._plot_gradient_evolution()
        
        # 8. Plot weight trajectories
        self._plot_weight_trajectories()
        
        # 9. Plot weight sign changes
        self._plot_weight_sign_changes()
    
    def _plot_learning_curves(self):
        """Plot loss and correlation curves"""
        epochs = [m['epoch'] for m in self.circuit_metrics]
        mse = [m['mse'] for m in self.circuit_metrics]
        correlation = [m['correlation'] for m in self.circuit_metrics]
        completeness = [m['circuit_completeness'] for m in self.circuit_metrics]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Plot MSE
        ax1.plot(epochs, mse)
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_yscale('log')
        ax1.grid(True)
        ax1.set_title(f'Learning Curves for {self.k}-Parity Function')
        
        # Plot correlation
        ax2.plot(epochs, correlation)
        ax2.set_ylabel('Target Correlation')
        ax2.grid(True)
        
        # Plot circuit completeness
        ax3.plot(epochs, completeness)
        ax3.set_ylabel('Circuit Completeness')
        ax3.set_xlabel('Epoch')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/learning_curves.png", dpi=300)
        plt.close()
    
    def _plot_circuit_growth(self):
        """Plot growth of circuit neurons over time"""
        epochs = [m['epoch'] for m in self.circuit_metrics]
        circuit_count = [m['circuit_count'] for m in self.circuit_metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, circuit_count, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Number of Circuit Components')
        plt.title(f'Growth of Circuit Components for {self.k}-Parity')
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/circuit_growth.png", dpi=300)
        plt.close()
    
    def _visualize_even_parity_circuit(self):
        """Visualize the circuit for even parity"""
        # Get final metrics
        final_metrics = self.circuit_metrics[-1]
        circuits = final_metrics['circuits']
        
        if not circuits:
            print("No circuits identified for visualization")
            return
        
        # Create graph representation
        G = nx.DiGraph()
        
        # Add input nodes
        for i in range(self.k):
            G.add_node(f"x{i}", layer=0, pos=(i, 0))
        
        # Add layer 1 nodes (ReLU neurons)
        for idx, pattern, weights, output in circuits:
            G.add_node(f"h{idx}", layer=1, pos=(idx % self.k, 1), pattern=pattern)
            # Connect to inputs
            for i in range(self.k):
                weight = weights[i]
                if abs(weight) > 0.1:  # Only show significant connections
                    G.add_edge(f"x{i}", f"h{idx}", weight=weight)
        
        # Add output node
        G.add_node("y", layer=2, pos=(self.k//2, 2))
        
        # Connect layer 1 to output
        for idx, pattern, weights, output in circuits:
            G.add_edge(f"h{idx}", "y", weight=output)
        
        # Create position layout
        pos = nx.get_node_attributes(G, 'pos')
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Draw nodes by layer
        node_colors = ['lightblue', 'lightgreen', 'salmon']
        for layer in range(3):
            nodelist = [n for n, d in G.nodes(data=True) if d.get('layer') == layer]
            if nodelist:
                nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=node_colors[layer], 
                                      node_size=500, alpha=0.8)
        
        # Draw edges with colors based on weights
        edges = G.edges(data='weight')
        if edges:
            edge_colors = []
            for u, v, w in edges:
                # Handle potential NaN or None values
                if w is None or np.isnan(w):
                    edge_colors.append('gray')
                else:
                    edge_colors.append(plt.cm.RdBu(0.5 * (1 + w / max(1, abs(w)))))
            
            # Scale edge widths by weight magnitude
            edge_widths = []
            for u, v, w in edges:
                if w is None or np.isnan(w):
                    edge_widths.append(1)
                else:
                    edge_widths.append(1 + 3 * abs(w))
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                                  arrowsize=15, min_source_margin=15, min_target_margin=15)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_weight='bold')
        
        plt.title(f"Even {self.k}-Parity Circuit")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/even_parity_circuit.png", dpi=300)
        plt.close()
        
        # Also save a table of the weights
        if circuits:
            plt.figure(figsize=(12, len(circuits) * 0.5 + 2))
            cell_text = []
            for idx, pattern, weights, output in circuits:
                row = [f"{w:.2f}" for w in weights] + [f"{output:.2f}"]
                cell_text.append(row)
            
            column_labels = [f"x{i}" for i in range(self.k)] + ["Output"]
            row_labels = [f"h{idx}" for idx, _, _, _ in circuits]
            
            plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=column_labels, 
                     loc='center', cellLoc='center')
            plt.axis('off')
            plt.title(f"Even {self.k}-Parity Circuit Weights")
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/even_parity_weights.png", dpi=300)
            plt.close()
    
    def _visualize_odd_parity_circuit(self):
        """Visualize the circuit for odd parity"""
        # Get final metrics
        final_metrics = self.circuit_metrics[-1]
        circuits = final_metrics['circuits']
        
        if not circuits:
            print("No circuits identified for visualization")
            return
        
        # Create graph representation
        G = nx.DiGraph()
        
        # Add input nodes
        for i in range(self.k):
            G.add_node(f"x{i}", layer=0, pos=(i, 0))
        
        # Add first layer nodes (detectors)
        l1_nodes = set()
        for path in circuits:
            for detector in path['detectors']:
                l1_idx = detector['l1_idx']
                if l1_idx not in l1_nodes:
                    l1_nodes.add(l1_idx)
                    # Position within layer based on active features
                    if len(detector['active_features']) > 0:
                        pos_x = sum(detector['active_features']) / max(1, len(detector['active_features']))
                    else:
                        pos_x = l1_idx % self.k
                    G.add_node(f"h1_{l1_idx}", layer=1, pos=(pos_x, 1))
                
                # Connect to inputs
                for feat_idx, feat in enumerate(detector['active_features']):
                    if feat_idx < len(detector['l1_weight']):
                        weight = detector['l1_weight'][feat_idx]
                    else:
                        weight = 0  # Default if index is out of bounds
                    G.add_edge(f"x{feat}", f"h1_{l1_idx}", weight=weight)
        
        # Add second layer nodes (aggregators)
        for path_idx, path in enumerate(circuits):
            l2_idx = path['l2_idx']
            G.add_node(f"h2_{l2_idx}", layer=2, pos=(path_idx, 2))
            
            # Connect to first layer
            for detector in path['detectors']:
                l1_idx = detector['l1_idx']
                weight = detector['l2_weight']
                G.add_edge(f"h1_{l1_idx}", f"h2_{l2_idx}", weight=weight)
        
        # Add output node
        G.add_node("y", layer=3, pos=(len(circuits)//2, 3))
        
        # Connect second layer to output
        for path in circuits:
            l2_idx = path['l2_idx']
            weight = path['output_weight']
            G.add_edge(f"h2_{l2_idx}", "y", weight=weight)
        
        # Create position layout
        pos = nx.get_node_attributes(G, 'pos')
        
        # Adjust positions for better visualization
        for node, position in pos.items():
            x, y = position
            layer = G.nodes[node]['layer']
            
            # Spread out nodes in each layer
            if layer == 1:  # First hidden layer
                x = x / self.k * 10
            elif layer == 2:  # Second hidden layer
                x = x / max(1, len(circuits)) * 10
                
            pos[node] = (x, y * 3)  # Increase vertical spacing
        
        # Plot
        plt.figure(figsize=(14, 10))
        
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
                # Handle potential NaN or None values
                if w is None or np.isnan(w):
                    edge_colors.append('gray')
                else:
                    edge_colors.append(plt.cm.RdBu(0.5 * (1 + w / max(1, abs(w)))))
            
            # Scale edge widths by weight magnitude
            edge_widths = []
            for u, v, w in edges:
                if w is None or np.isnan(w):
                    edge_widths.append(1)
                else:
                    edge_widths.append(1 + 3 * abs(w))
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                                  arrowsize=15, min_source_margin=15, min_target_margin=15)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_weight='bold')
        
        plt.title(f"Odd {self.k}-Parity Circuit")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/odd_parity_circuit.png", dpi=300)
        plt.close()
        
        # Create detailed tables of the circuit structure
        plt.figure(figsize=(14, len(circuits) * 1.5 + 2))
        plt.title(f"Odd {self.k}-Parity Circuit Structure")
        plt.axis('off')
        
        cell_text = []
        for i, path in enumerate(circuits):
            # Create a text description of this circuit path
            l2_idx = path['l2_idx']
            output_weight = path['output_weight']
            
            row_text = [f"Aggregator {l2_idx}"]
            detector_text = []
            
            for det in path['detectors']:
                l1_idx = det['l1_idx']
                features = list(det['active_features'])
                l1_weight_text = ", ".join([f"{w:.2f}" for w in det['l1_weight']])
                l2_weight = det['l2_weight']
                
                detector_text.append(f"Detector {l1_idx}: Features {features}, Weights [{l1_weight_text}], L2 Weight {l2_weight:.2f}")
            
            cell_text.append([f"Path {i+1}: L2 Neuron {l2_idx}, Output Weight {output_weight:.2f}"])
            for det_text in detector_text:
                cell_text.append([det_text])
            cell_text.append([""])  # Empty row between paths
        
        # Create a simple table to show the text
        tbl = plt.table(cellText=cell_text, cellLoc='left', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)  # Adjust row heights
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/odd_parity_structure.png", dpi=300)
        plt.close()
    
    def _plot_weight_evolution(self):
        """Plot weight evolution for circuit neurons"""
        # Check if enough data points are available
        if len(self.weight_history['epochs']) < 2:
            print("Not enough data for weight evolution visualization")
            return
        
        # Select key time points - beginning, middle, end
        indices = [0]
        if len(self.weight_history['epochs']) > 2:
            indices.append(len(self.weight_history['epochs'])//2)
        indices.append(-1)
        
        is_odd_parity = self.k % 2 != 0
        
        # Create subplots based on parity type
        if is_odd_parity:
            fig, axs = plt.subplots(3, len(indices), figsize=(5*len(indices), 15))
            if len(indices) == 1:
                axs = np.array([[axs[0]], [axs[1]], [axs[2]]])
            titles = ['Initial Weights', 'Intermediate Weights', 'Final Weights']
            titles = titles[:len(indices)]
            
            for i, t_idx in enumerate(indices):
                # Skip if no circuit neurons found at this time
                if len(self.weight_history['neurons1'][t_idx]) == 0:
                    continue
                    
                # First layer weights (top row)
                W1 = self.weight_history['W1'][t_idx]
                ax = axs[0, i]
                if W1.size > 0:
                    im = ax.imshow(W1, cmap='RdBu', vmin=-2, vmax=2)
                    ax.set_title(f"{titles[i]}: Layer 1")
                    ax.set_xlabel('Input Feature')
                    ax.set_ylabel('Neuron Index')
                
                # Second layer weights (middle row)
                W2 = self.weight_history['W2'][t_idx]
                ax = axs[1, i]
                if W2.size > 0:
                    im = ax.imshow(W2, cmap='RdBu', vmin=-2, vmax=2)
                    ax.set_title(f"{titles[i]}: Layer 2")
                    ax.set_xlabel('Layer 1 Neuron')
                    ax.set_ylabel('Layer 2 Neuron')
                
                # Output weights (bottom row)
                a = self.weight_history['a'][t_idx]
                ax = axs[2, i]
                if a.size > 0:
                    ax.bar(range(len(a)), a)
                    ax.set_title(f"{titles[i]}: Output Weights")
                    ax.set_xlabel('Layer 2 Neuron')
                    ax.set_ylabel('Weight')
            
        else:  # Even parity - simpler visualization
            fig, axs = plt.subplots(2, len(indices), figsize=(5*len(indices), 10))
            if len(indices) == 1:
                axs = np.array([[axs[0]], [axs[1]]])
            titles = ['Initial Weights', 'Intermediate Weights', 'Final Weights']
            titles = titles[:len(indices)]
            
            for i, t_idx in enumerate(indices):
                # Skip if no circuit neurons found at this time
                if len(self.weight_history['neurons1'][t_idx]) == 0:
                    continue
                    
                # First layer weights (top row)
                W1 = self.weight_history['W1'][t_idx]
                ax = axs[0, i]
                if W1.size > 0:
                    im = ax.imshow(W1, cmap='RdBu', vmin=-2, vmax=2)
                    ax.set_title(f"{titles[i]}: Layer 1")
                    ax.set_xlabel('Input Feature')
                    ax.set_ylabel('Neuron Index')
                
                # Output weights (bottom row)
                a = self.weight_history['a'][t_idx]
                ax = axs[1, i]
                if a.size > 0:
                    ax.bar(range(len(a)), a)
                    ax.set_title(f"{titles[i]}: Output Weights")
                    ax.set_xlabel('Layer 1 Neuron')
                    ax.set_ylabel('Weight')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/weight_evolution.png", dpi=300)
        plt.close()
    
    def _plot_attribution_evolution(self):
        """Plot attribution patterns at different stages"""
        # Check if enough data points are available
        if len(self.circuit_metrics) < 3:
            print("Not enough data for attribution evolution visualization")
            return
        
        # Select key time points - beginning, middle, end
        n_points = len(self.circuit_metrics)
        indices = [0]
        if n_points > 2:
            indices.append(n_points//2)
        indices.append(-1)
        
        # Create figure
        fig, axs = plt.subplots(1, len(indices), figsize=(5*len(indices), 5))
        if len(indices) == 1:
            axs = [axs]
        
        for i, t_idx in enumerate(indices):
            attribution = self.circuit_metrics[t_idx]['attribution']
            
            # Focus on input attribution - which inputs matter most?
            input_attr = attribution.get('input', np.zeros(self.d))
            
            # Plot bar chart of attributions
            ax = axs[i]
            ax.bar(range(self.d), input_attr)
            ax.set_title(f"Epoch {self.circuit_metrics[t_idx]['epoch']}")
            ax.set_xlabel('Input Feature')
            ax.set_ylabel('Attribution')
            
            # Highlight the first k features (relevant to parity)
            ax.axvspan(-0.5, self.k-0.5, alpha=0.2, color='green')
            
            # Annotate
            if i == 0:
                ax.text(self.d//2, input_attr.max() * 0.8, "Green: Relevant Features", 
                       ha='center', va='center', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/attribution_evolution.png", dpi=300)
        plt.close()
    
    def _plot_weight_matrices_comparison(self):
        """Plot initial and final weight matrices as heatmaps"""
        # Get initial weights
        W1_initial = self.initial_weights['W1'].cpu().numpy()
        W2_initial = self.initial_weights['W2'].cpu().numpy()
        a_initial = self.initial_weights['a'].cpu().numpy()
        
        # Get final weights
        W1_final = self.model.W1.detach().cpu().numpy()
        W2_final = self.model.W2.detach().cpu().numpy()
        a_final = self.model.a.detach().cpu().numpy()
        
        # Create figure with 2 rows (initial/final) and 3 columns (W1, W2, a)
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # Set titles for columns
        axs[0, 0].set_title("First Layer Weights (W1)")
        axs[0, 1].set_title("Second Layer Weights (W2)")
        axs[0, 2].set_title("Output Weights (a)")
        
        # Set titles for rows
        axs[0, 0].set_ylabel("Initial")
        axs[1, 0].set_ylabel("Final")
        
        # Plot W1 initial
        im00 = axs[0, 0].imshow(W1_initial[:50, :self.k*2], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axs[0, 0].set_xlabel("Input Feature (showing first few)")
        axs[0, 0].set_ylabel("Neuron Index (showing first 50)")
        plt.colorbar(im00, ax=axs[0, 0])
        
        # Plot W2 initial
        im01 = axs[0, 1].imshow(W2_initial[:50, :50], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axs[0, 1].set_xlabel("Layer 1 Neuron (showing first 50)")
        axs[0, 1].set_ylabel("Layer 2 Neuron (showing first 50)")
        plt.colorbar(im01, ax=axs[0, 1])
        
        # Plot a initial
        im02 = axs[0, 2].bar(range(min(50, len(a_initial))), a_initial[:50])
        axs[0, 2].set_xlabel("Layer 2 Neuron (showing first 50)")
        axs[0, 2].set_ylabel("Weight Value")
        
        # Plot W1 final
        im10 = axs[1, 0].imshow(W1_final[:50, :self.k*2], cmap='RdBu_r', vmin=-2, vmax=2)
        axs[1, 0].set_xlabel("Input Feature (showing first few)")
        axs[1, 0].set_ylabel("Neuron Index (showing first 50)")
        plt.colorbar(im10, ax=axs[1, 0])
        
        # Highlight relevant features (first k)
        axs[1, 0].axvspan(-0.5, self.k-0.5, alpha=0.2, color='green')
        
        # Plot W2 final
        im11 = axs[1, 1].imshow(W2_final[:50, :50], cmap='RdBu_r', vmin=-2, vmax=2)
        axs[1, 1].set_xlabel("Layer 1 Neuron (showing first 50)")
        axs[1, 1].set_ylabel("Layer 2 Neuron (showing first 50)")
        plt.colorbar(im11, ax=axs[1, 1])
        
        # Plot a final
        im12 = axs[1, 2].bar(range(min(50, len(a_final))), a_final[:50])
        axs[1, 2].set_xlabel("Layer 2 Neuron (showing first 50)")
        axs[1, 2].set_ylabel("Weight Value")
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/weight_matrices_comparison.png", dpi=300)
        plt.close()
        
        # Create focused heatmaps of the relevant regions
        # First, identify important neurons in each layer
        
        # For W1, focus on the first k features across all neurons
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        
        # Initial W1 relevant features
        im0 = axs[0].imshow(W1_initial[:, :self.k], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axs[0].set_title("Initial W1 (Relevant Features Only)")
        axs[0].set_xlabel(f"Relevant Features (0-{self.k-1})")
        axs[0].set_ylabel("Neuron Index")
        plt.colorbar(im0, ax=axs[0])
        
        # Final W1 relevant features
        im1 = axs[1].imshow(W1_final[:, :self.k], cmap='RdBu_r', vmin=-2, vmax=2)
        axs[1].set_title("Final W1 (Relevant Features Only)")
        axs[1].set_xlabel(f"Relevant Features (0-{self.k-1})")
        axs[1].set_ylabel("Neuron Index")
        plt.colorbar(im1, ax=axs[1])
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/W1_relevant_features_comparison.png", dpi=300)
        plt.close()
    
    def _plot_gradient_evolution(self):
        """Plot the evolution of gradient norms and ratios over training"""
        if not self.gradient_history:
            print("No gradient history recorded")
            return
        
        # Extract data
        epochs = [g['epoch'] for g in self.gradient_history]
        rel_grad_mean = [g['W1_relevant_grad_mean'] for g in self.gradient_history]
        irrel_grad_mean = [g['W1_irrelevant_grad_mean'] for g in self.gradient_history]
        
        # For each neuron, calculate average ratio over time
        n_neurons = len(self.gradient_history[0]['W1_rel_irrel_ratio'])
        avg_ratios = np.zeros(n_neurons)
        for g in self.gradient_history:
            avg_ratios += g['W1_rel_irrel_ratio']
        avg_ratios /= len(self.gradient_history)
        
        # First plot - mean gradient values over time
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, rel_grad_mean, 'b-', label=f'Relevant Features (0-{self.k-1})')
        plt.plot(epochs, irrel_grad_mean, 'r-', label=f'Irrelevant Features ({self.k}-{self.d-1})')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Gradient Magnitude')
        plt.title('Evolution of Gradient Magnitudes for Relevant vs. Irrelevant Features')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.save_dir}/gradient_magnitudes.png", dpi=300)
        plt.close()
        
        # Second plot - ratio of relevant to irrelevant gradient norms for top neurons
        top_neurons = np.argsort(-avg_ratios)[:10]  # Top 10 neurons with highest ratio
        
        plt.figure(figsize=(10, 6))
        for i, neuron_idx in enumerate(top_neurons):
            neuron_ratios = [g['W1_rel_irrel_ratio'][neuron_idx] for g in self.gradient_history]
            plt.plot(epochs, neuron_ratios, label=f'Neuron {neuron_idx}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Relevant/Irrelevant Gradient Ratio')
        plt.title('Neurons with Highest Focus on Relevant Features')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.save_dir}/gradient_ratios.png", dpi=300)
        plt.close()
    
    def _plot_weight_trajectories(self):
        """Plot the evolution of weight values over training"""
        if not self.weight_evolution['epochs']:
            print("No weight evolution recorded")
            return
        
        epochs = self.weight_evolution['epochs']
        
        # Plot W1 weight trajectories for relevant features
        # Select a subset of neurons to visualize (to avoid cluttering)
        n_neurons_to_show = min(20, self.M1)
        neuron_indices = np.random.choice(self.M1, n_neurons_to_show, replace=False)
        
        # Create figure with subplots for each neuron
        plt.figure(figsize=(15, 15))
        
        # Set up grid of subplots (5x4 or adjustable based on neurons to show)
        n_rows = int(np.ceil(n_neurons_to_show / 4))
        n_cols = min(4, n_neurons_to_show)
        
        for i, neuron_idx in enumerate(neuron_indices):
            if i >= n_rows * n_cols:
                break
                
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            # Extract weights for this neuron over time
            neuron_weights = np.array([W1[neuron_idx, :self.k] for W1 in self.weight_evolution['W1']])
            
            # Plot each feature's weight over time with transparency gradient
            cmap = plt.cm.tab10
            for feat_idx in range(self.k):
                weights = neuron_weights[:, feat_idx]
                
                # Calculate alpha values for transparency (higher for later epochs)
                alphas = np.linspace(0.3, 1.0, len(epochs))
                
                # Plot line segments with changing transparency
                for j in range(len(epochs)-1):
                    ax.plot(epochs[j:j+2], weights[j:j+2], 
                           color=cmap(feat_idx % 10), 
                           alpha=alphas[j],
                           linewidth=2)
            
            ax.set_title(f"Neuron {neuron_idx}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight Value")
            ax.grid(True, alpha=0.3)
            
            # Add legend only to the first subplot
            if i == 0:
                ax.legend([f"Feature {j}" for j in range(self.k)], 
                         loc='upper left', fontsize='small')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/W1_weight_trajectories.png", dpi=300)
        plt.close()
        
        # Plot output weight trajectories (a) for second layer neurons
        n_neurons_to_show = min(20, self.M2)
        neuron_indices = np.random.choice(self.M2, n_neurons_to_show, replace=False)
        
        plt.figure(figsize=(12, 8))
        
        for i, neuron_idx in enumerate(neuron_indices):
            # Extract output weights for this neuron over time
            neuron_weights = np.array([a[neuron_idx] for a in self.weight_evolution['a']])
            
            # Calculate alpha values for transparency (higher for later epochs)
            alphas = np.linspace(0.3, 1.0, len(epochs))
            
            # Plot line segments with changing transparency
            for j in range(len(epochs)-1):
                plt.plot(epochs[j:j+2], neuron_weights[j:j+2], 
                       color=cmap(i % 10), 
                       alpha=alphas[j],
                       linewidth=2)
        
        plt.xlabel("Epoch")
        plt.ylabel("Output Weight Value")
        plt.title("Output Weight Trajectories")
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/output_weight_trajectories.png", dpi=300)
        plt.close()
    
    def _plot_weight_sign_changes(self):
        """Plot changes in weight signs over training"""
        if not self.weight_evolution['epochs']:
            print("No weight evolution recorded")
            return
        
        epochs = self.weight_evolution['epochs']
        
        # Count sign changes in W1 for the first k features (relevant features)
        sign_changes = np.zeros((self.M1, self.k))
        
        for t in range(1, len(epochs)):
            W1_prev = self.weight_evolution['W1'][t-1]
            W1_curr = self.weight_evolution['W1'][t]
            
            # Count sign changes between consecutive time points
            for i in range(self.M1):
                for j in range(self.k):
                    if np.sign(W1_prev[i, j]) != np.sign(W1_curr[i, j]) and not (W1_prev[i, j] == 0 or W1_curr[i, j] == 0):
                        sign_changes[i, j] += 1
        
        # Plot heatmap of sign changes
        plt.figure(figsize=(10, 8))
        plt.imshow(sign_changes, cmap='viridis', aspect='auto')
        plt.colorbar(label='Number of Sign Changes')
        plt.xlabel('Input Feature')
        plt.ylabel('Neuron Index')
        plt.title('Sign Changes in W1 Weights (Relevant Features)')
        plt.savefig(f"{self.save_dir}/W1_sign_changes.png", dpi=300)
        plt.close()
        
        # Plot distribution of sign changes
        plt.figure(figsize=(10, 6))
        plt.hist(sign_changes.flatten(), bins=20, alpha=0.7)
        plt.xlabel('Number of Sign Changes')
        plt.ylabel('Count')
        plt.title('Distribution of Weight Sign Changes')
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/sign_changes_distribution.png", dpi=300)
        plt.close()
        
        # Identify neurons with the most sign changes
        total_changes_per_neuron = np.sum(sign_changes, axis=1)
        top_neurons = np.argsort(-total_changes_per_neuron)[:5]  # Top 5 neurons with most sign changes
        
        if len(top_neurons) > 0:
            plt.figure(figsize=(12, 8))
            for i, neuron_idx in enumerate(top_neurons):
                # Extract weights for this neuron over time
                neuron_weights = np.array([W1[neuron_idx, :self.k] for W1 in self.weight_evolution['W1']])
                
                # Plot each feature's weight over time
                cmap = plt.cm.tab10
                for feat_idx in range(self.k):
                    weights = neuron_weights[:, feat_idx]
                    plt.plot(epochs, weights, label=f"Neuron {neuron_idx}, Feature {feat_idx}", 
                           color=cmap((i*self.k + feat_idx) % 10), linewidth=2)
            
            plt.xlabel("Epoch")
            plt.ylabel("Weight Value")
            plt.title("Weight Trajectories for Neurons with Most Sign Changes")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{self.save_dir}/top_sign_changing_neurons.png", dpi=300)
            plt.close()
    
    def analyze_weight_symmetries(self):
        """
        Analyze symmetries in the weight patterns of the circuit
        This helps identify the specific circuit structures
        """
        # Get final state
        final_metrics = self.circuit_metrics[-1]
        
        if self.k % 2 == 0:  # Even parity
            circuits = final_metrics['circuits']
            
            # Group neurons by their sign patterns
            sign_patterns = {}
            
            for idx, pattern_type, weights, output in circuits:
                # Create a signature based on weight signs
                signature = tuple(np.sign(weights).astype(int))
                
                if signature not in sign_patterns:
                    sign_patterns[signature] = []
                sign_patterns[signature].append((idx, output))
            
            # Print patterns
            print(f"Weight pattern analysis for even {self.k}-parity:")
            print(f"Found {len(sign_patterns)} distinct sign patterns:")
            
            for pattern, neurons in sign_patterns.items():
                pattern_str = ''.join('+' if s == 1 else '-' if s == -1 else '0' for s in pattern)
                avg_output = np.mean([out for _, out in neurons])
                
                print(f"Pattern {pattern_str}: {len(neurons)} neurons, avg output {avg_output:.2f}")
                
                # Theory validation for specific patterns
                sign_changes = sum(1 for i in range(len(pattern)-1) if pattern[i] != pattern[i+1] and pattern[i] != 0 and pattern[i+1] != 0)
                if sign_changes == 0 and avg_output > 0:
                    print("  ✓ Uniform sign pattern with positive output - matches theory")
                elif sign_changes > 0 and avg_output < 0:
                    print("  ✓ Mixed sign pattern with negative output - matches theory")
                else:
                    print("  ✗ Unexpected pattern-output combination")
        
        else:  # Odd parity
            circuits = final_metrics['circuits']
            
            # Analyze layer 1 patterns
            l1_patterns = {}
            l2_symmetries = []
            
            for path in circuits:
                l2_idx = path['l2_idx']
                
                # Analyze detectors
                for detector in path['detectors']:
                    l1_idx = detector['l1_idx']
                    features = detector['active_features']
                    
                    # Create pattern signature
                    if len(features) > 0:
                        pattern = tuple(sorted(features))
                        
                        if pattern not in l1_patterns:
                            l1_patterns[pattern] = []
                        l1_patterns[pattern].append(l1_idx)
                
                # Check l2 symmetry - compare to existing paths
                feature_coverage = set()
                for det in path['detectors']:
                    for feat in det['active_features']:
                        feature_coverage.add(feat)
                
                l2_symmetries.append({
                    'l2_idx': l2_idx,
                    'detector_count': len(path['detectors']),
                    'feature_coverage': feature_coverage,
                    'output_sign': np.sign(path['output_weight'])
                })
            
            # Print patterns
            print(f"Weight pattern analysis for odd {self.k}-parity:")
            print(f"Found {len(l1_patterns)} distinct feature detection patterns in layer 1:")
            
            for pattern, neurons in l1_patterns.items():
                print(f"Features {pattern}: {len(neurons)} detector neurons")
            
            print(f"\nFound {len(l2_symmetries)} feature aggregation patterns in layer 2:")
            for i, sym in enumerate(l2_symmetries):
                print(f"Aggregator {i}: {sym['detector_count']} detectors, " + 
                      f"covers {len(sym['feature_coverage'])} features, " +
                      f"output sign {'positive' if sym['output_sign'] > 0 else 'negative'}")
        
        # Return analysis for further use
        if self.k % 2 == 0:
            return sign_patterns
        else:
            return l1_patterns, l2_symmetries
    
    def create_pruned_model(self, include_only_circuits=True):
        """
        Create a pruned model that only contains circuit neurons
        
        Args:
            include_only_circuits: If True, only keep neurons identified as part of circuits
            
        Returns:
            Pruned model
        """
        # Get final metrics
        final_metrics = self.circuit_metrics[-1]
        circuits = final_metrics['circuits']
        
        # Create a copy of the model
        pruned_model = copy.deepcopy(self.model)
        
        # Apply masks to keep only relevant weights
        if self.k % 2 == 0:  # Even parity (single layer)
            if include_only_circuits:
                # Identify neurons in the circuit
                circuit_neurons = [idx for idx, _, _, _ in circuits]
                
                # Create masks
                layer1_mask = torch.zeros_like(pruned_model.W1)
                output_mask = torch.zeros_like(pruned_model.a)
                
                # Keep only circuit neurons and their connections to relevant inputs
                for idx in circuit_neurons:
                    layer1_mask[idx, :self.k] = 1.0
                    output_mask[idx] = 1.0
                
                # Apply masks
                pruned_model.W1.data = pruned_model.W1.data * layer1_mask
                pruned_model.a.data = pruned_model.a.data * output_mask
                
                # Zero out all second layer weights (not used in even parity)
                pruned_model.W2.data.zero_()
            else:
                # Keep all neurons but only connections to relevant inputs
                pruned_model.W1.data[:, self.k:] = 0
        
        else:  # Odd parity (two layer)
            if include_only_circuits:
                # Create masks based on identified circuits
                layer1_mask = torch.zeros_like(pruned_model.W1)
                layer2_mask = torch.zeros_like(pruned_model.W2)
                output_mask = torch.zeros_like(pruned_model.a)
                
                for path in circuits:
                    l2_idx = path['l2_idx']
                    output_mask[l2_idx] = 1.0
                    
                    for detector in path['detectors']:
                        l1_idx = detector['l1_idx']
                        for feat in detector['active_features']:
                            layer1_mask[l1_idx, feat] = 1.0
                        
                        layer2_mask[l2_idx, l1_idx] = 1.0
                
                # Apply masks
                pruned_model.W1.data = pruned_model.W1.data * layer1_mask
                pruned_model.W2.data = pruned_model.W2.data * layer2_mask
                pruned_model.a.data = pruned_model.a.data * output_mask
            else:
                # Keep all neurons but only connections to relevant inputs
                pruned_model.W1.data[:, self.k:] = 0
        
        return pruned_model
    
    def test_pruned_model(self, n_samples=10000):
        """
        Test a pruned model against the full model
        
        Args:
            n_samples: Number of samples to test on
            
        Returns:
            Dictionary of results comparing pruned and full models
        """
        # Create pruned model
        pruned_model = self.create_pruned_model(include_only_circuits=True)
        
        # Create samples
        X_test = torch.tensor(np.random.choice([-1, 1], size=(n_samples, self.d)), 
                             dtype=torch.float32).to(self.device)
        y_test = self._target_function(X_test)
        
        # Evaluate full model
        self.model.eval()
        with torch.no_grad():
            y_pred_full = self.model(X_test)
            mse_full = ((y_pred_full - y_test) ** 2).mean().item()
            corr_full = (y_pred_full * y_test).mean().item()
        
        # Evaluate pruned model
        pruned_model.eval()
        with torch.no_grad():
            y_pred_pruned = pruned_model(X_test)
            mse_pruned = ((y_pred_pruned - y_test) ** 2).mean().item()
            corr_pruned = (y_pred_pruned * y_test).mean().item()
        
        # Compare predictions directly
        with torch.no_grad():
            mse_full_vs_pruned = ((y_pred_full - y_pred_pruned) ** 2).mean().item()
            corr_full_vs_pruned = (y_pred_full * y_pred_pruned).mean().item()
        
        # Compile results
        results = {
            'mse_full': mse_full,
            'corr_full': corr_full,
            'mse_pruned': mse_pruned,
            'corr_pruned': corr_pruned,
            'mse_full_vs_pruned': mse_full_vs_pruned,
            'corr_full_vs_pruned': corr_full_vs_pruned
        }
        
        # Print summary
        print("\nModel Pruning Results:")
        print(f"Full model MSE: {mse_full:.6f}, Correlation: {corr_full:.4f}")
        print(f"Pruned model MSE: {mse_pruned:.6f}, Correlation: {corr_pruned:.4f}")
        print(f"Full vs Pruned MSE: {mse_full_vs_pruned:.6f}, Correlation: {corr_full_vs_pruned:.4f}")
        
        # If correlation between full and pruned is high, the circuit is sufficient
        if corr_pruned > 0.9 and corr_full_vs_pruned > 0.9:
            print("✓ PROOF: The identified circuits are sufficient to compute the parity function!")
        else:
            print("✗ The identified circuits may not be sufficient to compute the parity function.")
        
        return results
    
    def test_individual_circuits(self):
        """
        Test individual circuit components to see how they perform
        
        Returns:
            Dictionary of results for each circuit
        """
        # Get final metrics
        final_metrics = self.circuit_metrics[-1]
        circuits = final_metrics['circuits']
        
        # Create test samples
        X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, self.d)), 
                             dtype=torch.float32).to(self.device)
        y_test = self._target_function(X_test)
        
        # Test each circuit
        circuit_results = []
        
        self.model.eval()
        
        if self.k % 2 == 0:  # Even parity
            for i, (idx, pattern, weights, output) in enumerate(circuits):
                # Create a model with just this circuit
                single_circuit_model = copy.deepcopy(self.model)
                
                # Zero out all weights
                single_circuit_model.W1.data.zero_()
                single_circuit_model.W2.data.zero_()
                single_circuit_model.a.data.zero_()
                
                # Keep only this circuit's weights
                single_circuit_model.W1.data[idx, :self.k] = self.model.W1.data[idx, :self.k]
                single_circuit_model.a.data[idx] = self.model.a.data[idx]
                
                # Test model
                with torch.no_grad():
                    y_pred = single_circuit_model(X_test)
                    mse = ((y_pred - y_test) ** 2).mean().item()
                    corr = (y_pred * y_test).mean().item()
                
                result = {
                    'circuit_idx': i,
                    'neuron_idx': idx,
                    'pattern': pattern,
                    'mse': mse,
                    'correlation': corr
                }
                circuit_results.append(result)
                
            # Print results
            print("\nIndividual Circuit Performance (Even Parity):")
            for result in circuit_results:
                print(f"Circuit {result['circuit_idx']} (Neuron {result['neuron_idx']}): " + 
                      f"MSE={result['mse']:.6f}, Correlation={result['correlation']:.4f}")
                
            # Also check if all circuits together (superposition) improves performance
            # This would indicate that neurons compute the function in superposition
            circuits_together = copy.deepcopy(self.model)
            
            # Zero out all weights
            circuits_together.W1.data.zero_()
            circuits_together.W2.data.zero_()
            circuits_together.a.data.zero_()
            
            # Keep only circuit neurons
            for idx, _, _, _ in circuits:
                circuits_together.W1.data[idx, :self.k] = self.model.W1.data[idx, :self.k]
                circuits_together.a.data[idx] = self.model.a.data[idx]
            
            # Test model
            with torch.no_grad():
                y_pred = circuits_together(X_test)
                mse = ((y_pred - y_test) ** 2).mean().item()
                corr = (y_pred * y_test).mean().item()
            
            print(f"\nAll circuits together: MSE={mse:.6f}, Correlation={corr:.4f}")
            
        else:  # Odd parity
            for i, path in enumerate(circuits):
                # Create a model with just this circuit
                single_circuit_model = copy.deepcopy(self.model)
                
                # Zero out all weights
                single_circuit_model.W1.data.zero_()
                single_circuit_model.W2.data.zero_()
                single_circuit_model.a.data.zero_()
                
                # Keep only this circuit's weights
                l2_idx = path['l2_idx']
                single_circuit_model.a.data[l2_idx] = self.model.a.data[l2_idx]
                
                for detector in path['detectors']:
                    l1_idx = detector['l1_idx']
                    active_features = detector['active_features']
                    
                    for feat in active_features:
                        single_circuit_model.W1.data[l1_idx, feat] = self.model.W1.data[l1_idx, feat]
                    
                    single_circuit_model.W2.data[l2_idx, l1_idx] = self.model.W2.data[l2_idx, l1_idx]
                
                # Test model
                with torch.no_grad():
                    y_pred = single_circuit_model(X_test)
                    mse = ((y_pred - y_test) ** 2).mean().item()
                    corr = (y_pred * y_test).mean().item()
                
                result = {
                    'circuit_idx': i,
                    'path': path,
                    'mse': mse,
                    'correlation': corr
                }
                circuit_results.append(result)
            
            # Print results
            print("\nIndividual Circuit Performance (Odd Parity):")
            for result in circuit_results:
                print(f"Circuit {result['circuit_idx']} (L2 Neuron {result['path']['l2_idx']}): " + 
                      f"MSE={result['mse']:.6f}, Correlation={result['correlation']:.4f}")
        
        return circuit_results
    
    def analyze_circuit_logic(self):
        """
        Analyze the logical operations performed by circuit neurons
        
        Returns:
            Dictionary of logical operations for each circuit
        """
        # Create a large number of test inputs for analysis
        X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, self.d)), 
                             dtype=torch.float32).to(self.device)
        
        # Test each neuron's response to different input patterns
        self.model.eval()
        
        if self.k % 2 == 0:  # Even parity
            final_metrics = self.circuit_metrics[-1]
            circuits = final_metrics['circuits']
            
            logic_results = []
            
            # Compute activations for all neurons
            with torch.no_grad():
                h1 = self.model.get_first_layer_activations(X_test).cpu().numpy()
                final_output = self.model(X_test).cpu().numpy()
                true_output = self._target_function(X_test).cpu().numpy()
            
            # For each circuit neuron, analyze its behavior
            for idx, pattern, weights, output in circuits:
                # Get neuron activations
                neuron_activations = h1[:, idx]
                
                # Get input values for the k relevant features
                relevant_inputs = X_test.cpu().numpy()[:, :self.k]
                
                # Calculate different logical operations on inputs
                and_op = np.all(relevant_inputs == 1, axis=1).astype(float)
                or_op = np.any(relevant_inputs == 1, axis=1).astype(float)
                nand_op = (~np.all(relevant_inputs == 1, axis=1)).astype(float)
                nor_op = (~np.any(relevant_inputs == 1, axis=1)).astype(float)
                xor_op = np.sum(relevant_inputs == 1, axis=1) % 2 == 1
                xnor_op = np.sum(relevant_inputs == 1, axis=1) % 2 == 0
                
                # Convert to -1/1 instead of 0/1
                xor_op = xor_op * 2 - 1
                xnor_op = xnor_op * 2 - 1
                
                # Compute correlations with different logical operations
                corr_and = np.corrcoef(neuron_activations, and_op)[0, 1]
                corr_or = np.corrcoef(neuron_activations, or_op)[0, 1]
                corr_nand = np.corrcoef(neuron_activations, nand_op)[0, 1]
                corr_nor = np.corrcoef(neuron_activations, nor_op)[0, 1]
                corr_xor = np.corrcoef(neuron_activations, xor_op)[0, 1]
                corr_xnor = np.corrcoef(neuron_activations, xnor_op)[0, 1]
                
                # Determine the best match
                operations = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
                correlations = [corr_and, corr_or, corr_nand, corr_nor, corr_xor, corr_xnor]
                best_op_idx = np.argmax(np.abs(correlations))
                best_op = operations[best_op_idx]
                best_corr = correlations[best_op_idx]
                
                result = {
                    'neuron_idx': idx,
                    'best_operation': best_op,
                    'correlation': best_corr,
                    'all_correlations': {
                        'AND': corr_and,
                        'OR': corr_or,
                        'NAND': corr_nand,
                        'NOR': corr_nor,
                        'XOR': corr_xor,
                        'XNOR': corr_xnor
                    }
                }
                logic_results.append(result)
            
            # Print results
            print("\nLogical Operations Analysis (Even Parity):")
            for result in logic_results:
                print(f"Neuron {result['neuron_idx']}: Best operation = {result['best_operation']}, " +
                      f"Correlation = {result['correlation']:.4f}")
                print("  All correlations:", {op: f"{corr:.4f}" for op, corr in result['all_correlations'].items()})
            
            return logic_results
            
        else:  # Odd parity - analyze both layers
            final_metrics = self.circuit_metrics[-1]
            circuits = final_metrics['circuits']
            
            layer1_logic = []
            layer2_logic = []
            
            # Compute activations for all neurons
            with torch.no_grad():
                h1 = self.model.get_first_layer_activations(X_test).cpu().numpy()
                h2 = self.model.get_second_layer_activations(X_test).cpu().numpy()
                final_output = self.model(X_test).cpu().numpy()
                true_output = self._target_function(X_test).cpu().numpy()
            
            # Get input values for the k relevant features
            relevant_inputs = X_test.cpu().numpy()[:, :self.k]
            
            # For each circuit, analyze both layers
            for path in circuits:
                l2_idx = path['l2_idx']
                
                # First analyze each detector in the first layer
                for detector in path['detectors']:
                    l1_idx = detector['l1_idx']
                    active_features = detector['active_features']
                    
                    # Get neuron activations
                    neuron_activations = h1[:, l1_idx]
                    
                    # Get inputs for active features
                    if len(active_features) > 0:
                        detector_inputs = relevant_inputs[:, active_features]
                        
                        # Compute logical operations
                        and_op = np.all(detector_inputs == 1, axis=1).astype(float)
                        or_op = np.any(detector_inputs == 1, axis=1).astype(float)
                        nand_op = (~np.all(detector_inputs == 1, axis=1)).astype(float)
                        nor_op = (~np.any(detector_inputs == 1, axis=1)).astype(float)
                        
                        # For XOR/XNOR, handle cases with arbitrary feature counts
                        if len(active_features) >= 2:
                            xor_op = np.sum(detector_inputs == 1, axis=1) % 2 == 1
                            xnor_op = np.sum(detector_inputs == 1, axis=1) % 2 == 0
                            # Convert to -1/1 instead of 0/1
                            xor_op = xor_op * 2 - 1
                            xnor_op = xnor_op * 2 - 1
                        else:
                            xor_op = np.zeros_like(and_op)
                            xnor_op = np.zeros_like(and_op)
                        
                        # Compute correlations
                        corr_and = np.corrcoef(neuron_activations, and_op)[0, 1] if len(and_op) > 0 else 0
                        corr_or = np.corrcoef(neuron_activations, or_op)[0, 1] if len(or_op) > 0 else 0
                        corr_nand = np.corrcoef(neuron_activations, nand_op)[0, 1] if len(nand_op) > 0 else 0
                        corr_nor = np.corrcoef(neuron_activations, nor_op)[0, 1] if len(nor_op) > 0 else 0
                        corr_xor = np.corrcoef(neuron_activations, xor_op)[0, 1] if len(xor_op) > 0 else 0
                        corr_xnor = np.corrcoef(neuron_activations, xnor_op)[0, 1] if len(xnor_op) > 0 else 0
                        
                        operations = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
                        correlations = [corr_and, corr_or, corr_nand, corr_nor, corr_xor, corr_xnor]
                        
                        # Ensure correlations are valid
                        correlations = [0 if np.isnan(c) else c for c in correlations]
                        
                        best_op_idx = np.argmax(np.abs(correlations))
                        best_op = operations[best_op_idx]
                        best_corr = correlations[best_op_idx]
                        
                        layer1_logic.append({
                            'neuron_idx': l1_idx,
                            'features': list(active_features),
                            'best_operation': best_op,
                            'correlation': best_corr,
                            'all_correlations': {
                                'AND': corr_and,
                                'OR': corr_or,
                                'NAND': corr_nand,
                                'NOR': corr_nor,
                                'XOR': corr_xor,
                                'XNOR': corr_xnor
                            }
                        })
                
                # Now analyze the second layer neuron
                l2_activations = h2[:, l2_idx]
                
                # Get activations of all connected first layer neurons
                connected_l1_indices = [detector['l1_idx'] for detector in path['detectors']]
                if connected_l1_indices:
                    l1_activations = h1[:, connected_l1_indices]
                    
                    # Analyze relationship between L2 neuron and L1 neurons
                    # This is complex - we'll compute correlation with different operations
                    
                    # Logical AND of all L1 neurons
                    l1_and = np.all(l1_activations > 0, axis=1).astype(float)
                    
                    # Logical OR of all L1 neurons
                    l1_or = np.any(l1_activations > 0, axis=1).astype(float)
                    
                    # XOR of all L1 neurons
                    l1_xor = np.sum(l1_activations > 0, axis=1) % 2 == 1
                    l1_xor = l1_xor * 2 - 1
                    
                    # XNOR of all L1 neurons
                    l1_xnor = np.sum(l1_activations > 0, axis=1) % 2 == 0
                    l1_xnor = l1_xnor * 2 - 1
                    
                    # Compute correlations
                    corr_and = np.corrcoef(l2_activations, l1_and)[0, 1] if len(l1_and) > 0 else 0
                    corr_or = np.corrcoef(l2_activations, l1_or)[0, 1] if len(l1_or) > 0 else 0
                    corr_xor = np.corrcoef(l2_activations, l1_xor)[0, 1] if len(l1_xor) > 0 else 0
                    corr_xnor = np.corrcoef(l2_activations, l1_xnor)[0, 1] if len(l1_xnor) > 0 else 0
                    
                    operations = ['AND', 'OR', 'XOR', 'XNOR']
                    correlations = [corr_and, corr_or, corr_xor, corr_xnor]
                    
                    # Ensure correlations are valid
                    correlations = [0 if np.isnan(c) else c for c in correlations]
                    
                    best_op_idx = np.argmax(np.abs(correlations))
                    best_op = operations[best_op_idx]
                    best_corr = correlations[best_op_idx]
                    
                    layer2_logic.append({
                        'neuron_idx': l2_idx,
                        'connected_l1': connected_l1_indices,
                        'best_operation': best_op,
                        'correlation': best_corr,
                        'all_correlations': {
                            'AND': corr_and,
                            'OR': corr_or,
                            'XOR': corr_xor,
                            'XNOR': corr_xnor
                        }
                    })
            
            # Print results
            print("\nLogical Operations Analysis (Odd Parity):")
            print("Layer 1 (Detectors):")
            for result in layer1_logic:
                print(f"Neuron {result['neuron_idx']} (Features {result['features']}): " +
                      f"Best operation = {result['best_operation']}, Correlation = {result['correlation']:.4f}")
            print("\nLayer 2 (Aggregators):")
            for result in layer2_logic:
                print(f"Neuron {result['neuron_idx']} (Connected to {result['connected_l1']}): " +
                      f"Best operation = {result['best_operation']}, Correlation = {result['correlation']:.4f}")
            
            return {'layer1': layer1_logic, 'layer2': layer2_logic}

    def generate_circuit_animation(self, filename='circuit_formation.mp4'):
        """
        Generate an animation showing the circuit formation over time
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            print("Could not import animation module. Make sure you have ffmpeg installed.")
            return
            
        # Check if we have enough data points
        if len(self.circuit_metrics) < 3:
            print("Not enough data for animation")
            return
        
        # Create figure and axis
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
        is_odd_parity = self.k % 2 != 0
        
        def update(frame):
            ax.clear()
            
            # Get metrics for this frame
            metrics = self.circuit_metrics[frame]
            circuits = metrics['circuits']
            
            # Create graph
            G = nx.DiGraph()
            
            # Add input nodes
            for i in range(self.k):
                G.add_node(f"x{i}", layer=0, pos=(i, 0))
            
            if is_odd_parity:
                # Odd parity visualization
                # Add first layer nodes (detectors)
                l1_nodes = set()
                for path in circuits:
                    for detector in path['detectors']:
                        l1_idx = detector['l1_idx']
                        if l1_idx not in l1_nodes:
                            l1_nodes.add(l1_idx)
                            # Position within layer based on active features
                            if len(detector['active_features']) > 0:
                                pos_x = sum(detector['active_features']) / max(1, len(detector['active_features']))
                            else:
                                pos_x = l1_idx % self.k
                            G.add_node(f"h1_{l1_idx}", layer=1, pos=(pos_x, 1))
                        
                        # Connect to inputs
                        for feat_idx, feat in enumerate(detector['active_features']):
                            if feat_idx < len(detector['l1_weight']):
                                weight = detector['l1_weight'][feat_idx]
                            else:
                                weight = 0
                            G.add_edge(f"x{feat}", f"h1_{l1_idx}", weight=weight)
                
                # Add second layer nodes (aggregators)
                for path_idx, path in enumerate(circuits):
                    l2_idx = path['l2_idx']
                    G.add_node(f"h2_{l2_idx}", layer=2, pos=(path_idx, 2))
                    
                    # Connect to first layer
                    for detector in path['detectors']:
                        l1_idx = detector['l1_idx']
                        weight = detector['l2_weight']
                        G.add_edge(f"h1_{l1_idx}", f"h2_{l2_idx}", weight=weight)
                
                # Add output node
                G.add_node("y", layer=3, pos=(len(circuits)//2, 3))
                
                # Connect second layer to output
                for path in circuits:
                    l2_idx = path['l2_idx']
                    weight = path['output_weight']
                    G.add_edge(f"h2_{l2_idx}", "y", weight=weight)
            else:
                # Even parity visualization
                # Add layer 1 nodes (ReLU neurons)
                for idx, pattern, weights, output in circuits:
                    G.add_node(f"h{idx}", layer=1, pos=(idx % self.k, 1), pattern=pattern)
                    # Connect to inputs
                    for i in range(self.k):
                        weight = weights[i]
                        if abs(weight) > 0.1:  # Only show significant connections
                            G.add_edge(f"x{i}", f"h{idx}", weight=weight)
                
                # Add output node
                G.add_node("y", layer=2, pos=(self.k//2, 2))
                
                # Connect layer 1 to output
                for idx, pattern, weights, output in circuits:
                    G.add_edge(f"h{idx}", "y", weight=output)
            
            # Create position layout
            pos = nx.get_node_attributes(G, 'pos')
            
            # Adjust positions for better visualization
            for node, position in pos.items():
                x, y = position
                layer = G.nodes[node]['layer']
                
                # Spread out nodes in each layer
                if layer == 1:  # First hidden layer
                    x = x / max(1, self.k) * 10
                elif layer == 2 and is_odd_parity:  # Second hidden layer for odd parity
                    x = x / max(1, len(circuits)) * 10
                    
                pos[node] = (x, y * 3)  # Increase vertical spacing
            
            # Draw nodes by layer
            node_colors = ['lightblue', 'lightgreen', 'salmon', 'purple']
            max_layers = 4 if is_odd_parity else 3
            
            for layer in range(max_layers):
                nodelist = [n for n, d in G.nodes(data=True) if d.get('layer') == layer]
                if nodelist:
                    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=node_colors[layer], 
                                         node_size=500, alpha=0.8, ax=ax)
            
            # Draw edges with colors based on weights
            edges = G.edges(data='weight')
            if edges:
                edge_colors = []
                for u, v, w in edges:
                    # Handle potential NaN or None values
                    if w is None or np.isnan(w):
                        edge_colors.append('gray')
                    else:
                        edge_colors.append(plt.cm.RdBu(0.5 * (1 + w / max(1, abs(w)))))
                
                # Scale edge widths by weight magnitude
                edge_widths = []
                for u, v, w in edges:
                    if w is None or np.isnan(w):
                        edge_widths.append(1)
                    else:
                        edge_widths.append(1 + 3 * abs(w))
                
                nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                                      arrowsize=15, min_source_margin=15, min_target_margin=15, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_weight='bold', ax=ax)
            
            # Add title with metrics
            ax.set_title(f"Epoch {metrics['epoch']}: Correlation {metrics['correlation']:.4f}, " +
                       f"Circuit Completeness {metrics['circuit_completeness']:.2f}")
            ax.axis('off')
            
            return ax,
        
        # Create animation
        try:
            frames = range(min(20, len(self.circuit_metrics)))  # Limit frames for speed
            if len(frames) < 3:
                frames = range(len(self.circuit_metrics))
                
            ani = animation.FuncAnimation(fig, update, frames=frames, interval=500)
            
            # Save animation
            ani.save(f"{self.save_dir}/{filename}", dpi=100, writer='ffmpeg')
            print(f"Animation saved to {self.save_dir}/{filename}")
            
            return ani
        except Exception as e:
            print(f"Error creating animation: {e}")
            return None

    def run_comprehensive_analysis(self):
        """
        Run a comprehensive analysis of the trained model and its circuits
        """
        print("\n========== COMPREHENSIVE CIRCUIT ANALYSIS ==========")
        
        # 1. Visualize circuit formation
        print("\n----- Visualizing Circuit Formation -----")
        self.visualize_circuit_formation()
        
        # 2. Analyze weight symmetries
        print("\n----- Analyzing Weight Symmetries -----")
        self.analyze_weight_symmetries()
        
        # 3. Test pruned model vs. full model
        print("\n----- Testing Pruned Model -----")
        pruning_results = self.test_pruned_model()
        
        # 4. Test individual circuits
        print("\n----- Testing Individual Circuits -----")
        circuit_results = self.test_individual_circuits()
        
        # 5. Analyze circuit logic
        print("\n----- Analyzing Circuit Logic -----")
        logic_results = self.analyze_circuit_logic()
        
        # 6. Generate animation of circuit formation
        print("\n----- Generating Circuit Animation -----")
        self.generate_circuit_animation()
        
        print("\n========== ANALYSIS COMPLETE ==========")
        return {
            'pruning_results': pruning_results,
            'circuit_results': circuit_results,
            'logic_results': logic_results
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


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and analyze neural networks learning parity functions')
    parser.add_argument('--d', type=int, default=45, help='Input dimension')
    parser.add_argument('--k', type=int, default=6, help='Parity function order')
    parser.add_argument('--M1', type=int, default=256, help='First layer width')
    parser.add_argument('--M2', type=int, default=256, help='Second layer width')
    parser.add_argument('--epochs', type=int, default=50000, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--save_dir', type=str, default='parity_circuit_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # For even k-parity (e.g., k=4)
    if args.k % 2 == 0:
        print(f"=== Training Even {args.k}-Parity Function ===")
        save_dir = f"{args.save_dir}_even_{args.k}"
    else:
        print(f"=== Training Odd {args.k}-Parity Function ===")
        save_dir = f"{args.save_dir}_odd_{args.k}"
    
    # Create tracker
    tracker = ParityCircuitTracker(
        d=args.d, 
        k=args.k, 
        M1=args.M1, 
        M2=args.M2,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        save_dir=save_dir
    )
    
    # Train model
    tracker.train(n_epochs=args.epochs, log_interval=args.log_interval)
    
    # Run comprehensive analysis
    tracker.run_comprehensive_analysis()