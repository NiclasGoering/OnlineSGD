import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from scipy.optimize import curve_fit
import itertools

class TwoHiddenLayerReLUNet(torch.nn.Module):
    def __init__(self, input_dim, hidden1_width, hidden2_width):
        super(TwoHiddenLayerReLUNet, self).__init__()
        # Initialize layers with careful scaling - NO BIASES as requested
        self.W1 = torch.nn.Parameter(torch.randn(hidden1_width, input_dim) / np.sqrt(input_dim))
        self.W2 = torch.nn.Parameter(torch.randn(hidden2_width, hidden1_width) / np.sqrt(hidden1_width))
        self.a = torch.nn.Parameter(torch.randn(1, hidden2_width) / np.sqrt(hidden2_width))
        
    def forward(self, x):
        # First hidden layer
        h1 = torch.relu(torch.matmul(x, self.W1.t()))
        # Second hidden layer
        h2 = torch.relu(torch.matmul(h1, self.W2.t()))
        # Output layer
        output = torch.matmul(h2, self.a.t())
        return output.squeeze()
    
    def get_first_layer_activations(self, x):
        return torch.relu(torch.matmul(x, self.W1.t()))
    
    def get_second_layer_activations(self, x):
        h1 = torch.relu(torch.matmul(x, self.W1.t()))
        return torch.relu(torch.matmul(h1, self.W2.t()))

class RandomProjectionAnalyzer:
    """Class to analyze random projection properties in parity learning"""
    
    def __init__(self, d_values, k_values, n_trials=10):
        self.d_values = d_values
        self.k_values = k_values
        self.n_trials = n_trials
        self.results = {}
    
    def generate_all_subset_masks(self, d, k, max_subsets=1000):
        """Generate binary masks for all k-subsets of [d] (or a sample if too many)"""
        all_indices = list(range(d))
        if d < 30:  # For small d, we can generate all subsets
            all_subsets = list(itertools.combinations(all_indices, k))
            if len(all_subsets) > max_subsets:
                # Randomly sample if too many
                all_subsets = random.sample(all_subsets, max_subsets)
        else:
            # For large d, generate random subsets
            all_subsets = []
            for _ in range(max_subsets):
                subset = sorted(random.sample(all_indices, k))
                if subset not in all_subsets:
                    all_subsets.append(subset)
        
        # Create feature masks
        masks = np.ones((len(all_subsets), d)) * -1  # Default to -1
        for i, subset in enumerate(all_subsets):
            masks[i, subset] = 1  # Set relevant features to 1
            
        return masks
    
    def random_projections_experiment(self, d, k, n_projections_range):
        """
        Test how many random projections are needed to separate k-sparse patterns
        
        We'll test different numbers of random projections and measure the
        separation quality directly in accordance with Lemma 2.3.
        """
        # Generate masks for all k-subsets
        subset_masks = self.generate_all_subset_masks(d, k)
        n_subsets = subset_masks.shape[0]
        
        # Calculate "true" separation in original space
        # For each mask, compute its correlation with all other masks
        original_correlations = np.zeros((n_subsets, n_subsets))
        for i in range(n_subsets):
            for j in range(i+1, n_subsets):
                corr = np.sum(subset_masks[i] * subset_masks[j]) / d
                original_correlations[i, j] = corr
                original_correlations[j, i] = corr
        
        # Track separation quality for different projection dimensions
        separation_quality = []
        
        # Test different projection dimensions
        for n_proj in n_projections_range:
            quality_samples = []
            
            # Run multiple trials
            for _ in range(5):
                # Generate random projection matrix
                projection_matrix = np.random.randn(n_proj, d) / np.sqrt(d)
                
                # Project masks into lower dimension
                projected_masks = np.matmul(subset_masks, projection_matrix.T)
                
                # Calculate correlations in projected space
                projected_correlations = np.zeros((n_subsets, n_subsets))
                for i in range(n_subsets):
                    for j in range(i+1, n_subsets):
                        # Normalize vectors before computing correlation
                        v1 = projected_masks[i] / np.linalg.norm(projected_masks[i])
                        v2 = projected_masks[j] / np.linalg.norm(projected_masks[j])
                        corr = np.dot(v1, v2)
                        projected_correlations[i, j] = corr
                        projected_correlations[j, i] = corr
                
                # Measure preservation of separations
                # Define quality as correlation between original and projected correlations
                i_upper, j_upper = np.triu_indices(n_subsets, 1)
                orig_corr_values = original_correlations[i_upper, j_upper]
                proj_corr_values = projected_correlations[i_upper, j_upper]
                
                quality = np.corrcoef(orig_corr_values, proj_corr_values)[0, 1]
                quality_samples.append(quality)
            
            # Average quality across trials
            avg_quality = np.mean(quality_samples)
            separation_quality.append(avg_quality)
        
        return n_projections_range, separation_quality
    
    def analyze_scaling(self):
        """Analyze how required projections scale with dimension for each k"""
        # Define projection dimensions to test
        projection_dims = np.logspace(1, 4, 20).astype(int)
        projection_dims = np.unique(projection_dims)  # Remove duplicates
        
        for k in self.k_values:
            self.results[k] = {'d_values': [], 'required_projections': []}
            
            for d in self.d_values:
                if d <= k:
                    continue  # Skip invalid configurations
                
                print(f"Testing d={d}, k={k}")
                
                # Run experiment
                n_projections, quality = self.random_projections_experiment(
                    d, k, projection_dims[projection_dims < d*10])
                
                # Find required projections for good separation (quality > 0.9)
                required_idx = np.argmax(np.array(quality) > 0.9)
                if required_idx == 0 and quality[0] <= 0.9:
                    # No projection dimension achieved required quality
                    required_projections = projection_dims[-1] * 2  # Estimate higher
                else:
                    required_projections = n_projections[required_idx]
                
                self.results[k]['d_values'].append(d)
                self.results[k]['required_projections'].append(required_projections)
                
                print(f"  Required projections = {required_projections}")
        
        return self.results
    
    def fit_power_law(self, d_values, projections):
        """Fit a power law curve to the data: projections = C * d^alpha"""
        # Convert to log space
        log_d = np.log(d_values)
        log_proj = np.log(projections)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_d, log_proj, 1)
        alpha = coeffs[0]  # Exponent
        C = np.exp(coeffs[1])  # Coefficient
        
        return C, alpha
    
    def visualize_scaling(self):
        """Create visualization of projection scaling with dimension"""
        plt.figure(figsize=(12, 10))
        
        theoretical_ratios = []
        empirical_ratios = []
        k_values_plot = []
        
        for k in self.k_values:
            d_values = np.array(self.results[k]['d_values'])
            projections = np.array(self.results[k]['required_projections'])
            
            if len(d_values) < 2:  # Need at least 2 points for fitting
                continue
                
            k_values_plot.append(k)
            
            plt.loglog(d_values, projections, 'o-', linewidth=2, label=f'k={k} (empirical)')
            
            # Fit power law
            C, alpha = self.fit_power_law(d_values, projections)
            
            # Generate smooth curve for fitted model
            d_smooth = np.logspace(np.log10(min(d_values)), np.log10(max(d_values)), 100)
            projections_smooth = C * d_smooth**alpha
            
            plt.loglog(d_smooth, projections_smooth, '--', linewidth=1, 
                     label=f'k={k} fit: {C:.2f}·d^{alpha:.3f}')
            
            # Also plot the theoretical k/2 scaling for comparison
            theoretical = d_values**(k/2) / np.power(5, k/2)  # Scaled for visibility
            plt.loglog(d_values, theoretical, ':', linewidth=1, 
                     label=f'k={k} theoretical: d^{k/2}')
            
            print(f"k={k}: Fitted power law = {C:.2f} * d^{alpha:.3f}")
            print(f"       Theoretical exponent = {k/2}")
            print(f"       Ratio of empirical to theoretical: {alpha/(k/2):.3f}")
            
            theoretical_ratios.append(k/2)
            empirical_ratios.append(alpha)
        
        plt.xlabel('Dimension (d)', fontsize=14)
        plt.ylabel('Required Projections', fontsize=14)
        plt.title('Random Projection Dimension Scaling (Lemma 2.3)', fontsize=16)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.savefig("random_projection_scaling.png", dpi=300)
        plt.close()
        
        # Create ratio plot to directly compare with theoretical prediction
        if len(k_values_plot) > 0:
            plt.figure(figsize=(10, 8))
            
            width = 0.35
            x = np.arange(len(k_values_plot))
            
            plt.bar(x - width/2, empirical_ratios, width, label='Empirical Exponent', alpha=0.7)
            plt.bar(x + width/2, theoretical_ratios, width, label='Theoretical Exponent (k/2)', alpha=0.7)
            
            plt.axhline(y=1.0, linestyle='--', color='r', alpha=0.5)
            
            plt.xlabel('Sparsity (k)', fontsize=14)
            plt.ylabel('Exponent (α)', fontsize=14)
            plt.title('Validation of Lemma 2.3: Empirical vs Theoretical Scaling', fontsize=16)
            plt.xticks(x, k_values_plot)
            plt.grid(True, axis='y', alpha=0.7)
            plt.legend(fontsize=12)
            
            plt.savefig("random_projection_exponents.png", dpi=300)
            plt.close()
            
            # Plot ratio of empirical to theoretical
            plt.figure(figsize=(10, 8))
            ratios = [e/t for e, t in zip(empirical_ratios, theoretical_ratios)]
            
            plt.bar(k_values_plot, ratios, alpha=0.7)
            plt.axhline(y=1.0, linestyle='--', color='r', alpha=0.7, 
                      label='Perfect match (ratio = 1)')
            
            plt.xlabel('Sparsity (k)', fontsize=14)
            plt.ylabel('Ratio (Empirical/Theoretical)', fontsize=14)
            plt.title('Validation of Lemma 2.3: Ratio of Empirical to Theoretical Exponent', fontsize=16)
            plt.grid(True, axis='y', alpha=0.7)
            plt.legend(fontsize=12)
            
            plt.savefig("random_projection_ratio.png", dpi=300)
            plt.close()


class TwoLayerParityPhaseAnalyzer:
    def __init__(self, d=30, k=6, hidden1_width=128, hidden2_width=128, learning_rate=0.01, 
                 batch_size=512, device=None, save_dir="parity_phase_transition"):
        self.d = d
        self.k = k
        self.hidden1_width = hidden1_width
        self.hidden2_width = hidden2_width
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.is_odd_parity = (k % 2 == 1)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize model, optimizer, and loss function
        self.model = TwoHiddenLayerReLUNet(d, hidden1_width, hidden2_width).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        
        # History storage
        self.history = {
            'epochs': [],
            'loss': [],
            'correlation': [],
            # First layer metrics
            'W1_rel_norm': [],  # ||W_{1,rel}||_F
            'W1_irr_norm': [],  # ||W_{1,irr}||_F
            'weight_ratio_L1': [],  # Z_t^(1) = ||W_{1,irr}||_F^2 / ||W_{1,rel}||_F^2
            'signal_strength_L1': [],  # Ψ_t^(1)
            'alignment_scores_L1': [],  # α_i^(1)(t)
            # Second layer metrics
            'W2_positive_ratio': [],  # Proportion of W2 > 0
            'W2_negative_ratio': [],  # Proportion of W2 < 0
            'W2_pos_neg_ratio': [],  # Ratio of positive to negative magnitude
            'L2_activation_sparsity': [],  # Proportion of inactive L2 neurons
            # Correlation metrics
            'empirical_correlation': [],  # Ĉ_A^(t)
        }
        
        # Test data for consistent evaluation
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, d)), 
                                  dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)
        
        # Calculate theoretical critical time
        self.T_c_theoretical = int(np.ceil((d**(k/2)) / learning_rate))
        
        print(f"TwoLayerParityPhaseAnalyzer initialized on {self.device}")
        print(f"Analyzing {k}-parity function in {d} dimensions ({'odd' if self.is_odd_parity else 'even'} parity)")
        print(f"Network: {d} → {hidden1_width} → {hidden2_width} → 1")
        print(f"Theoretical critical time T_c = {self.T_c_theoretical}")
    
    def _target_function(self, x):
        """Compute the k-sparse parity function on the first k inputs"""
        return torch.prod(x[:, :self.k], dim=1)
    
    def compute_layer1_metrics(self):
        """Compute metrics for first layer"""
        W1 = self.model.W1.detach().cpu().numpy()
        W1_rel = W1[:, :self.k]
        W1_irr = W1[:, self.k:]
        
        # Compute norms
        rel_norm = np.linalg.norm(W1_rel, 'fro')
        irr_norm = np.linalg.norm(W1_irr, 'fro')
        
        # Compute weight ratio Z_t^(1)
        ratio = (irr_norm**2) / (rel_norm**2) if rel_norm > 0 else float('inf')
        
        # Compute signal strength Ψ_t^(1)
        signal_strength = np.sum(np.abs(np.sum(W1_rel, axis=1)))
        
        # Compute alignment scores α_i^(1)(t)
        alignment_scores = []
        for i in range(self.hidden1_width):
            row_sum = np.sum(W1_rel[i])
            row_norm = np.linalg.norm(W1_rel[i])
            
            score = np.abs(row_sum) / row_norm if row_norm > 0 else 0
            alignment_scores.append(score)
        
        return {
            'W1_rel_norm': rel_norm,
            'W1_irr_norm': irr_norm,
            'weight_ratio_L1': ratio,
            'signal_strength_L1': signal_strength,
            'alignment_scores_L1': alignment_scores
        }
    
    def compute_layer2_metrics(self, num_samples=500):
        """Compute metrics for second layer"""
        # Second layer weights
        W2 = self.model.W2.detach().cpu().numpy()
        
        # Calculate statistics
        positive_ratio = np.mean(W2 > 0)
        negative_ratio = np.mean(W2 < 0)
        
        # Calculate ratio of total positive magnitude to total negative magnitude
        pos_magnitude = np.sum(np.abs(W2[W2 > 0]))
        neg_magnitude = np.sum(np.abs(W2[W2 < 0]))
        pos_neg_ratio = pos_magnitude / max(neg_magnitude, 1e-10)
        
        # Generate random inputs for activation analysis
        X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                       dtype=torch.float32).to(self.device)
        
        # Get layer 2 activations
        self.model.eval()
        with torch.no_grad():
            h2 = self.model.get_second_layer_activations(X).cpu().numpy()
        
        # Calculate activation sparsity
        activation_sparsity = 1.0 - np.mean(h2 > 0)
        
        return {
            'W2_positive_ratio': positive_ratio,
            'W2_negative_ratio': negative_ratio,
            'W2_pos_neg_ratio': pos_neg_ratio,
            'L2_activation_sparsity': activation_sparsity
        }
    
    def compute_empirical_correlation(self, num_samples=1000):
        """Compute empirical correlations Ĉ_A^(t) for various subsets A"""
        # Generate random samples
        X = torch.tensor(np.random.choice([-1, 1], size=(num_samples, self.d)), 
                       dtype=torch.float32).to(self.device)
        y = self._target_function(X)
        
        # Calculate empirical correlation for true subset S
        true_subset_corr = torch.mean(y * torch.prod(X[:, :self.k], dim=1)).item()
        
        # Calculate empirical correlation for a few random subsets
        random_subset_corrs = []
        for _ in range(5):  # Sample 5 random subsets
            indices = np.random.choice(self.d, self.k, replace=False)
            subset_prod = torch.prod(X[:, indices], dim=1)
            corr = torch.mean(y * subset_prod).item()
            random_subset_corrs.append(corr)
        
        return {
            'true_subset': true_subset_corr,
            'random_subsets': random_subset_corrs
        }
    
    def train(self, n_epochs=10000, eval_interval=100, verbose=True):
        """Train the network and track relevant metrics for phase transition analysis"""
        print(f"Starting training for {n_epochs} epochs...")
        
        # For timing
        start_time = time.time()
        
        # Initial metrics
        self._compute_and_store_metrics(0)
        
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
            
            # Evaluation and metrics
            if epoch % eval_interval == 0 or epoch == n_epochs:
                self._compute_and_store_metrics(epoch)
                
                if verbose and (epoch % (eval_interval * 10) == 0 or epoch == n_epochs):
                    elapsed = time.time() - start_time
                    Z_t = self.history['weight_ratio_L1'][-1]
                    W2_pos = self.history['W2_positive_ratio'][-1]
                    corr = self.history['correlation'][-1]
                    print(f"Epoch {epoch}: Loss={loss.item():.6f}, Correlation={corr:.4f}, Z_t^(1)={Z_t:.4f}, W2_pos={W2_pos:.2f}, Time={elapsed:.1f}s")
        
        print("Training completed!")
        return self.history
    
    def _compute_and_store_metrics(self, epoch):
        """Compute and store all metrics for the current epoch"""
        self.model.eval()
        with torch.no_grad():
            # Compute test loss and correlation
            preds = self.model(self.X_test)
            test_loss = self.criterion(preds, self.y_test).item()
            correlation = torch.corrcoef(torch.stack([preds, self.y_test]))[0, 1].item()
            
            # Compute layer metrics
            layer1_metrics = self.compute_layer1_metrics()
            layer2_metrics = self.compute_layer2_metrics()
            
            # Compute correlation metrics
            empirical_correlation = self.compute_empirical_correlation()
            
            # Store metrics
            self.history['epochs'].append(epoch)
            self.history['loss'].append(test_loss)
            self.history['correlation'].append(correlation)
            
            # Store first layer metrics
            self.history['W1_rel_norm'].append(layer1_metrics['W1_rel_norm'])
            self.history['W1_irr_norm'].append(layer1_metrics['W1_irr_norm'])
            self.history['weight_ratio_L1'].append(layer1_metrics['weight_ratio_L1'])
            self.history['signal_strength_L1'].append(layer1_metrics['signal_strength_L1'])
            self.history['alignment_scores_L1'].append(layer1_metrics['alignment_scores_L1'])
            
            # Store second layer metrics
            self.history['W2_positive_ratio'].append(layer2_metrics['W2_positive_ratio'])
            self.history['W2_negative_ratio'].append(layer2_metrics['W2_negative_ratio'])
            self.history['W2_pos_neg_ratio'].append(layer2_metrics['W2_pos_neg_ratio'])
            self.history['L2_activation_sparsity'].append(layer2_metrics['L2_activation_sparsity'])
            
            # Store correlation metrics
            self.history['empirical_correlation'].append(empirical_correlation)
    
    def visualize_phase_transition(self):
        """Create visualizations that demonstrate the phase transition"""
        epochs = self.history['epochs']
        
        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # Plot 1: First Layer Weight Norms
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['W1_rel_norm'], 'b-', linewidth=2, label='||W1,rel||_F')
        ax1.plot(epochs, self.history['W1_irr_norm'], 'r-', linewidth=2, label='||W1,irr||_F')
        
        # Add vertical line at theoretical critical time
        ax1.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Frobenius Norm', fontsize=14)
        ax1.set_title('First Layer Weight Norms', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True)
        
        # Plot 2: First Layer Weight Ratio (Z_t^(1))
        ax2 = axes[0, 1]
        ax2.semilogy(epochs, self.history['weight_ratio_L1'], 'k-', linewidth=2, label='Z_t^(1)')
        
        # Add horizontal line at Z_t = 1
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Z_t = 1')
        
        # Add vertical line at theoretical critical time
        ax2.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Weight Ratio Z_t^(1) (log scale)', fontsize=14)
        ax2.set_title('First Layer Weight Ratio', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True)
        
        # Plot 3: First Layer Signal Strength (Ψ_t^(1))
        ax3 = axes[0, 2]
        ax3.plot(epochs, self.history['signal_strength_L1'], 'm-', linewidth=2, label='Ψ_t^(1)')
        
        # Add vertical line at theoretical critical time
        ax3.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax3.set_xlabel('Epoch', fontsize=14)
        ax3.set_ylabel('Signal Strength Ψ_t^(1)', fontsize=14)
        ax3.set_title('First Layer Signal Strength', fontsize=16)
        ax3.legend(fontsize=12)
        ax3.grid(True)
        
        # Plot 4: Second Layer Weight Positivity
        ax4 = axes[1, 0]
        ax4.plot(epochs, self.history['W2_positive_ratio'], 'b-', linewidth=2, label='W2 > 0')
        ax4.plot(epochs, self.history['W2_negative_ratio'], 'r-', linewidth=2, label='W2 < 0')
        
        # Add horizontal line at 0.5 (random initialization)
        ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Random Init (50%)')
        
        # Add vertical line at theoretical critical time
        ax4.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax4.set_xlabel('Epoch', fontsize=14)
        ax4.set_ylabel('Proportion', fontsize=14)
        ax4.set_title('Second Layer Weight Sign Distribution', fontsize=16)
        ax4.legend(fontsize=12)
        ax4.grid(True)
        
        # Plot 5: Second Layer Positive/Negative Magnitude Ratio
        ax5 = axes[1, 1]
        ax5.semilogy(epochs, self.history['W2_pos_neg_ratio'], 'c-', linewidth=2, label='|W2+|/|W2-|')
        
        # Add horizontal line at ratio = 1
        ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Equal Magnitude')
        
        # Add vertical line at theoretical critical time
        ax5.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax5.set_xlabel('Epoch', fontsize=14)
        ax5.set_ylabel('Magnitude Ratio (log scale)', fontsize=14)
        ax5.set_title('Second Layer Pos/Neg Weight Magnitude Ratio', fontsize=16)
        ax5.legend(fontsize=12)
        ax5.grid(True)
        
        # Plot 6: Layer 2 Activation Sparsity
        ax6 = axes[1, 2]
        ax6.plot(epochs, self.history['L2_activation_sparsity'], 'g-', linewidth=2, label='L2 Sparsity')
        
        # Add vertical line at theoretical critical time
        ax6.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax6.set_xlabel('Epoch', fontsize=14)
        ax6.set_ylabel('Proportion Inactive', fontsize=14)
        ax6.set_title('Second Layer Activation Sparsity', fontsize=16)
        ax6.legend(fontsize=12)
        ax6.grid(True)
        
        plt.suptitle(f"{self.k}-Parity ({'Odd' if self.is_odd_parity else 'Even'}) Phase Transition Analysis (d={self.d})", 
                   fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{self.save_dir}/phase_transition_analysis_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()


def run_analysis_for_parity_type(d, parity_type, hidden1_width=128, hidden2_width=128):
    """Run analysis for either even or odd parity"""
    if parity_type == "even":
        k = 4  # Even parity
    else:
        k = 5  # Odd parity
    
    # Create save directory
    save_dir = f"parity_analysis_{parity_type}_d{d}_k{k}"
    
    # Calculate required number of epochs based on theoretical T_c
    learning_rate = 0.01
    T_c = int(np.ceil((d**(k/2)) / learning_rate))
    n_epochs = max(5000, 3 * T_c)  # Ensure we go well past T_c
    
    # Create analyzer
    analyzer = TwoLayerParityPhaseAnalyzer(
        d=d,
        k=k,
        hidden1_width=hidden1_width,
        hidden2_width=hidden2_width,
        learning_rate=learning_rate,
        batch_size=512,
        save_dir=save_dir
    )
    
    # Train and analyze
    analyzer.train(n_epochs=n_epochs, eval_interval=max(1, n_epochs//100), verbose=True)
    
    # Create visualizations
    analyzer.visualize_phase_transition()
    
    print(f"\nAnalysis complete for {parity_type} parity. Results saved to {save_dir}")
    
    return analyzer


def test_random_projection_lemma():
    """
    Test Lemma 2.3 (Random Projection Reduction) empirically
    
    This function tests whether the effective dimensionality of the correlation search
    space is indeed reduced from O(d^k) to O(d^(k/2)) through random projections.
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING LEMMA 2.3: RANDOM PROJECTION REDUCTION")
    print(f"{'='*80}\n")
    
    # Define dimensions and sparsity values to test
    d_values = [20, 30, 40, 45]
    k_values = [ 3, 4,5,6,7]
    
    # Create analyzer
    analyzer = RandomProjectionAnalyzer(d_values, k_values)
    
    # Run analysis
    analyzer.analyze_random_projection_scaling()
    
    # Create visualizations
    analyzer.visualize_scaling()
    
    print(f"\nRandom projection analysis complete.")
    print(f"See 'random_projection_scaling.png' and 'random_projection_ratio.png' for results.")


def run_parity_analysis_and_projection_test():
    """Run both parity analysis and random projection tests"""
    # First, verify the random projection lemma (Lemma 2.3)
    test_random_projection_lemma()
    
    # Then run parity analysis for even and odd parity
    d = 30  # Input dimension
    hidden1_width = 256
    hidden2_width = 256
    
    print(f"\n{'='*80}")
    print(f"COMPARING EVEN VS ODD PARITY FUNCTIONS WITH TWO-LAYER NETWORK")
    print(f"{'='*80}\n")
    
    # Run even parity analysis
    print(f"\n-- ANALYZING EVEN PARITY --\n")
    even_analyzer = run_analysis_for_parity_type(d, "even", hidden1_width, hidden2_width)
    
    # Run odd parity analysis
    print(f"\n-- ANALYZING ODD PARITY --\n")
    odd_analyzer = run_analysis_for_parity_type(d, "odd", hidden1_width, hidden2_width)
    
    # Create comparison plots
    plt.figure(figsize=(16, 12))
    
    # Plot W2 negativity comparison
    plt.subplot(2, 2, 1)
    plt.plot(even_analyzer.history['epochs'], even_analyzer.history['W2_negative_ratio'], 'b-', linewidth=2, label='Even Parity (k=4)')
    plt.plot(odd_analyzer.history['epochs'], odd_analyzer.history['W2_negative_ratio'], 'r-', linewidth=2, label='Odd Parity (k=5)')
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Initial (50%)')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('W2 Negative Proportion', fontsize=14)
    plt.title('Second Layer Negativity Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Plot L2 sparsity comparison
    plt.subplot(2, 2, 2)
    plt.plot(even_analyzer.history['epochs'], even_analyzer.history['L2_activation_sparsity'], 'b-', linewidth=2, label='Even Parity (k=4)')
    plt.plot(odd_analyzer.history['epochs'], odd_analyzer.history['L2_activation_sparsity'], 'r-', linewidth=2, label='Odd Parity (k=5)')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('L2 Activation Sparsity', fontsize=14)
    plt.title('Second Layer Activation Sparsity Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Plot W1 ratio comparison
    plt.subplot(2, 2, 3)
    plt.semilogy(even_analyzer.history['epochs'], even_analyzer.history['weight_ratio_L1'], 'b-', linewidth=2, label='Even Parity (k=4)')
    plt.semilogy(odd_analyzer.history['epochs'], odd_analyzer.history['weight_ratio_L1'], 'r-', linewidth=2, label='Odd Parity (k=5)')
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('First Layer Weight Ratio (log scale)', fontsize=14)
    plt.title('First Layer Weight Ratio Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Plot correlation comparison
    plt.subplot(2, 2, 4)
    plt.plot(even_analyzer.history['epochs'], even_analyzer.history['correlation'], 'b-', linewidth=2, label='Even Parity (k=4)')
    plt.plot(odd_analyzer.history['epochs'], odd_analyzer.history['correlation'], 'r-', linewidth=2, label='Odd Parity (k=5)')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Test Correlation', fontsize=14)
    plt.title('Learning Performance Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plt.suptitle('Even vs. Odd Parity Learning with Two-Layer ReLU Network', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"even_vs_odd_parity_comparison_d{d}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Run both the random projection analysis and parity analysis
    run_parity_analysis_and_projection_test()