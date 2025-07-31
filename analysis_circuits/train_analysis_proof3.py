import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

class TwoLayerReLUNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_width):
        super(TwoLayerReLUNet, self).__init__()
        self.W1 = torch.nn.Parameter(torch.randn(hidden_width, input_dim) / np.sqrt(input_dim))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_width))
        self.a = torch.nn.Parameter(torch.randn(1, hidden_width) / np.sqrt(hidden_width))
        
    def forward(self, x):
        h = torch.relu(torch.matmul(x, self.W1.t()) + self.b1)
        output = torch.matmul(h, self.a.t())
        return output.squeeze()
    
    def get_activations(self, x):
        return torch.relu(torch.matmul(x, self.W1.t()) + self.b1)


class ParityPhaseTransitionAnalyzer:
    def __init__(self, d=30, k=6, hidden_width=128, learning_rate=0.01, 
                 batch_size=512, device=None, save_dir="parity_phase_transition"):
        self.d = d
        self.k = k
        self.hidden_width = hidden_width
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize model, optimizer, and loss function
        self.model = TwoLayerReLUNet(d, hidden_width).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        
        # History storage
        self.history = {
            'epochs': [],
            'loss': [],
            'correlation': [],
            'weight_ratio': [],  # Z_t = ||W_{1,irr}||_F^2 / ||W_{1,rel}||_F^2
            'signal_strength': [],  # Ψ_t
            'alignment_scores': [],  # α_i^(t)
            'empirical_correlation': [],  # Ĉ_A^(t)
            'W1_rel_norm': [],  # ||W_{1,rel}||_F
            'W1_irr_norm': []   # ||W_{1,irr}||_F
        }
        
        # Test data for consistent evaluation
        self.X_test = torch.tensor(np.random.choice([-1, 1], size=(1000, d)), 
                                  dtype=torch.float32).to(self.device)
        self.y_test = self._target_function(self.X_test)
        
        # Calculate theoretical critical time
        self.T_c_theoretical = int(np.ceil((d**(k/2)) / learning_rate))
        
        print(f"ParityPhaseTransitionAnalyzer initialized on {self.device}")
        print(f"Analyzing {k}-parity function in {d} dimensions")
        print(f"Network: {d} → {hidden_width} → 1")
        print(f"Theoretical critical time T_c = {self.T_c_theoretical}")
    
    def _target_function(self, x):
        """Compute the k-sparse parity function on the first k inputs"""
        return torch.prod(x[:, :self.k], dim=1)
    
    def compute_weight_ratio(self):
        """Compute the weight ratio Z_t = ||W_{1,irr}||_F^2 / ||W_{1,rel}||_F^2"""
        W1 = self.model.W1.detach().cpu().numpy()
        W1_rel = W1[:, :self.k]
        W1_irr = W1[:, self.k:]
        
        rel_norm = np.linalg.norm(W1_rel, 'fro')**2
        irr_norm = np.linalg.norm(W1_irr, 'fro')**2
        
        # Avoid division by zero
        if rel_norm == 0:
            return float('inf')
        
        return irr_norm / rel_norm
    
    def compute_signal_strength(self):
        """Compute signal strength Ψ_t = ∑_{i=1}^M |∑_{l∈S} W_{1,il}^(t)|"""
        W1 = self.model.W1.detach().cpu().numpy()
        W1_rel = W1[:, :self.k]
        
        # Sum of absolute values of row sums
        signal_strength = np.sum(np.abs(np.sum(W1_rel, axis=1)))
        
        return signal_strength
    
    def compute_alignment_scores(self):
        """Compute alignment scores α_i^(t) for each neuron"""
        W1 = self.model.W1.detach().cpu().numpy()
        W1_rel = W1[:, :self.k]
        
        # Calculate alignment score for each neuron
        alignment_scores = []
        for i in range(self.hidden_width):
            row_sum = np.sum(W1_rel[i])
            row_norm = np.linalg.norm(W1_rel[i])
            
            # Avoid division by zero
            if row_norm == 0:
                alignment_scores.append(0)
            else:
                alignment_scores.append(np.abs(row_sum) / row_norm)
        
        return alignment_scores
    
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
        
        # Initialize arrays to store norms over time
        epochs = []
        w1_rel_norms = []
        w1_irr_norms = []
        
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
                    Z_t = self.history['weight_ratio'][-1]
                    corr = self.history['correlation'][-1]
                    print(f"Epoch {epoch}: Loss={loss.item():.6f}, Correlation={corr:.4f}, Z_t={Z_t:.4f}, Time={elapsed:.1f}s")
        
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
            
            # Compute weight norms
            W1 = self.model.W1.detach().cpu().numpy()
            W1_rel_norm = np.linalg.norm(W1[:, :self.k], 'fro')
            W1_irr_norm = np.linalg.norm(W1[:, self.k:], 'fro')
            
            # Compute other metrics
            weight_ratio = self.compute_weight_ratio()
            signal_strength = self.compute_signal_strength()
            alignment_scores = self.compute_alignment_scores()
            empirical_correlation = self.compute_empirical_correlation()
            
            # Store metrics
            self.history['epochs'].append(epoch)
            self.history['loss'].append(test_loss)
            self.history['correlation'].append(correlation)
            self.history['weight_ratio'].append(weight_ratio)
            self.history['signal_strength'].append(signal_strength)
            self.history['alignment_scores'].append(alignment_scores)
            self.history['empirical_correlation'].append(empirical_correlation)
            self.history['W1_rel_norm'].append(W1_rel_norm)
            self.history['W1_irr_norm'].append(W1_irr_norm)
    
    def visualize_phase_transition(self):
        """Create visualizations that demonstrate the phase transition"""
        epochs = self.history['epochs']
        W1_rel_norms = self.history['W1_rel_norm']
        W1_irr_norms = self.history['W1_irr_norm']
        weight_ratios = self.history['weight_ratio']
        signal_strengths = self.history['signal_strength']
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Weight norms over time
        ax1 = axes[0, 0]
        ax1.plot(epochs, W1_rel_norms, 'b-', linewidth=2, label='||W1,rel||_F')
        ax1.plot(epochs, W1_irr_norms, 'r-', linewidth=2, label='||W1,irr||_F')
        
        # Add vertical line at theoretical critical time
        ax1.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Frobenius Norm', fontsize=14)
        ax1.set_title('Weight Norms Over Time', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True)
        
        # Plot 2: Weight ratio Z_t over time
        ax2 = axes[0, 1]
        ax2.semilogy(epochs, weight_ratios, 'k-', linewidth=2, label='Z_t')
        
        # Add horizontal line at Z_t = 1
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Z_t = 1')
        
        # Add vertical line at theoretical critical time
        ax2.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Weight Ratio Z_t (log scale)', fontsize=14)
        ax2.set_title('Weight Ratio Z_t = ||W1,irr||²_F / ||W1,rel||²_F Over Time', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True)
        
        # Plot 3: Signal strength Ψ_t over time
        ax3 = axes[1, 0]
        ax3.plot(epochs, signal_strengths, 'm-', linewidth=2, label='Ψ_t')
        
        # Add vertical line at theoretical critical time
        ax3.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax3.set_xlabel('Epoch', fontsize=14)
        ax3.set_ylabel('Signal Strength Ψ_t', fontsize=14)
        ax3.set_title('Signal Strength Ψ_t Over Time', fontsize=16)
        ax3.legend(fontsize=12)
        ax3.grid(True)
        
        # Plot 4: Average alignment score over time
        ax4 = axes[1, 1]
        avg_alignment = [np.mean(scores) for scores in self.history['alignment_scores']]
        ax4.plot(epochs, avg_alignment, 'c-', linewidth=2, label='Avg. α_i')
        
        # Add vertical line at theoretical critical time
        ax4.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        ax4.set_xlabel('Epoch', fontsize=14)
        ax4.set_ylabel('Average Alignment Score', fontsize=14)
        ax4.set_title('Average Neuron Alignment Score Over Time', fontsize=16)
        ax4.legend(fontsize=12)
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/phase_transition_analysis_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Create a second figure for empirical correlations
        plt.figure(figsize=(12, 8))
        
        # Extract correlation data
        true_subset_corrs = [ec['true_subset'] for ec in self.history['empirical_correlation']]
        
        # Calculate average of random subset correlations
        random_subset_avgs = []
        for ec in self.history['empirical_correlation']:
            random_subset_avgs.append(np.mean(np.abs(ec['random_subsets'])))
        
        plt.plot(epochs, true_subset_corrs, 'b-', linewidth=2, label='True Subset S')
        plt.plot(epochs, random_subset_avgs, 'r-', linewidth=2, label='Random Subsets (Avg. Abs)')
        
        # Add vertical line at theoretical critical time
        plt.axvline(x=self.T_c_theoretical, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical T_c = {self.T_c_theoretical}')
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Empirical Correlation', fontsize=14)
        plt.title('Empirical Correlations Ĉ_A Over Time', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/empirical_correlations_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        # Create a third figure for alignment score distribution at different phases
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Find indices for initial, near-critical, and final states
        initial_idx = 0
        
        # Find the epoch closest to theoretical T_c
        critical_idx = min(range(len(epochs)), key=lambda i: abs(epochs[i] - self.T_c_theoretical))
        
        final_idx = -1
        
        # Plot alignment score distributions
        axes[0].hist(self.history['alignment_scores'][initial_idx], bins=20, alpha=0.7)
        axes[0].set_title(f'Initial Alignment Scores (Epoch {epochs[initial_idx]})', fontsize=14)
        axes[0].set_xlabel('Alignment Score α_i', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].grid(True)
        
        axes[1].hist(self.history['alignment_scores'][critical_idx], bins=20, alpha=0.7)
        axes[1].set_title(f'Near-Critical Alignment Scores (Epoch {epochs[critical_idx]})', fontsize=14)
        axes[1].set_xlabel('Alignment Score α_i', fontsize=12)
        axes[1].grid(True)
        
        axes[2].hist(self.history['alignment_scores'][final_idx], bins=20, alpha=0.7)
        axes[2].set_title(f'Final Alignment Scores (Epoch {epochs[final_idx]})', fontsize=14)
        axes[2].set_xlabel('Alignment Score α_i', fontsize=12)
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/alignment_distributions_d{self.d}_k{self.k}.png", dpi=300)
        plt.close()
        
        print(f"Visualizations saved to {self.save_dir}")
    
    def find_empirical_critical_time(self):
        """Find the empirical critical time from the data"""
        epochs = self.history['epochs']
        W1_rel_norms = self.history['W1_rel_norm']
        
        # Find where the rel norm starts increasing significantly
        rel_norm_changes = np.diff(W1_rel_norms)
        
        # Apply a threshold to identify a "sharp" increase
        threshold = np.percentile(rel_norm_changes, 95)  # Top 5% of changes
        critical_indices = np.where(rel_norm_changes > threshold)[0]
        
        if len(critical_indices) == 0:
            return None
        
        # Get the first occurrence
        critical_idx = critical_indices[0]
        empirical_Tc = epochs[critical_idx + 1]  # +1 because of diff
        
        return empirical_Tc

    def verify_theoretical_bounds(self):
        """Verify if the theoretical bounds from the proof hold empirically"""
        # Find empirical critical time
        empirical_Tc = self.find_empirical_critical_time()
        
        if empirical_Tc is None:
            print("Could not determine empirical critical time")
            return
        
        # Calculate theoretical bounds
        theoretical_Tc = self.T_c_theoretical
        
        # Compute ratio of empirical to theoretical
        ratio = empirical_Tc / theoretical_Tc
        
        print(f"\nPhase Transition Analysis Results:")
        print(f"{'='*40}")
        print(f"Theoretical critical time T_c = {theoretical_Tc}")
        print(f"Empirical critical time T_c ≈ {empirical_Tc}")
        print(f"Ratio of empirical to theoretical: {ratio:.2f}")
        
        # Check if ratio is within reasonable bounds (e.g., 0.1 to 10)
        if 0.1 <= ratio <= 10:
            print(f"VERIFIED: Empirical critical time is within an order of magnitude of theoretical prediction!")
        else:
            print(f"NOT VERIFIED: Empirical critical time differs significantly from theoretical prediction.")
        
        # Check if weight ratio decays at the expected rate after T_c
        epochs = np.array(self.history['epochs'])
        weight_ratios = np.array(self.history['weight_ratio'])
        
        # Only consider epochs after empirical T_c
        post_critical_indices = epochs > empirical_Tc
        if np.any(post_critical_indices):
            post_critical_epochs = epochs[post_critical_indices]
            post_critical_ratios = weight_ratios[post_critical_indices]
            
            # Fit exponential decay model
            if len(post_critical_epochs) >= 2:
                from scipy.optimize import curve_fit
                
                def exp_decay(t, a, b):
                    return a * np.exp(-b * t)
                
                try:
                    # Normalize epochs relative to T_c
                    normalized_epochs = post_critical_epochs - empirical_Tc
                    
                    # Fit the model
                    params, _ = curve_fit(exp_decay, normalized_epochs, post_critical_ratios)
                    
                    decay_rate = params[1]
                    theoretical_decay_rate = self.learning_rate / self.d
                    
                    print(f"\nPost-Critical Weight Ratio Decay Analysis:")
                    print(f"{'='*40}")
                    print(f"Empirical decay rate: {decay_rate:.6f}")
                    print(f"Theoretical decay rate η/d: {theoretical_decay_rate:.6f}")
                    print(f"Ratio of empirical to theoretical: {decay_rate/theoretical_decay_rate:.2f}")
                    
                    # Check if decay rates are within an order of magnitude
                    if 0.1 <= decay_rate/theoretical_decay_rate <= 10:
                        print(f"VERIFIED: Post-critical decay rate is within an order of magnitude of theoretical prediction!")
                    else:
                        print(f"NOT VERIFIED: Post-critical decay rate differs significantly from theoretical prediction.")
                        
                except:
                    print("Could not fit exponential decay model to post-critical weight ratios")
        
        # Save results to file
        with open(f"{self.save_dir}/theoretical_verification_d{self.d}_k{self.k}.txt", 'w') as f:
            f.write(f"Phase Transition Analysis Results for {self.k}-Parity in {self.d} dimensions\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"Theoretical critical time T_c = {theoretical_Tc}\n")
            f.write(f"Empirical critical time T_c ≈ {empirical_Tc}\n")
            f.write(f"Ratio of empirical to theoretical: {ratio:.2f}\n\n")
            
            if 0.1 <= ratio <= 10:
                f.write(f"VERIFIED: Empirical critical time is within an order of magnitude of theoretical prediction!\n\n")
            else:
                f.write(f"NOT VERIFIED: Empirical critical time differs significantly from theoretical prediction.\n\n")
            
            if 'decay_rate' in locals() and 'theoretical_decay_rate' in locals():
                f.write(f"Post-Critical Weight Ratio Decay Analysis:\n")
                f.write(f"{'='*40}\n")
                f.write(f"Empirical decay rate: {decay_rate:.6f}\n")
                f.write(f"Theoretical decay rate η/d: {theoretical_decay_rate:.6f}\n")
                f.write(f"Ratio of empirical to theoretical: {decay_rate/theoretical_decay_rate:.2f}\n\n")
                
                if 0.1 <= decay_rate/theoretical_decay_rate <= 10:
                    f.write(f"VERIFIED: Post-critical decay rate is within an order of magnitude of theoretical prediction!\n")
                else:
                    f.write(f"NOT VERIFIED: Post-critical decay rate differs significantly from theoretical prediction.\n")


def run_phase_transition_analysis(d, k, hidden_width=128, learning_rate=0.01, n_epochs=5000):
    """Run a complete analysis of phase transition in k-sparse parity learning"""
    print(f"\n{'='*80}")
    print(f"Running phase transition analysis for {k}-parity in {d} dimensions")
    print(f"{'='*80}\n")
    
    # Create save directory
    save_dir = f"parity_phase_transition_d{d}_k{k}"
    
    # Create analyzer
    analyzer = ParityPhaseTransitionAnalyzer(
        d=d,
        k=k,
        hidden_width=hidden_width,
        learning_rate=learning_rate,
        batch_size=512,
        save_dir=save_dir
    )
    
    # Train and analyze
    analyzer.train(n_epochs=n_epochs, eval_interval=max(1, n_epochs//10000))
    
    # Create visualizations
    analyzer.visualize_phase_transition()
    
    # Verify theoretical bounds
    analyzer.verify_theoretical_bounds()
    
    print(f"\nAnalysis complete. Results saved to {save_dir}")
    
    return analyzer


if __name__ == "__main__":
    # Run the analysis with parameters that match the proof assumptions
    d = 50  # Input dimension
    k = 5   # Parity function order (k-sparse parity)
    
    # Calculate required number of epochs based on theoretical T_c
    learning_rate = 0.001
    T_c = int(np.ceil((d**(k/2))))  #int(np.ceil((d**(k/2)) / learning_rate))
    n_epochs = 50000000 #max(25000, 3 * T_c)  # Ensure we go well past T_c
    
    # Run the analysis
    analyzer = run_phase_transition_analysis(
        d=d,
        k=k,
        hidden_width=1024,
        learning_rate=learning_rate,
        n_epochs=n_epochs
    )