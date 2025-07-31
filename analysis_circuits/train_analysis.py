import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import os
from tqdm import tqdm
import seaborn as sns
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
d = 45              # Input dimension
k = 6               # Monomial order (x_1 * x_2 * ... * x_k)
M = 4096             # Network width (M = Θ(d))
batch_size = 512    # Batch size for training (approximating online SGD)
n_epochs = 5000000  # Maximum number of epochs
learning_rate = 0.001
log_interval = 500   # How often to log metrics
save_path = f"monomial_k{k}_d{d}_results"
os.makedirs(save_path, exist_ok=True)

# Define the target monomial function: f*(x) = x_1 * x_2 * ... * x_k
def target_function(x):
    return torch.prod(x[:, :k], dim=1)

# Define a two-layer ReLU network without bias
class TwoLayerReLUNet(torch.nn.Module):
    def __init__(self, input_dim, width):
        super(TwoLayerReLUNet, self).__init__()
        # Initialize layers
        self.w = torch.nn.Parameter(torch.randn(width, input_dim) / np.sqrt(input_dim))
        self.a = torch.nn.Parameter(torch.randn(width) / np.sqrt(width))
        
    def forward(self, x):
        # Two-layer ReLU network without bias: sum_j a_j * σ(w_j^T x)
        hidden = torch.relu(torch.matmul(x, self.w.t()))
        output = torch.matmul(hidden, self.a)
        return output

# Function to calculate all metrics
def calculate_metrics(model, X_test, S_subsets):
    model.eval()
    with torch.no_grad():
        # Get model parameters
        a = model.a.detach()
        w = model.w.detach()
        
        # Target values
        y_true = target_function(X_test)
        
        # Model predictions
        y_pred = model(X_test)
        
        # 1. Training error
        error = torch.mean((y_true - y_pred) ** 2).item()
        
        # 2. Target correlation
        correlation = torch.mean(y_true * y_pred).item()
        
        # 3. Per-neuron metrics
        alpha = torch.abs(a)  # Output weight magnitude
        w_norm = torch.sqrt(torch.sum(w ** 2, dim=1))  # Weight norm
        w_S = w[:, :k]  # Weights for the support of the monomial
        w_S_norm = torch.sqrt(torch.sum(w_S ** 2, dim=1))  # Support weight norm
        beta = w_S_norm / w_norm  # Alignment with target support
        
        # Calculate neuron-target correlation
        activations = torch.relu(torch.matmul(X_test, w.t()))
        rho = torch.mean(y_true.unsqueeze(1) * activations, dim=0)
        
        # 4. Feature correlations for different subset orders
        feature_correlations = {}
        for subset in S_subsets:
            subset_str = '_'.join(map(str, subset))
            if len(subset) > 0:
                feature = torch.prod(X_test[:, subset], dim=1)
                corr = torch.mean(feature * y_pred).item()
                feature_correlations[subset_str] = corr
        
        # 5. Product of alpha and beta
        alpha_beta = alpha * beta
        
        # 6. Specialized neuron fraction (neurons with α_j(t)β_j(t) > τ)
        tau = 0.1  # Threshold for specialization
        specialized_count = torch.sum(alpha_beta > tau).item()
        specialized_fraction = specialized_count / len(alpha_beta)
        
        # Store all metrics
        metrics = {
            'error': error,
            'correlation': correlation,
            'alpha': alpha.cpu().numpy(),
            'beta': beta.cpu().numpy(),
            'rho': rho.cpu().numpy(),
            'alpha_beta': alpha_beta.cpu().numpy(),
            'feature_correlations': feature_correlations,
            'specialized_count': specialized_count,
            'specialized_fraction': specialized_fraction
        }
        
        return metrics

# Generate all subsets of {0, 1, ..., k-1} for feature correlations
def generate_subsets(k):
    subsets = []
    # Generate all subsets of size l for l in {1, 2, ..., k}
    for l in range(1, k+1):
        subset_l = []
        def backtrack(start, curr):
            if len(curr) == l:
                subset_l.append(curr[:])
                return
            for i in range(start, k):
                curr.append(i)
                backtrack(i+1, curr)
                curr.pop()
        backtrack(0, [])
        subsets.extend(subset_l)
    return subsets

# Create visualization functions
def plot_neuron_distribution(metrics_history, save_path):
    # Select 4 time points
    epochs = len(metrics_history)
    time_points = [0, epochs//10, epochs//3, epochs-1]
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    for i, t in enumerate(time_points):
        alpha = metrics_history[t]['alpha']
        beta = metrics_history[t]['beta']
        
        # Create 2D histogram
        h, xedges, yedges, im = axs[i].hist2d(alpha, beta, bins=30, cmap='viridis', norm=LogNorm())
        
        # Add colorbar
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        axs[i].set_title(f'Epoch {t}')
        axs[i].set_xlabel('α (output weight magnitude)')
        axs[i].set_ylabel('β (input weight alignment)')
        
    plt.tight_layout()
    plt.savefig(f"{save_path}/neuron_distribution.png", dpi=300)
    plt.close()

def plot_learning_curves(metrics_history, save_path):
    epochs = range(len(metrics_history))
    
    # Extract metrics over time
    errors = [m['error'] for m in metrics_history]
    correlations = [m['correlation'] for m in metrics_history]
    specialized_fractions = [m['specialized_fraction'] for m in metrics_history]
    
    # Create plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot error
    axs[0].plot(epochs, errors)
    axs[0].set_ylabel('Mean Squared Error')
    axs[0].set_yscale('log')
    axs[0].grid(True)
    
    # Plot correlation
    axs[1].plot(epochs, correlations)
    axs[1].set_ylabel('Target Correlation')
    axs[1].grid(True)
    
    # Plot specialized neuron fraction
    axs[2].plot(epochs, specialized_fractions)
    axs[2].set_ylabel('Specialized Neuron Fraction')
    axs[2].set_xlabel('Epoch')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/learning_curves.png", dpi=300)
    plt.close()

def plot_feature_learning(metrics_history, S_subsets, save_path):
    epochs = range(len(metrics_history))
    
    # Group features by order (size of subset)
    features_by_order = {}
    for subset in S_subsets:
        subset_str = '_'.join(map(str, subset))
        order = len(subset)
        if order not in features_by_order:
            features_by_order[order] = []
        features_by_order[order].append(subset_str)
    
    # Plot correlation for each feature order
    plt.figure(figsize=(12, 8))
    
    for order in range(1, k+1):
        # Average correlation across all features of this order
        if order in features_by_order:
            order_correlations = []
            for t in range(len(metrics_history)):
                correlations = [metrics_history[t]['feature_correlations'].get(subset_str, 0) 
                               for subset_str in features_by_order[order]]
                avg_correlation = np.mean(correlations)
                order_correlations.append(avg_correlation)
            
            plt.plot(epochs, order_correlations, label=f'Order {order}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Feature Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/feature_learning.png", dpi=300)
    plt.close()

def plot_alpha_beta_growth(metrics_history, save_path):
    epochs = range(len(metrics_history))
    
    # Track top 5 neurons that end up with highest alpha_beta
    final_alpha_beta = metrics_history[-1]['alpha_beta']
    top_indices = np.argsort(final_alpha_beta)[-5:]
    
    # Plot alpha_beta growth for these neurons
    plt.figure(figsize=(10, 6))
    
    for i in top_indices:
        alpha_beta_values = [m['alpha_beta'][i] for m in metrics_history]
        plt.plot(epochs, alpha_beta_values, label=f'Neuron {i}')
    
    plt.xlabel('Epoch')
    plt.ylabel('α_j(t)β_j(t)')
    plt.title('Growth of α_j(t)β_j(t) for Top Specialized Neurons')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f"{save_path}/alpha_beta_growth.png", dpi=300)
    plt.close()

# NEW FUNCTION: Plot alpha*beta for all neurons with color coding and transparency
def plot_alpha_beta_all_neurons(metrics_history, save_path):
    # Select 4 time points for visualization
    epochs = len(metrics_history)
    time_points = [0, epochs//10, epochs//3, epochs-1]
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    # Create colormap for alpha*beta values
    cmap = plt.cm.viridis
    
    for i, t in enumerate(time_points):
        alpha_beta = metrics_history[t]['alpha_beta']
        
        # Normalize for color mapping
        norm = plt.Normalize(np.min(alpha_beta), np.max(alpha_beta))
        colors = cmap(norm(alpha_beta))
        
        # Add transparency based on value (higher values more opaque)
        # Scale transparency between 0.2 and 0.9
        min_alpha, max_alpha = 0.2, 0.9
        transparency = min_alpha + (max_alpha - min_alpha) * norm(alpha_beta)
        
        # Set transparency
        colors[:, 3] = transparency
        
        # Create scatter plot with neuron indices on x-axis
        neuron_indices = np.arange(len(alpha_beta))
        axs[i].scatter(neuron_indices, alpha_beta, c=colors, s=50, edgecolor='none')
        
        # Add horizontal line at specialization threshold
        axs[i].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Specialization Threshold')
        
        # Set plot properties
        axs[i].set_title(f'Epoch {t*log_interval if t > 0 else 0}')
        axs[i].set_xlabel('Neuron Index')
        axs[i].set_ylabel('α_j(t)β_j(t)')
        axs[i].set_yscale('log')
        axs[i].grid(True, alpha=0.3)
        
        if i == 0:
            axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/alpha_beta_all_neurons.png", dpi=300)
    plt.close()

def plot_neuron_clustering(metrics_history, save_path):
    # Analyze final state for clustering
    final_metrics = metrics_history[-1]
    X = np.column_stack((final_metrics['alpha'], final_metrics['beta']))
    
    # Apply K-means clustering with 2 clusters (specialized vs. non-specialized)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Determine which cluster is specialized (higher mean alpha_beta product)
    mean_alpha_beta = [np.mean(final_metrics['alpha_beta'][clusters == i]) for i in range(2)]
    specialized_cluster = np.argmax(mean_alpha_beta)
    
    # Create colormap
    colors = ['blue' if c != specialized_cluster else 'red' for c in clusters]
    
    # Create two plots: original and with log axes
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
    # Original plot
    axs[0].scatter(final_metrics['alpha'], final_metrics['beta'], c=colors, alpha=0.6)
    axs[0].set_xlabel('α (output weight magnitude)')
    axs[0].set_ylabel('β (input weight alignment)')
    axs[0].set_title('Neuron Clustering: Red = Specialized, Blue = Non-specialized')
    axs[0].grid(True, alpha=0.3)
    
    # Log-scaled plot
    axs[1].scatter(final_metrics['alpha'], final_metrics['beta'], c=colors, alpha=0.6)
    axs[1].set_xlabel('α (output weight magnitude)')
    axs[1].set_ylabel('β (input weight alignment)')
    axs[1].set_title('Neuron Clustering (Log Scales)')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/neuron_clustering.png", dpi=300)
    plt.close()

def plot_phase_transition(metrics_history, save_path):
    epochs = range(len(metrics_history))
    
    # Define metrics for detecting phase transition
    errors = np.array([m['error'] for m in metrics_history])
    specialized_fractions = np.array([m['specialized_fraction'] for m in metrics_history])
    correlations = np.array([m['correlation'] for m in metrics_history])
    
    # Calculate rate of change (use smoothed derivatives)
    window = 5
    if len(epochs) > window:
        error_deriv = np.convolve(-np.log(errors), np.ones(window)/window, mode='valid')
        specialized_deriv = np.convolve(specialized_fractions, np.ones(window)/window, mode='valid')
        correlation_deriv = np.convolve(correlations, np.ones(window)/window, mode='valid')
        
        # Plot derivatives to identify phase transitions
        plt.figure(figsize=(12, 8))
        plt.plot(epochs[window-1:], error_deriv, label='Error Rate of Change')
        plt.plot(epochs[window-1:], specialized_deriv, label='Specialization Rate of Change')
        plt.plot(epochs[window-1:], correlation_deriv, label='Correlation Rate of Change')
        plt.xlabel('Epoch')
        plt.ylabel('Rate of Change')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/phase_transition.png", dpi=300)
        plt.close()

# Main training loop
def train_and_analyze():
    print(f"Analyzing k={k} monomial in d={d} dimensions with {M} neurons")
    
    # Generate all subsets for feature correlation analysis
    S_subsets = generate_subsets(k)
    
    # Initialize model
    model = TwoLayerReLUNet(d, M).to(device)
    
    # Use MSE loss and basic SGD
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Store metrics history
    metrics_history = []
    
    # Generate test dataset for consistent metric calculation
    X_test = torch.tensor(np.random.choice([-1, 1], size=(10000, d)), dtype=torch.float32).to(device)
    
    # Measure initial metrics
    initial_metrics = calculate_metrics(model, X_test, S_subsets)
    metrics_history.append(initial_metrics)
    
    # Training loop
    for epoch in tqdm(range(n_epochs)):
        model.train()
        
        # Generate a batch of random inputs (approximating online SGD)
        X = torch.tensor(np.random.choice([-1, 1], size=(batch_size, d)), dtype=torch.float32).to(device)
        y = target_function(X)
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log metrics at intervals
        if (epoch + 1) % log_interval == 0:
            metrics = calculate_metrics(model, X_test, S_subsets)
            metrics_history.append(metrics)
            
            # Print progress
            if (epoch + 1) % (log_interval * 10) == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Error: {metrics['error']:.6f}, Correlation: {metrics['correlation']:.6f}, Specialized: {metrics['specialized_fraction']:.4f}")
            
            # Early stopping if nearly perfect correlation
            if metrics['correlation'] > 0.99:
                print(f"Reached high correlation at epoch {epoch+1}")
                break
    
    # Final metrics
    final_metrics = calculate_metrics(model, X_test, S_subsets)
    metrics_history.append(final_metrics)
    
    # Save all metrics data
    np.savez_compressed(
        f"{save_path}/metrics_data.npz",
        metrics_history=metrics_history,
        d=d,
        k=k,
        M=M
    )
    
    # Create all plots
    print("Creating visualizations...")
    plot_neuron_distribution(metrics_history, save_path)
    plot_learning_curves(metrics_history, save_path)
    plot_feature_learning(metrics_history, S_subsets, save_path)
    plot_alpha_beta_growth(metrics_history, save_path)
    plot_neuron_clustering(metrics_history, save_path)
    plot_phase_transition(metrics_history, save_path)
    
    # Add the new visualization
    plot_alpha_beta_all_neurons(metrics_history, save_path)
    
    # Calculate learning times for each order
    feature_orders = {}
    for subset in S_subsets:
        subset_str = '_'.join(map(str, subset))
        order = len(subset)
        if order not in feature_orders:
            feature_orders[order] = []
        feature_orders[order].append(subset_str)
    
    # Define threshold for considering a feature "learned"
    threshold = 0.9
    learning_times = {}
    
    for order in range(1, k+1):
        if order in feature_orders:
            for t in range(len(metrics_history)):
                avg_correlation = np.mean([metrics_history[t]['feature_correlations'].get(subset_str, 0) 
                                          for subset_str in feature_orders[order]])
                if avg_correlation > threshold and order not in learning_times:
                    learning_times[order] = t * log_interval
                    break
    
    # Calculate specialized neuron emergence time (T_discovery)
    specialized_threshold = 0.05  # 5% of neurons specialized
    T_discovery = None
    for t in range(len(metrics_history)):
        if metrics_history[t]['specialized_fraction'] > specialized_threshold:
            T_discovery = t * log_interval
            break
    
    # Print discovery and learning times
    print("\nAnalysis Results:")
    print(f"T_discovery (Manifold Discovery Time): {T_discovery} epochs")
    print("\nFeature Learning Times:")
    for order in sorted(learning_times.keys()):
        print(f"Order {order}: {learning_times[order]} epochs")
    
    # If we have times for multiple orders, check scaling
    if len(learning_times) > 1:
        orders = sorted(learning_times.keys())
        for i in range(1, len(orders)):
            ratio = learning_times[orders[i]] / learning_times[orders[i-1]]
            print(f"Learning time ratio Order {orders[i]}/Order {orders[i-1]}: {ratio:.2f}")
    
    print(f"\nAll results saved to {save_path}")
    return metrics_history, learning_times, T_discovery

if __name__ == "__main__":
    start_time = time.time()
    metrics_history, learning_times, T_discovery = train_and_analyze()
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")