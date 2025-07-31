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
d = 30              # Input dimension
k = 4               # Monomial order (x_1 * x_2 * ... * x_k)
M1 = 512           # First hidden layer width
M2 = 512            # Second hidden layer width
batch_size = 512    # Batch size for training (approximating online SGD)
n_epochs = 1000000  # Maximum number of epochs
learning_rate = 0.01
log_interval = 50   # How often to log metrics
save_path = f"monomial_k{k}_d{d}_depth2_results"
os.makedirs(save_path, exist_ok=True)

# Define the target monomial function: f*(x) = x_1 * x_2 * ... * x_k
def target_function(x):
    return torch.prod(x[:, :k], dim=1)

# Define a network with 2 hidden layers
class DepthTwoReLUNet(torch.nn.Module):
    def __init__(self, input_dim, hidden1_width, hidden2_width):
        super(DepthTwoReLUNet, self).__init__()
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

# Function to calculate all metrics
def calculate_metrics(model, X_test, S_subsets):
    model.eval()
    with torch.no_grad():
        # Get model parameters
        a = model.a.detach()
        W1 = model.W1.detach()
        W2 = model.W2.detach()
        
        # Get activations
        h1 = model.get_first_layer_activations(X_test)
        h2 = model.get_second_layer_activations(X_test)
        
        # Target values
        y_true = target_function(X_test)
        
        # Model predictions
        y_pred = model(X_test)
        
        # 1. Training error
        error = torch.mean((y_true - y_pred) ** 2).item()
        
        # 2. Target correlation
        correlation = torch.mean(y_true * y_pred).item()
        
        # 3. Per-neuron metrics
        # First layer
        W1_S = W1[:, :k]  # Weights for the support of the monomial
        W1_norm = torch.sqrt(torch.sum(W1 ** 2, dim=1))
        W1_S_norm = torch.sqrt(torch.sum(W1_S ** 2, dim=1))
        beta1 = W1_S_norm / W1_norm  # Alignment with target support
        
        # Calculate first layer correlation with target
        rho1 = torch.mean(y_true.unsqueeze(1) * h1, dim=0)
        
        # Second layer
        alpha2 = torch.abs(a)  # Output weight magnitude
        
        # Calculate second layer correlation with target
        rho2 = torch.mean(y_true.unsqueeze(1) * h2, dim=0)
        
        # Layer interaction metrics
        W2_norms = torch.sqrt(torch.sum(W2 ** 2, dim=1))
        
        # Pattern detector scores: degree to which first layer neurons detect patterns
        # Higher scores indicate neurons that are more specialized
        detector_scores = torch.zeros(W1.shape[0])
        for i in range(W1.shape[0]):
            # How aligned is this neuron with the k relevant input features?
            detector_scores[i] = beta1[i] * torch.sum(torch.abs(W2[:, i])) / W1.shape[0]
        
        # Aggregator scores: degree to which second layer neurons aggregate patterns
        # Higher scores indicate neurons that are more specialized for aggregation
        aggregator_scores = torch.zeros(W2.shape[0])
        for i in range(W2.shape[0]):
            # How well does this neuron aggregate pattern detectors?
            top_detectors = torch.topk(torch.abs(W2[i]), min(10, W2.shape[1])).indices
            aggregator_scores[i] = torch.mean(beta1[top_detectors]) * alpha2[i]
        
        # 4. Feature correlations for different subset orders
        feature_correlations = {}
        for subset in S_subsets:
            subset_str = '_'.join(map(str, subset))
            if len(subset) > 0:
                feature = torch.prod(X_test[:, subset], dim=1)
                corr = torch.mean(feature * y_pred).item()
                feature_correlations[subset_str] = corr
        
        # 5. Specialized neuron fractions
        # First layer: Pattern detectors
        tau1 = 0.1  # Threshold for first layer specialization
        specialized_count1 = torch.sum(detector_scores > tau1).item()
        specialized_fraction1 = specialized_count1 / len(detector_scores)
        
        # Second layer: Aggregators
        tau2 = 0.1  # Threshold for second layer specialization
        specialized_count2 = torch.sum(aggregator_scores > tau2).item()
        specialized_fraction2 = specialized_count2 / len(aggregator_scores)
        
        # Store all metrics
        metrics = {
            'error': error,
            'correlation': correlation,
            'beta1': beta1.cpu().numpy(),
            'rho1': rho1.cpu().numpy(),
            'alpha2': alpha2.cpu().numpy(),
            'rho2': rho2.cpu().numpy(),
            'detector_scores': detector_scores.cpu().numpy(),
            'aggregator_scores': aggregator_scores.cpu().numpy(),
            'feature_correlations': feature_correlations,
            'specialized_count1': specialized_count1,
            'specialized_fraction1': specialized_fraction1,
            'specialized_count2': specialized_count2,
            'specialized_fraction2': specialized_fraction2,
            'W1': W1.cpu().numpy(),
            'W2': W2.cpu().numpy(),
            'a': a.cpu().numpy()
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
    
    # First layer: beta1 distribution
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    for i, t in enumerate(time_points):
        beta1 = metrics_history[t]['beta1']
        
        # Create histogram
        hist, bins, _ = axs[i].hist(beta1, bins=50, alpha=0.7)
        
        axs[i].set_title(f'Epoch {t*log_interval if t > 0 else 0}')
        axs[i].set_xlabel('β (First Layer Alignment)')
        axs[i].set_ylabel('Count')
        
    plt.tight_layout()
    plt.savefig(f"{save_path}/first_layer_alignment_distribution.png", dpi=300)
    plt.close()
    
    # Second layer: alpha2 distribution
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    for i, t in enumerate(time_points):
        alpha2 = metrics_history[t]['alpha2']
        
        # Create histogram
        hist, bins, _ = axs[i].hist(alpha2, bins=50, alpha=0.7)
        
        axs[i].set_title(f'Epoch {t*log_interval if t > 0 else 0}')
        axs[i].set_xlabel('α (Second Layer Output Weight)')
        axs[i].set_ylabel('Count')
        
    plt.tight_layout()
    plt.savefig(f"{save_path}/second_layer_weight_distribution.png", dpi=300)
    plt.close()

def plot_learning_curves(metrics_history, save_path):
    epochs = range(len(metrics_history))
    epoch_nums = [e * log_interval for e in epochs]
    
    # Extract metrics over time
    errors = [m['error'] for m in metrics_history]
    correlations = [m['correlation'] for m in metrics_history]
    specialized_fractions1 = [m['specialized_fraction1'] for m in metrics_history]
    specialized_fractions2 = [m['specialized_fraction2'] for m in metrics_history]
    
    # Create plot
    fig, axs = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    
    # Plot error
    axs[0].plot(epoch_nums, errors)
    axs[0].set_ylabel('Mean Squared Error')
    axs[0].set_yscale('log')
    axs[0].grid(True)
    
    # Plot correlation
    axs[1].plot(epoch_nums, correlations)
    axs[1].set_ylabel('Target Correlation')
    axs[1].grid(True)
    
    # Plot first layer specialized neuron fraction
    axs[2].plot(epoch_nums, specialized_fractions1)
    axs[2].set_ylabel('First Layer Specialized Fraction')
    axs[2].grid(True)
    
    # Plot second layer specialized neuron fraction
    axs[3].plot(epoch_nums, specialized_fractions2)
    axs[3].set_ylabel('Second Layer Specialized Fraction')
    axs[3].set_xlabel('Epoch')
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/learning_curves.png", dpi=300)
    plt.close()

def plot_feature_learning(metrics_history, S_subsets, save_path):
    epochs = range(len(metrics_history))
    epoch_nums = [e * log_interval for e in epochs]
    
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
            
            plt.plot(epoch_nums, order_correlations, label=f'Order {order}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Feature Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/feature_learning.png", dpi=300)
    plt.close()

def plot_layer_specialization(metrics_history, save_path):
    epochs = range(len(metrics_history))
    epoch_nums = [e * log_interval for e in epochs]
    
    # Track top 5 neurons from each layer that end up most specialized
    final_detector_scores = metrics_history[-1]['detector_scores']
    final_aggregator_scores = metrics_history[-1]['aggregator_scores']
    
    top_detectors = np.argsort(final_detector_scores)[-5:]
    top_aggregators = np.argsort(final_aggregator_scores)[-5:]
    
    # Plot detector scores for top first layer neurons
    plt.figure(figsize=(10, 6))
    for i in top_detectors:
        detector_scores = [m['detector_scores'][i] for m in metrics_history]
        plt.plot(epoch_nums, detector_scores, label=f'Detector {i}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Detector Score')
    plt.title('Pattern Detector Scores for Top Specialized First Layer Neurons')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f"{save_path}/detector_score_growth.png", dpi=300)
    plt.close()
    
    # Plot aggregator scores for top second layer neurons
    plt.figure(figsize=(10, 6))
    for i in top_aggregators:
        aggregator_scores = [m['aggregator_scores'][i] for m in metrics_history]
        plt.plot(epoch_nums, aggregator_scores, label=f'Aggregator {i}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Aggregator Score')
    plt.title('Pattern Aggregator Scores for Top Specialized Second Layer Neurons')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f"{save_path}/aggregator_score_growth.png", dpi=300)
    plt.close()

def plot_neuron_clustering(metrics_history, save_path):
    # Analyze final state for clustering
    final_metrics = metrics_history[-1]
    
    # First layer clustering: based on detector scores and beta1
    X1 = np.column_stack((final_metrics['detector_scores'], final_metrics['beta1']))
    
    # Apply K-means clustering with 2 clusters
    kmeans1 = KMeans(n_clusters=2, random_state=42)
    clusters1 = kmeans1.fit_predict(X1)
    
    # Determine which cluster is specialized
    mean_detector_score = [np.mean(final_metrics['detector_scores'][clusters1 == i]) for i in range(2)]
    specialized_cluster1 = np.argmax(mean_detector_score)
    
    # Create colormap
    colors1 = ['blue' if c != specialized_cluster1 else 'red' for c in clusters1]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(final_metrics['detector_scores'], final_metrics['beta1'], c=colors1, alpha=0.6)
    plt.xlabel('Detector Score')
    plt.ylabel('β (First Layer Alignment)')
    plt.title('First Layer Neuron Clustering: Red = Specialized Pattern Detectors')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}/first_layer_clustering.png", dpi=300)
    plt.close()
    
    # Second layer clustering: based on aggregator scores and alpha2
    X2 = np.column_stack((final_metrics['aggregator_scores'], final_metrics['alpha2']))
    
    # Apply K-means clustering with 2 clusters
    kmeans2 = KMeans(n_clusters=2, random_state=42)
    clusters2 = kmeans2.fit_predict(X2)
    
    # Determine which cluster is specialized
    mean_aggregator_score = [np.mean(final_metrics['aggregator_scores'][clusters2 == i]) for i in range(2)]
    specialized_cluster2 = np.argmax(mean_aggregator_score)
    
    # Create colormap
    colors2 = ['blue' if c != specialized_cluster2 else 'red' for c in clusters2]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(final_metrics['aggregator_scores'], final_metrics['alpha2'], c=colors2, alpha=0.6)
    plt.xlabel('Aggregator Score')
    plt.ylabel('α (Second Layer Output Weight)')
    plt.title('Second Layer Neuron Clustering: Red = Specialized Pattern Aggregators')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}/second_layer_clustering.png", dpi=300)
    plt.close()

def plot_phase_transition(metrics_history, save_path):
    epochs = range(len(metrics_history))
    epoch_nums = [e * log_interval for e in epochs]
    
    # Define metrics for detecting phase transition
    errors = np.array([m['error'] for m in metrics_history])
    specialized_fractions1 = np.array([m['specialized_fraction1'] for m in metrics_history])
    specialized_fractions2 = np.array([m['specialized_fraction2'] for m in metrics_history])
    correlations = np.array([m['correlation'] for m in metrics_history])
    
    # Calculate rate of change (use smoothed derivatives)
    window = 5
    if len(epochs) > window:
        error_deriv = np.convolve(-np.log(errors + 1e-10), np.ones(window)/window, mode='valid')
        specialized_deriv1 = np.convolve(specialized_fractions1, np.ones(window)/window, mode='valid')
        specialized_deriv2 = np.convolve(specialized_fractions2, np.ones(window)/window, mode='valid')
        correlation_deriv = np.convolve(correlations, np.ones(window)/window, mode='valid')
        
        # Plot derivatives to identify phase transitions
        plt.figure(figsize=(12, 8))
        plt.plot(epoch_nums[window-1:], error_deriv, label='Error Rate of Change')
        plt.plot(epoch_nums[window-1:], specialized_deriv1, label='First Layer Specialization Rate')
        plt.plot(epoch_nums[window-1:], specialized_deriv2, label='Second Layer Specialization Rate')
        plt.plot(epoch_nums[window-1:], correlation_deriv, label='Correlation Rate of Change')
        plt.xlabel('Epoch')
        plt.ylabel('Rate of Change')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/phase_transition.png", dpi=300)
        plt.close()

def plot_weight_analysis(metrics_history, save_path):
    # Analyze the final state of the weights to understand network specialization
    final_metrics = metrics_history[-1]
    
    # First layer: Analyze top detectors and their weights
    detector_scores = final_metrics['detector_scores']
    top_detectors = np.argsort(detector_scores)[-10:]  # Top 10 detectors
    
    W1 = final_metrics['W1']
    
    # Plot heatmap of weights for the top detectors
    plt.figure(figsize=(12, 8))
    # Only show weights for the first k features (relevant to the monomial)
    plt.imshow(W1[top_detectors, :k], cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Input Feature Index (0 to k-1)')
    plt.ylabel('Detector Neuron Index')
    plt.title('First Layer Weights for Top Pattern Detectors (Relevant Features Only)')
    plt.savefig(f"{save_path}/top_detector_weights.png", dpi=300)
    plt.close()
    
    # Second layer: Analyze top aggregators and their connections to detectors
    aggregator_scores = final_metrics['aggregator_scores']
    top_aggregators = np.argsort(aggregator_scores)[-10:]  # Top 10 aggregators
    
    W2 = final_metrics['W2']
    
    # Plot heatmap of weights for the top aggregators to top detectors
    plt.figure(figsize=(12, 8))
    plt.imshow(W2[top_aggregators][:, top_detectors], cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Detector Neuron Index')
    plt.ylabel('Aggregator Neuron Index')
    plt.title('Second Layer Weights: How Top Aggregators Connect to Top Detectors')
    plt.savefig(f"{save_path}/aggregator_detector_connections.png", dpi=300)
    plt.close()
    
    # Also visualize output weights for top aggregators
    a = final_metrics['a']
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_aggregators, a[top_aggregators])
    plt.xlabel('Aggregator Neuron Index')
    plt.ylabel('Output Weight Value')
    plt.title('Output Weights for Top Aggregator Neurons')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}/top_aggregator_output_weights.png", dpi=300)
    plt.close()

# Main training loop
def train_and_analyze():
    print(f"Analyzing k={k} monomial in d={d} dimensions with {M1} first layer and {M2} second layer neurons")
    
    # Generate all subsets for feature correlation analysis
    S_subsets = generate_subsets(k)
    
    # Initialize model
    model = DepthTwoReLUNet(d, M1, M2).to(device)
    
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
                print(f"Epoch {epoch+1}/{n_epochs}, Error: {metrics['error']:.6f}, Correlation: {metrics['correlation']:.6f}")
                print(f"First Layer Specialized: {metrics['specialized_fraction1']:.4f}, Second Layer Specialized: {metrics['specialized_fraction2']:.4f}")
            
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
        M1=M1,
        M2=M2
    )
    
    # Create all plots
    print("Creating visualizations...")
    plot_neuron_distribution(metrics_history, save_path)
    plot_learning_curves(metrics_history, save_path)
    plot_feature_learning(metrics_history, S_subsets, save_path)
    plot_layer_specialization(metrics_history, save_path)
    plot_neuron_clustering(metrics_history, save_path)
    plot_phase_transition(metrics_history, save_path)
    plot_weight_analysis(metrics_history, save_path)
    
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
    
    # Calculate specialized neuron emergence times
    # First layer
    specialized_threshold1 = 0.05  # 5% of neurons specialized
    T_discovery1 = None
    for t in range(len(metrics_history)):
        if metrics_history[t]['specialized_fraction1'] > specialized_threshold1:
            T_discovery1 = t * log_interval
            break
    
    # Second layer
    specialized_threshold2 = 0.05  # 5% of neurons specialized
    T_discovery2 = None
    for t in range(len(metrics_history)):
        if metrics_history[t]['specialized_fraction2'] > specialized_threshold2:
            T_discovery2 = t * log_interval
            break
    
    # Print discovery and learning times
    print("\nAnalysis Results:")
    print(f"First Layer Specialization Time: {T_discovery1} epochs")
    print(f"Second Layer Specialization Time: {T_discovery2} epochs")
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
    return metrics_history, learning_times, T_discovery1, T_discovery2

if __name__ == "__main__":
    start_time = time.time()
    metrics_history, learning_times, T_discovery1, T_discovery2 = train_and_analyze()
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")