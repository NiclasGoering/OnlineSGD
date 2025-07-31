import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

# --- Activation Function ---
# The paper mentions tanh and ReLU. We'll use ReLU here, but it can be easily swapped.
def activation_function(x):
    """The activation function phi for the hidden layer."""
    return torch.relu(x)

# --- Dataset Generation ---
def generate_k_sparse_parity_dataset(P, d, k, S, device):
    """
    Generates a dataset for the k-sparse parity problem.

    Args:
        P (int): Number of data points (dataset size).
        d (int): Input dimension.
        k (int): Sparsity of the parity function.
        S (list or np.ndarray): The set of k indices defining the parity.
        device (torch.device): The device to move the tensors to.

    Returns:
        tuple: A tuple containing:
            - X (torch.Tensor): Input data of shape (P, d) with values in {-1, 1}.
            - Y (torch.Tensor): Target labels of shape (P, 1).
    """
    if len(S) != k:
        raise ValueError(f"Length of sparse indices S ({len(S)}) must be equal to k ({k}).")

    # Generate random binary inputs {-1, 1}
    X = torch.randint(0, 2, (P, d), device=device, dtype=torch.float32) * 2 - 1

    # Compute the parity labels
    # Y = product of elements in columns specified by S
    Y = torch.prod(X[:, S], dim=1).unsqueeze(1)

    return X, Y

# --- Neural Network Model ---
class TwoLayerNet(nn.Module):
    """
    A two-layer neural network as defined in the paper: f(x) = sum(a_i * phi(w_i^T * x)).
    """
    def __init__(self, d, N, sigma_v, sigma_w, gamma):
        """
        Initializes the network weights.

        Args:
            d (int): Input dimension.
            N (int): Width of the hidden layer.
            sigma_v (float): Std deviation parameter for the first layer weights (w).
            sigma_w (float): Std deviation parameter for the second layer weights (a).
            gamma (float): Scaling exponent for the second layer weights.
        """
        super().__init__()
        self.d = d
        self.N = N
        self.gamma = gamma
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w

        # First layer weights (w): shape (N, d)
        # w_ij ~ N(0, sigma_v^2 / d)
        w_std = self.sigma_v / np.sqrt(self.d)
        self.w = nn.Parameter(torch.randn(self.N, self.d) * w_std)

        # Second layer weights (a): shape (N, 1)
        # a_i ~ N(0, sigma_w^2 / N^gamma)
        a_std = self.sigma_w / np.sqrt(self.N**self.gamma)
        self.a = nn.Parameter(torch.randn(self.N, 1) * a_std)

    def forward(self, x):
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # x @ self.w.T gives pre-activations of shape (batch_size, N)
        pre_activations = x @ self.w.T
        # phi(pre_activations) gives hidden layer activations
        hidden_activations = activation_function(pre_activations)
        # hidden_activations @ self.a gives the final output
        output = hidden_activations @ self.a
        return output

# --- Training and Evaluation ---
def train_model(model, X_train, Y_train, hyperparams):
    """
    Trains the model using full-batch gradient descent with Langevin noise.

    Args:
        model (TwoLayerNet): The model to train.
        X_train (torch.Tensor): Training input data.
        Y_train (torch.Tensor): Training target labels.
        hyperparams (dict): Dictionary of hyperparameters.
        
    Returns:
        float: The final training error (MSE).
    """
    # Unpack hyperparameters
    eta = hyperparams['learning_rate']
    max_epochs = hyperparams['max_epochs']
    kappa_0 = hyperparams['kappa_0']
    train_err_threshold = hyperparams['train_err_threshold']
    
    # Extract model parameters for loss calculation
    N = model.N
    d = model.d
    gamma = model.gamma
    sigma_w = model.sigma_w
    sigma_v = model.sigma_v

    # Calculate kappa based on the formula
    kappa = kappa_0 * (N**(1 - gamma))

    # Use SGD optimizer for a direct implementation of Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    
    print("Starting training...")
    final_train_mse = -1.0
    for epoch in tqdm(range(max_epochs)):
        optimizer.zero_grad()

        # --- Full-batch forward pass ---
        Y_pred = model(X_train)

        # --- Calculate the modified Action ---
        # S_int term, normalized by P: (1 / 2*kappa) * mean((f(x) - y)^2)
        interaction_loss_avg = (1.0 / (2.0 * kappa)) * torch.mean((Y_pred - Y_train)**2)

        # S_prior term (L2 regularization) - NOW NORMALIZED by number of parameters
        # This balances the two loss terms.
        prior_loss_a = (N**gamma / (2.0 * sigma_w**2)) * torch.mean(model.a**2)
        prior_loss_w = (d / (2.0 * sigma_v**2)) * torch.mean(model.w**2)
        
        prior_loss = prior_loss_a + prior_loss_w

        # Total loss is the modified objective
        total_loss = interaction_loss_avg + prior_loss

        # --- Backward pass and optimization step ---
        total_loss.backward()
        
        optimizer.step()

        # --- Langevin Dynamics: Add Gaussian Noise ---
        # The noise std is sqrt(2*T*eta) where T=kappa.
        with torch.no_grad():
            noise_std = np.sqrt(2 * kappa * eta)
            # Add noise to first layer weights (w)
            model.w.add_(torch.randn_like(model.w) * noise_std)
            # Add noise to second layer weights (a)
            model.a.add_(torch.randn_like(model.a) * noise_std)


        # --- Check for convergence ---
        if (epoch + 1) % 100 == 0:
            # Calculate plain MSE for an intuitive convergence check
            with torch.no_grad():
                train_mse = torch.mean((model(X_train) - Y_train)**2).item()
            
            if torch.isnan(torch.tensor(train_mse)):
                print(f"\nNaN detected at epoch {epoch+1}. Stopping training.")
                final_train_mse = float('nan')
                break

            if train_mse < train_err_threshold:
                print(f"\nTraining converged at epoch {epoch+1} with Train MSE: {train_mse:.6f}")
                final_train_mse = train_mse
                break
    
    print("Training finished.")
    # If training finished by reaching max_epochs, calculate final train_mse
    if final_train_mse < 0:
        with torch.no_grad():
            final_train_mse = torch.mean((model(X_train) - Y_train)**2).item()

    return final_train_mse


def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the model's generalization error on the test set.

    Args:
        model (TwoLayerNet): The trained model.
        X_test (torch.Tensor): Test input data.
        Y_test (torch.Tensor): Test target labels.

    Returns:
        float: The generalization error (MSE).
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        Y_pred = model(X_test)
        gen_error = torch.mean((Y_pred - Y_test)**2).item()
    return gen_error

# --- Main Execution ---
def main():
    """Main function to run the experiment."""
    # --- HYPERPARAMETERS (Tweak these) ---
    hyperparams = {
        'd': 30,                      # Input dimension
        'k': 4,                       # Sparsity of the parity function
        'N': 1024,                    # Width of the hidden layer
        'gamma': 0.0,                 # Scaling exponent for weights 'a'
        'sigma_v': 1.0,               # Std deviation param for weights 'w'
        'sigma_w': 1.0,               # Std deviation param for weights 'a'
        'kappa_0': 0.000001,              # Base noise parameter
        'learning_rate': 1e-6,        # Can use a slightly larger LR now that loss is balanced.
        'max_epochs': 100000,         # Maximum number of training epochs
        'train_err_threshold': 1e-4,  # Stop training if train MSE is below this
    }

    # --- EXPERIMENT SETUP ---
    # Define the range of dataset sizes (P) to iterate over
    P_values = [ 100, 1000, 5000, 10000, 50000, 100000]
    
    # Path to save the results. PLEASE PROVIDE YOUR PATH HERE.
    # Example: "/home/user/documents/k_sparse_parity_results"
    SAVE_PATH = "/home/goring/OnlineSGD/langevin/resultsd30k2" 
    
    # Create the directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fix the sparse indices for the entire experiment for consistency
    np.random.seed(42)
    S = np.random.choice(hyperparams['d'], hyperparams['k'], replace=False)
    print(f"Target parity function uses indices: {S}")

    # Generate a large, fixed test set for consistent evaluation
    P_test = 10000
    X_test, Y_test = generate_k_sparse_parity_dataset(P_test, hyperparams['d'], hyperparams['k'], S, device)

    # --- RUN EXPERIMENT ---
    results = {'P_values': [], 'gen_errors': [], 'train_errors': []}

    for P in P_values:
        print("-" * 50)
        print(f"Training with dataset size P = {P}")

        # Generate training data
        X_train, Y_train = generate_k_sparse_parity_dataset(P, hyperparams['d'], hyperparams['k'], S, device)

        # Instantiate a new model for each run to ensure fair comparison
        torch.manual_seed(P) # Seed for reproducibility, changes per run
        model = TwoLayerNet(
            d=hyperparams['d'],
            N=hyperparams['N'],
            sigma_v=hyperparams['sigma_v'],
            sigma_w=hyperparams['sigma_w'],
            gamma=hyperparams['gamma']
        ).to(device)

        # Train the model and get final training error
        train_error = train_model(model, X_train, Y_train, hyperparams)

        # Evaluate and store results
        gen_error = evaluate_model(model, X_test, Y_test)
        
        print(f"P = {P}, Final Train Error (MSE): {train_error:.6f}, Generalization Error (MSE): {gen_error:.6f}")
        
        results['P_values'].append(P)
        results['gen_errors'].append(gen_error)
        results['train_errors'].append(train_error)

    # --- SAVE RESULTS ---
    # Save numerical data to a JSON file
    json_path = os.path.join(SAVE_PATH, "gen_error_vs_P.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {json_path}")

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['P_values'], results['gen_errors'], marker='o', linestyle='-', label='Generalization Error')
    plt.plot(results['P_values'], results['train_errors'], marker='x', linestyle='--', label='Train Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dataset Size (P)')
    plt.ylabel('Error (MSE)')
    plt.title('Error vs. Dataset Size for k-Sparse Parity')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plot_path = os.path.join(SAVE_PATH, "gen_error_vs_P.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == '__main__':
    main()
