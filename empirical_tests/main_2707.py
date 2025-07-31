import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def generate_k_sparse_parity_data(P, d, k, device='cpu'):
    """
    Generates a dataset for the k-sparse parity problem on the specified device.

    The k-sparse parity problem is a binary classification task where the label
    is the parity (product) of k specific features from the input vector.
    The other d-k features are noise.

    Args:
        P (int): The number of data points (dataset size).
        d (int): The dimensionality of the input vectors.
        k (int): The number of relevant features for parity calculation.
        device (str): The device to create the tensors on ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Input data of shape (P, d) with values in {-1, 1}.
            - torch.Tensor: Labels of shape (P, 1) with values in {-1, 1}.
    """
    if k > d:
        raise ValueError("k (number of sparse features) cannot be greater than d (dimensionality).")

    # Generate random binary data directly on the target device
    X = torch.randint(2, size=(P, d), device=device, dtype=torch.float32) * 2 - 1

    # Select k features for the parity calculation (we'll use the first k for simplicity)
    relevant_features = X[:, :k]

    # Calculate the parity (product of the k features)
    # y = 1 if the number of -1s is even, -1 if it's odd.
    y = torch.prod(relevant_features, dim=1).unsqueeze(1)

    return X, y


class TwoLayerNet(nn.Module):
    """
    A two-layer neural network as specified by the user.
    f(x) = sum_i a_i * phi(w_i^T * x)
    """

    def __init__(self, d, N, g_w, g_a, gamma_scaling_exponent):
        """
        Initializes the network parameters.

        Args:
            d (int): Input dimension.
            N (int): Number of neurons in the hidden layer (width).
            g_w (float): Variance scaling factor for weights w.
            g_a (float): Variance scaling factor for weights a.
            gamma_scaling_exponent (float): Exponent for N in sigma_a scaling.
        """
        super().__init__()
        self.d = d
        self.N = N

        # Calculate standard deviations based on the provided formulas
        sigma_w_sq = g_w / d
        sigma_a_sq = g_a / (N ** gamma_scaling_exponent)

        self.sigma_w = np.sqrt(sigma_w_sq)
        self.sigma_a = np.sqrt(sigma_a_sq)

        # Initialize weights
        # w has shape (d, N) so each column is a neuron's weight vector
        self.w = nn.Parameter(torch.randn(d, N) * self.sigma_w)
        # a has shape (N, 1)
        self.a = nn.Parameter(torch.randn(N, 1) * self.sigma_a)

        # Activation function (using tanh as a smooth version of a step-like function)
        self.phi = torch.relu

    def forward(self, x):
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input data of shape (P, d).

        Returns:
            torch.Tensor: Network output of shape (P, 1).
        """
        # x @ self.w results in shape (P, N), where each element (p, i) is w_i^T * x_p
        pre_activation = x @ self.w
        # self.phi(...) results in shape (P, N)
        post_activation = self.phi(pre_activation)
        # post_activation @ self.a results in shape (P, 1)
        output = post_activation @ self.a
        return output


def train_with_langevin(model, X_train, y_train, X_test, y_test, hyperparams, save_dir):
    """
    Trains the model using full-batch Langevin Gradient Descent.

    Args:
        model (TwoLayerNet): The neural network model to train.
        X_train (torch.Tensor): Training data.
        y_train (torch.Tensor): Training labels.
        X_test (torch.Tensor): Test data.
        y_test (torch.Tensor): Test labels.
        hyperparams (dict): Dictionary containing all hyperparameters.
        save_dir (Path): Directory to save logs and plots.

    Returns:
        dict: A dictionary containing the final train and test errors (both MSE and 0-1).
    """
    # Unpack hyperparameters
    eta = hyperparams['eta']
    kappa_0 = hyperparams['kappa_0']
    gamma_scaling_exponent = hyperparams['gamma_scaling_exponent']
    epochs = hyperparams['epochs']
    log_interval = hyperparams['log_interval']

    P_train = X_train.shape[0]
    N = model.N

    # Calculate temperature T based on kappa
    kappa = kappa_0 * (N ** (1 - gamma_scaling_exponent))
    T = 2 * (kappa ** 2)

    # Loss function (Mean Squared Error)
    loss_fn = nn.MSELoss(reduction='sum') # Sum over the batch

    # Training loop
    print(f"\n--- Training with P = {P_train}, N = {N}, T = {T:.4f} ---")
    start_time = time.time()

    for epoch in range(epochs + 1):
        # --- Forward pass ---
        y_pred_train = model(X_train)
        
        # The total loss over the batch, as in the formula
        loss = loss_fn(y_pred_train, y_train)

        # --- Gradient calculation ---
        # Zero out previous gradients
        model.zero_grad()
        # Compute gradients of the loss with respect to model parameters (w and a)
        loss.backward()

        # --- Manual Langevin Update ---
        with torch.no_grad():
            # Update for 'a'
            grad_a = model.a.grad
            # In the formula, the loss is sum over mu, but MSELoss(reduction='sum') already does that.
            # The formula has 1/P, so we add it here.
            nabla_a_loss = grad_a / P_train
            
            # Generate Gaussian noise (will be on the same device as model.a)
            noise_a = torch.randn_like(model.a) * np.sqrt(2 * T * eta)
            
            # Weight decay term
            decay_a = (T / (model.sigma_a ** 2)) * model.a
            
            # Update rule for 'a'
            delta_a = -eta * (decay_a + nabla_a_loss) + noise_a
            model.a.add_(delta_a)

            # Update for 'w'
            grad_w = model.w.grad
            nabla_w_loss = grad_w / P_train
            
            noise_w = torch.randn_like(model.w) * np.sqrt(2 * T * eta)
            decay_w = (T / (model.sigma_w ** 2)) * model.w
            
            delta_w = -eta * (decay_w + nabla_w_loss) + noise_w
            model.w.add_(delta_w)

        # --- Logging ---
        if epoch % log_interval == 0:
            with torch.no_grad():
                # Evaluate on test set
                y_pred_test = model(X_test)
                test_loss = loss_fn(y_pred_test, y_test) / X_test.shape[0]

                # Calculate 0-1 classification error
                train_preds_class = torch.sign(y_pred_train)
                train_error_01 = (train_preds_class != y_train).float().mean()

                test_preds_class = torch.sign(y_pred_test)
                test_error_01 = (test_preds_class != y_test).float().mean()
                
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch {epoch:5d}/{epochs} | "
                    f"Train Loss (MSE): {loss.item()/P_train:.4f} | "
                    f"Test Loss (MSE): {test_loss.item():.4f} | "
                    f"Train Err (0-1): {train_error_01:.4f} | "
                    f"Test Err (0-1): {test_error_01:.4f} | "
                    f"Time: {elapsed_time:.2f}s"
                )

    # Final evaluation
    final_train_loss = loss.item() / P_train
    final_test_loss = test_loss.item()
    final_train_error_01 = train_error_01.item()
    final_test_error_01 = test_error_01.item()
    
    print(f"--- Finished training for P = {P_train} ---")
    print(f"Final Train Error: {final_train_error_01:.4f}, Final Test Error: {final_test_error_01:.4f}")

    return {
        "train_mse": final_train_loss,
        "test_mse": final_test_loss,
        "train_error_01": final_train_error_01,
        "test_error_01": final_test_error_01,
    }


def main():
    """
    Main function to run the experiment.
    """
    # --- Device Configuration ---
    # Automatically select GPU if available, otherwise fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Using device: {device} ---")

    # ----------------------------------------------------------------------
    # --- Hyperparameter Configuration ---
    # ----------------------------------------------------------------------
    # All parameters can be changed here.
    hyperparams = {
        # Data parameters
        "d": 40,  # Dimensionality
        "k": 4,   # Sparsity for parity
        # Network parameters
        "N": 500, # Network width
        "g_w": 1.0, # Variance scale for w
        "g_a": 1.0, # Variance scale for a
        "gamma_scaling_exponent": 2.0, # Exponent for N in sigma_a scaling
        # Langevin/Training parameters
        "eta": 5e-4, # Learning rate
        "kappa_0": 1.0, # Base for noise temperature
        "epochs": 2000000, # Total training epochs
        # Logging
        "log_interval": 10000, # How often to print progress
    }
    
    # --- Experiment Configuration ---
    # Directory to save results. It will be created if it doesn't exist.
    save_dir = Path("/home/goring/OnlineSGD/empirical_tests/results_2707_d40k4_kappa1")
    
    # List of training set sizes (P) to iterate over
    P_values = [ 1000, 10000, 50000,100_000, 500_000,1000000,5000000]
    
    # Fixed test set size
    P_test = 10000
    
    # ----------------------------------------------------------------------
    
    # Create save directory
    save_dir.mkdir(exist_ok=True)
    
    # Generate a fixed test set for consistent evaluation
    print(f"Generating fixed test set of size {P_test}...")
    X_test, y_test = generate_k_sparse_parity_data(P_test, hyperparams['d'], hyperparams['k'], device=device)

    # Store results for all P values
    results_history = []

    for P_train in P_values:
        # Generate training data for the current P
        X_train, y_train = generate_k_sparse_parity_data(P_train, hyperparams['d'], hyperparams['k'], device=device)

        # Instantiate the model
        model = TwoLayerNet(
            d=hyperparams['d'],
            N=hyperparams['N'],
            g_w=hyperparams['g_w'],
            g_a=hyperparams['g_a'],
            gamma_scaling_exponent=hyperparams['gamma_scaling_exponent']
        )
        # Move model to the selected device (GPU)
        model.to(device)
        
        # SPEEDUP: Compile the model for maximum performance on modern GPUs
        # This will JIT-compile the forward pass into an optimized kernel.
        # The first run will be a bit slower due to compilation overhead.
        if device == 'cuda':
            print("Compiling model for GPU...")
            model = torch.compile(model)

        # Run training
        final_errors = train_with_langevin(
            model, X_train, y_train, X_test, y_test, hyperparams, save_dir
        )
        
        results_history.append({
            "P": P_train,
            **final_errors
        })

    # --- Save results to JSON ---
    json_path = save_dir / "training_results.json"
    print(f"\nSaving final results to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(results_history, f, indent=4)

    # --- Plot results ---
    plot_path = save_dir / "error_vs_P.png"
    print(f"Saving plot to {plot_path}")
    
    p_vals = [r['P'] for r in results_history]
    train_errs = [r['train_error_01'] for r in results_history]
    test_errs = [r['test_error_01'] for r in results_history]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(p_vals, train_errs, marker='o', linestyle='-', label='Train Error (0-1)')
    ax.plot(p_vals, test_errs, marker='s', linestyle='--', label='Test Error (0-1)')
    
    ax.set_xlabel("Training Set Size (P)")
    ax.set_ylabel("Final 0-1 Classification Error")
    ax.set_title("Train/Test Error vs. Training Set Size")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    
    plt.savefig(plot_path)
    # Convert tensors to cpu for plotting if they are on gpu
    plt.show()


if __name__ == '__main__':
    main()