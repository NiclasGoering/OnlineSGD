#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import sys
import glob
import numpy as np
import json
import yaml
import traceback
from datetime import datetime
import time
from typing import List, Dict, Tuple, Any, Callable, Union
from tqdm import tqdm
import itertools
from functools import partial
from torch.cuda.amp import GradScaler
import gzip
import io

# Import the DeepNN class
from helpers.FFNN_sgd import DeepNN

# Ensure prints flush immediately
print = partial(print, flush=True)

# Define phi functions that can be referenced in the YAML
PHI_FUNCTIONS = {
    "id": lambda x: x,                     # Identity function
    "tanh": torch.tanh,                    # Hyperbolic tangent
    "square": lambda x: x**2,              # Square
    "cube": lambda x: x**3,                # Cube
    "relu": lambda x: torch.relu(x),       # ReLU
    "sigmoid": torch.sigmoid,              # Sigmoid
    "sin": torch.sin,                      # Sine
    "cos": torch.cos,                      # Cosine
    "exp": torch.exp,                      # Exponential
    "log": lambda x: torch.log(torch.abs(x) + 1e-10),  # Log with safety
    "sqrt": lambda x: torch.sqrt(torch.abs(x) + 1e-10),  # Square root with safety
}


def load_checkpoint(checkpoint_file):
    """Load checkpoint file with completed configuration IDs"""
    completed_configs = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            completed_configs = set(line.strip() for line in f)
        print(f"Loaded {len(completed_configs)} completed configurations from checkpoint")
    return completed_configs


def update_checkpoint(checkpoint_file, unique_id):
    """Add a completed configuration ID to the checkpoint file"""
    with open(checkpoint_file, 'a') as f:
        f.write(f"{unique_id}\n")


def generate_unique_id(function: Dict, lr: float, hidden_size: int, 
                      depth: int, mode: str, align: bool, input_dim: int, batch_size: int, 
                      exp_num: int, distribution: str, load_factor_enabled: bool = False) -> str:
    """Generate a unique identifier for this run configuration."""
    # Just use the function name instead of detailed term representation
    # This creates much shorter filenames that won't exceed OS limits
    
    align_suffix = "_align" if align else ""
    load_factor_suffix = "_loadfactor" if load_factor_enabled else ""
    
    unique_id = (
        f"{function['name']}"  # Just use the name from YAML
        f"_d{input_dim}"
        f"_h{hidden_size}"
        f"_depth{depth}"
        f"_lr{lr}"
        f"_b{batch_size}"  # Added batch size to unique ID
        f"_mode{mode}"
        f"_dist{distribution}"
        f"_exp{exp_num}"
        f"{align_suffix}"
        f"{load_factor_suffix}"
    )
    return unique_id


def get_term_descriptions(complex_terms):
    """
    Generate human-readable descriptions of the complex terms for logging and debugging.
    Corrected to properly show phi functions.
    """
    descriptions = []
    for term_dict in complex_terms:
        term_indices = term_dict["indices"]
        phi_name = term_dict.get("phi", "id")
        coefficient = term_dict.get("coefficient", 1.0)
        
        indices_str = f"[{','.join(map(str, term_indices))}]"
        
        if phi_name == "id" and coefficient == 1.0:
            desc = f"{indices_str}"
        elif phi_name == "id":
            desc = f"{coefficient} * {indices_str}"
        else:
            desc = f"{coefficient} * {phi_name}({indices_str})"
        
        descriptions.append(desc)
    
    return descriptions


def try_compile_model(model, gpu_id):
    """Attempt to use torch.compile() with proper error handling"""
    use_compile = True  # Set to False to disable torch.compile() entirely
    
    if not use_compile:
        print(f"[GPU {gpu_id}] torch.compile() is disabled")
        return model
    
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        
        # Set to a safer backend with fewer optimizations but more stability
        compiled_model = torch.compile(
            model, 
            mode="reduce-overhead",
            fullgraph=False,  # Don't try to compile the entire model as one graph
            dynamic=True      # Allow for dynamic shapes
        )
        print(f"[GPU {gpu_id}] Using torch.compile() for model acceleration")
        return compiled_model
    except Exception as e:
        print(f"[GPU {gpu_id}] torch.compile() not available: {str(e)}")
        return model


def evaluate_test_loss(model, X_test, y_test):
    """
    Safely evaluate test loss using batched processing to handle compiled models.
    Returns the mean loss across all batches.
    """
    batch_size = 10000  # Larger batch size for H100s
    total_batches = (X_test.shape[0] + batch_size - 1) // batch_size
    total_loss = 0.0
    total_samples = 0
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, X_test.shape[0])
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        with torch.amp.autocast('cuda'):
            # Clone outputs to prevent CUDA graph overwrite issues
            outputs = model(X_batch).detach().clone()
            batch_loss = torch.mean((outputs - y_batch) ** 2).item()
        
        # Weighted average
        total_loss = (total_loss * total_samples + batch_loss * batch_size_actual) / (total_samples + batch_size_actual)
        total_samples += batch_size_actual
    
    return total_loss


def save_compressed_model(model, save_path):
    """Save model in a compressed format to save storage space"""
    # Get state dict
    state_dict = model.state_dict()
    
    # Use BytesIO as an in-memory buffer
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    
    # Get the bytes from the buffer
    buffer.seek(0)
    model_bytes = buffer.getvalue()
    
    # Compress and save
    with gzip.open(save_path, 'wb') as f:
        f.write(model_bytes)
    
    print(f"Model saved and compressed at {save_path}")


def load_compressed_model(load_path, model):
    """Load a compressed model file"""
    with gzip.open(load_path, 'rb') as f:
        buffer = f.read()
    
    state_dict = torch.load(io.BytesIO(buffer))
    model.load_state_dict(state_dict)
    return model


def extract_activations_from_deepnn(model: nn.Module, X: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract only post-activation outputs from DeepNN model.
    
    Args:
        model: The DeepNN model
        X: Input tensor
        
    Returns:
        List of post-activation outputs (after ReLU) for each layer
    """
    model.eval()
    
    # Enable activation storage in the model
    if hasattr(model, 'enable_activation_storage'):
        model.enable_activation_storage()
    else:
        print("Warning: Model does not have enable_activation_storage method")
        return []
    
    # Process data in batches to avoid memory issues
    batch_size = 25000  # Large batch size for H100
    post_activations = []
    
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            end_idx = min(i + batch_size, X.shape[0])
            X_batch = X[i:end_idx]
            
            # Forward pass to collect activations
            _ = model(X_batch)
            
            # Check if activations were stored
            if not hasattr(model, 'hidden_activations') or model.hidden_activations is None:
                print("Warning: No activations were stored by the model")
                return []
            
            # Extract only the input and post-ReLU activations
            batch_post_activations = []
            
            # Add input
            batch_post_activations.append(model.hidden_activations[0])
            
            # Add only post-ReLU activations (which are at odd indices after the input)
            for idx in range(2, len(model.hidden_activations), 2):
                if idx < len(model.hidden_activations):
                    batch_post_activations.append(model.hidden_activations[idx])
            
            # Initialize all_activations with the right shapes on first batch
            if not post_activations:
                post_activations = [
                    torch.empty((X.shape[0], *act.shape[1:]), device=X.device) 
                    for act in batch_post_activations
                ]
            
            # Store post-activations for this batch
            for j, act in enumerate(batch_post_activations):
                if j < len(post_activations):
                    # Handle different dimensional activations
                    if act.dim() > 2:
                        # Flatten spatial dimensions for consistency
                        act = act.reshape(act.shape[0], -1)
                    
                    post_activations[j][i:end_idx] = act
    
    # Disable activation storage when done
    if hasattr(model, 'disable_activation_storage'):
        model.disable_activation_storage()
    
    # Print layer shapes for debugging
    for i, act in enumerate(post_activations):
        if i == 0:
            print(f"Input shape: {act.shape}")
        else:
            print(f"Layer {i} (post-ReLU) shape: {act.shape}")
    
    return post_activations


def compute_load_factor_per_term(
    model: nn.Module,
    X: torch.Tensor,
    complex_terms: List[Dict],
    device: torch.device,
    num_workers: int = 4
) -> Dict[str, List[float]]:
    """
    Calculate the load factor for each layer with individual terms.
    Returns a dictionary of term-specific load factors per layer.
    """
    print("\nExtracting layer activations...")
    activations = extract_activations_from_deepnn(model, X)
    
    if not activations:
        print("WARNING: Specialized extraction failed. Returning empty results.")
        return {"error": "Could not extract activations from the model"}
    
    print(f"Extracted activations from {len(activations)} layers")
    
    # Results dict: key is term index, value is list of load factors per layer
    term_specific_results = {}
    
    # Calculate load factors for each individual term
    for term_idx, term_dict in enumerate(complex_terms):
        term_key = f"term{term_idx}"
        
        # Calculate true values for this single term
        true_value = calculate_true_term_value(
            term_dict, X, device
        )
        
        # Calculate load factors for each layer with this term
        layer_results = []
        for layer_idx, layer_activations in enumerate(activations):
            # Calculate load factor for this layer and term
            load_factor = calculate_load_factor_for_layer_robust(
                layer_activations, true_value, device
            )
            layer_results.append(load_factor)
        
        # Store results indexed by term
        term_specific_results[term_key] = layer_results
        
    return term_specific_results


def calculate_true_term_value(
    term_dict: Dict,
    X: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Calculate the true value for a specific term.
    
    Args:
        term_dict: Term dictionary with indices, phi, coefficient
        X: Input tensor
        device: Device to place tensors on
        
    Returns:
        Tensor with true values for the term
    """
    # Process data in batches to avoid memory issues for large datasets
    batch_size = 10000  # Large batch size for H100
    n_samples = X.shape[0]
    true_values = torch.zeros(n_samples, device=device)
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        X_batch = X[i:end_idx]
        
        # Get term parameters
        term_indices = term_dict["indices"]
        phi_name = term_dict.get("phi", "id")
        coefficient = term_dict.get("coefficient", 1.0)
        phi_func = PHI_FUNCTIONS[phi_name]
        
        # Calculate raw term value
        term_value = torch.ones(X_batch.shape[0], device=device)
        for dim_idx in term_indices:
            # Convert from 1-indexed to 0-indexed
            term_value = term_value * X_batch[:, dim_idx-1]
        
        # Apply phi function and coefficient
        term_output = coefficient * phi_func(term_value)
        true_values[i:end_idx] = term_output
    
    return true_values


def calculate_load_factor_for_layer_robust(
    layer_activations: torch.Tensor,
    true_values: torch.Tensor,
    device: torch.device,
    lambda_reg: float = 1e-5  # Regularization parameter
) -> float:
    """
    Calculate the load factor (minimum possible RMSE) for a single layer and term combination
    using direct ridge regression solution for better numerical stability.
    
    Args:
        layer_activations: Activations from a specific layer
        true_values: True values for the term combination
        device: Device to place tensors on
        lambda_reg: Regularization parameter for ridge regression
        
    Returns:
        Load factor (minimum possible RMSE)
    """
    # Clone activations and true values to ensure we don't modify the originals
    X = layer_activations.clone()
    y = true_values.clone().view(-1, 1)
    
    # Get dimensions
    n_samples = X.shape[0]
    input_dim = X.shape[1]
    
    # Use batched processing for large datasets
    batch_size = 10000  # Large batch size for H100
    
    # For very large feature dimensions, use iterative computation
    if input_dim > 10000:
        # Compute X^T X in batches
        XTX = torch.zeros(input_dim, input_dim, device=device)
        XTy = torch.zeros(input_dim, 1, device=device)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]
            
            # Update X^T X and X^T y
            XTX += torch.matmul(X_batch.T, X_batch)
            XTy += torch.matmul(X_batch.T, y_batch)
    else:
        # For smaller dimensions, we can compute directly
        XTX = torch.matmul(X.T, X)
        XTy = torch.matmul(X.T, y)
    
    # Add regularization for stability
    XTX += lambda_reg * torch.eye(input_dim, device=device)
    
    # Try solving the linear system with Cholesky decomposition first (fastest)
    try:
        L = torch.linalg.cholesky(XTX)
        # Solve L @ L.T @ weights = XTy
        temp = torch.linalg.solve_triangular(L, XTy, upper=False)
        weights = torch.linalg.solve_triangular(L.T, temp, upper=True)
    except Exception as e:
        # If Cholesky fails, fall back to SVD (more stable but slower)
        print(f"Cholesky decomposition failed, falling back to SVD: {str(e)}")
        try:
            weights = torch.linalg.solve(XTX, XTy)
        except Exception as e2:
            print(f"Linear solve failed, using pseudo-inverse: {str(e2)}")
            # Use SVD as a last resort
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            # Apply stronger regularization if SVD is needed
            S_inv = torch.where(S > 1e-4, 1/S, torch.zeros_like(S))
            weights = Vh.T @ (S_inv.unsqueeze(1) * (U.T @ y))
    
    # Compute predictions and MSE in batches to avoid memory issues
    total_squared_error = 0.0
    total_samples = 0
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        X_batch = X[i:end_idx]
        y_batch = y[i:end_idx]
        batch_size_actual = end_idx - i
        
        # Compute predictions
        pred_batch = torch.matmul(X_batch, weights)
        
        # Compute squared error
        squared_error = torch.sum((pred_batch - y_batch) ** 2).item()
        
        # Accumulate error
        total_squared_error += squared_error
        total_samples += batch_size_actual
    
    # Compute RMSE
    mse = total_squared_error / total_samples
    rmse = np.sqrt(mse)
    
    return rmse


def freeze_layer(model: nn.Module, layer_index: int):
    """
    Freeze a specific layer in the DeepNN model.
    
    Args:
        model: The DeepNN model
        layer_index: The index of the layer to freeze (0-indexed, excluding the input)
    """
    # For DeepNN, we need to map layer_index to actual parameters
    # Each layer has a Linear + ReLU, so we need to find the corresponding linear layer
    linear_idx = 0
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            if linear_idx == layer_index:
                # Freeze this layer's parameters
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"Frozen layer {layer_index} (Linear layer at index {i})")
                return
            linear_idx += 1
    
    print(f"Warning: Could not find layer {layer_index} to freeze")


def unfreeze_all_layers(model: nn.Module):
    """
    Unfreeze all layers in the model.
    
    Args:
        model: The neural network model
    """
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            for param in layer.parameters():
                param.requires_grad = True
    
    print("All layers unfrozen")


def compute_load_factor(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    complex_terms: List[Dict],
    device: torch.device,
    num_workers: int = 4  # This parameter is ignored now
) -> Dict[str, List[float]]:
    """
    Calculate the load factor for each layer with various combinations of terms.
    Updated to use the specialized extraction for DeepNN models and ridge regression for stability.
    """
    # Debug: Print model structure to understand what layers it has
    print("\nModel structure:")
    for name, module in model.named_modules():
        print(f"{name}: {type(module).__name__}")
    
    # Extract activations from each layer - use specialized function for DeepNN
    print("\nExtracting layer activations...")
    activations = extract_activations_from_deepnn(model, X)
    
    # If we couldn't extract activations with the specialized function, try the general one
    if not activations:
        print("WARNING: Specialized extraction failed. Returning empty results.")
        return {"error": "Could not extract activations from the model"}
    
    print(f"Extracted activations from {len(activations)} layers")
    
    # Generate all possible combinations of terms
    all_combinations = get_all_term_combinations(complex_terms)
    print(f"Generated {len(all_combinations)} term combinations")
    
    # Calculate load factors sequentially - no multiprocessing
    print("Calculating load factors sequentially for all combinations...")
    combined_results = {}
    
    # Process all combinations
    for combo_idx, combo in enumerate(all_combinations):
        # Generate a descriptive key for this combination
        combo_desc = "+".join([f"term{i}" for i in combo])
        
        # Calculate true values for this combination
        true_value = calculate_true_combination_value(combo, X, complex_terms, device)
        
        # Calculate load factors for each layer with this combination
        layer_results = []
        for layer_idx, layer_activations in enumerate(activations):
            # Calculate load factor for this layer and combination
            load_factor = calculate_load_factor_for_layer_robust(
                layer_activations, true_value, device
            )
            layer_results.append(load_factor)
        
        # Store results indexed by combination
        combined_results[combo_desc] = layer_results
        
        # Log progress regularly
        if (combo_idx + 1) % 5 == 0 or combo_idx == len(all_combinations) - 1:
            print(f"Processed {combo_idx + 1}/{len(all_combinations)} combinations")
    
    return combined_results


def get_all_term_combinations(complex_terms: List[Dict]) -> List[Tuple[int, ...]]:
    """
    Generate all possible combinations of terms.
    
    Args:
        complex_terms: List of term dictionaries
        
    Returns:
        List of combinations as tuples of term indices
    """
    n_terms = len(complex_terms)
    all_combinations = []
    
    # Generate all possible combinations (excluding empty set)
    for r in range(1, n_terms + 1):
        for combo in itertools.combinations(range(n_terms), r):
            all_combinations.append(combo)
    
    return all_combinations


def calculate_true_combination_value(
    combo: Tuple[int, ...],
    X: torch.Tensor,
    complex_terms: List[Dict],
    device: torch.device
) -> torch.Tensor:
    """
    Calculate the true value for a specific combination of terms.
    
    Args:
        combo: Tuple of term indices to include
        X: Input tensor
        complex_terms: List of term dictionaries
        device: Device to place tensors on
        
    Returns:
        Tensor with true values for the combination
    """
    # Process data in batches to avoid memory issues for large datasets
    batch_size = 10000  # Large batch size for H100
    n_samples = X.shape[0]
    true_values = torch.zeros(n_samples, device=device)
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        X_batch = X[i:end_idx]
        
        # Start with zeros for this batch
        batch_values = torch.zeros(end_idx - i, device=device)
        
        # Add each term in the combination
        for term_idx in combo:
            term_dict = complex_terms[term_idx]
            term_indices = term_dict["indices"]
            phi_name = term_dict.get("phi", "id")
            coefficient = term_dict.get("coefficient", 1.0)
            phi_func = PHI_FUNCTIONS[phi_name]
            
            # Calculate raw term value
            term_value = torch.ones(X_batch.shape[0], device=device)
            for dim_idx in term_indices:
                # Convert from 1-indexed to 0-indexed
                term_value = term_value * X_batch[:, dim_idx-1]
            
            # Apply phi function and coefficient
            term_output = coefficient * phi_func(term_value)
            batch_values = batch_values + term_output
        
        # Store results for this batch
        true_values[i:end_idx] = batch_values
    
    return true_values


def calculate_term_test_losses(
    model: nn.Module,
    X: torch.Tensor,
    complex_terms: List[Dict],
    device: torch.device
) -> List[float]:
    """
    Calculate term-specific errors by creating separate test cases where only one term
    contributes at a time.
    
    Args:
        model: The neural network model
        X: Input tensor
        complex_terms: List of term dictionaries with indices, phi, coefficient
        device: Device to place tensors on
        
    Returns:
        List of test losses for each term
    """
    model.eval()
    batch_size = 10000  # Use large batch size for H100
    
    # Calculate term-specific errors
    term_errors = []
    
    with torch.no_grad():
        for term_idx, term_dict in enumerate(complex_terms):
            # Get this term's info
            term_indices = term_dict["indices"]
            phi_name = term_dict.get("phi", "id")
            coefficient = term_dict.get("coefficient", 1.0)
            
            # Get the phi function
            phi_func = PHI_FUNCTIONS[phi_name]
            
            # Calculate true value for this term
            term_value = torch.ones(X.shape[0], device=device)
            for dim_idx in term_indices:
                # Convert from 1-indexed to 0-indexed
                term_value = term_value * X[:, dim_idx-1]
            
            # Apply phi function and coefficient
            true_term_output = coefficient * phi_func(term_value)
            
            # Run the model on batches
            total_error = 0.0
            total_samples = 0
            
            for i in range(0, X.shape[0], batch_size):
                end_idx = min(i + batch_size, X.shape[0])
                batch_size_actual = end_idx - i
                batch_X = X[i:end_idx]
                batch_true = true_term_output[i:end_idx]
                
                # Create a test case where only this term contributes
                # Start with a tensor of ones
                isolated_X = torch.ones_like(batch_X)
                
                # Copy only the features used by this term
                for dim_idx in term_indices:
                    # Convert from 1-indexed to 0-indexed
                    isolated_X[:, dim_idx-1] = batch_X[:, dim_idx-1]
                
                # Run model on isolated input
                with torch.amp.autocast('cuda'):
                    isolated_output = model(isolated_X).detach()
                
                # Calculate error for this term
                batch_error = torch.mean((isolated_output - batch_true) ** 2).item()
                
                # Update running average
                total_error = (total_error * total_samples + batch_error * batch_size_actual) / (total_samples + batch_size_actual)
                total_samples += batch_size_actual
            
            term_errors.append(total_error)
    
    return term_errors


def calculate_progressive_term_isolation(
    model: nn.Module,
    X: torch.Tensor,
    complex_terms: List[Dict],
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Calculate term learning metrics using the Progressive Term Isolation approach.
    Returns raw correlation values between model outputs and each term's contribution.
    
    Args:
        model: The neural network model
        X: Input tensor
        complex_terms: List of term dictionaries
        device: Device to place tensors on
        
    Returns:
        Dictionary with PTI metrics for each term
    """
    model.eval()
    batch_size = 10000  # Use large batch size for H100
    
    # Sort terms by complexity (number of indices)
    sorted_indices = sorted(range(len(complex_terms)), key=lambda i: len(complex_terms[i]["indices"]))
    
    # Initialize metrics
    pti_metrics = {
        'correlation_ratios': [],  # Correlation between model and term's unique contribution
        'residual_mse': [],        # MSE when predicting the residual
    }
    
    # Calculate true values for all terms
    true_term_values = []
    full_function = torch.zeros(X.shape[0], device=device)
    
    with torch.no_grad():
        for term_dict in complex_terms:
            term_indices = term_dict["indices"]
            phi_name = term_dict.get("phi", "id")
            coefficient = term_dict.get("coefficient", 1.0)
            phi_func = PHI_FUNCTIONS[phi_name]
            
            # Calculate true value for this term
            term_value = torch.ones(X.shape[0], device=device)
            for dim_idx in term_indices:
                # Convert from 1-indexed to 0-indexed
                term_value = term_value * X[:, dim_idx-1]
            
            # Apply phi function and coefficient
            true_term_output = coefficient * phi_func(term_value)
            true_term_values.append(true_term_output)
            full_function += true_term_output
    
    # Calculate residuals and correlations progressively
    with torch.no_grad():
        learned_contribution = torch.zeros(X.shape[0], device=device)
        
        for i in sorted_indices:
            term_dict = complex_terms[i]
            true_term = true_term_values[i]
            term_indices = term_dict["indices"]
            
            # Calculate residual (target minus what's been learned)
            residual = full_function - learned_contribution
            
            # Create test points that isolate this term's features
            isolated_X = torch.ones_like(X)
            for dim_idx in term_indices:
                # Convert from 1-indexed to 0-indexed
                isolated_X[:, dim_idx-1] = X[:, dim_idx-1]
            
            # Predict on isolated inputs
            model_outputs = []
            for j in range(0, X.shape[0], batch_size):
                end_idx = min(j + batch_size, X.shape[0])
                batch_X = isolated_X[j:end_idx]
                
                with torch.amp.autocast('cuda'):
                    batch_output = model(batch_X).detach()
                model_outputs.append(batch_output)
            
            model_output = torch.cat(model_outputs, dim=0)
            
            # Calculate correlation with residual
            model_mean = torch.mean(model_output)
            residual_mean = torch.mean(residual)
            
            numerator = torch.mean((model_output - model_mean) * (residual - residual_mean))
            denominator = torch.sqrt(torch.var(model_output) * torch.var(residual) + 1e-10)
            
            correlation = (numerator / denominator).item()
            
            # Calculate MSE on residual
            mse = torch.mean((model_output - true_term) ** 2).item()
            
            # Store raw metrics only
            pti_metrics['correlation_ratios'].append(correlation)
            pti_metrics['residual_mse'].append(mse)
            
            # Update accumulated contribution
            learned_contribution = learned_contribution + true_term
    
    # Reorder metrics to match original term order
    reordered_metrics = {
        'correlation_ratios': [0] * len(complex_terms),
        'residual_mse': [0] * len(complex_terms)
    }
    
    for i, orig_idx in enumerate(sorted_indices):
        for key in reordered_metrics:
            reordered_metrics[key][orig_idx] = pti_metrics[key][i]
    
    return reordered_metrics


def estimate_lipschitz_constant(model: nn.Module, X: torch.Tensor, device: torch.device) -> float:
    """
    Estimate the Lipschitz constant of the model by computing the maximum gradient norm.
    
    Args:
        model: The neural network model
        X: Input tensor, a sample of data points to calculate gradients
        device: Device to place tensors on
        
    Returns:
        Estimated Lipschitz constant
    """
    model.eval()
    batch_size = 10000  # Smaller batch size to avoid memory issues with gradient calculations
    max_grad_norm = 0.0
    
    # Create a smaller sample for efficiency if X is very large
    if X.shape[0] > 2000:
        # Use a random sample of 2000 examples
        indices = torch.randperm(X.shape[0])[:2000]
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Process in batches to avoid memory issues
    for i in range(0, len(X_sample), batch_size):
        X_batch = X_sample[i:i+batch_size].clone().detach().requires_grad_(True)
        
        # Forward pass
        with torch.amp.autocast('cuda'):
            outputs = model(X_batch)
        
        # For scalar output (regression)
        if outputs.dim() <= 1 or outputs.shape[1] == 1:
            # Compute gradients
            grads = torch.autograd.grad(
                outputs.sum(), X_batch, create_graph=False, retain_graph=False
            )[0]
            
            # Compute gradient norms for this batch
            batch_grad_norms = torch.norm(grads.view(grads.shape[0], -1), dim=1)
            current_max = torch.max(batch_grad_norms).item()
            max_grad_norm = max(max_grad_norm, current_max)
    
    return max_grad_norm


def calculate_lipschitz_normalized_loss(
    model: nn.Module, 
    X: torch.Tensor, 
    y: torch.Tensor, 
    lipschitz_constant: float
) -> Tuple[float, float]:
    """
    Calculate the Lipschitz normalized MSE loss.
    
    Args:
        model: The neural network model
        X: Input tensor
        y: Target tensor
        lipschitz_constant: The pre-calculated Lipschitz constant
        
    Returns:
        Tuple of (normalized_loss, regular_mse_loss)
    """
    model.eval()
    with torch.no_grad():
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(X)
            mse_loss = torch.mean((outputs - y) ** 2).item()
        
        # Square the Lipschitz constant for MSE (since MSE is squared)
        lipschitz_constant_squared = lipschitz_constant ** 2
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Normalize the loss
        normalized_loss = mse_loss / (lipschitz_constant_squared + epsilon)
        
        return normalized_loss, mse_loss


def generate_staircase_data(
    n_samples: int,
    input_dim: int,
    complex_terms: List[Dict],
    device: torch.device,
    distribution: str = "binary",
    test: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data for a complex staircase function with specified input distribution.
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Total input dimension
        complex_terms: List of term dictionaries with indices, phi, coefficient
        device: Device to place tensors on
        distribution: Input distribution type ("binary" or "normal")
        test: If True, generates test data with fixed seed
    
    Returns:
        Tuple of (X, y) tensors
    """
    # Find max dimension referenced in terms to ensure input_dim is sufficient
    all_dims = []
    for term_dict in complex_terms:
        all_dims.extend(term_dict["indices"])
    max_dim_needed = max(all_dims) if all_dims else 1
    
    if max_dim_needed > input_dim:
        raise ValueError(f"Function requires dimension {max_dim_needed} but input_dim is only {input_dim}")
    
    # Set seed for reproducibility of test set
    seed = 42 if test else int(time.time())
    torch.manual_seed(seed)
    
    # Generate input data based on specified distribution
    if distribution.lower() == "binary":
        # Binary inputs (-1 or 1)
        X = 2 * torch.randint(0, 2, (n_samples, input_dim), device=device).float() - 1
    elif distribution.lower() == "normal":
        # Standard normal distribution with scaling factor for better training stability
        X = torch.randn((n_samples, input_dim), device=device)  # Scale down inputs
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Use 'binary' or 'normal'.")
    
    # Start with zeros for target values
    y = torch.zeros(n_samples, device=device)
    
    # Add each term (product of selected dimensions with phi and coefficient)
    for term_dict in complex_terms:
        term_indices = term_dict["indices"]
        phi_name = term_dict.get("phi", "id")
        coefficient = term_dict.get("coefficient", 1.0)
        
        # Get the phi function
        phi_func = PHI_FUNCTIONS[phi_name]
        
        # Calculate raw term value (product of selected dimensions)
        term_value = torch.ones(n_samples, device=device)
        for dim_idx in term_indices:
            # Convert from 1-indexed to 0-indexed
            term_value = term_value * X[:, dim_idx-1]
        
        # Apply scaling correction for normal distribution
        if distribution.lower() == "normal" and len(term_indices) > 1:
            # Scale down product terms based on their order
            # This compensates for the growing variance of products of normal variables
            scale_factor = 1.0 / (len(term_indices) ** 0.5)
            term_value = term_value * scale_factor
        
        # Apply phi function and coefficient
        term_output = coefficient * phi_func(term_value)
        y = y + term_output
    
    return X, y


def generate_staircase_data_for_term(
    n_samples: int,
    input_dim: int, 
    term: Dict,
    device: torch.device,
    distribution: str = "binary"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data for a single term instead of the full function.
    Used in layer-term-wise training.
    """
    # Find max dimension referenced in term to ensure input_dim is sufficient
    term_indices = term["indices"]
    max_dim_needed = max(term_indices) if term_indices else 1
    
    if max_dim_needed > input_dim:
        raise ValueError(f"Term requires dimension {max_dim_needed} but input_dim is only {input_dim}")
    
    # Set seed for reproducibility
    seed = int(time.time())
    torch.manual_seed(seed)
    
    # Generate input data based on specified distribution
    if distribution.lower() == "binary":
        # Binary inputs (-1 or 1)
        X = 2 * torch.randint(0, 2, (n_samples, input_dim), device=device).float() - 1
    elif distribution.lower() == "normal":
        # Standard normal distribution
        X = torch.randn((n_samples, input_dim), device=device)
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Use 'binary' or 'normal'.")
    
    # Calculate target values for just this term
    y = torch.zeros(n_samples, device=device)
    
    # Get term parameters
    phi_name = term.get("phi", "id")
    coefficient = term.get("coefficient", 1.0)
    
    # Get the phi function
    phi_func = PHI_FUNCTIONS[phi_name]
    
    # Calculate raw term value (product of selected dimensions)
    term_value = torch.ones(n_samples, device=device)
    for dim_idx in term_indices:
        # Convert from 1-indexed to 0-indexed
        term_value = term_value * X[:, dim_idx-1]
    
    # Apply scaling correction for normal distribution
    if distribution.lower() == "normal" and len(term_indices) > 1:
        # Scale down product terms based on their order
        scale_factor = 1.0 / (len(term_indices) ** 0.5)
        term_value = term_value * scale_factor
    
    # Apply phi function and coefficient
    y = coefficient * phi_func(term_value)
    
    return X, y


def train_model(
    gpu_id: int, 
    config: Dict,
    results_dir: str,
    checkpoint_file: str,
    function: Dict,
    lr: float, 
    batch_size: int, 
    hidden_size: int, 
    depth: int, 
    mode: str, 
    align: bool, 
    input_dim: int, 
    exp_num: int
) -> None:
    """
    Train a model with the given configuration on a specific GPU.
    Enhanced version supporting layer-wise and layer-term-wise training.
    """
    try:
        # Set up the device
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # Enable optimizations for H100 GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
        # Enable Flash Attention and other H100 optimizations if available
        try:
            torch.backends.cuda.enable_flash_sdp = True
            torch.backends.cuda.enable_mem_efficient_sdp = True
            torch.backends.cuda.enable_math_sdp = True
            print(f"[GPU {gpu_id}] Enabled H100-specific optimizations")
        except AttributeError:
            pass
        
        print(f"[GPU {gpu_id}] Starting training for configuration")
        
        # Get input distribution from config
        input_distribution = config['base_config'].get('input_distribution', 'binary')
        print(f"[GPU {gpu_id}] Using {input_distribution} input distribution")
        
        # Get early stopping configuration
        early_stopping_enabled = config['base_config'].get('early_stopping', {}).get('enabled', False)
        early_stopping_threshold = config['base_config'].get('early_stopping', {}).get('threshold', 1e-4)
        early_stopping_patience = config['base_config'].get('early_stopping', {}).get('patience', 1000)
        
        # Get the new layer-wise training parameters
        layer_term_wise = config['base_config'].get('layer_term_wise', False)
        layer_wise = config['base_config'].get('layer_wise', False)
        layer_freeze_threshold = config['base_config'].get('layer_freeze_threshold', 0.15)  # Changed to 0.15 as requested
        
        if layer_term_wise and layer_wise:
            print(f"[GPU {gpu_id}] Warning: Both layer_term_wise and layer_wise are enabled. Defaulting to layer_term_wise.")
            layer_wise = False
        
        if layer_term_wise:
            print(f"[GPU {gpu_id}] Using layer-term-wise training with threshold {layer_freeze_threshold}")
        elif layer_wise:
            print(f"[GPU {gpu_id}] Using layer-wise training with threshold {layer_freeze_threshold}")
        
        # Get load factor calculation configuration
        calculate_load_factor = config['base_config'].get('calculate_load_factor', False)
        load_factor_test_epochs = config['base_config'].get('load_factor_test_epochs', 1000)
        load_factor_workers = config['base_config'].get('load_factor_workers', 4)
        
        # Generate unique ID for this run
        unique_id = generate_unique_id(
            function, lr, hidden_size, depth, mode, align, input_dim, batch_size, 
            exp_num, input_distribution, calculate_load_factor
        )
        
        if layer_term_wise:
            unique_id += "_layertermwise"
        elif layer_wise:
            unique_id += "_layerwise"
            
        print(f"[GPU {gpu_id}] Training: {unique_id}")
        
        # Get complex terms from function definition
        complex_terms = function['terms']
        
        # Check if we have enough layers for terms in layer-term-wise mode
        if layer_term_wise and len(complex_terms) > depth:
            print(f"[GPU {gpu_id}] Error: Not enough layers ({depth}) for terms ({len(complex_terms)}) in layer-term-wise mode")
            return
        
        # Print the term descriptions for clarity
        term_descriptions = get_term_descriptions(complex_terms)
        print(f"[GPU {gpu_id}] Complex terms: {term_descriptions}")
        
        # Generate test data once (will be reused for all evaluations)
        n_test = config['base_config'].get('test_set_size', 10000)  # Larger test set for H100
        
        try:
            X_test, y_test = generate_staircase_data(
                n_samples=n_test,
                input_dim=input_dim,
                complex_terms=complex_terms,
                device=device,
                distribution=input_distribution,
                test=True
            )
            
            # Pin test data for faster access if not already on GPU
            if X_test.device.type != 'cuda':
                X_test = X_test.pin_memory()
            if y_test.device.type != 'cuda':
                y_test = y_test.pin_memory()
            
        except ValueError as e:
            print(f"[GPU {gpu_id}] Error generating test data: {str(e)}")
            return
        
        # Create model
        model = DeepNN(
            d=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            mode=mode,
            alignment=align,
            base_width=config['sweeps'].get('base_width', 256),
            gamma=1.0
        ).to(device)
        
        # Use improved compile function with better error handling
        model = try_compile_model(model, gpu_id)
        
        # Create optimizer with explicit float conversion for scientific notation
        optimizer_type = config['base_config'].get('optimizer', 'sgd').lower()
        if optimizer_type == 'sgd':
            momentum = float(config['base_config'].get('optimizer_params', {}).get('sgd', {}).get('momentum', 0.0))
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_type == 'adam':
            adam_params = config['base_config'].get('optimizer_params', {}).get('adam', {})
            beta1 = float(adam_params.get('beta1', 0.9))
            beta2 = float(adam_params.get('beta2', 0.999))
            eps = float(adam_params.get('eps', 1e-8))
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Training parameters
        total_iterations = int(float(config['base_config'].get('epochs', 1000)))
        
        # Get gradient clipping settings
        use_gradient_clipping = config['base_config'].get('gradient_clipping', {}).get('enabled', False)
        max_grad_norm = config['base_config'].get('gradient_clipping', {}).get('max_norm', 1.0)
        if use_gradient_clipping:
            print(f"[GPU {gpu_id}] Using gradient clipping with max norm: {max_grad_norm}")
        
        # Super-batch size: how many data points to generate at once
        # This reduces the overhead of frequently generating new data
        super_batch_size = batch_size * 100  # Generate 100 batches worth of data at once
        
        # STORAGE OPTIMIZATION: Only store stats at specific intervals
        stat_interval = 1000  # Record training and lipschitz stats every 100 epochs
        eval_interval = 5000  # How often to evaluate on test set
        save_interval = eval_interval * 5  # How often to save results to disk
        
        # Storage for summarized training history
        train_stats = []  # Will store [iter_num, mean_loss, variance_loss]
        lipschitz_metrics = []  # Will store [iter_num, lipschitz_constant, norm_loss, raw_loss]
        test_metrics = []  # Will store [iter_num, test_loss]
        term_test_metrics = []  # Will store [iter_num, [term1_loss, term2_loss, ...]]
        
        # Storage for our advanced metrics
        term_pti_metrics = []  # Will store [iter_num, {pti_metrics}]
        term_gradient_metrics = []  # Will store [iter_num, {gradient_metrics}]
        
        # Containers for collecting batch metrics
        current_train_losses = []
        current_lip_constants = []
        current_lip_norm_losses = []
        current_lip_raw_losses = []
        
        # Initialize mixed precision scaler
        scaler = GradScaler()
        
        # Initialize early stopping variables
        below_threshold_count = 0
        early_stopping_triggered = False
        
        # Initialize layer-wise and layer-term-wise training state
        frozen_layers = set()
        active_term_index = 0 if layer_term_wise else None
        active_layer_index = 0 if (layer_term_wise or layer_wise) else None
        
        # Save initial model if requested
        if "initial" in config['base_config'].get('save_models', []):
            models_dir = os.path.join(results_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            initial_model_path = os.path.join(models_dir, f"{unique_id}_initial.pth.gz")
            save_compressed_model(model, initial_model_path)
        
        # Training loop
        iteration = 0
        
        # Calculate initial load factors if layer-wise training is enabled
        if layer_term_wise or layer_wise:
            print(f"[GPU {gpu_id}] Calculating initial per-term load factors...")
            model.eval()
            term_load_factors = compute_load_factor_per_term(
                model=model,
                X=X_test,
                complex_terms=complex_terms,
                device=device,
                num_workers=load_factor_workers
            )
            
            # Save the initial term-specific load factors
            savable_load_factors = {}
            for term_key, layer_factors in term_load_factors.items():
                savable_load_factors[term_key] = np.array(layer_factors)
            
            # Add metadata about initial training state
            metadata = {
                'iteration': 0,
                'frozen_layers': list(frozen_layers),
                'active_term_index': active_term_index if active_term_index is not None else -1,
                'active_layer_index': active_layer_index if active_layer_index is not None else -1,
                'mode': 'layer_term_wise' if layer_term_wise else 'layer_wise',
                'threshold': layer_freeze_threshold
            }
            
            # Save to file
            initial_term_load_file = os.path.join(results_dir, f"{unique_id}_termloadfactor_initial.npz")
            np.savez_compressed(
                initial_term_load_file,
                load_factors=savable_load_factors,
                term_descriptions=term_descriptions,
                metadata=np.array([str(metadata)])
            )
            
            print(f"[GPU {gpu_id}] Initial term-specific load factors saved to {initial_term_load_file}")
            
            model.train()
            
            # Log initial load factors
            for term_key, layer_factors in term_load_factors.items():
                # Skip the input layer (index 0)
                trainable_layer_factors = layer_factors[1:]
                print(f"[GPU {gpu_id}] Initial load factors for {term_key}: {[round(factor, 6) for factor in trainable_layer_factors]}")
        
        # Calculate initial load factors if enabled (for the standard load factor calculation)
        if calculate_load_factor:
            print(f"[GPU {gpu_id}] Calculating initial load factors...")
            
            model.eval()  # Switch to evaluation mode for load factor calculation
            
            # Calculate load factors for all term combinations
            load_factors = compute_load_factor(
                model=model,
                X=X_test,
                y=y_test,
                complex_terms=complex_terms,
                device=device,
                num_workers=load_factor_workers
            )
            
            # Save load factors to a separate file
            load_factor_file = os.path.join(results_dir, f"{unique_id}_loadfactor_initial.npz")
            np.savez_compressed(
                load_factor_file, 
                iteration=0,
                load_factors=load_factors,
                term_descriptions=term_descriptions
            )
            
            print(f"[GPU {gpu_id}] Initial load factors saved to {load_factor_file}")
            model.train()  # Switch back to training mode
        
        # Generate data in chunks to reduce overhead while still respecting online SGD
        while iteration < total_iterations and not early_stopping_triggered:
            # Calculate how many samples to generate in this chunk
            samples_to_generate = min(super_batch_size, (total_iterations - iteration) * batch_size)
            
            # Generate a large chunk of training data based on active term if using layer-term-wise
            if layer_term_wise and active_term_index is not None:
                # For layer-term-wise, generate data only for the active term
                X_chunk, y_chunk = generate_staircase_data_for_term(
                    n_samples=samples_to_generate,
                    input_dim=input_dim,
                    term=complex_terms[active_term_index],
                    device=device,
                    distribution=input_distribution
                )
            else:
                # For standard or layer-wise, generate data for the full function
                X_chunk, y_chunk = generate_staircase_data(
                    n_samples=samples_to_generate,
                    input_dim=input_dim,
                    complex_terms=complex_terms,
                    device=device,
                    distribution=input_distribution
                )
            
            # Process this chunk in small batches (respecting online SGD)
            num_super_batches = (len(X_chunk) + batch_size - 1) // batch_size
            
            for i in range(num_super_batches):
                if iteration >= total_iterations or early_stopping_triggered:
                    break
                    
                # Get a small batch from the super-batch
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X_chunk))
                X_batch = X_chunk[start_idx:end_idx]
                y_batch = y_chunk[start_idx:end_idx]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with autocast (mixed precision)
                with torch.amp.autocast('cuda'):
                    outputs = model(X_batch)
                    loss = torch.mean((outputs - y_batch) ** 2)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                
                # If any layers are frozen, zero out their gradients
                if frozen_layers:
                    linear_idx = 0
                    for layer in model.layers:
                        if isinstance(layer, nn.Linear):
                            if linear_idx in frozen_layers:
                                for param in layer.parameters():
                                    if param.grad is not None:
                                        param.grad.zero_()
                            linear_idx += 1
                
                # Apply gradient clipping if enabled
                if use_gradient_clipping:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    # Clip gradients to max_norm
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Update parameters with scaler to maintain mixed precision
                scaler.step(optimizer)
                
                # Scaler update
                scaler.update()
                
                # Record loss for this batch
                train_loss = loss.item()
                current_train_losses.append(train_loss)
                
                # Calculate Lipschitz constant and normalized loss for this batch (without backprop)
                model.eval()
                batch_lip_constant = estimate_lipschitz_constant(model, X_batch, device)
                batch_norm_loss, batch_raw_loss = calculate_lipschitz_normalized_loss(
                    model, X_batch, y_batch, batch_lip_constant
                )
                model.train()
                
                # Record Lipschitz metrics for this batch
                current_lip_constants.append(batch_lip_constant)
                current_lip_norm_losses.append(batch_norm_loss)
                current_lip_raw_losses.append(batch_raw_loss)
                
                # Check if we need to evaluate layer load factors for layer-wise training
                check_layer_factors = (
                    (layer_term_wise or layer_wise) and
                    iteration % load_factor_test_epochs == 0 and 
                    iteration > 0
                )
                
                if check_layer_factors:
                    print(f"[GPU {gpu_id}] Checking layer load factors at iteration {iteration}...")
                    
                    model.eval()
                    term_load_factors = compute_load_factor_per_term(
                        model=model,
                        X=X_test,
                        complex_terms=complex_terms,
                        device=device,
                        num_workers=load_factor_workers
                    )
                    
                    # Save the term-specific load factors
                    savable_load_factors = {}
                    for term_key, layer_factors in term_load_factors.items():
                        savable_load_factors[term_key] = np.array(layer_factors)
                    
                    # Add metadata about current training state
                    metadata = {
                        'iteration': int(iteration),
                        'frozen_layers': list(frozen_layers),
                        'active_term_index': active_term_index if active_term_index is not None else -1,
                        'active_layer_index': active_layer_index if active_layer_index is not None else -1,
                        'mode': 'layer_term_wise' if layer_term_wise else 'layer_wise',
                        'threshold': layer_freeze_threshold
                    }
                    
                    # Save to file
                    term_load_file = os.path.join(results_dir, f"{unique_id}_termloadfactor_iter{iteration}.npz")
                    np.savez_compressed(
                        term_load_file,
                        load_factors=savable_load_factors,
                        term_descriptions=term_descriptions,
                        metadata=np.array([str(metadata)])
                    )
                    
                    print(f"[GPU {gpu_id}] Term-specific load factors saved to {term_load_file}")
                    
                    model.train()
                    
                    # Process load factors based on training mode
                    if layer_term_wise and active_term_index is not None:
                        # Get active term key
                        active_term_key = f"term{active_term_index}"
                        
                        # Get load factors for the active term, skipping input layer (index 0)
                        active_term_factors = term_load_factors[active_term_key][1:]
                        
                        # Check if active layer meets threshold for active term
                        # Note: active_layer_index is 0-indexed but we need to add 1 because
                        # we skipped the input layer in active_term_factors
                        if active_layer_index < len(active_term_factors):
                            current_factor = active_term_factors[active_layer_index]
                            
                            print(f"[GPU {gpu_id}] Layer {active_layer_index+1} load factor for {active_term_key}: {current_factor:.6f}")
                            
                            if current_factor < layer_freeze_threshold:
                                # Freeze the current layer
                                freeze_layer(model, active_layer_index)
                                frozen_layers.add(active_layer_index)
                                
                                # Move to next term and layer
                                active_term_index += 1
                                active_layer_index += 1
                                
                                # Check if we've finished all terms
                                if active_term_index >= len(complex_terms):
                                    print(f"[GPU {gpu_id}] All terms have been learned by their respective layers!")
                                    active_term_index = None
                                    active_layer_index = None
                                else:
                                    print(f"[GPU {gpu_id}] Moving to term {active_term_index} and layer {active_layer_index+1}")
                    
                    elif layer_wise and active_layer_index is not None:
                        # We focus on term0 for layer-wise training
                        term0_key = "term0"
                        
                        # Get load factors for term0, skipping input layer (index 0)
                        term0_factors = term_load_factors[term0_key][1:]
                        
                        if active_layer_index < len(term0_factors):
                            current_factor = term0_factors[active_layer_index]
                            
                            print(f"[GPU {gpu_id}] Layer {active_layer_index+1} load factor for {term0_key}: {current_factor:.6f}")
                            
                            if current_factor < layer_freeze_threshold:
                                # Freeze the current layer
                                freeze_layer(model, active_layer_index)
                                frozen_layers.add(active_layer_index)
                                
                                # Move to next layer
                                active_layer_index += 1
                                
                                # Check if we've finished all layers
                                if active_layer_index >= len(term0_factors):
                                    print(f"[GPU {gpu_id}] All layers have reached threshold on term0!")
                                    active_layer_index = None
                
                # Check if we need to calculate standard load factors
                calculate_load_now = False
                
                # Calculate standard load factors at initial iteration, specific epochs, and final iteration
                if calculate_load_factor:
                    if (iteration == 1 or  # Initial iteration
                        (iteration % load_factor_test_epochs == 0 and iteration > 0) or  # Every load_factor_test_epochs iterations
                        iteration == total_iterations - 1 or  # Final iteration
                        early_stopping_triggered):  # Early stopping
                        calculate_load_now = True
                
                if calculate_load_now:
                    print(f"[GPU {gpu_id}] Calculating standard load factors at iteration {iteration}...")
                    
                    model.eval()  # Switch to evaluation mode for load factor calculation
                    
                    # Calculate load factors for all term combinations
                    load_factors = compute_load_factor(
                        model=model,
                        X=X_test,
                        y=y_test,
                        complex_terms=complex_terms,
                        device=device,
                        num_workers=load_factor_workers
                    )
                    
                    # Save load factors to a separate file
                    load_factor_file = os.path.join(results_dir, f"{unique_id}_loadfactor_iter{iteration}.npz")
                    np.savez_compressed(
                        load_factor_file, 
                        iteration=iteration,
                        load_factors=load_factors,
                        term_descriptions=term_descriptions
                    )
                    
                    print(f"[GPU {gpu_id}] Standard load factors saved to {load_factor_file}")
                    model.train()  # Switch back to training mode
                
                # Increment iteration counter
                iteration += 1
                
                # Early stopping check
                if early_stopping_enabled:
                    if train_loss < early_stopping_threshold:
                        below_threshold_count += 1
                        if below_threshold_count >= early_stopping_patience:
                            print(f"[GPU {gpu_id}] {unique_id} - Early stopping triggered at iteration {iteration}. "
                                  f"Loss below {early_stopping_threshold} for {early_stopping_patience} consecutive iterations.")
                            early_stopping_triggered = True
                            break
                    else:
                        below_threshold_count = 0  # Reset the counter if loss is above threshold
                
                # Record metrics at fixed stat_interval (100 iterations)
                if iteration % stat_interval == 0:
                    # Calculate mean and variance for training losses
                    if current_train_losses:
                        mean_train_loss = sum(current_train_losses) / len(current_train_losses)
                        variance_train = sum((x - mean_train_loss) ** 2 for x in current_train_losses) / len(current_train_losses) if len(current_train_losses) > 1 else 0.0
                        
                        # Store training stats - use float32 for iteration number
                        train_stats.append([float(iteration), mean_train_loss, variance_train])
                        
                        # Reset training loss buffer
                        current_train_losses = []
                    
                    # Calculate mean for Lipschitz metrics
                    if current_lip_constants:
                        mean_lip_constant = sum(current_lip_constants) / len(current_lip_constants)
                        mean_lip_norm_loss = sum(current_lip_norm_losses) / len(current_lip_norm_losses)
                        mean_lip_raw_loss = sum(current_lip_raw_losses) / len(current_lip_raw_losses)
                        
                        # Store Lipschitz metrics - use float32 for iteration number
                        lipschitz_metrics.append([float(iteration), mean_lip_constant, mean_lip_norm_loss, mean_lip_raw_loss])
                        
                        # Reset Lipschitz metric buffers
                        current_lip_constants = []
                        current_lip_norm_losses = []
                        current_lip_raw_losses = []
                    
                    # Log progress
                    print(f"[GPU {gpu_id}] {unique_id} - Iteration {iteration}/{total_iterations}, "
                          f"Train Loss: {mean_train_loss:.6f}, Lipschitz Norm Loss: {mean_lip_norm_loss:.6f}")
                    
                    # Log layer freezing status
                    if frozen_layers:
                        print(f"[GPU {gpu_id}] Frozen layers: {sorted(frozen_layers)}")
                    
                    if layer_term_wise and active_term_index is not None:
                        print(f"[GPU {gpu_id}] Currently training layer {active_layer_index+1} on term {active_term_index}")
                    elif layer_wise and active_layer_index is not None:
                        print(f"[GPU {gpu_id}] Currently monitoring layer {active_layer_index+1} for freezing")
                
                # Only evaluate on test set at specific intervals OR at the final iteration
                should_evaluate = (
                    (iteration % eval_interval == 0) or 
                    (iteration == 1) or 
                    (iteration == total_iterations) or
                    early_stopping_triggered
                )
                
                if should_evaluate:
                    model.eval()
                    with torch.no_grad():
                        # Calculate full test loss, being careful with compiled model
                        test_loss = evaluate_test_loss(model, X_test, y_test)
                    
                    # Record test metrics - use float32 for iteration number
                    test_metrics.append([float(iteration), test_loss])
                    
                    # Calculate original term-specific losses
                    term_losses = calculate_term_test_losses(
                        model=model,
                        X=X_test,
                        complex_terms=complex_terms,
                        device=device
                    )
                    
                    # Store term test losses with iteration number
                    term_test_metrics.append([float(iteration), term_losses])
                    
                    # Calculate Progressive Term Isolation metrics
                    pti_results = calculate_progressive_term_isolation(
                        model=model,
                        X=X_test,
                        complex_terms=complex_terms,
                        device=device
                    )
                    
                    # Store PTI metrics with iteration number
                    term_pti_metrics.append([float(iteration), pti_results])
                    
                    # Log test results with individual term losses
                    print(f"[GPU {gpu_id}] {unique_id} - Iteration {iteration}/{total_iterations}, "
                          f"Test Loss: {test_loss:.6f}")
                    
                    # Log term metrics in a more readable format
                    term_loss_strings = []
                    for i, (desc, loss) in enumerate(zip(term_descriptions, term_losses)):
                        # Include PTI correlation
                        corr = pti_results['correlation_ratios'][i]
                        term_loss_strings.append(f"Term {i} ({desc}): MSE={loss:.6f}, Corr={corr:.3f}")
                    
                    print(f"[GPU {gpu_id}] Term Metrics: {' | '.join(term_loss_strings)}")
                    
                    # Save current results at checkpoints using compressed NPZ
                    if iteration % save_interval == 0 or iteration == total_iterations or early_stopping_triggered:
                        result_file = os.path.join(results_dir, f"{unique_id}.npz")
                        
                        # Convert lists to numpy arrays with float32 precision for numerical stability
                        train_stats_np = np.array(train_stats, dtype=np.float32)
                        test_metrics_np = np.array(test_metrics, dtype=np.float32)
                        
                        # Extract iterations and term losses for efficient storage
                        term_iterations = np.array([t[0] for t in term_test_metrics], dtype=np.float32)
                        term_losses_array = np.array([t[1] for t in term_test_metrics], dtype=np.float32)
                        
                        # Process PTI metrics efficiently
                        pti_iterations = np.array([t[0] for t in term_pti_metrics], dtype=np.float32)
                        pti_correlation_ratios = np.array([t[1]['correlation_ratios'] for t in term_pti_metrics], dtype=np.float32)
                        pti_residual_mse = np.array([t[1]['residual_mse'] for t in term_pti_metrics], dtype=np.float32)
                        
                        # Process Lipschitz metrics
                        lipschitz_iterations = np.array([t[0] for t in lipschitz_metrics], dtype=np.float32)
                        lipschitz_constants = np.array([t[1] for t in lipschitz_metrics], dtype=np.float32)
                        lipschitz_normalized_losses = np.array([t[2] for t in lipschitz_metrics], dtype=np.float32)
                        lipschitz_raw_losses = np.array([t[3] for t in lipschitz_metrics], dtype=np.float32)
                        
                        # Store term descriptions for context
                        term_descriptions_array = np.array([desc for desc in term_descriptions])
                        
                        # Get final metrics
                        final_train_loss = float(train_stats[-1][1]) if train_stats else float('nan')
                        final_test_loss = test_metrics[-1][1] if test_metrics else float('nan')
                        final_term_losses = term_test_metrics[-1][1] if term_test_metrics else []
                        final_lipschitz_constant = float(lipschitz_constants[-1]) if len(lipschitz_constants) > 0 else float('nan')
                        final_normalized_loss = float(lipschitz_normalized_losses[-1]) if len(lipschitz_normalized_losses) > 0 else float('nan')
                        
                        # Store metadata
                        metadata = {
                            'function_name': function['name'],
                            'input_dim': input_dim,
                            'input_distribution': input_distribution,
                            'hidden_size': hidden_size,
                            'depth': depth,
                            'learning_rate': lr,
                            'batch_size': batch_size,  # Include batch size in metadata
                            'mode': mode,
                            'alignment': align,
                            'experiment_num': exp_num,
                            'final_train_loss': final_train_loss,
                            'final_test_loss': final_test_loss,
                            'final_lipschitz_constant': final_lipschitz_constant,
                            'final_normalized_loss': final_normalized_loss,
                            'optimizer': optimizer_type,
                            'unique_id': unique_id,
                            'gpu_id': gpu_id,
                            'total_iterations': total_iterations,
                            'current_iteration': iteration,
                            'final_term_losses': final_term_losses,
                            'term_descriptions': term_descriptions,
                            'layer_term_wise': layer_term_wise,
                            'layer_wise': layer_wise,
                            'layer_freeze_threshold': layer_freeze_threshold,
                            'frozen_layers': list(frozen_layers)
                        }
                        
                        # Create a dictionary of data to save
                        save_data = {
                            # Original metrics
                            'train_stats': train_stats_np,
                            'test_metrics': test_metrics_np,
                            'term_iterations': term_iterations,
                            'term_losses': term_losses_array,
                            'term_descriptions': term_descriptions_array,
                            
                            # PTI metrics
                            'pti_iterations': pti_iterations,
                            'pti_correlation_ratios': pti_correlation_ratios,
                            'pti_residual_mse': pti_residual_mse,
                            
                            # Lipschitz metrics
                            'lipschitz_iterations': lipschitz_iterations,
                            'lipschitz_constants': lipschitz_constants,
                            'lipschitz_normalized_losses': lipschitz_normalized_losses,
                            'lipschitz_raw_losses': lipschitz_raw_losses,
                            
                            # Metadata
                            'metadata': np.array([str(metadata)])
                        }
                        
                        # Save NPZ file with all metrics
                        np.savez_compressed(result_file, **save_data)
                        print(f"[GPU {gpu_id}] Saved results to {result_file}")
                    
                    # Switch back to training mode
                    model.train()
        
        # Save final model if requested
        if "final" in config['base_config'].get('save_models', []):
            models_dir = os.path.join(results_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            final_model_path = os.path.join(models_dir, f"{unique_id}_final.pth.gz")
            save_compressed_model(model, final_model_path)
        
        # Calculate and save final term-specific load factors
        if (layer_term_wise or layer_wise) and not early_stopping_triggered:
            print(f"[GPU {gpu_id}] Calculating final term-specific load factors...")
            
            model.eval()
            final_term_load_factors = compute_load_factor_per_term(
                model=model,
                X=X_test,
                complex_terms=complex_terms,
                device=device,
                num_workers=load_factor_workers
            )
            
            # Convert load factors to a savable format
            savable_load_factors = {}
            for term_key, layer_factors in final_term_load_factors.items():
                savable_load_factors[term_key] = np.array(layer_factors)
            
            # Add metadata about final training state
            metadata = {
                'iteration': int(iteration),
                'frozen_layers': list(frozen_layers),
                'active_term_index': active_term_index if active_term_index is not None else -1,
                'active_layer_index': active_layer_index if active_layer_index is not None else -1,
                'mode': 'layer_term_wise' if layer_term_wise else 'layer_wise',
                'threshold': layer_freeze_threshold,
                'final': True
            }
            
            # Save to file
            final_term_load_file = os.path.join(results_dir, f"{unique_id}_termloadfactor_final.npz")
            np.savez_compressed(
                final_term_load_file,
                load_factors=savable_load_factors,
                term_descriptions=term_descriptions,
                metadata=np.array([str(metadata)])
            )
            
            print(f"[GPU {gpu_id}] Final term-specific load factors saved to {final_term_load_file}")
            model.train()
        
        # Calculate final standard load factors if enabled and not already calculated
        if calculate_load_factor and not early_stopping_triggered:
            print(f"[GPU {gpu_id}] Calculating final standard load factors...")
            
            model.eval()
            
            # Calculate load factors for all term combinations
            load_factors = compute_load_factor(
                model=model,
                X=X_test,
                y=y_test,
                complex_terms=complex_terms,
                device=device,
                num_workers=load_factor_workers
            )
            
            # Save load factors to a separate file
            load_factor_file = os.path.join(results_dir, f"{unique_id}_loadfactor_final.npz")
            np.savez_compressed(
                load_factor_file, 
                iteration=iteration,
                load_factors=load_factors,
                term_descriptions=term_descriptions
            )
            
            print(f"[GPU {gpu_id}] Final standard load factors saved to {load_factor_file}")
        
        print(f"[GPU {gpu_id}] Completed: {unique_id}")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error in training: {str(e)}")
        traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_load_layerwise.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    print(f"Loading config from: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = config['base_config']['base_results_dir']
    results_dir = os.path.join(base_results_dir, f"complex_leap_exp_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up checkpoint file
    checkpoint_file = os.path.join(results_dir, f"checkpoint_{timestamp}.txt")
    
    # Check if we should restart from a previous checkpoint
    restart_checkpoint = config['base_config'].get('restart_checkpoint')
    if restart_checkpoint and os.path.exists(restart_checkpoint):
        checkpoint_file = restart_checkpoint
        print(f"Restarting from checkpoint: {restart_checkpoint}")
    
    # Save config for reference
    with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available. Running on CPU.")
        num_gpus = 1
    
    print(f"Using {num_gpus} GPU(s)")
    
    # Generate all combinations of hyperparameters
    functions = config['functions']
    
    # Get hyperparameters for sweeping
    learning_rates = config['sweeps']['learning_rates']
    batch_sizes = config['sweeps']['batch_sizes']
    hidden_sizes = config['sweeps']['hidden_sizes']
    depths = config['sweeps']['depths']
    modes = config['sweeps']['modes']
    alignments = config['sweeps']['alignment']
    num_experiments = config['base_config']['num_experiments']
    
    # Get input distribution type
    input_distribution = config['base_config'].get('input_distribution', 'binary')
    print(f"Using input distribution: {input_distribution}")
    
    # Generate all job configurations
    all_combinations = []
    for function in functions:
        # Get dimensions specific to this function
        dimensions = function.get('dimensions', [10])  # Default to [10] if not specified
        
        for lr, batch_size, hidden_size, depth, mode, align, input_dim, exp_num in itertools.product(
            learning_rates, batch_sizes, hidden_sizes, depths, modes, alignments, dimensions, 
            range(1, num_experiments + 1)
        ):
            all_combinations.append((function, lr, batch_size, hidden_size, depth, mode, align, input_dim, exp_num))
    
    print(f"Total configurations to train: {len(all_combinations)}")
    
    # Load checkpoint to skip completed configurations
    completed_configs = load_checkpoint(checkpoint_file)
    remaining_configs = 0
    
    # Choose GPU 0 by default for the main training
    primary_gpu_id = 0
    
    # Process each configuration sequentially
    for combo_idx, combo in enumerate(all_combinations):
        function, lr, batch_size, hidden_size, depth, mode, align, input_dim, exp_num = combo
        
        # Generate unique ID for this run
        unique_id = generate_unique_id(
            function, lr, hidden_size, depth, mode, align, input_dim, batch_size, 
            exp_num, input_distribution, config['base_config'].get('calculate_load_factor', False)
        )
        
        # Append training mode suffix if needed
        if config['base_config'].get('layer_term_wise', False):
            unique_id += "_layertermwise"
        elif config['base_config'].get('layer_wise', False):
            unique_id += "_layerwise"
        
        # Skip if already completed
        if unique_id in completed_configs:
            print(f"Skipping completed configuration: {unique_id}")
            continue
        
        remaining_configs += 1
        
        print(f"Processing configuration {combo_idx + 1}/{remaining_configs}: {unique_id}")
        
        # Run training for this configuration on the primary GPU
        train_model(
            gpu_id=primary_gpu_id,
            config=config,
            results_dir=results_dir,
            checkpoint_file=checkpoint_file,
            function=function,
            lr=lr,
            batch_size=batch_size,
            hidden_size=hidden_size,
            depth=depth,
            mode=mode,
            align=align,
            input_dim=input_dim,
            exp_num=exp_num
        )
        
        # Mark as completed
        update_checkpoint(checkpoint_file, unique_id)
        completed_configs.add(unique_id)
        
        print(f"Completed configuration: {unique_id}")
    
    print(f"All training completed. Results saved to {results_dir}")


if __name__ == "__main__":
    main()