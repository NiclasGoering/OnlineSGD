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

def calculate_term_gradient_alignment(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    complex_terms: List[Dict],
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Calculate term-specific gradient alignment metrics.
    This measures how aligned the model's gradient is with the gradient for learning each term.
    
    Args:
        model: The neural network model
        X: Input tensor
        y: Target tensor (full function values)
        complex_terms: List of term dictionaries
        device: Device to place tensors on
        
    Returns:
        Dictionary with gradient alignment metrics for each term
    """
    # Use large batch for gradient calculations on H100s
    batch_size = 10000
    X_batch = X[:batch_size] if X.shape[0] > batch_size else X
    y_batch = y[:batch_size] if y.shape[0] > batch_size else y
    
    # Initialize metrics
    alignment_metrics = {
        'gradient_alignment': [],  # Cosine similarity between term gradient and full gradient
        'gradient_magnitude_ratio': [],  # Ratio of term gradient magnitude to full gradient
    }
    
    # Calculate true values for all terms in the batch
    true_term_values = []
    with torch.no_grad():
        for term_dict in complex_terms:
            term_indices = term_dict["indices"]
            phi_name = term_dict.get("phi", "id")
            coefficient = term_dict.get("coefficient", 1.0)
            phi_func = PHI_FUNCTIONS[phi_name]
            
            # Calculate true value for this term
            term_value = torch.ones(X_batch.shape[0], device=device)
            for dim_idx in term_indices:
                # Convert from 1-indexed to 0-indexed
                term_value = term_value * X_batch[:, dim_idx-1]
            
            # Apply phi function and coefficient
            true_term_output = coefficient * phi_func(term_value)
            true_term_values.append(true_term_output)
    
    # Define MSE loss
    mse_loss = lambda pred, target: torch.mean((pred - target) ** 2)
    
    # Calculate gradient for full function (efficiently reusing the model forward pass)
    model.zero_grad()
    output_full = model(X_batch)
    loss_full = mse_loss(output_full, y_batch)
    loss_full.backward()
    
    # Store gradients for full function (flattened for efficient dot products)
    grad_vector_full = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
    grad_norm_full = torch.norm(grad_vector_full)
    
    # Calculate alignment for each term
    for i, true_term in enumerate(true_term_values):
        # Calculate gradient for this term
        model.zero_grad()
        output_term = model(X_batch)  # Reuse the same input to avoid recomputation
        loss_term = mse_loss(output_term, true_term)
        loss_term.backward()
        
        # Store gradients for this term (flattened)
        grad_vector_term = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
        grad_norm_term = torch.norm(grad_vector_term)
        
        # Calculate cosine similarity (alignment)
        if grad_norm_full > 0 and grad_norm_term > 0:
            alignment = torch.dot(grad_vector_full, grad_vector_term) / (grad_norm_full * grad_norm_term)
            alignment = alignment.item()
        else:
            alignment = 0.0
        
        # Calculate gradient magnitude ratio
        magnitude_ratio = (grad_norm_term / (grad_norm_full + 1e-10)).item()
        
        # Store metrics
        alignment_metrics['gradient_alignment'].append(alignment)
        alignment_metrics['gradient_magnitude_ratio'].append(magnitude_ratio)
    
    # Reset gradients after analysis
    model.zero_grad()
    
    return alignment_metrics

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

def generate_unique_id(function: Dict, lr: float, hidden_size: int, 
                      depth: int, mode: str, align: bool, input_dim: int, batch_size: int, 
                      exp_num: int, distribution: str) -> str:
    """Generate a unique identifier for this run configuration."""
    # Just use the function name instead of detailed term representation
    # This creates much shorter filenames that won't exceed OS limits
    
    align_suffix = "_align" if align else ""
    
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
    )
    return unique_id

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

def train_model(
    gpu_id: int,
    queue: mp.Queue,
    config: Dict,
    results_dir: str,
    checkpoint_file: str
) -> None:
    """
    Train a model with the given configuration on a specific GPU.
    Enhanced version supporting complex staircase functions.
    Fixed to handle torch.compile() correctly.
    """
    # Import torch and other required modules inside the function
    # This ensures they are properly imported in the worker process
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler
    from helpers.FFNN import DeepNN
    
    try:
        # Load checkpoint to avoid reprocessing completed configurations
        completed_configs = load_checkpoint(checkpoint_file)
        
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
        
        print(f"[GPU {gpu_id}] Worker started")
        
        # Get input distribution from config
        input_distribution = config['base_config'].get('input_distribution', 'binary')
        print(f"[GPU {gpu_id}] Using {input_distribution} input distribution")
        
        # Check if gradient alignment calculation is enabled
        calculate_gradient_alignment = config['base_config'].get('calculate_gradient_alignment', True)
        print(f"[GPU {gpu_id}] Gradient alignment calculation is {'enabled' if calculate_gradient_alignment else 'disabled'}")
        
        # Process queue items until we receive None
        while True:
            item = queue.get()
            if item is None:
                break
                
            try:
                # Unpack configuration - now includes batch_size
                function, lr, batch_size, hidden_size, depth, mode, align, input_dim, exp_num = item
                
                # Generate unique ID for this run
                unique_id = generate_unique_id(
                    function, lr, hidden_size, depth, mode, align, input_dim, batch_size, 
                    exp_num, input_distribution
                )
                
                # Skip if already completed
                if unique_id in completed_configs:
                    print(f"[GPU {gpu_id}] Skipping completed configuration: {unique_id}")
                    continue
                
                print(f"[GPU {gpu_id}] Training: {unique_id}")
                
                # Get complex terms from function definition
                complex_terms = function['terms']
                
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
                    continue
                
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
                
                # Training loop
                iteration = 0
                
                # Calculate initial Lipschitz constant
                initial_lip_constant = estimate_lipschitz_constant(model, X_test, device)
                # Calculate initial normalized loss
                initial_norm_loss, initial_raw_loss = calculate_lipschitz_normalized_loss(
                    model, X_test, y_test, initial_lip_constant
                )
                
                # Store initial Lipschitz metrics
                current_lip_constants.append(initial_lip_constant)
                current_lip_norm_losses.append(initial_norm_loss)
                current_lip_raw_losses.append(initial_raw_loss)
                
                # Generate data in chunks to reduce overhead while still respecting online SGD
                while iteration < total_iterations:
                    # Calculate how many samples to generate in this chunk
                    samples_to_generate = min(super_batch_size, (total_iterations - iteration) * batch_size)
                    
                    # Generate a large chunk of training data
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
                        if iteration >= total_iterations:
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
                        
                        # Apply gradient clipping if enabled
                        if use_gradient_clipping:
                            # Unscale gradients before clipping
                            scaler.unscale_(optimizer)
                            # Clip gradients to max_norm
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        # Update parameters with scaler to maintain mixed precision
                        scaler.step(optimizer)
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
                        
                        # Increment iteration counter
                        iteration += 1
                        
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
                        
                        # Only evaluate on test set at specific intervals OR at the final iteration
                        should_evaluate = (
                            (iteration % eval_interval == 0) or 
                            (iteration == 1) or 
                            (iteration == total_iterations)
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
                            
                            # Calculate Term Gradient Alignment metrics conditionally
                            gradient_results = None
                            if calculate_gradient_alignment:
                                model.train()
                                gradient_results = calculate_term_gradient_alignment(
                                    model=model,
                                    X=X_test,
                                    y=y_test,
                                    complex_terms=complex_terms,
                                    device=device
                                )
                                
                                # Store gradient metrics with iteration number
                                term_gradient_metrics.append([float(iteration), gradient_results])
                                
                                # Switch back to eval mode for logging
                                model.eval()
                            
                            # Get the most recent Lipschitz metrics
                            if lipschitz_metrics:
                                recent_lip_constant = lipschitz_metrics[-1][1]
                                recent_lip_norm_loss = lipschitz_metrics[-1][2]
                            else:
                                recent_lip_constant = 0.0
                                recent_lip_norm_loss = 0.0
                                
                            # Get the most recent training loss
                            if train_stats:
                                recent_train_loss = train_stats[-1][1]
                            else:
                                recent_train_loss = 0.0
                            
                            # Log test results with individual term losses
                            print(f"[GPU {gpu_id}] {unique_id} - Iteration {iteration}/{total_iterations}, "
                                  f"Train Loss: {recent_train_loss:.6f}, Test Loss: {test_loss:.6f}, "
                                  f"Lipschitz: {recent_lip_constant:.4f}, Norm Loss: {recent_lip_norm_loss:.8f}")
                            
                            # Log term metrics in a more readable format
                            term_loss_strings = []
                            for i, (desc, loss) in enumerate(zip(term_descriptions, term_losses)):
                                # Include PTI correlation
                                corr = pti_results['correlation_ratios'][i]
                                # Include gradient alignment if available
                                if gradient_results:
                                    gradient_align = gradient_results['gradient_alignment'][i]
                                    term_loss_strings.append(f"Term {i+1} ({desc}): MSE={loss:.6f}, Corr={corr:.3f}, Grad={gradient_align:.3f}")
                                else:
                                    term_loss_strings.append(f"Term {i+1} ({desc}): MSE={loss:.6f}, Corr={corr:.3f}")
                            
                            print(f"[GPU {gpu_id}] Term Metrics: {' | '.join(term_loss_strings)}")
                            
                            # Save current results at checkpoints using compressed NPZ
                            if iteration % save_interval == 0 or iteration == total_iterations:
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
                                
                                # Process gradient alignment metrics if calculated
                                grad_iterations = None
                                grad_alignment = None
                                grad_magnitude_ratio = None
                                
                                if calculate_gradient_alignment and term_gradient_metrics:
                                    grad_iterations = np.array([t[0] for t in term_gradient_metrics], dtype=np.float32)
                                    grad_alignment = np.array([t[1]['gradient_alignment'] for t in term_gradient_metrics], dtype=np.float32)
                                    grad_magnitude_ratio = np.array([t[1]['gradient_magnitude_ratio'] for t in term_gradient_metrics], dtype=np.float32)
                                
                                # Store term descriptions for context
                                term_descriptions_array = np.array([desc for desc in term_descriptions])
                                
                                # Store metadata as a dictionary
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
                                    'final_train_loss': float(train_stats[-1][1]) if train_stats else float('nan'),
                                    'final_test_loss': test_loss,
                                    'final_lipschitz_constant': float(lipschitz_constants[-1]) if len(lipschitz_constants) > 0 else float('nan'),
                                    'final_normalized_loss': float(lipschitz_normalized_losses[-1]) if len(lipschitz_normalized_losses) > 0 else float('nan'),
                                    'optimizer': optimizer_type,
                                    'unique_id': unique_id,
                                    'gpu_id': gpu_id,
                                    'total_iterations': total_iterations,
                                    'current_iteration': iteration,
                                    'final_term_losses': term_losses,
                                    'term_descriptions': term_descriptions,
                                    'calculate_gradient_alignment': calculate_gradient_alignment
                                }
                                
                                # Store complex terms in a serializable format for reconstruction
                                serializable_terms = []
                                for term_dict in complex_terms:
                                    serializable_term = {
                                        'indices': term_dict['indices'],
                                        'phi': term_dict.get('phi', 'id'),
                                        'coefficient': float(term_dict.get('coefficient', 1.0))
                                    }
                                    serializable_terms.append(serializable_term)
                                
                                metadata['complex_terms'] = serializable_terms
                                
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
                                
                                # Add gradient alignment metrics if available
                                if calculate_gradient_alignment and grad_iterations is not None:
                                    save_data.update({
                                        'grad_iterations': grad_iterations,
                                        'grad_alignment': grad_alignment,
                                        'grad_magnitude_ratio': grad_magnitude_ratio
                                    })
                                
                                # Save NPZ file with all metrics
                                np.savez_compressed(result_file, **save_data)
                            
                            # Switch back to training mode
                            model.train()
                        
                        # Log training progress occasionally without writing to disk
                        elif iteration % 1000 == 0:
                            if train_stats:
                                avg_train_loss = train_stats[-1][1]
                                avg_lip_loss = lipschitz_metrics[-1][2] if lipschitz_metrics else 0.0
                                print(f"[GPU {gpu_id}] {unique_id} - Iteration {iteration}/{total_iterations}, "
                                      f"Train Loss: {avg_train_loss:.6f}, Lip Norm Loss: {avg_lip_loss:.6f}")
                
                # Ensure we don't miss the final metrics
                if iteration % stat_interval != 0 and current_train_losses:
                    # Calculate mean and variance for final training losses
                    mean_train_loss = sum(current_train_losses) / len(current_train_losses)
                    variance_train = sum((x - mean_train_loss) ** 2 for x in current_train_losses) / len(current_train_losses) if len(current_train_losses) > 1 else 0.0
                    
                    # Store final training stats - use float32 for iteration number
                    train_stats.append([float(iteration), mean_train_loss, variance_train])
                    
                    # Calculate mean for final Lipschitz metrics
                    if current_lip_constants:
                        mean_lip_constant = sum(current_lip_constants) / len(current_lip_constants)
                        mean_lip_norm_loss = sum(current_lip_norm_losses) / len(current_lip_norm_losses)
                        mean_lip_raw_loss = sum(current_lip_raw_losses) / len(current_lip_raw_losses)
                        
                        # Store final Lipschitz metrics - use float32 for iteration number
                        lipschitz_metrics.append([float(iteration), mean_lip_constant, mean_lip_norm_loss, mean_lip_raw_loss])
                    
                    print(f"[GPU {gpu_id}] Recorded final metrics at iteration {iteration}")
                
                # Ensure we have test and Lipschitz metrics for the final iteration
                if (not test_metrics or test_metrics[-1][0] != iteration):
                    model.eval()
                    with torch.no_grad():
                        # Calculate final test loss
                        test_loss = evaluate_test_loss(model, X_test, y_test)
                    
                    # Record final test metrics
                    test_metrics.append([float(iteration), test_loss])
                    print(f"[GPU {gpu_id}] Recorded final test metrics at iteration {iteration}")
                    
                    # Calculate final term-specific metrics
                    term_losses = calculate_term_test_losses(
                        model=model,
                        X=X_test,
                        complex_terms=complex_terms,
                        device=device
                    )
                    
                    # Store final term test losses
                    term_test_metrics.append([float(iteration), term_losses])
                    
                    # Calculate final PTI metrics
                    pti_results = calculate_progressive_term_isolation(
                        model=model,
                        X=X_test,
                        complex_terms=complex_terms,
                        device=device
                    )
                    
                    # Store final PTI metrics
                    term_pti_metrics.append([float(iteration), pti_results])
                    
                    # Calculate final gradient alignment metrics if enabled
                    if calculate_gradient_alignment:
                        model.train()
                        gradient_results = calculate_term_gradient_alignment(
                            model=model,
                            X=X_test,
                            y=y_test,
                            complex_terms=complex_terms,
                            device=device
                        )
                        
                        # Store final gradient metrics
                        term_gradient_metrics.append([float(iteration), gradient_results])
                        model.eval()
                
                # Final save with complete results
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
                
                # Process gradient alignment metrics if calculated
                grad_iterations = None
                grad_alignment = None
                grad_magnitude_ratio = None
                
                if calculate_gradient_alignment and term_gradient_metrics:
                    grad_iterations = np.array([t[0] for t in term_gradient_metrics], dtype=np.float32)
                    grad_alignment = np.array([t[1]['gradient_alignment'] for t in term_gradient_metrics], dtype=np.float32)
                    grad_magnitude_ratio = np.array([t[1]['gradient_magnitude_ratio'] for t in term_gradient_metrics], dtype=np.float32)
                
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
                    'calculate_gradient_alignment': calculate_gradient_alignment
                }
                
                # Store complex terms in a serializable format for reconstruction
                serializable_terms = []
                for term_dict in complex_terms:
                    serializable_term = {
                        'indices': term_dict['indices'],
                        'phi': term_dict.get('phi', 'id'),
                        'coefficient': float(term_dict.get('coefficient', 1.0))
                    }
                    serializable_terms.append(serializable_term)
                
                metadata['complex_terms'] = serializable_terms
                
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
                
                # Add gradient alignment metrics if available
                if calculate_gradient_alignment and grad_iterations is not None:
                    save_data.update({
                        'grad_iterations': grad_iterations,
                        'grad_alignment': grad_alignment,
                        'grad_magnitude_ratio': grad_magnitude_ratio
                    })
                
                # Save NPZ file with all metrics
                np.savez_compressed(result_file, **save_data)
                
                # Mark this configuration as completed
                update_checkpoint(checkpoint_file, unique_id)
                completed_configs.add(unique_id)
                
                print(f"[GPU {gpu_id}] Completed: {unique_id}")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing item {item}: {str(e)}")
                traceback.print_exc()
        
        print(f"[GPU {gpu_id}] Worker finished")
    
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {str(e)}")
        traceback.print_exc()

def load_and_analyze_results(results_dir, pattern='*.npz'):
    """
    Load and analyze results in a memory-efficient way.
    This can be used to extract key metrics without loading all data.
    """
    result_files = glob.glob(os.path.join(results_dir, pattern))
    
    summary = []
    for result_file in result_files:
        try:
            # Load only the metadata from the NPZ file to minimize memory usage
            with np.load(result_file, allow_pickle=True) as data:
                metadata_str = str(data['metadata'][0])
                # Convert string representation back to dictionary
                metadata = eval(metadata_str)
                
                # Create a summary entry with just the key metrics
                summary_entry = {
                    'unique_id': metadata['unique_id'],
                    'function_name': metadata['function_name'],
                    'input_distribution': metadata.get('input_distribution', 'binary'),
                    'batch_size': metadata.get('batch_size', 0),  # Include batch size
                    'final_test_loss': metadata['final_test_loss'],
                    'final_lipschitz_constant': metadata.get('final_lipschitz_constant', float('nan')),
                    'final_normalized_loss': metadata.get('final_normalized_loss', float('nan')),
                    'final_term_losses': metadata.get('final_term_losses', []),
                    'term_descriptions': metadata.get('term_descriptions', [])
                }
                summary.append(summary_entry)
        except Exception as e:
            print(f"Error loading {result_file}: {str(e)}")
    
    return summary

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_leap.py <config_file.yaml>")
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
    
    # Number of worker processes
    num_workers = min(config['base_config'].get('num_workers', 8), num_gpus * 10)
    
    # Generate all combinations of hyperparameters
    functions = config['functions']
    
    # Get hyperparameters for sweeping
    learning_rates = config['sweeps']['learning_rates']
    batch_sizes = config['sweeps']['batch_sizes']  # New: iteratable batch sizes
    hidden_sizes = config['sweeps']['hidden_sizes']
    depths = config['sweeps']['depths']
    modes = config['sweeps']['modes']
    alignments = config['sweeps']['alignment']
    num_experiments = config['base_config']['num_experiments']
    
    # Get input distribution type
    input_distribution = config['base_config'].get('input_distribution', 'binary')
    print(f"Using input distribution: {input_distribution}")
    
    # Generate all job configurations - now includes batch_size in combinations
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
    remaining_configs = len(all_combinations) - len(completed_configs)
    print(f"Remaining configurations to process: {remaining_configs}")
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("spawn method already set")
    
    # Create a queue and add all combinations
    queue = mp.Queue()
    for combo in all_combinations:
        queue.put(combo)
    
    # Add None markers to signal workers to exit
    for _ in range(num_workers):
        queue.put(None)
    
    # Start worker processes
    processes = []
    for i in range(num_workers):
        gpu_id = i % num_gpus
        p = mp.Process(target=train_model, args=(gpu_id, queue, config, results_dir, checkpoint_file))
        p.start()
        processes.append(p)
    
    # Wait for all workers to finish
    for p in processes:
        p.join()
    
    print(f"All training completed. Results saved to {results_dir}")

if __name__ == "__main__":
    main()