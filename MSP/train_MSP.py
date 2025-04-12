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
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import itertools
from functools import partial
from torch.cuda.amp import GradScaler

# Import the DeepNN class
from helpers.FFNN import DeepNN

# Ensure prints flush immediately
print = partial(print, flush=True)

def calculate_gal(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    function_terms: List[List[int]] = None
) -> Dict:
    """
    Calculate Gradient Alignment (GAL) for each layer and each term.
    Memory-optimized version that doesn't store raw gradients.
    """
    model.eval()
    batch_size = 100  # Use manageable batch size
    total_batches = (X.shape[0] + batch_size - 1) // batch_size
    
    # Generate term outputs for each individual MSP term
    term_outputs = []
    if function_terms:
        for term_dims in function_terms:
            term_y = torch.zeros_like(y, device=device)
            term_value = torch.ones(X.shape[0], device=device)
            for dim_idx in term_dims:
                # Convert from 1-indexed to 0-indexed
                term_value = term_value * X[:, dim_idx-1]
            term_y = term_value
            term_outputs.append(term_y)
    
    # Store gradients for full function and each term
    full_grads = []
    term_grads = [[] for _ in range(len(function_terms))] if function_terms else []
    
    # Randomly permuted versions for "noise"
    y_perm = y[torch.randperm(y.shape[0])]
    term_perm_outputs = [t[torch.randperm(t.shape[0])] for t in term_outputs]
    
    # Track gradients per layer
    layer_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_names.append(name)
    
    # Get gradients for real target
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, X.shape[0])
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        y_perm_batch = y_perm[start_idx:end_idx]
        
        # Full function gradients
        model.zero_grad()
        outputs = model(X_batch)
        loss = torch.mean((outputs - y_batch) ** 2)
        loss.backward()
        
        # Save gradients for each layer
        batch_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                batch_grads[name] = param.grad.detach().clone()
                
        full_grads.append(batch_grads)
        
        # Permuted full function gradients
        model.zero_grad()
        outputs = model(X_batch)
        loss = torch.mean((outputs - y_perm_batch) ** 2)
        loss.backward()
        
        # Save permuted gradients for each layer
        batch_perm_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                batch_perm_grads[name] = param.grad.detach().clone()
        
        # Individual term gradients
        if function_terms:
            for t_idx, term_y in enumerate(term_outputs):
                term_y_batch = term_y[start_idx:end_idx]
                term_perm_batch = term_perm_outputs[t_idx][start_idx:end_idx]
                
                # Real term gradients
                model.zero_grad()
                outputs = model(X_batch)
                loss = torch.mean((outputs - term_y_batch) ** 2)
                loss.backward()
                
                term_batch_grads = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        term_batch_grads[name] = param.grad.detach().clone()
                
                # Store only if this is the first batch
                if i == 0:
                    term_grads[t_idx].append(term_batch_grads)
                
                # Permuted term gradients
                model.zero_grad()
                outputs = model(X_batch)
                loss = torch.mean((outputs - term_perm_batch) ** 2)
                loss.backward()
    
    # Calculate GAL per layer for full function
    gal_per_layer = {}
    full_grad_avg = {name: torch.zeros_like(full_grads[0][name]) for name in layer_names}
    full_perm_avg = {name: torch.zeros_like(full_grads[0][name]) for name in layer_names}
    
    # Calculate average gradients (Γf(θ))
    for batch_grads in full_grads:
        for name in layer_names:
            full_grad_avg[name] += batch_grads[name] / len(full_grads)
    
    # Calculate GAL for full function
    for name in layer_names:
        gal_per_layer[name] = torch.norm(full_grad_avg[name] - full_perm_avg[name])**2
    
    # Calculate GAL per layer for each term
    term_gal_per_layer = []
    for t_idx in range(len(function_terms)):
        term_gal = {}
        for name in layer_names:
            if term_grads[t_idx]:
                term_gal[name] = torch.norm(term_grads[t_idx][0][name] - full_perm_avg[name])**2
            else:
                term_gal[name] = torch.tensor(0.0, device=device)
        term_gal_per_layer.append(term_gal)
    
    # Full function GAL (across all layers)
    full_gal = sum(gal_per_layer.values()).item()
    
    # Term GALs (across all layers)
    term_gals = [sum(term_gal.values()).item() for term_gal in term_gal_per_layer]
    
    # STORAGE OPTIMIZATION: Remove raw gradient estimates
    
    # Create result dictionary with only essential information
    result = {
        'full_gal': full_gal,
        'term_gals': term_gals,
        'gal_per_layer': {name: gal_per_layer[name].item() for name in layer_names},
        'term_gal_per_layer': [{name: term_gal[name].item() for name in layer_names} for term_gal in term_gal_per_layer]
    }
    
    return result

def generate_staircase_data(
    n_samples: int,
    input_dim: int,
    staircase_terms: List[List[int]],
    device: torch.device,
    test: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data for a staircase function with binary inputs.
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Total input dimension
        staircase_terms: List of terms, where each term is a list of dimensions (1-indexed)
        device: Device to place tensors on
        test: If True, generates test data with fixed seed
    
    Returns:
        Tuple of (X, y) tensors
    """
    # Find max dimension referenced in terms to ensure input_dim is sufficient
    all_dims = [dim for term in staircase_terms for dim in term]
    max_dim_needed = max(all_dims) if all_dims else 1
    
    if max_dim_needed > input_dim:
        raise ValueError(f"Function requires dimension {max_dim_needed} but input_dim is only {input_dim}")
    
    # Set seed for reproducibility of test set
    seed = 42 if test else int(time.time())
    torch.manual_seed(seed)
    
    # Generate binary input data (0s and 1s)
    X = 2 * torch.randint(0, 2, (n_samples, input_dim), device=device).float() - 1
    
    # Start with zeros for target values
    y = torch.zeros(n_samples, device=device)
    
    # Add each term (product of selected dimensions)
    for term_dims in staircase_terms:
        term_value = torch.ones(n_samples, device=device)
        for dim_idx in term_dims:
            # Convert from 1-indexed to 0-indexed
            term_value = term_value * X[:, dim_idx-1]
        y = y + term_value
    
    return X, y

def generate_unique_id(function: Dict, lr: float, hidden_size: int, 
                      depth: int, mode: str, align: bool, input_dim: int, exp_num: int,
                      lr_constant: float = None) -> str:
    """Generate a unique identifier for this run configuration."""
    # Just use the function name instead of detailed term representation
    # This creates much shorter filenames that won't exceed OS limits
    
    align_suffix = "_align" if align else ""
    lr_const_part = f"_C{lr_constant}" if lr_constant is not None else ""
    
    unique_id = (
        f"{function['name']}"  # Just use the name from YAML
        f"_d{input_dim}"
        f"_h{hidden_size}"
        f"_depth{depth}"
        f"_lr{lr}"
        f"_mode{mode}"
        f"_exp{exp_num}"
        f"{align_suffix}"
        f"{lr_const_part}"
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

def get_leap_complexity(function_name):
    """Extract leap complexity from function name"""
    if function_name.startswith('leap_'):
        # Extract the number after 'leap_'
        parts = function_name.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
    return 1  # Default to 1 if not specified

def calculate_paper_learning_rate(input_dim, leap, kappa, lr_constant):
    """Calculate learning rate according to the paper's formula"""
    D = max(leap, 2)  # From paper, we use max(Leap, 2)
    # Formula: η1 = 1/(C1·κ·d^(D/2)·log(d)^C1)
    return 1.0 / (lr_constant * kappa * (input_dim ** (D/2)) * (np.log(input_dim) ** lr_constant))

def train_model(
    gpu_id: int,
    queue: mp.Queue,
    config: Dict,
    results_dir: str,
    checkpoint_file: str
) -> None:
    """
    Train a model with the given configuration on a specific GPU.
    Storage-optimized version.
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
        
        # Process queue items until we receive None
        while True:
            item = queue.get()
            if item is None:
                break
                
            try:
                # Unpack configuration
                function, lr, hidden_size, depth, mode, align, input_dim, exp_num, lr_constant = item
                
                # Check if we should use paper's learning rate formula
                use_paper_lr = config['base_config'].get('use_paper_lr', False)
                if use_paper_lr:
                    # Get second-layer initialization scale (kappa)
                    kappa = config['base_config'].get('second_layer_init_scale', 1e-3)
                    
                    # Get leap complexity from function name
                    leap = get_leap_complexity(function['name'])
                    
                    # Calculate learning rate according to the paper's formula
                    lr = calculate_paper_learning_rate(input_dim, leap, kappa, lr_constant)
                    print(f"[GPU {gpu_id}] Using paper's LR formula with C={lr_constant}, leap={leap}: {lr:.8f}")
                
                # Generate unique ID for this run
                unique_id = generate_unique_id(
                    function, lr, hidden_size, depth, mode, align, input_dim, exp_num, 
                    lr_constant if use_paper_lr else None
                )
                
                # Skip if already completed
                if unique_id in completed_configs:
                    print(f"[GPU {gpu_id}] Skipping completed configuration: {unique_id}")
                    continue
                
                print(f"[GPU {gpu_id}] Training: {unique_id}")
                
                # Generate test data once (will be reused for all evaluations)
                staircase_terms = function['terms']
                n_test = config['base_config'].get('test_set_size', 2000)
                
                try:
                    X_test, y_test = generate_staircase_data(
                        n_samples=n_test,
                        input_dim=input_dim,
                        staircase_terms=staircase_terms,
                        device=device,
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
                
                # Try to use torch.compile() for H100 acceleration if available
                try:
                    import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                    model = torch.compile(model, mode="reduce-overhead")
                    print(f"[GPU {gpu_id}] Using torch.compile() for model acceleration")
                except Exception as e:
                    print(f"[GPU {gpu_id}] torch.compile() not available: {str(e)}")
                
                # Get initialization scale for second layer weights
                kappa = config['base_config'].get('second_layer_init_scale', 1e-3)
                
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
                batch_size = config['base_config'].get('batch_size', 32)
                
                # Super-batch size: how many data points to generate at once
                # This reduces the overhead of frequently generating new data
                super_batch_size = batch_size * 10000  # Generate 10k batches worth of data at once
                
                # STORAGE OPTIMIZATION: Only store stats at specific intervals
                stat_interval = 100  # How often to aggregate stats (every N iterations)
                eval_interval = 5000  # How often to evaluate on test set
                save_interval = eval_interval * 5  # How often to save results to disk
                
                # Storage for summarized training history (much smaller)
                train_stats = []  # Will store [iter_num, mean_loss, min_loss, max_loss] 
                test_metrics = []  # Will store [iter_num, test_loss]
                gal_metrics = []   # Will store [iter_num, gal_metrics]
                
                # Running stats tracking
                current_losses = []
                
                # Initialize mixed precision scaler
                scaler = GradScaler()
                
                # Training loop
                iteration = 0
                
                # Generate data in chunks to reduce overhead while still respecting online SGD
                while iteration < total_iterations:
                    # Calculate how many samples to generate in this chunk
                    samples_to_generate = min(super_batch_size, (total_iterations - iteration) * batch_size)
                    
                    # Generate a large chunk of training data
                    X_chunk, y_chunk = generate_staircase_data(
                        n_samples=samples_to_generate,
                        input_dim=input_dim,
                        staircase_terms=staircase_terms,
                        device=device
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
                        scaler.step(optimizer)
                        scaler.update()
                        
                        # Record batch loss
                        batch_loss = loss.item()
                        current_losses.append(batch_loss)
                        
                        # STORAGE OPTIMIZATION: Record stats only at intervals
                        if len(current_losses) >= stat_interval:
                            mean_loss = sum(current_losses) / len(current_losses)
                            min_loss = min(current_losses)
                            max_loss = max(current_losses)
                            
                            # Just store the summary stats
                            train_stats.append([iteration, mean_loss, min_loss, max_loss])
                            
                            # Reset current losses
                            current_losses = []
                        
                        # Increment iteration counter
                        iteration += 1
                        
                        # Only evaluate on test set at specific intervals
                        should_evaluate = (
                            (iteration % eval_interval == 0) or 
                            (iteration == 1) or 
                            (iteration == total_iterations)
                        )
                        
                        if should_evaluate:
                            model.eval()
                            with torch.no_grad():
                                with torch.amp.autocast('cuda'):
                                    test_outputs = model(X_test)
                                    test_loss = torch.mean((test_outputs - y_test) ** 2).item()
                            
                            # Record test metrics
                            test_metrics.append([iteration, test_loss])
                            
                            # Calculate recent average loss (over the last 100 iterations or fewer)
                            recent_losses = current_losses if current_losses else [train_stats[-1][1]]
                            avg_train_loss = sum(recent_losses) / len(recent_losses)
                            
                            # Calculate GAL metrics for full function and each term
                            gal_results = calculate_gal(
                                model=model, 
                                X=X_test, 
                                y=y_test, 
                                device=device,
                                function_terms=staircase_terms
                            )
                            
                            # Store GAL results with iteration number
                            gal_data = {
                                'iteration': iteration,
                                'full_gal': gal_results['full_gal'],
                                'term_gals': gal_results['term_gals'],
                                # Store per-layer metrics but not raw gradients
                                'gal_per_layer': gal_results['gal_per_layer'],
                                'term_gal_per_layer': gal_results['term_gal_per_layer']
                            }
                            gal_metrics.append(gal_data)
                            
                            # Log test results with GAL information
                            print(f"[GPU {gpu_id}] {unique_id} - Iteration {iteration}/{total_iterations}, "
                                  f"Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}, "
                                  f"Full GAL: {gal_results['full_gal']:.6f}")
                            
                            # Save current results at checkpoints using compressed NPZ
                            # STORAGE OPTIMIZATION: Use compressed format, half precision
                            if iteration % save_interval == 0 or iteration == total_iterations:
                                result_file = os.path.join(results_dir, f"{unique_id}.npz")
                                
                                # Convert lists to numpy arrays with float16 precision
                                train_stats_np = np.array(train_stats, dtype=np.float16)
                                test_metrics_np = np.array(test_metrics, dtype=np.float16)
                                
                                # Store metadata as a dictionary
                                metadata = {
                                    'function_name': function['name'],
                                    'input_dim': input_dim,
                                    'hidden_size': hidden_size,
                                    'depth': depth,
                                    'learning_rate': lr,
                                    'mode': mode,
                                    'alignment': align,
                                    'experiment_num': exp_num,
                                    'lr_constant': lr_constant if use_paper_lr else None,
                                    'final_train_loss': float(train_stats[-1][1]) if train_stats else float('nan'),
                                    'final_test_loss': test_loss,
                                    'optimizer': optimizer_type,
                                    'batch_size': batch_size,
                                    'unique_id': unique_id,
                                    'gpu_id': gpu_id,
                                    'total_iterations': total_iterations,
                                    'current_iteration': iteration,
                                    'final_full_gal': gal_results['full_gal'],
                                    'final_term_gals': gal_results['term_gals']
                                }
                                
                                # Pack GAL metrics into a compact format for storage
                                gal_iterations = np.array([g['iteration'] for g in gal_metrics], dtype=np.int32)
                                full_gals = np.array([g['full_gal'] for g in gal_metrics], dtype=np.float16)
                                
                                # Compact storage of term GALs
                                # Store as a 2D array [num_evals x num_terms]
                                if gal_metrics and 'term_gals' in gal_metrics[0] and gal_metrics[0]['term_gals']:
                                    num_terms = len(gal_metrics[0]['term_gals'])
                                    term_gals_array = np.zeros((len(gal_metrics), num_terms), dtype=np.float16)
                                    for i, g in enumerate(gal_metrics):
                                        term_gals_array[i] = g['term_gals']
                                else:
                                    term_gals_array = np.array([], dtype=np.float16)
                                
                                # Save NPZ file with half precision arrays and metadata
                                np.savez_compressed(
                                    result_file,
                                    train_stats=train_stats_np,  # [iter, mean, min, max]
                                    test_metrics=test_metrics_np,  # [iter, loss]
                                    gal_iterations=gal_iterations,
                                    full_gals=full_gals,
                                    term_gals=term_gals_array,
                                    metadata=np.array([str(metadata)])  # Wrap in array for storage
                                )
                            
                            # Switch back to training mode
                            model.train()
                        
                        # Log training progress occasionally without writing to disk
                        elif iteration % 1000 == 0:
                            if current_losses:
                                avg_train_loss = sum(current_losses) / len(current_losses)
                            elif train_stats:
                                avg_train_loss = train_stats[-1][1]
                            else:
                                avg_train_loss = float('nan')
                                
                            print(f"[GPU {gpu_id}] {unique_id} - Iteration {iteration}/{total_iterations}, "
                                  f"Train Loss: {avg_train_loss:.6f}")
                
                # Final save with complete results
                result_file = os.path.join(results_dir, f"{unique_id}.npz")
                
                # Convert lists to numpy arrays with float16 precision
                train_stats_np = np.array(train_stats, dtype=np.float16)
                test_metrics_np = np.array(test_metrics, dtype=np.float16)
                
                # Get final metrics
                final_train_loss = float(train_stats[-1][1]) if train_stats else float('nan')
                final_test_loss = test_metrics[-1][1] if test_metrics else float('nan')
                final_full_gal = gal_metrics[-1]['full_gal'] if gal_metrics else float('nan')
                final_term_gals = gal_metrics[-1]['term_gals'] if gal_metrics else []
                
                # Store metadata
                metadata = {
                    'function_name': function['name'],
                    'input_dim': input_dim,
                    'hidden_size': hidden_size,
                    'depth': depth,
                    'learning_rate': lr,
                    'mode': mode,
                    'alignment': align,
                    'experiment_num': exp_num,
                    'lr_constant': lr_constant if use_paper_lr else None,
                    'final_train_loss': final_train_loss,
                    'final_test_loss': final_test_loss,
                    'optimizer': optimizer_type,
                    'batch_size': batch_size,
                    'unique_id': unique_id,
                    'gpu_id': gpu_id,
                    'total_iterations': total_iterations,
                    'current_iteration': iteration,
                    'final_full_gal': final_full_gal,
                    'final_term_gals': final_term_gals
                }
                
                # Pack GAL metrics into a compact format for storage
                gal_iterations = np.array([g['iteration'] for g in gal_metrics], dtype=np.int32)
                full_gals = np.array([g['full_gal'] for g in gal_metrics], dtype=np.float16)
                
                # Compact storage of term GALs
                if gal_metrics and 'term_gals' in gal_metrics[0] and gal_metrics[0]['term_gals']:
                    num_terms = len(gal_metrics[0]['term_gals'])
                    term_gals_array = np.zeros((len(gal_metrics), num_terms), dtype=np.float16)
                    for i, g in enumerate(gal_metrics):
                        term_gals_array[i] = g['term_gals']
                else:
                    term_gals_array = np.array([], dtype=np.float16)
                
                # Save NPZ file with half precision arrays and metadata
                np.savez_compressed(
                    result_file,
                    train_stats=train_stats_np,  # [iter, mean, min, max]
                    test_metrics=test_metrics_np,  # [iter, loss]
                    gal_iterations=gal_iterations,
                    full_gals=full_gals,
                    term_gals=term_gals_array,
                    metadata=np.array([str(metadata)])  # Wrap in array for storage
                )
                
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
                    'final_test_loss': metadata['final_test_loss'],
                    'final_full_gal': metadata['final_full_gal']
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
    results_dir = os.path.join(base_results_dir, f"leap_exp_{timestamp}")
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
    num_workers = min(config['base_config'].get('num_workers', 8), num_gpus * 2)
    
    # Generate all combinations of hyperparameters
    functions = config['functions']
    
    # Check if we should use paper's learning rate formula
    use_paper_lr = config['base_config'].get('use_paper_lr', False)
    
    # Get learning rate parameters
    if use_paper_lr:
        # Get LR constants to sweep
        lr_constants = config['sweeps'].get('lr_constants', [5.0])
        learning_rates = [0.0]  # Dummy value, will be replaced by formula
    else:
        # Use explicit learning rates from config
        learning_rates = config['sweeps']['learning_rates']
        lr_constants = [None]  # No LR constant needed
    
    hidden_sizes = config['sweeps']['hidden_sizes']
    depths = config['sweeps']['depths']
    modes = config['sweeps']['modes']
    alignments = config['sweeps']['alignment']
    num_experiments = config['base_config']['num_experiments']
    
    # Generate all job configurations
    all_combinations = []
    for function in functions:
        # Get dimensions specific to this function
        dimensions = function.get('dimensions', [10])  # Default to [10] if not specified
        
        for lr, hidden_size, depth, mode, align, input_dim, exp_num, lr_constant in itertools.product(
            learning_rates, hidden_sizes, depths, modes, alignments, dimensions, 
            range(1, num_experiments + 1), lr_constants
        ):
            all_combinations.append((function, lr, hidden_size, depth, mode, align, input_dim, exp_num, lr_constant))
    
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