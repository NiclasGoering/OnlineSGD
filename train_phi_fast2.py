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
import threading
import queue
from typing import List, Dict, Tuple, Any, Callable, Union, Optional
from tqdm import tqdm
import itertools
from functools import partial
from torch.cuda.amp import GradScaler
import gzip
import io
import heapq
from collections import defaultdict

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

# ===== NEW: RAM Disk Cache Implementation =====
class RAMDiskCache:
    """In-memory cache for results with background flush to disk"""
    def __init__(self, max_items=50, flush_interval=60):
        self.cache = {}
        self.max_items = max_items
        self.flush_interval = flush_interval  # seconds
        self.lock = threading.Lock()
        self.last_flush_time = time.time()
        self.active_writes = []  # Track active write threads
        
        # Start background flush thread
        self.shutdown_flag = False
        self.flush_thread = threading.Thread(target=self._periodic_flush)
        self.flush_thread.daemon = True
        self.flush_thread.start()
    
    def _periodic_flush(self):
        """Periodically flush cache to disk"""
        while not self.shutdown_flag:
            time.sleep(5)  # Check every 5 seconds
            current_time = time.time()
            if (current_time - self.last_flush_time > self.flush_interval) or (len(self.cache) > self.max_items / 2):
                self.flush_to_disk()
            
            # Clean up completed write threads
            with self.lock:
                self.active_writes = [t for t in self.active_writes if t.is_alive()]
    
    def add(self, key, data, result_dir):
        """Add item to cache"""
        with self.lock:
            self.cache[key] = (data, result_dir)
            
            # If cache is full, flush oldest items
            if len(self.cache) >= self.max_items:
                self.flush_to_disk(force=True)
    
    def flush_to_disk(self, force=False):
        """Flush cache to disk in a background thread"""
        items_to_flush = {}
        with self.lock:
            # If forced flush or time threshold or cache size threshold
            if force or time.time() - self.last_flush_time > self.flush_interval or len(self.cache) > self.max_items / 2:
                items_to_flush = self.cache.copy()
                self.cache.clear()
                self.last_flush_time = time.time()
        
        if items_to_flush:
            # Start a background thread to write items to disk
            t = threading.Thread(target=self._write_items_to_disk, args=(items_to_flush,))
            t.daemon = True
            t.start()
            with self.lock:
                self.active_writes.append(t)
    
    def _write_items_to_disk(self, items):
        """Write items from cache to disk"""
        for key, (data, result_dir) in items.items():
            try:
                result_file = os.path.join(result_dir, f"{key}.npz")
                np.savez_compressed(result_file, **data)
                print(f"Successfully saved {result_file} from RAM cache")
            except Exception as e:
                print(f"Error saving {key} from RAM cache: {e}")
                traceback.print_exc()
    
    def shutdown(self):
        """Shutdown cache and flush remaining items"""
        self.shutdown_flag = True
        self.flush_to_disk(force=True)
        self.flush_thread.join(timeout=30)
        
        # Wait for active writes to complete
        for t in self.active_writes:
            t.join(timeout=10)
        
        print(f"RAM disk cache shutdown. All items flushed to disk.")

# Create global RAM cache instance
global_ram_cache = RAMDiskCache(max_items=100, flush_interval=120)

class AsyncMetricsCalculator:
    """Handles asynchronous calculation of expensive metrics"""
    def __init__(self, device):
        self.device = device
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.shutdown_flag = False
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def _worker_loop(self):
        while not self.shutdown_flag:
            try:
                task = self.request_queue.get(timeout=1.0)
                if task is None:
                    break
                
                metric_type, args, kwargs = task
                
                if metric_type == "lipschitz":
                    model, X = args
                    result = self._estimate_lipschitz_constant(model, X)
                    self.result_queue.put(("lipschitz", result))
                elif metric_type == "term_test":
                    model, X, complex_terms = args
                    result = self._calculate_term_test_losses(model, X, complex_terms)
                    self.result_queue.put(("term_test", result))
                elif metric_type == "pti":
                    model, X, complex_terms = args
                    result = self._calculate_progressive_term_isolation(model, X, complex_terms)
                    self.result_queue.put(("pti", result))
                elif metric_type == "gradient_align":
                    model, X, y, complex_terms = args
                    result = self._calculate_term_gradient_alignment(model, X, y, complex_terms)
                    self.result_queue.put(("gradient_align", result))
                
                self.request_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in metrics worker: {e}")
                traceback.print_exc()
    
    def request_lipschitz(self, model, X):
        """Queue a Lipschitz constant calculation"""
        self.request_queue.put(("lipschitz", (model, X), {}))
    
    def request_term_test(self, model, X, complex_terms):
        """Queue term test losses calculation"""
        self.request_queue.put(("term_test", (model, X, complex_terms), {}))
    
    def request_pti(self, model, X, complex_terms):
        """Queue PTI metrics calculation"""
        self.request_queue.put(("pti", (model, X, complex_terms), {}))
    
    def request_gradient_align(self, model, X, y, complex_terms):
        """Queue gradient alignment calculation"""
        self.request_queue.put(("gradient_align", (model, X, y, complex_terms), {}))
    
    def get_results(self, block=False, timeout=None):
        """Get all available results without blocking by default"""
        results = {}
        try:
            while True:
                metric_type, value = self.result_queue.get(block=block, timeout=timeout)
                results[metric_type] = value
                self.result_queue.task_done()
        except queue.Empty:
            pass
        return results
    
    def shutdown(self):
        """Shutdown the worker thread"""
        self.shutdown_flag = True
        self.request_queue.put(None)
        self.thread.join()
    
    
    def _create_clone_model(self, model):
            """Create a clone of the model directly using the DeepNN class, handling compiled models"""
            # Import the specific DeepNN class
            from helpers.FFNN_sgd import DeepNN
            
            try:
                # Create new instance with the correct parameters
                cloned_model = DeepNN(
                    d=model.input_dim,
                    hidden_size=model.hidden_size,
                    depth=model.depth,
                    mode=model.mode,
                    alignment=model.alignment,
                    base_width=model.base_width,
                    gamma=model.gamma
                ).to(self.device)
                
                # Get the state dict of the original model
                state_dict = model.state_dict()
                
                # Check if model has been compiled (indicated by "_orig_mod" prefix)
                is_compiled = any("_orig_mod" in key for key in state_dict.keys())
                
                if is_compiled:
                    # Create a new state dict with the correct keys
                    fixed_state_dict = {}
                    for key, value in state_dict.items():
                        # Remove the "_orig_mod." prefix from keys
                        if "_orig_mod." in key:
                            new_key = key.replace("_orig_mod.", "")
                            fixed_state_dict[new_key] = value
                    
                    # Load the fixed state dict into the model
                    cloned_model.load_state_dict(fixed_state_dict)
                else:
                    # Load the state dict directly
                    cloned_model.load_state_dict(state_dict)
                    
                return cloned_model
                
            except Exception as e:
                print(f"Error in model cloning: {e}")
                print(f"Model attributes: input_dim={model.input_dim}, hidden_size={model.hidden_size}, "
                    f"depth={model.depth}, mode={model.mode}, alignment={model.alignment}, "
                    f"base_width={model.base_width}, gamma={model.gamma}")
                
                # Debug state dict keys
                try:
                    state_dict_keys = list(model.state_dict().keys())
                    print(f"Original model state_dict keys: {state_dict_keys}")
                    
                    # Print expected keys for the clone model
                    clone = DeepNN(
                        d=model.input_dim,
                        hidden_size=model.hidden_size, 
                        depth=model.depth,
                        mode=model.mode,
                        alignment=model.alignment,
                        base_width=model.base_width,
                        gamma=model.gamma
                    )
                    clone_keys = list(clone.state_dict().keys())
                    print(f"Expected clone state_dict keys: {clone_keys}")
                except Exception as debug_err:
                    print(f"Error during debugging: {debug_err}")
                    
                traceback.print_exc()
                raise
            
    def _estimate_lipschitz_constant(self, model, X):
        """Estimate Lipschitz constant (copied from original function)"""
        with torch.no_grad():
            # Clone model for safe calculation using our helper method
            temp_model = self._create_clone_model(model)
            return estimate_lipschitz_constant(temp_model, X, self.device)
    
    def _calculate_term_test_losses(self, model, X, complex_terms):
        """Calculate term test losses (copied from original function)"""
        with torch.no_grad():
            # Clone model for safe calculation using our helper method
            temp_model = self._create_clone_model(model)
            return calculate_term_test_losses(temp_model, X, complex_terms, self.device)
    
    def _calculate_progressive_term_isolation(self, model, X, complex_terms):
        """Calculate PTI metrics (copied from original function)"""
        with torch.no_grad():
            # Clone model for safe calculation using our helper method
            temp_model = self._create_clone_model(model)
            return calculate_progressive_term_isolation(temp_model, X, complex_terms, self.device)
    
    def _calculate_term_gradient_alignment(self, model, X, y, complex_terms):
        """Calculate gradient alignment metrics (copied from original function)"""
        # Clone model for safe calculation using our helper method
        temp_model = self._create_clone_model(model)
        return calculate_term_gradient_alignment(temp_model, X, y, complex_terms, self.device)

# ===== OPTIMIZATION 2: Asynchronous File I/O =====
def async_save_npz(save_data, result_file):
    """Save NPZ file in a separate thread to avoid blocking main thread"""
    def _save_thread_func(data, filename):
        try:
            np.savez_compressed(filename, **data)
            print(f"Successfully saved {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            traceback.print_exc()
    
    # Clone data to avoid race conditions with mutation in main thread
    data_copy = save_data.copy()
    
    # Start separate thread
    save_thread = threading.Thread(target=_save_thread_func, args=(data_copy, result_file))
    save_thread.daemon = True
    save_thread.start()
    return save_thread

# ===== NEW: Shared Data Generator (Optimization 1) =====
class SharedDataGeneratorService:
    """
    Centralized data generation service that all workers can use.
    This eliminates redundant data generation across workers.
    """
    def __init__(self, input_distribution, num_generator_processes=2):
        # Queues for request and response
        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue(maxsize=50)  # Buffer up to 50 data batches
        
        # Track running processes
        self.processes = []
        self.shutdown_flag = mp.Value('i', 0)
        
        # Data caching
        self.cache_lock = threading.Lock()
        self.test_data_cache = {}  # Cache for test data
        
        # Start generator processes
        self.input_distribution = input_distribution
        for i in range(num_generator_processes):
            p = mp.Process(target=self._data_generator_process, 
                          args=(i, self.request_queue, self.response_queue, 
                                self.shutdown_flag, input_distribution))
            p.daemon = True
            p.start()
            self.processes.append(p)
        
        print(f"Started shared data generator service with {num_generator_processes} processes")
    
    def _data_generator_process(self, proc_id, request_queue, response_queue, shutdown_flag, distribution):
        """Process that generates data based on requests"""
        print(f"Data generator process {proc_id} started")
        
        while not shutdown_flag.value:
            try:
                # Get request with timeout to check shutdown flag periodically
                try:
                    request = request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if request is None:
                    break
                
                # Unpack request
                request_id, input_dim, complex_terms, n_samples, is_test = request
                
                # Generate data
                device = torch.device('cpu')  # Always generate on CPU
                try:
                    X, y = generate_staircase_data(
                        n_samples=n_samples,
                        input_dim=input_dim,
                        complex_terms=complex_terms,
                        device=device,
                        distribution=distribution,
                        test=is_test
                    )
                    
                    # Pin memory for faster transfer to GPU
                    X = X.pin_memory()
                    y = y.pin_memory()
                    
                    # Put result in response queue
                    response_queue.put((request_id, X, y))
                    
                except Exception as e:
                    print(f"Error in data generator process {proc_id}: {e}")
                    traceback.print_exc()
                    response_queue.put((request_id, None, None))
                    
            except Exception as e:
                print(f"Error in data generator process {proc_id}: {e}")
                traceback.print_exc()
        
        print(f"Data generator process {proc_id} shutting down")
    
    def request_data(self, input_dim, complex_terms, n_samples, is_test=False):
        """Request data to be generated"""
        # For test data, check cache first
        if is_test:
            cache_key = (input_dim, str(complex_terms), self.input_distribution)
            with self.cache_lock:
                if cache_key in self.test_data_cache:
                    print(f"Test data cache hit for dim={input_dim}")
                    return self.test_data_cache[cache_key]
        
        # Generate unique request ID
        request_id = str(time.time()) + str(hash(str(complex_terms)) % 1000)
        
        # Send request
        self.request_queue.put((request_id, input_dim, complex_terms, n_samples, is_test))
        
        # Return the request ID so caller can match the response
        return request_id
    
    def get_data(self, request_id, timeout=10):
        """Get generated data for a specific request ID"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to find our response in the queue
                resp_id, X, y = self.response_queue.get(timeout=0.5)
                
                if resp_id == request_id:
                    # Found our response
                    
                    # If this is test data, cache it
                    if X is not None and y is not None and len(X) > 0 and "_test_" in request_id:
                        input_dim = X.shape[1]
                        with self.cache_lock:
                            key = (input_dim, request_id.split("_terms")[1], self.input_distribution)
                            self.test_data_cache[key] = (X, y)
                            print(f"Cached test data for dim={input_dim}")
                    
                    return X, y
                else:
                    # Not our response, put it back in the queue
                    self.response_queue.put((resp_id, X, y))
                    time.sleep(0.1)  # Short wait to avoid hammering the queue
                    
            except queue.Empty:
                # Queue is empty, wait a bit
                time.sleep(0.1)
        
        print(f"Timeout waiting for data with request_id {request_id}")
        return None, None
    
    def shutdown(self):
        """Shutdown all generator processes"""
        self.shutdown_flag.value = 1
        
        # Send termination signals
        for _ in self.processes:
            self.request_queue.put(None)
        
        # Wait for processes to terminate
        for p in self.processes:
            p.join(timeout=5)
        
        print("Shared data generator service shutdown complete")

# ===== Unmodified Data Generator Class (for compatibility) =====
class DataGenerator:
    """Generates data on CPU in background threads"""
    def __init__(self, input_dim, complex_terms, distribution="binary", num_workers=2, queue_size=10):
        self.input_dim = input_dim
        self.complex_terms = complex_terms
        self.distribution = distribution
        self.queue = queue.Queue(maxsize=queue_size)
        self.shutdown_flag = False
        self.workers = []
        
        # Start worker threads
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        while not self.shutdown_flag:
            try:
                task = None
                # Try to get a generation task with timeout to check shutdown flag periodically
                try:
                    task = self.task_queue.get(timeout=0.5)
                except (queue.Empty, AttributeError):
                    # Task queue not initialized yet or empty
                    continue
                
                if task is None:
                    break
                
                n_samples, is_test = task
                
                # Generate on CPU
                device = torch.device('cpu')
                X, y = generate_staircase_data(
                    n_samples=n_samples,
                    input_dim=self.input_dim,
                    complex_terms=self.complex_terms,
                    device=device,
                    distribution=self.distribution,
                    test=is_test
                )
                
                # Pin memory for faster GPU transfer later
                X = X.pin_memory()
                y = y.pin_memory()
                
                # Put result in queue
                self.queue.put((X, y))
                self.task_queue.task_done()
            except Exception as e:
                print(f"Error in data generation worker: {e}")
                traceback.print_exc()
    
    def request_data(self, n_samples, is_test=False):
        """Request a batch of data to be generated"""
        if not hasattr(self, 'task_queue'):
            self.task_queue = queue.Queue()
        self.task_queue.put((n_samples, is_test))
    
    def get_data(self, timeout=None):
        """Get generated data from the queue"""
        try:
            X, y = self.queue.get(timeout=timeout)
            self.queue.task_done()
            return X, y
        except queue.Empty:
            return None, None
    
    def shutdown(self):
        """Shutdown all worker threads"""
        self.shutdown_flag = True
        if hasattr(self, 'task_queue'):
            for _ in self.workers:
                self.task_queue.put(None)
        for worker in self.workers:
            worker.join()

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
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # Use bfloat16 for H100
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
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # Use bfloat16 for H100
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
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # Use bfloat16 for H100
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
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # Use bfloat16 for H100
                    isolated_output = model(isolated_X).detach()
                
                # Calculate error for this term
                batch_error = torch.mean((isolated_output - batch_true) ** 2).item()
                
                # Update running average
                total_error = (total_error * total_samples + batch_error * batch_size_actual) / (total_samples + batch_size_actual)
                total_samples += batch_size_actual
            
            term_errors.append(total_error)
    
    return term_errors

# ===== OPTIMIZATION 4: Improved Data Generation =====
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
    Optimized version to reduce CPU-GPU synchronization.
    
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
    is_cuda = device.type == 'cuda'
    # Use CPU for data generation, then transfer to target device
    gen_device = torch.device('cpu')
    
    if distribution.lower() == "binary":
        # Binary inputs (-1 or 1)
        X = 2 * torch.randint(0, 2, (n_samples, input_dim), device=gen_device).float() - 1
    elif distribution.lower() == "normal":
        # Standard normal distribution with scaling factor for better training stability
        X = torch.randn((n_samples, input_dim), device=gen_device)
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Use 'binary' or 'normal'.")
    
    # Start with zeros for target values
    y = torch.zeros(n_samples, device=gen_device)
    
    # Add each term (product of selected dimensions with phi and coefficient)
    for term_dict in complex_terms:
        term_indices = term_dict["indices"]
        phi_name = term_dict.get("phi", "id")
        coefficient = term_dict.get("coefficient", 1.0)
        
        # Get the phi function - use custom CPU implementation for efficiency
        if phi_name == "id":
            # Identity function
            phi_func = lambda x: x
        elif phi_name == "tanh":
            # Hyperbolic tangent
            phi_func = torch.tanh
        elif phi_name == "square":
            # Square
            phi_func = lambda x: x**2
        elif phi_name == "cube":
            # Cube
            phi_func = lambda x: x**3
        elif phi_name == "relu":
            # ReLU
            phi_func = lambda x: torch.relu(x)
        elif phi_name == "sigmoid":
            # Sigmoid
            phi_func = torch.sigmoid
        elif phi_name == "sin":
            # Sine
            phi_func = torch.sin
        elif phi_name == "cos":
            # Cosine
            phi_func = torch.cos
        elif phi_name == "exp":
            # Exponential
            phi_func = torch.exp
        elif phi_name == "log":
            # Log with safety
            phi_func = lambda x: torch.log(torch.abs(x) + 1e-10)
        elif phi_name == "sqrt":
            # Square root with safety
            phi_func = lambda x: torch.sqrt(torch.abs(x) + 1e-10)
        else:
            phi_func = PHI_FUNCTIONS[phi_name]
        
        # Calculate raw term value (product of selected dimensions)
        term_value = torch.ones(n_samples, device=gen_device)
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
    
    # Transfer to target device if necessary and make contiguous for efficient access
    if is_cuda:
        # Pin memory for faster transfer
        X = X.pin_memory()
        y = y.pin_memory()
        X = X.to(device, non_blocking=True).contiguous()
        y = y.to(device, non_blocking=True).contiguous()
    
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

# ===== OPTIMIZATION 5: Improved Checkpoint Loading =====
def load_checkpoint(checkpoint_file):
    """Load checkpoint file with completed configuration IDs - optimized with buffered I/O"""
    completed_configs = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', buffering=1024*1024) as f:  # Use 1MB buffer
                for line in f:
                    completed_configs.add(line.strip())
            print(f"Loaded {len(completed_configs)} completed configurations from checkpoint: {checkpoint_file}")
            if completed_configs:
                # Print a few examples for verification
                examples = list(completed_configs)[:5]
                print(f"First few completed configs: {examples}")
        except Exception as e:
            print(f"Error loading checkpoint file {checkpoint_file}: {e}")
            print("Starting with empty completed_configs set")
    else:
        print(f"Checkpoint file {checkpoint_file} does not exist. Starting with empty completed_configs set")
    return completed_configs

def update_checkpoint(checkpoint_file, unique_id):
    """Add a completed configuration ID to the checkpoint file"""
    with open(checkpoint_file, 'a') as f:
        f.write(f"{unique_id}\n")
        f.flush()  # Ensure write is flushed to disk

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

# ===== OPTIMIZATION 6: Improved Model Compilation =====
def try_compile_model(model, gpu_id):
    """
    Attempt to use torch.compile() with proper error handling.
    Optimized for H100 GPUs.
    """
    use_compile = True  # Set to True to enable torch.compile()
    
    if not use_compile:
        print(f"[GPU {gpu_id}] torch.compile() is disabled")
        return model
    
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        
        # H100-optimized settings
        compiled_model = torch.compile(
            model, 
            mode="max-autotune",  # More aggressive optimization for H100
            fullgraph=True,       # Try to compile the entire model as one graph for H100s
            dynamic=True          # Allow for dynamic shapes
        )
        print(f"[GPU {gpu_id}] Using torch.compile() with max-autotune for H100")
        return compiled_model
    except Exception as e:
        print(f"[GPU {gpu_id}] torch.compile() not available: {str(e)}")
        return model

# ===== OPTIMIZATION 7: Batched Evaluation =====
def evaluate_test_loss(model, X_test, y_test, batch_size=32768):
    """
    Safely evaluate test loss using batched processing to handle compiled models.
    Returns the mean loss across all batches.
    Optimized batch size for H100 GPUs.
    """
    total_batches = (X_test.shape[0] + batch_size - 1) // batch_size
    
    # Faster implementation with reduced synchronization
    total_squared_error = 0.0
    total_samples = 0
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, X_test.shape[0])
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # Use bfloat16 for H100
            # Clone outputs to prevent CUDA graph overwrite issues
            outputs = model(X_batch).detach()
            
            # Don't call .item() here to avoid synchronization
            squared_error = ((outputs - y_batch) ** 2).sum()
        
        # Convert to CPU only once at the end of the batch
        total_squared_error += squared_error.item()
        total_samples += batch_size_actual
    
    # Calculate final mean loss
    mean_loss = total_squared_error / total_samples
    return mean_loss

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

# ===== NEW: Priority-based Configuration Queuing (Optimization 5) =====
def calculate_config_priority(combo):
    """
    Calculate priority score for configuration - lower scores have higher priority
    for faster throughput of simpler jobs.
    """
    function, lr, batch_size, hidden_size, depth, mode, align, input_dim, exp_num = combo
    
    # Calculate priority based on model complexity
    model_size = depth * hidden_size
    term_complexity = len(function['terms'])
    dimension_factor = input_dim
    
    # Terms with more indices are more complex
    term_index_complexity = 0
    for term in function['terms']:
        term_index_complexity += len(term['indices'])
    
    # Calculate priority - lower is higher priority
    # We prioritize smaller models and simpler functions first
    priority = model_size * dimension_factor * (term_complexity + term_index_complexity)
    
    return priority

# ===== NEW: Dynamic Load Balancing (Optimization 4) =====
def worker_manager_process(shared_task_queue, worker_status, task_queue_size, num_workers):
    """
    Process that monitors worker status and redistributes work dynamically.
    Implements work stealing for dynamic load balancing.
    """
    print(f"Worker manager started - monitoring {num_workers} workers")
    
    while True:
        try:
            # Check if all workers are done
            all_done = True
            for i in range(num_workers):
                if worker_status[i] == 1:  # Worker is busy
                    all_done = False
                    break
            
            # If all workers are done and task queue is empty, we're finished
            if all_done and shared_task_queue.qsize() == 0:
                break
                
            # Log status occasionally
            if time.time() % 60 < 1:  # Log roughly once per minute
                active_workers = sum(1 for status in worker_status if status == 1)
                try:
                    queue_size = shared_task_queue.qsize()
                except:
                    queue_size = "unknown"
                print(f"Worker status: {active_workers}/{num_workers} active, queue size: {queue_size}")
            
            time.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            print(f"Error in worker manager: {e}")
            traceback.print_exc()
            time.sleep(10)  # Longer delay if error
    
    print("Worker manager shutting down - all tasks completed")

# ===== OPTIMIZATION 8: Multi-threaded Training =====


def train_model(
    worker_id: int,
    shared_task_queue: mp.Queue,
    worker_status: mp.Array,
    shared_data_service: SharedDataGeneratorService,
    config: Dict,
    results_dir: str,
    checkpoint_file: str,
    gpu_id: int
) -> None:
    """
    Train a model with the given configuration on a specific GPU.
    Enhanced version with dynamic load balancing and shared data generation.
    
    Args:
        worker_id: ID of this worker process
        shared_task_queue: Shared queue with tasks to process
        worker_status: Shared array to track worker status
        shared_data_service: Shared data generation service
        config: Configuration dictionary
        results_dir: Directory to save results
        checkpoint_file: File to track completed configurations
        gpu_id: ID of GPU to use
    """
    # Import torch and other required modules inside the function
    # This ensures they are properly imported in the worker process
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler
    from helpers.FFNN_sgd import DeepNN  # Use the correct DeepNN class
    
    # Set worker status to idle initially
    worker_status[worker_id] = 0
    
    try:
        # ====== CRITICAL OPTIMIZATION: Restrict to a single visible GPU ======
        # This avoids CUDA context contention between workers
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device('cuda:0')  # Now this is the only device we can see
        
        # Load checkpoint to avoid reprocessing completed configurations
        completed_configs = load_checkpoint(checkpoint_file)
        
        # Enable optimizations for H100 GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
        # Enable Flash Attention and other H100 optimizations
        try:
            torch.backends.cuda.enable_flash_sdp = True
            torch.backends.cuda.enable_mem_efficient_sdp = True
            torch.backends.cuda.enable_math_sdp = True
            print(f"[Worker {worker_id} GPU {gpu_id}] Enabled H100-specific optimizations")
        except AttributeError:
            pass
        
        print(f"[Worker {worker_id} GPU {gpu_id}] Worker started")
        
        # Get input distribution from config
        input_distribution = config['base_config'].get('input_distribution', 'binary')
        print(f"[Worker {worker_id} GPU {gpu_id}] Using {input_distribution} input distribution")
        
        # Check if gradient alignment calculation is enabled
        calculate_gradient_alignment = config['base_config'].get('calculate_gradient_alignment', True)
        print(f"[Worker {worker_id} GPU {gpu_id}] Gradient alignment calculation is {'enabled' if calculate_gradient_alignment else 'disabled'}")
        
        # Check if Lipschitz calculation is enabled (new flag)
        calculate_lipschitz = config['base_config'].get('calculate_lipschitz', True)
        print(f"[Worker {worker_id} GPU {gpu_id}] Lipschitz constant calculation is {'enabled' if calculate_lipschitz else 'disabled'}")
        
        # Get early stopping configuration
        early_stopping_enabled = config['base_config'].get('early_stopping', {}).get('enabled', False)
        early_stopping_threshold = config['base_config'].get('early_stopping', {}).get('threshold', 1e-4)
        early_stopping_patience = config['base_config'].get('early_stopping', {}).get('patience', 1000)
        
        if early_stopping_enabled:
            print(f"[Worker {worker_id} GPU {gpu_id}] Early stopping enabled with threshold {early_stopping_threshold} for {early_stopping_patience} iterations")
        
        # Get model saving configuration
        save_models = config['base_config'].get('save_models', [])
        if save_models:
            print(f"[Worker {worker_id} GPU {gpu_id}] Will save models at iterations: {save_models}")
            # Create models directory
            models_dir = os.path.join(results_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
        
        # Get train error saving frequency (new parameter)
        train_error_save_freq = config['base_config'].get('train_error_save_freq', 1000)
        print(f"[Worker {worker_id} GPU {gpu_id}] Will save training error every {train_error_save_freq} iterations")
        
        # Process queue items until we receive None or queue is empty
        while True:
            try:
                # Mark worker as idle while waiting for task
                worker_status[worker_id] = 0
                
                # Try to get a task
                try:
                    item = shared_task_queue.get(timeout=5)
                except queue.Empty:
                    continue
                
                if item is None:
                    print(f"[Worker {worker_id} GPU {gpu_id}] Received termination signal")
                    break
                
                # Mark worker as busy
                worker_status[worker_id] = 1
                
                # Unpack configuration - now includes batch_size
                function, lr, batch_size, hidden_size, depth, mode, align, input_dim, exp_num = item
                
                # NEW: Apply LR scaling by input dimension if enabled
                lr_dim_scaling = config['base_config'].get('lr_dim_scaling', False)
                if lr_dim_scaling:
                    original_lr = lr
                    lr = lr / input_dim  # Scale LR by 1/d
                    print(f"[Worker {worker_id} GPU {gpu_id}] LR scaling enabled: {original_lr} -> {lr:.6f} (scaled by 1/{input_dim})")
                
                # Generate unique ID for this run
                unique_id = generate_unique_id(
                    function, lr, hidden_size, depth, mode, align, input_dim, batch_size, 
                    exp_num, input_distribution
                )
                
                # Skip if already completed
                if unique_id in completed_configs:
                    print(f"[Worker {worker_id} GPU {gpu_id}] Skipping completed configuration: {unique_id}")
                    continue
                
                print(f"[Worker {worker_id} GPU {gpu_id}] Training: {unique_id}")
                
                # Get complex terms from function definition
                complex_terms = function['terms']
                
                # Print the term descriptions for clarity
                term_descriptions = get_term_descriptions(complex_terms)
                print(f"[Worker {worker_id} GPU {gpu_id}] Complex terms: {term_descriptions}")
                
                # Setup asynchronous metrics calculation - always needed for term correlation
                metrics_calculator = AsyncMetricsCalculator(device)
                
                # Generate test data using shared data service
                n_test = config['base_config'].get('test_set_size', 10000)  # Larger test set for H100
                
                try:
                    # Create a unique request ID for this test data
                    test_data_request_id = f"test_data_{unique_id}_terms{hash(str(complex_terms))}"
                    
                    # Request test data from shared service
                    request_id = shared_data_service.request_data(
                        input_dim=input_dim,
                        complex_terms=complex_terms,
                        n_samples=n_test,
                        is_test=True
                    )
                    
                    # Get the test data
                    X_test, y_test = shared_data_service.get_data(request_id, timeout=20)
                    
                    if X_test is None or y_test is None:
                        raise ValueError("Failed to get test data from shared service")
                    
                    # Move to the appropriate device
                    X_test = X_test.to(device, non_blocking=True)
                    y_test = y_test.to(device, non_blocking=True)
                    
                except ValueError as e:
                    print(f"[Worker {worker_id} GPU {gpu_id}] Error generating test data: {str(e)}")
                    continue
                
                # Create model - FIXED to use 'd' as parameter name
                model = DeepNN(
                    d=input_dim,  # Use 'd' parameter name
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
                    print(f"[Worker {worker_id} GPU {gpu_id}] Using gradient clipping with max norm: {max_grad_norm}")
                
                # Use exactly the batch size specified in the YAML configuration
                # No automatic scaling of batch sizes
                
                # Super-batch size: how many data points to generate at once
                # This reduces the overhead of frequently generating new data
                super_batch_size = batch_size * 200  # Generate 200 batches worth of data at once
                
                # STORAGE OPTIMIZATION: Only store stats at specific intervals
                # NOTE: User requires train_error_save_freq to be customizable
                stat_interval = train_error_save_freq  # Record training stats using user-specified frequency
                eval_interval = 5000  # How often to evaluate on test set (can be made less frequent)
                save_interval = 10000  # How often to save results to disk
                
                # Storage for summarized training history
                train_stats = []  # Will store [iter_num, mean_loss, variance_loss]
                lipschitz_metrics = []  # Will store [iter_num, lipschitz_constant, norm_loss, raw_loss]
                test_metrics = []  # Will store [iter_num, test_loss]
                term_test_metrics = []  # Will store [iter_num, [term1_loss, term2_loss, ...]]
                
                # Storage for our advanced metrics
                term_pti_metrics = []  # Will store [iter_num, {pti_metrics}]
                term_gradient_metrics = []  # Will store [iter_num, {gradient_metrics}]
                
                # Containers for collecting batch metrics with reduced synchronization
                current_train_losses = []
                
                # Initialize mixed precision scaler for bfloat16
                # Use bfloat16 for H100s as it's more numerically stable than fp16
                scaler = GradScaler(enabled=True)
                
                # Initialize early stopping variables
                below_threshold_count = 0
                early_stopping_triggered = False
                
                # Save initial model if requested
                if "initial" in save_models:
                    initial_model_path = os.path.join(models_dir, f"{unique_id}_initial.pth.gz")
                    save_compressed_model(model, initial_model_path)
                
                # Training loop
                iteration = 0
                active_save_threads = []  # Track active save threads
                initial_metrics_requested = False
                
                # Generate data in chunks to reduce overhead while still respecting online SGD
                while iteration < total_iterations and not early_stopping_triggered:
                    # Calculate how many samples to generate in this chunk
                    samples_to_generate = min(super_batch_size, (total_iterations - iteration) * batch_size)
                    
                    # Request data from shared data service
                    request_id = shared_data_service.request_data(
                        input_dim=input_dim,
                        complex_terms=complex_terms,
                        n_samples=samples_to_generate,
                        is_test=False
                    )
                    
                    # Get the generated data
                    X_chunk, y_chunk = shared_data_service.get_data(request_id, timeout=20)
                    if X_chunk is None or y_chunk is None:
                        print(f"[Worker {worker_id} GPU {gpu_id}] Error: Data generation timeout, retrying...")
                        continue
                    
                    # Transfer data to GPU in one operation
                    X_chunk = X_chunk.to(device, non_blocking=True)
                    y_chunk = y_chunk.to(device, non_blocking=True)
                    
                    # Request metrics calculation for the first iteration
                    if not initial_metrics_requested:
                        # Always request term test and PTI
                        metrics_calculator.request_term_test(model, X_test, complex_terms)
                        metrics_calculator.request_pti(model, X_test, complex_terms)
                        
                        # Conditionally request Lipschitz and gradient alignment
                        if calculate_lipschitz:
                            metrics_calculator.request_lipschitz(model, X_test)
                        if calculate_gradient_alignment:
                            model.train()  # Need train mode for gradient calculation
                            metrics_calculator.request_gradient_align(model, X_test, y_test, complex_terms)
                            model.eval()  # Switch back to eval mode
                            
                        initial_metrics_requested = True
                    
                    # Process this chunk in small batches (respecting online SGD)
                    num_super_batches = (len(X_chunk) + batch_size - 1) // batch_size
                    
                    for i in range(num_super_batches):
                        if iteration >= total_iterations or early_stopping_triggered:
                            break
                            
                        # Clean up completed save threads
                        active_save_threads = [t for t in active_save_threads if t.is_alive()]
                        
                        # Get a small batch from the super-batch
                        start_idx = i * batch_size
                        end_idx = min(start_idx + batch_size, len(X_chunk))
                        X_batch = X_chunk[start_idx:end_idx]
                        y_batch = y_chunk[start_idx:end_idx]
                        
                        # Zero gradients
                        optimizer.zero_grad()
                        
                        # Forward pass with autocast (mixed precision)
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # Use bfloat16 for H100
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
                        
                        # Record loss for this batch - use detach to avoid synchronization
                        current_train_losses.append(loss.detach())
                        
                        # Check for async metrics results
                        metrics_results = metrics_calculator.get_results()
                        
                        # Process Lipschitz results if available
                        if 'lipschitz' in metrics_results and calculate_lipschitz:
                            lip_constant = metrics_results['lipschitz']
                            
                            # We can now calculate normalized loss
                            with torch.no_grad():
                                norm_loss, raw_loss = calculate_lipschitz_normalized_loss(
                                    model, X_test, y_test, lip_constant
                                )
                            
                            # Record Lipschitz metrics - use float32 for iteration number
                            lipschitz_metrics.append([float(iteration), lip_constant, norm_loss, raw_loss])
                        
                        # Process term test results if available
                        if 'term_test' in metrics_results:
                            term_losses = metrics_results['term_test']
                            term_test_metrics.append([float(iteration), term_losses])
                        
                        # Process PTI results if available
                        if 'pti' in metrics_results:
                            pti_results = metrics_results['pti']
                            term_pti_metrics.append([float(iteration), pti_results])
                        
                        # Process gradient alignment results if available
                        if 'gradient_align' in metrics_results and calculate_gradient_alignment:
                            gradient_results = metrics_results['gradient_align']
                            term_gradient_metrics.append([float(iteration), gradient_results])
                        
                        # Early stopping check
                        if early_stopping_enabled:
                            # Get loss value without synchronization
                            if len(current_train_losses) > 0:
                                # Only check once we have enough data
                                current_loss = current_train_losses[-1].item()
                                
                                if current_loss < early_stopping_threshold:
                                    below_threshold_count += 1
                                    if below_threshold_count >= early_stopping_patience:
                                        print(f"[Worker {worker_id} GPU {gpu_id}] {unique_id} - Early stopping triggered at iteration {iteration}. "
                                              f"Loss below {early_stopping_threshold} for {early_stopping_patience} consecutive iterations.")
                                        early_stopping_triggered = True
                                        break
                                else:
                                    below_threshold_count = 0  # Reset the counter if loss is above threshold
                        
                        # Increment iteration counter
                        iteration += 1
                        
                        # Save model at specific iterations if requested
                        if save_models and str(iteration) in [str(m) for m in save_models if isinstance(m, (int, float))]:
                            model_path = os.path.join(models_dir, f"{unique_id}_iter{iteration}.pth.gz")
                            save_compressed_model(model, model_path)
                        
                        # Record metrics at user-specified train_error_save_freq (e.g., 1000 iterations)
                        if iteration % stat_interval == 0:
                            # Calculate mean and variance for training losses
                            if current_train_losses:
                                # Compute mean loss efficiently on GPU
                                stacked_losses = torch.stack(current_train_losses)
                                mean_train_loss = torch.mean(stacked_losses).item()
                                
                                # Compute variance efficiently on GPU
                                variance_train = torch.var(stacked_losses).item() if len(stacked_losses) > 1 else 0.0
                                
                                # Store training stats - use float32 for iteration number
                                train_stats.append([float(iteration), mean_train_loss, variance_train])
                                
                                # Reset training loss buffer
                                current_train_losses = []
                            
                            # Log progress
                            lc_str = ""
                            if lipschitz_metrics and calculate_lipschitz:
                                lc_str = f", Lipschitz: {lipschitz_metrics[-1][1]:.4f}"
                                
                            print(f"[Worker {worker_id} GPU {gpu_id}] {unique_id} - Iteration {iteration}/{total_iterations}, "
                                  f"Train Loss: {mean_train_loss:.6f}{lc_str}")
                        
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
                                # Calculate full test loss
                                test_loss = evaluate_test_loss(model, X_test, y_test, batch_size=32768)
                            
                            # Record test metrics - use float32 for iteration number
                            test_metrics.append([float(iteration), test_loss])
                            
                            # Queue metrics calculations - always request term metrics
                            metrics_calculator.request_term_test(model, X_test, complex_terms)
                            metrics_calculator.request_pti(model, X_test, complex_terms)
                            
                            # Conditionally request other metrics
                            if calculate_lipschitz:
                                metrics_calculator.request_lipschitz(model, X_test)
                            if calculate_gradient_alignment:
                                model.train()  # Need train mode for gradient calculation
                                metrics_calculator.request_gradient_align(model, X_test, y_test, complex_terms)
                                model.eval()  # Switch back to eval mode
                            
                            # Log test results
                            print(f"[Worker {worker_id} GPU {gpu_id}] {unique_id} - Iteration {iteration}/{total_iterations}, "
                                  f"Test Loss: {test_loss:.6f}")
                            
                            # Save current results at checkpoints using RAM cache
                            if iteration % save_interval == 0 or iteration == total_iterations or early_stopping_triggered:
                                # Convert lists to numpy arrays with float32 precision for numerical stability
                                train_stats_np = np.array(train_stats, dtype=np.float32)
                                test_metrics_np = np.array(test_metrics, dtype=np.float32)
                                
                                # Create a dictionary of data to save
                                save_data = {
                                    # Original metrics
                                    'train_stats': train_stats_np,
                                    'test_metrics': test_metrics_np,
                                }
                                
                                # Add term-specific metrics if available
                                if term_test_metrics:
                                    # Extract iterations and term losses for efficient storage
                                    term_iterations = np.array([t[0] for t in term_test_metrics], dtype=np.float32)
                                    term_losses_array = np.array([t[1] for t in term_test_metrics], dtype=np.float32)
                                    save_data.update({
                                        'term_iterations': term_iterations,
                                        'term_losses': term_losses_array,
                                    })
                                
                                # Add PTI metrics if available
                                if term_pti_metrics:
                                    # Process PTI metrics efficiently
                                    pti_iterations = np.array([t[0] for t in term_pti_metrics], dtype=np.float32)
                                    pti_correlation_ratios = np.array([t[1]['correlation_ratios'] for t in term_pti_metrics], dtype=np.float32)
                                    pti_residual_mse = np.array([t[1]['residual_mse'] for t in term_pti_metrics], dtype=np.float32)
                                    save_data.update({
                                        'pti_iterations': pti_iterations,
                                        'pti_correlation_ratios': pti_correlation_ratios,
                                        'pti_residual_mse': pti_residual_mse,
                                    })
                                
                                # Add Lipschitz metrics if available and enabled
                                if lipschitz_metrics and calculate_lipschitz:
                                    # Process Lipschitz metrics
                                    lipschitz_iterations = np.array([t[0] for t in lipschitz_metrics], dtype=np.float32)
                                    lipschitz_constants = np.array([t[1] for t in lipschitz_metrics], dtype=np.float32)
                                    lipschitz_normalized_losses = np.array([t[2] for t in lipschitz_metrics], dtype=np.float32)
                                    lipschitz_raw_losses = np.array([t[3] for t in lipschitz_metrics], dtype=np.float32)
                                    save_data.update({
                                        'lipschitz_iterations': lipschitz_iterations,
                                        'lipschitz_constants': lipschitz_constants,
                                        'lipschitz_normalized_losses': lipschitz_normalized_losses,
                                        'lipschitz_raw_losses': lipschitz_raw_losses,
                                    })
                                
                                # Add gradient alignment metrics if available and enabled
                                if calculate_gradient_alignment and term_gradient_metrics:
                                    # Process gradient alignment metrics
                                    grad_iterations = np.array([t[0] for t in term_gradient_metrics], dtype=np.float32)
                                    grad_alignment = np.array([t[1]['gradient_alignment'] for t in term_gradient_metrics], dtype=np.float32)
                                    grad_magnitude_ratio = np.array([t[1]['gradient_magnitude_ratio'] for t in term_gradient_metrics], dtype=np.float32)
                                    save_data.update({
                                        'grad_iterations': grad_iterations,
                                        'grad_alignment': grad_alignment,
                                        'grad_magnitude_ratio': grad_magnitude_ratio,
                                    })
                                
                                # Store term descriptions for context
                                term_descriptions_array = np.array([desc for desc in term_descriptions])
                                save_data['term_descriptions'] = term_descriptions_array
                                
                                # Get final metrics for metadata
                                final_train_loss = float(train_stats[-1][1]) if train_stats else float('nan')
                                final_test_loss = test_loss
                                final_term_losses = term_test_metrics[-1][1] if term_test_metrics and len(term_test_metrics) > 0 else []
                                final_lipschitz_constant = float(lipschitz_constants[-1]) if calculate_lipschitz and 'lipschitz_constants' in locals() and len(lipschitz_constants) > 0 else float('nan')
                                final_normalized_loss = float(lipschitz_normalized_losses[-1]) if calculate_lipschitz and 'lipschitz_normalized_losses' in locals() and len(lipschitz_normalized_losses) > 0 else float('nan')
                                
                                # Store metadata
                                metadata = {
                                    'function_name': function['name'],
                                    'input_dim': input_dim,
                                    'input_distribution': input_distribution,
                                    'hidden_size': hidden_size,
                                    'depth': depth,
                                    'learning_rate': lr,
                                    'original_learning_rate': lr * input_dim if lr_dim_scaling else lr,  # Store original LR if scaled
                                    'lr_dim_scaling': lr_dim_scaling,  # Record if LR scaling was used
                                    'batch_size': batch_size,  # Include actual batch size
                                    'mode': mode,
                                    'alignment': align,
                                    'experiment_num': exp_num,
                                    'final_train_loss': final_train_loss,
                                    'final_test_loss': final_test_loss,
                                    'final_lipschitz_constant': final_lipschitz_constant if calculate_lipschitz else float('nan'),
                                    'final_normalized_loss': final_normalized_loss if calculate_lipschitz else float('nan'),
                                    'optimizer': optimizer_type,
                                    'unique_id': unique_id,
                                    'gpu_id': gpu_id,
                                    'total_iterations': total_iterations,
                                    'current_iteration': iteration,
                                    'final_term_losses': final_term_losses,
                                    'term_descriptions': term_descriptions,
                                    'calculate_gradient_alignment': calculate_gradient_alignment,
                                    'calculate_lipschitz': calculate_lipschitz,
                                    'early_stopping_triggered': early_stopping_triggered,
                                    'train_error_save_freq': train_error_save_freq,
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
                                save_data['metadata'] = np.array([str(metadata)])
                                
                                # Save result file using RAM cache
                                global_ram_cache.add(unique_id, save_data, results_dir)
                            
                            # Switch back to training mode
                            model.train()
                        
                        # Log training progress occasionally without writing to disk
                        elif iteration % 5000 == 0:
                            if current_train_losses:
                                # Compute current mean loss without synchronization
                                current_mean = torch.mean(torch.stack(current_train_losses)).item()
                                print(f"[Worker {worker_id} GPU {gpu_id}] {unique_id} - Iteration {iteration}/{total_iterations}, "
                                      f"Current Train Loss: {current_mean:.6f}")
                
                # Shutdown metrics calculator if used
                metrics_calculator.shutdown()
                
                # Wait for any active save threads to complete
                for t in active_save_threads:
                    if t.is_alive():
                        t.join(timeout=60)  # Wait up to 60 seconds for each thread
                
                # Save final model if requested
                if "final" in save_models:
                    final_model_path = os.path.join(models_dir, f"{unique_id}_final.pth.gz")
                    save_compressed_model(model, final_model_path)
                
                # Final save with complete results
                # Convert lists to numpy arrays with float32 precision for numerical stability
                train_stats_np = np.array(train_stats, dtype=np.float32)
                test_metrics_np = np.array(test_metrics, dtype=np.float32)
                
                # Create a dictionary of data to save
                save_data = {
                    # Original metrics
                    'train_stats': train_stats_np,
                    'test_metrics': test_metrics_np,
                }
                
                # Add term-specific metrics if available
                if term_test_metrics:
                    # Extract iterations and term losses for efficient storage
                    term_iterations = np.array([t[0] for t in term_test_metrics], dtype=np.float32)
                    term_losses_array = np.array([t[1] for t in term_test_metrics], dtype=np.float32)
                    save_data.update({
                        'term_iterations': term_iterations,
                        'term_losses': term_losses_array,
                    })
                
                # Add PTI metrics if available
                if term_pti_metrics:
                    # Process PTI metrics efficiently
                    pti_iterations = np.array([t[0] for t in term_pti_metrics], dtype=np.float32)
                    pti_correlation_ratios = np.array([t[1]['correlation_ratios'] for t in term_pti_metrics], dtype=np.float32)
                    pti_residual_mse = np.array([t[1]['residual_mse'] for t in term_pti_metrics], dtype=np.float32)
                    save_data.update({
                        'pti_iterations': pti_iterations,
                        'pti_correlation_ratios': pti_correlation_ratios,
                        'pti_residual_mse': pti_residual_mse,
                    })
                
                # Add Lipschitz metrics if available and enabled
                if lipschitz_metrics and calculate_lipschitz:
                    # Process Lipschitz metrics
                    lipschitz_iterations = np.array([t[0] for t in lipschitz_metrics], dtype=np.float32)
                    lipschitz_constants = np.array([t[1] for t in lipschitz_metrics], dtype=np.float32)
                    lipschitz_normalized_losses = np.array([t[2] for t in lipschitz_metrics], dtype=np.float32)
                    lipschitz_raw_losses = np.array([t[3] for t in lipschitz_metrics], dtype=np.float32)
                    save_data.update({
                        'lipschitz_iterations': lipschitz_iterations,
                        'lipschitz_constants': lipschitz_constants,
                        'lipschitz_normalized_losses': lipschitz_normalized_losses,
                        'lipschitz_raw_losses': lipschitz_raw_losses,
                    })
                
                # Add gradient alignment metrics if available and enabled
                if calculate_gradient_alignment and term_gradient_metrics:
                    # Process gradient alignment metrics
                    grad_iterations = np.array([t[0] for t in term_gradient_metrics], dtype=np.float32)
                    grad_alignment = np.array([t[1]['gradient_alignment'] for t in term_gradient_metrics], dtype=np.float32)
                    grad_magnitude_ratio = np.array([t[1]['gradient_magnitude_ratio'] for t in term_gradient_metrics], dtype=np.float32)
                    save_data.update({
                        'grad_iterations': grad_iterations,
                        'grad_alignment': grad_alignment,
                        'grad_magnitude_ratio': grad_magnitude_ratio,
                    })
                
                # Store term descriptions for context
                term_descriptions_array = np.array([desc for desc in term_descriptions])
                save_data['term_descriptions'] = term_descriptions_array
                
                # Get final metrics for metadata
                final_train_loss = float(train_stats[-1][1]) if train_stats else float('nan')
                final_test_loss = test_metrics[-1][1] if test_metrics else float('nan')
                final_term_losses = term_test_metrics[-1][1] if term_test_metrics and len(term_test_metrics) > 0 else []
                final_lipschitz_constant = float(lipschitz_constants[-1]) if calculate_lipschitz and 'lipschitz_constants' in locals() and len(lipschitz_constants) > 0 else float('nan')
                final_normalized_loss = float(lipschitz_normalized_losses[-1]) if calculate_lipschitz and 'lipschitz_normalized_losses' in locals() and len(lipschitz_normalized_losses) > 0 else float('nan')
                
                # Store metadata
                metadata = {
                    'function_name': function['name'],
                    'input_dim': input_dim,
                    'input_distribution': input_distribution,
                    'hidden_size': hidden_size,
                    'depth': depth,
                    'learning_rate': lr,
                    'original_learning_rate': lr * input_dim if lr_dim_scaling else lr,  # Store original LR if scaled
                    'lr_dim_scaling': lr_dim_scaling,  # Record if LR scaling was used
                    'batch_size': batch_size,  # Include actual batch size
                    'mode': mode,
                    'alignment': align,
                    'experiment_num': exp_num,
                    'final_train_loss': final_train_loss,
                    'final_test_loss': final_test_loss,
                    'final_lipschitz_constant': final_lipschitz_constant if calculate_lipschitz else float('nan'),
                    'final_normalized_loss': final_normalized_loss if calculate_lipschitz else float('nan'),
                    'optimizer': optimizer_type,
                    'unique_id': unique_id,
                    'gpu_id': gpu_id,
                    'total_iterations': total_iterations,
                    'current_iteration': iteration,
                    'final_term_losses': final_term_losses,
                    'term_descriptions': term_descriptions,
                    'calculate_gradient_alignment': calculate_gradient_alignment,
                    'calculate_lipschitz': calculate_lipschitz,
                    'early_stopping_triggered': early_stopping_triggered,
                    'train_error_save_freq': train_error_save_freq,
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
                save_data['metadata'] = np.array([str(metadata)])
                
                # Force final save to disk without RAM cache
                result_file = os.path.join(results_dir, f"{unique_id}_final.npz")
                np.savez_compressed(result_file, **save_data)
                
                # Mark this configuration as completed
                update_checkpoint(checkpoint_file, unique_id)
                completed_configs.add(unique_id)
                
                print(f"[Worker {worker_id} GPU {gpu_id}] Completed: {unique_id}")
                
            except Exception as e:
                print(f"[Worker {worker_id} GPU {gpu_id}] Error processing item {item}: {str(e)}")
                traceback.print_exc()
            
            finally:
                # Mark worker as idle when finished with the task
                worker_status[worker_id] = 0
                
    except Exception as e:
        print(f"[Worker {worker_id} GPU {gpu_id}] Fatal error: {str(e)}")
        traceback.print_exc()
        
        # If we encounter a fatal error, mark this worker as idle
        # so the system knows it can assign new work
        worker_status[worker_id] = 0
    
    print(f"[Worker {worker_id} GPU {gpu_id}] Worker finished")


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
                    'term_descriptions': metadata.get('term_descriptions', []),
                    'early_stopping_triggered': metadata.get('early_stopping_triggered', False)
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
    
    # Add new flags if they don't exist
    if 'calculate_lipschitz' not in config['base_config']:
        config['base_config']['calculate_lipschitz'] = True
        print("Added calculate_lipschitz flag with default value: True")
    
    if 'train_error_save_freq' not in config['base_config']:
        config['base_config']['train_error_save_freq'] = 1000
        print("Added train_error_save_freq flag with default value: 1000")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = config['base_config']['base_results_dir']
    results_dir = os.path.join(base_results_dir, f"complex_leap_exp_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up checkpoint file
    checkpoint_file = os.path.join(results_dir, f"checkpoint_{timestamp}.txt")
    
    # Check if we should restart from a previous checkpoint
    restart_checkpoint = config['base_config'].get('restart_checkpoint')
    if restart_checkpoint:
        if os.path.exists(restart_checkpoint):
            checkpoint_file = restart_checkpoint
            print(f"Restarting from checkpoint: {restart_checkpoint}")
        else:
            print(f"WARNING: Restart checkpoint {restart_checkpoint} not found, using new checkpoint file")
    
    # Save config for reference
    with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available. Running on CPU.")
        num_gpus = 1
    
    print(f"Using {num_gpus} GPU(s)")
    
    # Use exactly the number of workers specified in the YAML
    # NOTE: For small batch sizes (e.g., 512), 2-3 workers per GPU is typically optimal on H100s
    # For larger batch sizes or memory-intensive models, 1 worker per GPU is better
    num_workers = config['base_config'].get('num_workers', num_gpus)
    
    print(f"Using {num_workers} workers ({num_workers/num_gpus:.1f} workers per GPU)")
    
    # Start the shared data generator service
    input_distribution = config['base_config'].get('input_distribution', 'binary')
    print(f"Using input distribution: {input_distribution}")
    shared_data_service = SharedDataGeneratorService(
        input_distribution=input_distribution,
        num_generator_processes=min(4, num_gpus)  # Use up to 4 data generator processes
    )
    
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
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("spawn method already set")
    
    # ===== NEW: OPTIMIZATION 4 & 5 - Dynamic Load Balancing and Priority Queuing =====
    # Create a priority queue and add ONLY UNCOMPLETED combinations with priority
    # Lower priority values are processed first (smaller/simpler jobs)
    shared_task_queue = mp.Queue()
    worker_status = mp.Array('i', [0] * num_workers)  # 0=idle, 1=busy
    
    # Sort jobs by priority (complexity) - process simpler jobs first
    prioritized_jobs = []
    jobs_added = 0
    jobs_skipped = 0
    
    # First, collect all uncompleted jobs with their priority
    for combo in all_combinations:
        function, lr, batch_size, hidden_size, depth, mode, align, input_dim, exp_num = combo
        
        # Generate unique ID to check if this job is already completed
        unique_id = generate_unique_id(
            function, lr, hidden_size, depth, mode, align, input_dim, batch_size, 
            exp_num, input_distribution
        )
        
        if unique_id in completed_configs:
            jobs_skipped += 1
            if jobs_skipped <= 5:  # Print only first 5 skipped jobs to avoid log spam
                print(f"Skipping completed configuration: {unique_id}")
        else:
            # Calculate priority based on job complexity
            priority = calculate_config_priority(combo)
            prioritized_jobs.append((priority, combo))
            jobs_added += 1
    
    # Sort by priority (lower values first)
    prioritized_jobs.sort(key=lambda x: x[0])
    
    # Add jobs to the shared queue in priority order
    for priority, combo in prioritized_jobs:
        shared_task_queue.put(combo)
    
    # Add termination signals
    for _ in range(num_workers):
        shared_task_queue.put(None)
    
    print(f"Added {jobs_added} jobs to the queue in priority order")
    print(f"Skipped {jobs_skipped} already completed jobs")
    
    # Print some priority examples if available
    if prioritized_jobs:
        print("\nPriority examples (lower priority runs first):")
        for i, (priority, combo) in enumerate(prioritized_jobs[:5]):
            function, lr, batch_size, hidden_size, depth, mode, align, input_dim, exp_num = combo
            print(f"  Priority {priority}: {function['name']}, d={input_dim}, h={hidden_size}, depth={depth}")
        
        print(f"  ...({len(prioritized_jobs)-5} more jobs)...\n")
    
    # Remaining configurations count calculation
    remaining_configs = len(all_combinations) - len(completed_configs)
    print(f"Remaining configurations to process: {remaining_configs}")
    
    # Sanity check
    if jobs_added != remaining_configs:
        print(f"WARNING: Mismatch between jobs_added ({jobs_added}) and calculated remaining_configs ({remaining_configs})")
    
    # Start the worker manager process for dynamic load balancing
    worker_manager = mp.Process(
        target=worker_manager_process, 
        args=(shared_task_queue, worker_status, remaining_configs, num_workers)
    )
    worker_manager.daemon = True
    worker_manager.start()
    
    # Start worker processes with GPU-specific assignment and dynamic load balancing
    processes = []
    for i in range(num_workers):
        # Assign workers to GPUs with better distribution
        gpu_id = i % num_gpus
        p = mp.Process(
            target=train_model, 
            args=(i, shared_task_queue, worker_status, shared_data_service, 
                  config, results_dir, checkpoint_file, gpu_id)
        )
        p.start()
        processes.append(p)
    
    # Wait for all workers to finish
    for p in processes:
        p.join()
    
    # Wait for worker manager to finish
    worker_manager.join(timeout=10)
    
    # Shutdown the shared data generator service
    shared_data_service.shutdown()
    
    # Flush RAM cache to disk
    global_ram_cache.flush_to_disk(force=True)
    global_ram_cache.shutdown()
    
    print(f"All training completed. Results saved to {results_dir}")

if __name__ == "__main__":
    main()