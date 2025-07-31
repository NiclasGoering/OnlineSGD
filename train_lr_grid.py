#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import sys
import numpy as np
import yaml
import traceback
from datetime import datetime
import time
import signal
import math
from typing import Dict, Tuple, List
import json
from helpers.FFNN_sgd import DeepNN  # Use your existing import

# Define stability criteria
MAX_GRAD_NORM = 100.0
LOSS_EXPLOSION_THRESHOLD = 1e10  # Consider loss exploded if it exceeds this
STABILITY_CHECK_ITERATIONS = 25000

# LR finder specific constants (these will be overridden by config)
INITIAL_LR = 1e-6  # Default, but will be replaced by config value
LR_DOUBLING_FACTOR = 2.0
COARSE_GRAIN_STEPS = 3
NUM_EXPERIMENTS = 3  # Default, but will be replaced by config value

def detect_training_instability(loss, grad_norm):
    """Check if training has become unstable"""
    # Convert loss to float if it's a tensor
    loss_value = loss.item() if torch.is_tensor(loss) else loss
    
    # Check loss stability
    if math.isnan(loss_value) or math.isinf(loss_value) or loss_value > LOSS_EXPLOSION_THRESHOLD:
        return True, f"Loss explosion: {loss_value}"
    
    # Check gradient stability
    if math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > MAX_GRAD_NORM:
        return True, f"Gradient explosion: {grad_norm}"
    
    return False, None

def calculate_gradient_norm(model):
    """Calculate overall gradient norm of the model"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def generate_k_parity_data(n_samples, k, input_dim, device):
    """Generate k-parity function data"""
    X = torch.randint(0, 2, (n_samples, input_dim), device=device) * 2.0 - 1.0  # Convert to {-1, 1}
    y = torch.prod(X[:, :k], dim=1)
    return X, y

def test_learning_rate(config, gpu_id, function_config, lr, hidden_size, depth, mode, input_dim, batch_size, exp_num):
    """Test if a specific learning rate is stable"""
    try:
        # Set device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device('cuda:0')
        
        # Set random seed for reproducibility
        torch.manual_seed(42 + exp_num)
        
        # Create model
        model = DeepNN(
            d=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            mode=mode,
            alignment=False,
            base_width=config['sweeps'].get('base_width', 256),
            gamma=1.0
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        criterion = torch.nn.MSELoss()
        
        # Training loop for stability check
        model.train()
        stability_info = {
            'is_stable': True,
            'max_grad_norm': 0.0,
            'max_loss': 0.0,
            'instability_iteration': None,
            'instability_reason': None
        }
        
        max_grad_norm = config['base_config']['lr_finder'].get('max_grad_norm', MAX_GRAD_NORM)
        stability_check_iterations = config['base_config']['lr_finder'].get('stability_check_iterations', STABILITY_CHECK_ITERATIONS)
        
        for iteration in range(stability_check_iterations):
            # Generate batch data
            X, y = generate_k_parity_data(batch_size, function_config['k'], input_dim, device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Calculate gradient norm
            grad_norm = calculate_gradient_norm(model)
            
            # Check for instability
            is_unstable, reason = detect_training_instability(loss, grad_norm)
            
            if is_unstable:
                stability_info['is_stable'] = False
                stability_info['instability_iteration'] = iteration
                stability_info['instability_reason'] = reason
                break
            
            # Update stability info
            stability_info['max_grad_norm'] = max(stability_info['max_grad_norm'], grad_norm)
            stability_info['max_loss'] = max(stability_info['max_loss'], loss.item())
            
            # Update parameters
            optimizer.step()
            
            # Print progress occasionally
            if iteration % 5000 == 0:
                print(f"[GPU {gpu_id}] LR test {lr:.2e} - Iteration {iteration}: Loss={loss.item():.6f}, GradNorm={grad_norm:.4f}")
        
        return stability_info
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error testing LR {lr}: {str(e)}")
        traceback.print_exc()
        return {'is_stable': False, 'instability_reason': f"Exception: {str(e)}"}

def find_optimal_lr(config, gpu_id, function_config, hidden_size, depth, mode, input_dim, batch_size, exp_num):
    """Find optimal learning rate for a specific configuration"""
    try:
        print(f"[GPU {gpu_id}] Finding optimal LR for: k={function_config['k']}, d={input_dim}, h={hidden_size}, "
              f"depth={depth}, mode={mode}, batch={batch_size}, exp={exp_num}")
        
        # Phase 1: Coarse search (exponential doubling)
        # Get initial LR from config
        initial_lr = config['base_config']['lr_finder'].get('initial_lr', INITIAL_LR)
        coarse_grain_steps = config['base_config']['lr_finder'].get('coarse_grain_steps', COARSE_GRAIN_STEPS)
        
        current_lr = initial_lr
        stable_lr = None
        unstable_lr = None
        
        while unstable_lr is None:
            print(f"[GPU {gpu_id}] Testing LR: {current_lr:.2e}")
            
            stability_info = test_learning_rate(
                config, gpu_id, function_config, current_lr, hidden_size, depth, mode, input_dim, batch_size, exp_num
            )
            
            if stability_info['is_stable']:
                stable_lr = current_lr
                current_lr *= LR_DOUBLING_FACTOR
            else:
                unstable_lr = current_lr
                break
            
            # Prevent infinite loop
            if current_lr > 10.0:
                print(f"[GPU {gpu_id}] Reached maximum LR=10.0 without instability")
                return {'optimal_lr': stable_lr, 'search_info': {'phase1_stable': stable_lr, 'phase1_unstable': None}}
        
        # Phase 2: Fine-grained binary search
        if stable_lr is None:
            print(f"[GPU {gpu_id}] No stable LR found, using minimum: {initial_lr}")
            return {'optimal_lr': initial_lr, 'search_info': {'phase1_stable': None, 'phase1_unstable': unstable_lr}}
        
        # Perform binary search between stable_lr and unstable_lr
        search_history = []
        
        for _ in range(coarse_grain_steps):
            mid_lr = (stable_lr + unstable_lr) / 2
            print(f"[GPU {gpu_id}] Binary search: Testing LR={mid_lr:.2e} between {stable_lr:.2e} and {unstable_lr:.2e}")
            
            stability_info = test_learning_rate(
                config, gpu_id, function_config, mid_lr, hidden_size, depth, mode, input_dim, batch_size, exp_num
            )
            
            search_history.append({
                'lr': mid_lr,
                'is_stable': stability_info['is_stable'],
                'max_grad_norm': stability_info['max_grad_norm'],
                'max_loss': stability_info['max_loss']
            })
            
            if stability_info['is_stable']:
                stable_lr = mid_lr
            else:
                unstable_lr = mid_lr
        
        print(f"[GPU {gpu_id}] Final optimal LR: {stable_lr:.2e}")
        
        return {
            'optimal_lr': stable_lr,
            'search_info': {
                'phase1_stable': stable_lr,
                'phase1_unstable': unstable_lr,
                'binary_search_history': search_history
            }
        }
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error in LR search: {str(e)}")
        traceback.print_exc()
        return {'optimal_lr': config['base_config']['lr_finder'].get('initial_lr', INITIAL_LR), 
                'search_info': {'error': str(e)}}

def lr_finder_worker(gpu_id, queue, config, results_dir):
    """Worker process for finding optimal learning rates"""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] LR finder worker started")
        
        while True:
            item = queue.get()
            if item is None:
                break
            
            function_config, hidden_size, depth, mode, input_dim, batch_size, exp_num = item
            
            # Find optimal LR for this configuration
            result = find_optimal_lr(
                config, gpu_id, function_config, hidden_size, depth, mode, input_dim, batch_size, exp_num
            )
            
            # Save result
            result_file = os.path.join(
                results_dir,
                f"lr_result_k{function_config['k']}_d{input_dim}_h{hidden_size}_depth{depth}_{mode}_b{batch_size}_exp{exp_num}.npy"
            )
            
            save_data = {
                'function': function_config,
                'hidden_size': hidden_size,
                'depth': depth,
                'mode': mode,
                'input_dim': input_dim,
                'batch_size': batch_size,
                'experiment_num': exp_num,
                'optimal_lr': result['optimal_lr'],
                'search_info': result['search_info']
            }
            
            np.save(result_file, save_data)
            print(f"[GPU {gpu_id}] Saved result to {result_file}")
        
        print(f"[GPU {gpu_id}] LR finder worker finished")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error in LR finder worker: {str(e)}")
        traceback.print_exc()

def main():
    if len(sys.argv) < 2:
        print("Usage: python lr_finder.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    print(f"Loading config from: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update global constants with config values
    global INITIAL_LR, COARSE_GRAIN_STEPS, NUM_EXPERIMENTS, MAX_GRAD_NORM, STABILITY_CHECK_ITERATIONS
    INITIAL_LR = config['base_config']['lr_finder'].get('initial_lr', INITIAL_LR)
    COARSE_GRAIN_STEPS = config['base_config']['lr_finder'].get('coarse_grain_steps', COARSE_GRAIN_STEPS)
    NUM_EXPERIMENTS = config['base_config']['lr_finder'].get('num_experiments', NUM_EXPERIMENTS)
    MAX_GRAD_NORM = config['base_config']['lr_finder'].get('max_grad_norm', MAX_GRAD_NORM)
    STABILITY_CHECK_ITERATIONS = config['base_config']['lr_finder'].get('stability_check_iterations', STABILITY_CHECK_ITERATIONS)
    
    # Get results directory from config, use default if not specified
    base_results_dir = config.get('base_config', {}).get('base_results_dir', 'lr_finder_results')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_results_dir, f"lr_search_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config for reference
    with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        sys.exit(1)
    
    print(f"Using {num_gpus} GPU(s)")
    num_workers = config.get('num_workers', num_gpus)
    print(f"Using {num_workers} workers")
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("spawn method already set")
    
    # Generate all configurations
    all_configurations = []
    
    for function_config in config['functions']:
        for hidden_size in config['sweeps']['hidden_sizes']:
            for depth in config['sweeps']['depths']:
                for mode in config['sweeps']['modes']:
                    for input_dim in function_config['dimensions']:
                        for batch_size in config['sweeps']['batch_sizes']:
                            for exp_num in range(1, NUM_EXPERIMENTS + 1):
                                all_configurations.append(
                                    (function_config, hidden_size, depth, mode, input_dim, batch_size, exp_num)
                                )
    
    print(f"Total configurations to test: {len(all_configurations)}")
    
    # Create queue and add all configurations
    queue = mp.Queue()
    
    for config_item in all_configurations:
        queue.put(config_item)
    
    # Add None markers to signal workers to exit
    for _ in range(num_workers):
        queue.put(None)
    
    # Start worker processes
    processes = []
    for i in range(num_workers):
        gpu_id = i % num_gpus
        p = mp.Process(target=lr_finder_worker, args=(gpu_id, queue, config, results_dir))
        p.start()
        processes.append(p)
    
    # Wait for all workers to finish
    try:
        for p in processes:
            p.join()
        print("All LR finder workers completed")
        
        # Aggregate results
        aggregate_results(results_dir, config)
        
    except KeyboardInterrupt:
        print("Received keyboard interrupt, terminating workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
    
    print(f"Learning rate search completed. Results saved to {results_dir}")

def aggregate_results(results_dir, config):
    """Aggregate all LR search results into a single summary file"""
    all_results = []
    
    # Load all result files
    for file in os.listdir(results_dir):
        if file.startswith("lr_result_") and file.endswith(".npy"):
            result = np.load(os.path.join(results_dir, file), allow_pickle=True).item()
            all_results.append(result)
    
    # Create summary data structure
    summary = {
        'configurations': {},
        'statistics': {
            'mean_optimal_lr_by_k': {},
            'mean_optimal_lr_by_width': {},
            'mean_optimal_lr_by_depth': {},
            'mean_optimal_lr_by_mode': {}
        }
    }
    
    # Organize results by configuration
    for result in all_results:
        key = f"k{result['function']['k']}_d{result['input_dim']}_h{result['hidden_size']}_depth{result['depth']}_{result['mode']}_b{result['batch_size']}"
        
        if key not in summary['configurations']:
            summary['configurations'][key] = {
                'experiments': [],
                'mean_optimal_lr': 0.0,
                'std_optimal_lr': 0.0
            }
        
        summary['configurations'][key]['experiments'].append({
            'exp_num': result['experiment_num'],
            'optimal_lr': result['optimal_lr'],
            'search_info': result['search_info']
        })
    
    # Calculate statistics
    for key, config_data in summary['configurations'].items():
        optimal_lrs = [exp['optimal_lr'] for exp in config_data['experiments']]
        config_data['mean_optimal_lr'] = np.mean(optimal_lrs)
        config_data['std_optimal_lr'] = np.std(optimal_lrs)
    
    # Group by various factors
    for result in all_results:
        k = result['function']['k']
        if k not in summary['statistics']['mean_optimal_lr_by_k']:
            summary['statistics']['mean_optimal_lr_by_k'][k] = []
        summary['statistics']['mean_optimal_lr_by_k'][k].append(result['optimal_lr'])
        
        width = result['hidden_size']
        if width not in summary['statistics']['mean_optimal_lr_by_width']:
            summary['statistics']['mean_optimal_lr_by_width'][width] = []
        summary['statistics']['mean_optimal_lr_by_width'][width].append(result['optimal_lr'])
        
        depth = result['depth']
        if depth not in summary['statistics']['mean_optimal_lr_by_depth']:
            summary['statistics']['mean_optimal_lr_by_depth'][depth] = []
        summary['statistics']['mean_optimal_lr_by_depth'][depth].append(result['optimal_lr'])
        
        mode = result['mode']
        if mode not in summary['statistics']['mean_optimal_lr_by_mode']:
            summary['statistics']['mean_optimal_lr_by_mode'][mode] = []
        summary['statistics']['mean_optimal_lr_by_mode'][mode].append(result['optimal_lr'])
    
    # Calculate means
    for k, values in summary['statistics']['mean_optimal_lr_by_k'].items():
        summary['statistics']['mean_optimal_lr_by_k'][k] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    for width, values in summary['statistics']['mean_optimal_lr_by_width'].items():
        summary['statistics']['mean_optimal_lr_by_width'][width] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    for depth, values in summary['statistics']['mean_optimal_lr_by_depth'].items():
        summary['statistics']['mean_optimal_lr_by_depth'][depth] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    for mode, values in summary['statistics']['mean_optimal_lr_by_mode'].items():
        summary['statistics']['mean_optimal_lr_by_mode'][mode] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    # Save summary
    summary_file = os.path.join(results_dir, "lr_search_summary.json")
    with open(summary_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            else:
                return obj
        
        json.dump(convert_to_native(summary), f, indent=4)
    
    print(f"Saved aggregate results to {summary_file}")
    
    # Create a simple text summary for quick reference
    with open(os.path.join(results_dir, "lr_search_summary.txt"), 'w') as f:
        f.write("Learning Rate Search Summary\n")
        f.write("===========================\n\n")
        
        f.write("Mean Optimal LR by k:\n")
        for k, stats in summary['statistics']['mean_optimal_lr_by_k'].items():
            f.write(f"  k={k}: {stats['mean']:.4e} ± {stats['std']:.4e} (n={stats['count']})\n")
        
        f.write("\nMean Optimal LR by width:\n")
        for width, stats in summary['statistics']['mean_optimal_lr_by_width'].items():
            f.write(f"  width={width}: {stats['mean']:.4e} ± {stats['std']:.4e} (n={stats['count']})\n")
        
        f.write("\nMean Optimal LR by depth:\n")
        for depth, stats in summary['statistics']['mean_optimal_lr_by_depth'].items():
            f.write(f"  depth={depth}: {stats['mean']:.4e} ± {stats['std']:.4e} (n={stats['count']})\n")
        
        f.write("\nMean Optimal LR by mode:\n")
        for mode, stats in summary['statistics']['mean_optimal_lr_by_mode'].items():
            f.write(f"  mode={mode}: {stats['mean']:.4e} ± {stats['std']:.4e} (n={stats['count']})\n")

if __name__ == "__main__":
    main()