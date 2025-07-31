#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import itertools

# Global configuration flags
PLOT_GRADIENT_ALIGNMENT = True  # Set to False to skip gradient alignment plots
LOG_X_AXIS = True               # Set to True for logarithmic x-axis
LOG_Y_AXIS = True               # Set to True for logarithmic y-axis

def load_results(results_dir: str) -> Dict:
    """Load experiment results from NPZ files in the specified directory."""
    npz_files = glob.glob(os.path.join(results_dir, "*.npz"))
    print(f"Found {len(npz_files)} result files in {results_dir}")
    
    results = {}
    
    for npz_file in npz_files:
        try:
            with np.load(npz_file, allow_pickle=True) as data:
                # Extract metadata
                metadata_str = str(data['metadata'][0])
                
                # Define nan in the local scope before eval
                nan = float('nan')
                
                # Use eval with the local dictionary including nan
                metadata = eval(metadata_str, {"__builtins__": {}}, {"nan": nan})
                
                unique_id = metadata['unique_id']
                
                # Load training and test metrics
                train_stats = data['train_stats']
                test_metrics = data['test_metrics']
                
                # Load Lipschitz metrics if available
                lipschitz_metrics = None
                if all(key in data for key in ['lipschitz_iterations', 'lipschitz_constants', 'lipschitz_normalized_losses']):
                    lipschitz_metrics = {
                        'iterations': data['lipschitz_iterations'],
                        'constants': data['lipschitz_constants'],
                        'normalized_losses': data['lipschitz_normalized_losses'],
                        'raw_losses': data['lipschitz_raw_losses'] if 'lipschitz_raw_losses' in data else None
                    }
                
                # Load term test losses if available
                term_losses = None
                term_descriptions = None
                
                if 'term_iterations' in data and 'term_losses' in data:
                    term_iterations = data['term_iterations']
                    term_losses_array = data['term_losses']
                    
                    # Store iterations and values separately
                    term_losses = {
                        'iterations': term_iterations,
                        'values': term_losses_array
                    }
                
                if 'term_descriptions' in data:
                    term_descriptions = data['term_descriptions']
                
                # Load Progressive Term Isolation metrics if available
                pti_metrics = None
                if all(key in data for key in ['pti_iterations', 'pti_correlation_ratios', 'pti_residual_mse']):
                    pti_metrics = {
                        'iterations': data['pti_iterations'],
                        'correlation_ratios': data['pti_correlation_ratios'],
                        'residual_mse': data['pti_residual_mse']
                    }
                
                # Load Gradient Alignment metrics if available and enabled
                grad_metrics = None
                if PLOT_GRADIENT_ALIGNMENT and all(key in data for key in ['grad_iterations', 'grad_alignment', 'grad_magnitude_ratio']):
                    grad_metrics = {
                        'iterations': data['grad_iterations'],
                        'alignment': data['grad_alignment'],
                        'magnitude_ratio': data['grad_magnitude_ratio']
                    }
                
                # Store in results dictionary
                results[unique_id] = {
                    'metadata': metadata,
                    'train_stats': train_stats,
                    'test_metrics': test_metrics,
                    'lipschitz_metrics': lipschitz_metrics,
                    'term_losses': term_losses,
                    'term_descriptions': term_descriptions,
                    'pti_metrics': pti_metrics,
                    'grad_metrics': grad_metrics
                }
        except Exception as e:
            print(f"Error loading {npz_file}: {str(e)}")
    
    return results

def group_results_by_function_batch_mode(results: Dict) -> Dict:
    """Group results by function name, batch size, and mode."""
    grouped = defaultdict(list)
    
    for unique_id, result_data in results.items():
        function_name = result_data['metadata'].get('function_name', 'unknown')
        batch_size = result_data['metadata'].get('batch_size', 0)
        mode = result_data['metadata'].get('mode', 'standard')
        key = (function_name, batch_size, mode)
        grouped[key].append((unique_id, result_data))
    
    return grouped

def get_hyperparameter_groups(results_for_group: List[Tuple[str, Dict]]) -> Tuple[List, Dict, List]:
    """
    Group results by architectures (hidden_size, depth) and learning rates.
    Returns sorted architectures and mapping to results.
    """
    # Extract unique architectures and learning rates
    architectures = set()
    learning_rates = set()
    dimensions = set()
    
    for _, result_data in results_for_group:
        metadata = result_data['metadata']
        hidden_size = metadata.get('hidden_size', None)
        depth = metadata.get('depth', None)
        lr = metadata.get('learning_rate', None)
        dim = metadata.get('input_dim', None)
        
        architectures.add((hidden_size, depth))
        learning_rates.add(lr)
        dimensions.add(dim)
    
    # Sort architectures: d1 & small H on left, d4 & high H on right
    sorted_architectures = sorted(list(architectures), key=lambda x: (x[1], x[0]))
    
    # Sort learning rates from smallest to highest
    sorted_learning_rates = sorted(list(learning_rates))
    
    # Sort dimensions
    sorted_dimensions = sorted(list(dimensions))
    
    # Create mapping from dimension to architecture to learning rate to result
    dim_to_arch_lr = defaultdict(lambda: defaultdict(dict))
    
    for unique_id, result_data in results_for_group:
        metadata = result_data['metadata']
        hidden_size = metadata.get('hidden_size', None)
        depth = metadata.get('depth', None)
        lr = metadata.get('learning_rate', None)
        dim = metadata.get('input_dim', None)
        
        arch_key = (hidden_size, depth)
        dim_to_arch_lr[dim][arch_key][lr] = result_data
    
    return sorted_architectures, sorted_learning_rates, sorted_dimensions, dim_to_arch_lr

def find_true_max_iteration(result_data):
    """Find the true maximum iteration across all data sources."""
    max_iter = 0
    
    # Check train_stats
    if result_data['train_stats'] is not None and len(result_data['train_stats']) > 0:
        if result_data['train_stats'].shape[0] > 0:
            max_iter = max(max_iter, result_data['train_stats'][-1, 0])
    
    # Check test_metrics
    if result_data['test_metrics'] is not None and len(result_data['test_metrics']) > 0:
        if result_data['test_metrics'].shape[0] > 0:
            max_iter = max(max_iter, result_data['test_metrics'][-1, 0])
    
    # Check lipschitz_metrics
    if result_data['lipschitz_metrics'] is not None:
        iterations = result_data['lipschitz_metrics']['iterations']
        if len(iterations) > 0:
            max_iter = max(max_iter, iterations[-1])
    
    # Check pti_metrics
    if result_data['pti_metrics'] is not None:
        iterations = result_data['pti_metrics']['iterations']
        if len(iterations) > 0:
            max_iter = max(max_iter, iterations[-1])
    
    # Check grad_metrics
    if result_data['grad_metrics'] is not None:
        iterations = result_data['grad_metrics']['iterations']
        if len(iterations) > 0:
            max_iter = max(max_iter, iterations[-1])
    
    # Check term_losses
    if result_data['term_losses'] is not None:
        iterations = result_data['term_losses']['iterations']
        if len(iterations) > 0:
            max_iter = max(max_iter, iterations[-1])
    
    return max_iter

def plot_function_batch_mode_results(function_name: str, batch_size: int, mode: str,
                                    results_for_group: List[Tuple[str, Dict]], 
                                    output_dir: str):
    """Create a grid of plots for a specific function, batch size, and mode."""
    # Group by architectures and learning rates
    architectures, learning_rates, dimensions, dim_to_arch_lr = get_hyperparameter_groups(results_for_group)
    
    # Define number of rows and columns
    # Each architecture gets 3 learning rates, small gap between architectures
    # Each dimension gets 4 rows (train, lipschitz, corr, grad if enabled)
    rows_per_dim = 4 if PLOT_GRADIENT_ALIGNMENT else 3
    n_rows = len(dimensions) * rows_per_dim
    
    # For columns: each architecture gets its own group of learning rates
    cols_per_arch = len(learning_rates)
    n_cols = len(architectures) * cols_per_arch
    
    # Create figure with a much larger size to prevent crowding
    # Scale the size based on the number of subplots
    fig_height = 6.5 * len(dimensions)  # 6.5 inches per dimension
    fig_width = 5 * len(architectures)  # 5 inches per architecture group
    
    # Ensure minimum size
    fig_height = max(14, fig_height)
    fig_width = max(18, fig_width)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create a GridSpec with much more spacing between plots
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.8, wspace=0.5)  # Increased spacing
    
    # Add a title for the entire figure with more margin
    fig.suptitle(f"{function_name} - Batch Size {batch_size} - Mode: {mode}", 
                 fontsize=20, y=0.99)  # Increased font and moved up
    
    # Set maximum iterations (for x-axis limits)
    max_iterations = 5000000
    
    # Define colors for different terms
    term_colors = plt.cm.tab10.colors
    
    # Add column headers for architectures and learning rates with more space
    for arch_idx, (hidden_size, depth) in enumerate(architectures):
        # Position the architecture label at the center of its learning rate group
        center_col = arch_idx * cols_per_arch + cols_per_arch // 2
        header_text = f"h={hidden_size}, d={depth}"
        
        fig.text(
            (center_col + 0.5) / n_cols, 0.97, 
            header_text, 
            ha='center', va='top', 
            fontsize=16, fontweight='bold'  # Increased font size
        )
        
        # Add learning rate labels for this architecture group
        for lr_idx, lr in enumerate(learning_rates):
            col = arch_idx * cols_per_arch + lr_idx
            lr_text = f"LR={lr}"
            
            fig.text(
                (col + 0.5) / n_cols, 0.955, 
                lr_text, 
                ha='center', va='top', 
                fontsize=14  # Increased font size
            )
    
    # Create plots for each dimension, architecture, and learning rate
    for dim_idx, dim in enumerate(dimensions):
        # Add row label for this dimension with more space
        row_pos = dim_idx * rows_per_dim + rows_per_dim // 2
        fig.text(
            0.005, 1.0 - (row_pos) / n_rows, 
            f"dim={dim}", 
            ha='left', va='center', 
            fontsize=16, fontweight='bold',  # Increased font size
            rotation=90
        )
        
        # For each architecture
        for arch_idx, arch in enumerate(architectures):
            # For each learning rate in this architecture group
            for lr_idx, lr in enumerate(learning_rates):
                # Calculate the column index in the grid
                col = arch_idx * cols_per_arch + lr_idx
                
                # Check if we have data for this combination
                if arch in dim_to_arch_lr[dim] and lr in dim_to_arch_lr[dim][arch]:
                    result_data = dim_to_arch_lr[dim][arch][lr]
                    
                    # Find the true maximum iteration across all data sources
                    true_max_iter = find_true_max_iteration(result_data)
                    
                    # 1. Training Error Plot (row 1 for this dimension)
                    ax_train = fig.add_subplot(gs[dim_idx * rows_per_dim, col])
                    train_stats = result_data['train_stats']
                    
                    if len(train_stats) > 0:
                        iterations = train_stats[:, 0]
                        mean_losses = train_stats[:, 1]
                        
                        ax_train.plot(iterations, mean_losses, 'b-', label='Train Loss', linewidth=2)
                        
                        # If the last train iteration is less than the true max, extend the line
                        last_train_iter = iterations[-1] if len(iterations) > 0 else 0
                        last_train_loss = mean_losses[-1] if len(mean_losses) > 0 else 0
                        
                        if last_train_iter < true_max_iter * 0.99:
                            # Add dotted extension line to true_max_iter
                            ax_train.plot(
                                [last_train_iter, true_max_iter], 
                                [last_train_loss, last_train_loss], 
                                'b--', linewidth=1.5, alpha=0.7
                            )
                            # Add marker at last actual point
                            ax_train.plot([last_train_iter], [last_train_loss], 'bo', markersize=4)
                    
                    if LOG_X_AXIS:
                        ax_train.set_xscale('log')
                    if LOG_Y_AXIS:
                        ax_train.set_yscale('log')
                    
                    ax_train.set_xlim(1, max_iterations)
                    # Only add ylabel on the left column for clarity
                    if col % cols_per_arch == 0:
                        ax_train.set_ylabel('Loss', fontsize=12)  # Increased font size
                    
                    ax_train.set_title("Training Error", fontsize=14, pad=5)  # Increased font size
                    ax_train.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
                    
                    # Reduce number of ticks to avoid crowding
                    if LOG_X_AXIS:
                        ax_train.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
                    
                    # Use a more reasonable font for tick labels
                    ax_train.tick_params(axis='both', which='major', labelsize=10)  # Increased from 8
                    
                    # Hide x-labels except for the bottom row
                    if dim_idx < len(dimensions) - 1 or (PLOT_GRADIENT_ALIGNMENT and rows_per_dim > 1):
                        ax_train.set_xticklabels([])
                    
                    # 2. Lipschitz Normalized Loss (row 2 for this dimension)
                    ax_lip = fig.add_subplot(gs[dim_idx * rows_per_dim + 1, col])
                    lipschitz_metrics = result_data['lipschitz_metrics']
                    
                    if lipschitz_metrics is not None:
                        lip_iterations = lipschitz_metrics['iterations']
                        lip_norm_losses = lipschitz_metrics['normalized_losses']
                        
                        # Plot normalized loss
                        ax_lip.plot(lip_iterations, lip_norm_losses, 'g-', label='Normalized Loss', linewidth=2)
                        
                        # If the last lipschitz iteration is less than the true max, extend the line
                        last_lip_iter = lip_iterations[-1] if len(lip_iterations) > 0 else 0
                        last_lip_loss = lip_norm_losses[-1] if len(lip_norm_losses) > 0 else 0
                        
                        if last_lip_iter < true_max_iter * 0.99:
                            # Add dotted extension line to true_max_iter
                            ax_lip.plot(
                                [last_lip_iter, true_max_iter], 
                                [last_lip_loss, last_lip_loss], 
                                'g--', linewidth=1.5, alpha=0.7
                            )
                            # Add marker at last actual point
                            ax_lip.plot([last_lip_iter], [last_lip_loss], 'go', markersize=4)
                    
                    if LOG_X_AXIS:
                        ax_lip.set_xscale('log')
                    if LOG_Y_AXIS:
                        ax_lip.set_yscale('log')
                    
                    ax_lip.set_xlim(1, max_iterations)
                    # Only add ylabel on the left column for clarity
                    if col % cols_per_arch == 0:
                        ax_lip.set_ylabel('Norm. Loss', fontsize=12)  # Increased font size
                    
                    ax_lip.set_title("Lipschitz Normalized Loss", fontsize=14, pad=5)  # Increased font size
                    ax_lip.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
                    
                    # Reduce number of ticks
                    if LOG_X_AXIS:
                        ax_lip.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
                    
                    # Use a more reasonable font for tick labels
                    ax_lip.tick_params(axis='both', which='major', labelsize=10)  # Increased from 8
                    
                    # Hide x-labels except for the bottom row
                    if dim_idx < len(dimensions) - 1 or rows_per_dim > 2:
                        ax_lip.set_xticklabels([])
                    
                    # 3. Correlation Ratios Plot (row 3 for this dimension)
                    ax_corr = fig.add_subplot(gs[dim_idx * rows_per_dim + 2, col])
                    pti_metrics = result_data['pti_metrics']
                    term_descriptions = result_data['term_descriptions']
                    
                    if pti_metrics is not None and term_descriptions is not None:
                        pti_iterations = pti_metrics['iterations']
                        pti_correlation_ratios = pti_metrics['correlation_ratios']
                        
                        # Plot each term's correlation
                        n_terms = pti_correlation_ratios.shape[1] if len(pti_correlation_ratios.shape) > 1 else 1
                        
                        for t_idx in range(n_terms):
                            # Get description for this term
                            term_desc = term_descriptions[t_idx] if term_descriptions is not None else f"Term {t_idx+1}"
                            term_label = f"{term_desc}"
                            
                            # Get correlation values for this term
                            if len(pti_correlation_ratios.shape) > 1:
                                term_corr_values = pti_correlation_ratios[:, t_idx]
                            else:
                                term_corr_values = pti_correlation_ratios
                            
                            # Make sure iterations and values have the same length
                            if len(pti_iterations) == len(term_corr_values):
                                ax_corr.plot(
                                    pti_iterations, term_corr_values,
                                    color=term_colors[t_idx % len(term_colors)],
                                    linestyle='-',
                                    linewidth=2,
                                    label=term_label
                                )
                                
                                # If the last correlation iteration is less than the true max, extend the line
                                last_corr_iter = pti_iterations[-1] if len(pti_iterations) > 0 else 0
                                last_corr_val = term_corr_values[-1] if len(term_corr_values) > 0 else 0
                                
                                if last_corr_iter < true_max_iter * 0.99:
                                    # Add dotted extension line to true_max_iter
                                    ax_corr.plot(
                                        [last_corr_iter, true_max_iter], 
                                        [last_corr_val, last_corr_val], 
                                        color=term_colors[t_idx % len(term_colors)],
                                        linestyle='--', linewidth=1.5, alpha=0.7
                                    )
                                    # Add marker at last actual point
                                    ax_corr.plot(
                                        [last_corr_iter], [last_corr_val], 
                                        'o', color=term_colors[t_idx % len(term_colors)], 
                                        markersize=4
                                    )
                    
                    if LOG_X_AXIS:
                        ax_corr.set_xscale('log')
                    ax_corr.set_ylim(-1.05, 1.05)  # Correlation ranges from -1 to 1
                    ax_corr.set_xlim(1, max_iterations)
                    ax_corr.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Zero line
                    
                    # Only show x-labels on the bottom row
                    if (dim_idx == len(dimensions) - 1) and (not PLOT_GRADIENT_ALIGNMENT or rows_per_dim == 3):
                        ax_corr.set_xlabel('Iterations', fontsize=12)  # Increased font size
                    else:
                        ax_corr.set_xticklabels([])
                    
                    # Only add ylabel on the left column for clarity
                    if col % cols_per_arch == 0:
                        ax_corr.set_ylabel('Correlation', fontsize=12)  # Increased font size
                    
                    ax_corr.set_title("Term Correlation Ratios", fontsize=14, pad=5)  # Increased font size
                    ax_corr.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
                    
                    # Reduce number of ticks
                    if LOG_X_AXIS:
                        ax_corr.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
                    
                    # Use a more reasonable font for tick labels
                    ax_corr.tick_params(axis='both', which='major', labelsize=10)  # Increased from 8
                    
                    # Add legend to first column of each dimension - placed outside the plot
                    if arch_idx == 0 and lr_idx == 0 and term_descriptions is not None:
                        # Create a more compact legend with smaller font
                        handles, labels = ax_corr.get_legend_handles_labels()
                        if handles:
                            # Use shorter labels for the legend (truncate long ones)
                            short_labels = []
                            for label in labels:
                                if len(label) > 20:
                                    short_labels.append(label[:18] + '...')
                                else:
                                    short_labels.append(label)
                            
                            # Create a legend with smaller font and placed to the right
                            legend = ax_corr.legend(
                                handles, short_labels,
                                loc='center left', fontsize='x-small',  # Increased font size
                                bbox_to_anchor=(1.05, 0.5), 
                                ncol=1,
                                handlelength=1.5, 
                                columnspacing=1.0
                            )
                    
                    # 4. Gradient Alignment Plot (row 4 for this dimension) - if enabled
                    if PLOT_GRADIENT_ALIGNMENT:
                        ax_grad = fig.add_subplot(gs[dim_idx * rows_per_dim + 3, col])
                        grad_metrics = result_data['grad_metrics']
                        
                        if grad_metrics is not None and term_descriptions is not None:
                            grad_iterations = grad_metrics['iterations']
                            grad_alignment = grad_metrics['alignment']
                            
                            # Plot each term's gradient alignment
                            n_terms = grad_alignment.shape[1] if len(grad_alignment.shape) > 1 else 1
                            
                            for t_idx in range(n_terms):
                                # Get description for this term
                                term_desc = term_descriptions[t_idx] if term_descriptions is not None else f"Term {t_idx+1}"
                                term_label = f"{term_desc}"
                                
                                # Get alignment values for this term
                                if len(grad_alignment.shape) > 1:
                                    term_align_values = grad_alignment[:, t_idx]
                                else:
                                    term_align_values = grad_alignment
                                
                                # Make sure iterations and values have the same length
                                if len(grad_iterations) == len(term_align_values):
                                    ax_grad.plot(
                                        grad_iterations, term_align_values,
                                        color=term_colors[t_idx % len(term_colors)],
                                        linestyle='-',
                                        linewidth=2,
                                        label=term_label
                                    )
                                    
                                    # If the last grad iteration is less than the true max, extend the line
                                    last_grad_iter = grad_iterations[-1] if len(grad_iterations) > 0 else 0
                                    last_grad_val = term_align_values[-1] if len(term_align_values) > 0 else 0
                                    
                                    if last_grad_iter < true_max_iter * 0.99:
                                        # Add dotted extension line to true_max_iter
                                        ax_grad.plot(
                                            [last_grad_iter, true_max_iter], 
                                            [last_grad_val, last_grad_val], 
                                            color=term_colors[t_idx % len(term_colors)],
                                            linestyle='--', linewidth=1.5, alpha=0.7
                                        )
                                        # Add marker at last actual point
                                        ax_grad.plot(
                                            [last_grad_iter], [last_grad_val], 
                                            'o', color=term_colors[t_idx % len(term_colors)], 
                                            markersize=4
                                        )
                        
                        if LOG_X_AXIS:
                            ax_grad.set_xscale('log')
                        ax_grad.set_ylim(-1.05, 1.05)  # Alignment ranges from -1 to 1
                        ax_grad.set_xlim(1, max_iterations)
                        ax_grad.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Zero line
                        
                        # Always show x-labels on the bottom row
                        if dim_idx == len(dimensions) - 1:
                            ax_grad.set_xlabel('Iterations', fontsize=12)  # Increased font size
                        
                        # Only add ylabel on the left column for clarity
                        if col % cols_per_arch == 0:
                            ax_grad.set_ylabel('Alignment', fontsize=12)  # Increased font size
                        
                        ax_grad.set_title("Gradient Alignment", fontsize=14, pad=5)  # Increased font size
                        ax_grad.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
                        
                        # Reduce number of ticks
                        if LOG_X_AXIS:
                            ax_grad.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
                        
                        # Use a more reasonable font for tick labels
                        ax_grad.tick_params(axis='both', which='major', labelsize=10)  # Increased from 8
                else:
                    # Create empty subplots if no data found
                    for row_offset in range(rows_per_dim):
                        ax = fig.add_subplot(gs[dim_idx * rows_per_dim + row_offset, col])
                        ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
                        ax.axis('off')
    
    # Adjust layout with very generous margins
    plt.subplots_adjust(left=0.07, right=0.95, top=0.94, bottom=0.05, hspace=0.8, wspace=0.5)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{function_name}_batch{batch_size}_mode{mode}_results.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')  # Increased DPI for better quality
    print(f"Saved plot to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)

def main():
    # Hard-coded parameters - modify these as needed
    results_dir = "/home/goring/OnlineSGD/results_MSP/15_04_leap8_max_standard/complex_leap_exp_20250416_153853"
    output_dir = "/home/goring/OnlineSGD/plots/plot_boost_power_law_standard"
    
    # Global flags - set these before running
    global PLOT_GRADIENT_ALIGNMENT, LOG_X_AXIS, LOG_Y_AXIS
    PLOT_GRADIENT_ALIGNMENT = False  # Set to False to skip gradient alignment plots
    LOG_X_AXIS = True               # Set to True for logarithmic x-axis
    LOG_Y_AXIS = True               # Set to True for logarithmic y-axis (training error)
    
    # Load all results
    results = load_results(results_dir)
    print(f"Loaded results for {len(results)} experiments")
    
    # Group results by function name, batch size, and mode
    grouped_results = group_results_by_function_batch_mode(results)
    print(f"Found {len(grouped_results)} different function-batch-mode combinations")
    
    # Plot results for each function-batch-mode combination
    for (function_name, batch_size, mode), results_for_combo in grouped_results.items():
        print(f"Plotting results for function: {function_name}, batch size: {batch_size}, mode: {mode}")
        plot_function_batch_mode_results(function_name, batch_size, mode, results_for_combo, output_dir)
    
    print("All plots completed")

if __name__ == "__main__":
   main()