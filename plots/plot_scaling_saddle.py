#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from collections import defaultdict
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

def load_results(results_dir):
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
                
                # Load training metrics
                train_stats = data['train_stats']
                
                # Store in results dictionary
                results[unique_id] = {
                    'metadata': metadata,
                    'train_stats': train_stats
                }
        except Exception as e:
            print(f"Error loading {npz_file}: {str(e)}")
    
    return results

def group_results_by_function_batch_mode(results):
    """Group results by function name, batch size, and mode."""
    grouped = defaultdict(list)
    
    for unique_id, result_data in results.items():
        function_name = result_data['metadata'].get('function_name', 'unknown')
        batch_size = result_data['metadata'].get('batch_size', 0)
        mode = result_data['metadata'].get('mode', 'standard')
        key = (function_name, batch_size, mode)
        grouped[key].append((unique_id, result_data))
    
    return grouped

def get_hyperparameter_groups(results_for_group):
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
    
    # Sort architectures by depth, then hidden size
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

def detect_saddle_point(iterations, losses, plot=False, unique_id=None):
    """
    Detect saddle point in the training loss curve.
    
    Args:
        iterations: Array of iteration/epoch numbers
        losses: Array of loss values (same length as iterations)
        plot: If True, create a debug plot showing the detected saddle
        unique_id: Identifier for the experiment (for plot title)
        
    Returns:
        saddle_epoch: The epoch where the saddle occurs, or None if no saddle is detected
        saddle_type: String describing the type of saddle ("clear", "weak", "none")
    """
    # Take log of losses for better analysis
    log_losses = np.log10(losses)
    
    # Check if we have enough data points
    if len(iterations) < 10:
        return None, "insufficient_data"
    
    # Smooth the loss curve to remove noise
    window_length = min(51, len(log_losses) - (len(log_losses) % 2) - 1)  # Must be odd
    if window_length < 5:
        window_length = 5
    
    try:
        # Try Savitzky-Golay filter (polynomial smoothing)
        smoothed_log_losses = savgol_filter(log_losses, window_length, 3)
    except Exception:
        # Fall back to Gaussian filter if Savgol fails
        smoothed_log_losses = gaussian_filter1d(log_losses, sigma=3)
    
    # Compute numerical derivative of the smoothed log loss
    derivative = np.gradient(smoothed_log_losses, iterations)
    
    # Smooth the derivative
    try:
        smoothed_derivative = savgol_filter(derivative, window_length, 3)
    except Exception:
        smoothed_derivative = gaussian_filter1d(derivative, sigma=3)
    
    # Determine if there's a plateau region followed by a sharp decline
    # First, find regions where the derivative is close to zero (plateau)
    plateau_mask = np.abs(smoothed_derivative) < 0.0001
    
    # Find regions where derivative is strongly negative (sharp decline)
    decline_mask = smoothed_derivative < -0.0005
    
    # Check if we have both a plateau and a decline
    if np.any(plateau_mask) and np.any(decline_mask):
        # Find the last point of the plateau before the decline starts
        plateau_indices = np.where(plateau_mask)[0]
        decline_indices = np.where(decline_mask)[0]
        
        # Find plateau points that come before decline points
        valid_plateau_indices = [i for i in plateau_indices if i < np.min(decline_indices)]
        
        if valid_plateau_indices:
            saddle_index = max(valid_plateau_indices)
            saddle_epoch = iterations[saddle_index]
            saddle_type = "clear"
        else:
            # Look for the steepest decline point
            steepest_decline_index = np.argmin(smoothed_derivative)
            saddle_epoch = iterations[steepest_decline_index]
            saddle_type = "weak"
    else:
        # No clear saddle pattern, try to find the point of maximum curvature
        # Compute second derivative
        second_derivative = np.gradient(smoothed_derivative, iterations)
        
        # Smooth the second derivative
        try:
            smoothed_second_derivative = savgol_filter(second_derivative, window_length, 3)
        except Exception:
            smoothed_second_derivative = gaussian_filter1d(second_derivative, sigma=3)
        
        # Find points of high negative curvature (transition points)
        curve_points = find_peaks(-smoothed_second_derivative)[0]
        
        if len(curve_points) > 0:
            # Find the most significant curvature point
            significance = -smoothed_second_derivative[curve_points]
            most_significant_idx = curve_points[np.argmax(significance)]
            saddle_epoch = iterations[most_significant_idx]
            saddle_type = "inflection"
        else:
            # If no saddle point is found
            saddle_epoch = None
            saddle_type = "none"
    
    # Optionally create a debug plot
    if plot and saddle_epoch is not None:
        plt.figure(figsize=(12, 10))
        
        # Plot original loss
        plt.subplot(3, 1, 1)
        plt.semilogy(iterations, losses, 'b-', label='Loss')
        if saddle_epoch is not None:
            plt.axvline(x=saddle_epoch, color='r', linestyle='--', label=f'Saddle at {saddle_epoch}')
        plt.title(f'Loss Curve - ID: {unique_id}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot log loss and its smoothed version
        plt.subplot(3, 1, 2)
        plt.plot(iterations, log_losses, 'b-', alpha=0.5, label='Log Loss')
        plt.plot(iterations, smoothed_log_losses, 'g-', label='Smoothed Log Loss')
        if saddle_epoch is not None:
            plt.axvline(x=saddle_epoch, color='r', linestyle='--', label=f'Saddle at {saddle_epoch}')
        plt.title('Log Loss and Smoothed Version')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot the derivative
        plt.subplot(3, 1, 3)
        plt.plot(iterations, derivative, 'b-', alpha=0.5, label='Derivative')
        plt.plot(iterations, smoothed_derivative, 'g-', label='Smoothed Derivative')
        if saddle_epoch is not None:
            saddle_idx = np.where(iterations == saddle_epoch)[0][0]
            plt.axvline(x=saddle_epoch, color='r', linestyle='--', label=f'Saddle at {saddle_epoch}')
            plt.axhline(y=smoothed_derivative[saddle_idx], color='k', linestyle=':', alpha=0.5)
        plt.title('Derivative of Log Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Derivative')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'saddle_detection_{unique_id}.png', dpi=150)
        plt.close()
    
    return saddle_epoch, saddle_type

def fit_power_law(x_values, y_values):
    """
    Fit a power law model of the form y = a * x^n to the data.
    Uses linear regression on log-transformed data.
    
    Returns:
    - a: coefficient
    - n: exponent
    - r_squared: goodness of fit
    - formula: string representation of the formula
    """
    # Filter out non-positive values for log transform
    valid_indices = np.where((x_values > 0) & (y_values > 0))[0]
    if len(valid_indices) < 2:
        return None
    
    x_valid = x_values[valid_indices]
    y_valid = y_values[valid_indices]
    
    # Log transform
    log_x = np.log(x_valid)
    log_y = np.log(y_valid)
    
    # Linear regression on log-transformed data
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    # Compute original parameters
    a = np.exp(intercept)
    n = slope
    r_squared = r_value**2
    
    return {
        'a': a,
        'n': n,
        'r_squared': r_squared,
        'formula': f"y = {a:.4f} * x^{n:.4f}"
    }

def create_simple_grid_plot(df, output_dir, threshold_or_method):
    """
    Create a simple grid plot showing n and a values vs depth for each function.
    Functions are arranged side by side in a grid.
    """
    # Get unique functions
    functions = sorted(df['function'].unique())
    n_functions = len(functions)
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(n_functions*4, 8))
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, n_functions, figure=fig)
    
    # Create a plot for each function
    for i, function in enumerate(functions):
        # Filter data for this function
        func_data = df[df['function'] == function]
        
        # Top row: n values vs depth
        ax1 = fig.add_subplot(gs[0, i])
        
        # Group by depth and calculate mean n value
        depth_n = func_data.groupby('depth')['n'].mean().reset_index()
        
        # Plot n vs depth
        ax1.plot(depth_n['depth'], depth_n['n'], 'o-', color='blue', linewidth=2, markersize=10)
        
        # Add individual data points with jitter
        depths = sorted(func_data['depth'].unique())
        for depth in depths:
            depth_data = func_data[func_data['depth'] == depth]
            # Add small jitter to x for better visibility
            jitter = np.random.normal(0, 0.05, size=len(depth_data))
            ax1.scatter(depth_data['depth'] + jitter, depth_data['n'], 
                     alpha=0.6, color='skyblue', edgecolor='navy')
        
        # Set title and labels
        ax1.set_title(f"{function}", fontsize=14)
        if i == 0:
            ax1.set_ylabel('Exponent (n)', fontsize=12)
        ax1.set_xticks(depths)
        ax1.set_ylim(0, max(df['n']) * 1.1)
        ax1.grid(True, linestyle='--', alpha=0.6)
            
        # Bottom row: a values vs depth
        ax2 = fig.add_subplot(gs[1, i])
        
        # Group by depth and calculate mean a value
        depth_a = func_data.groupby('depth')['a'].mean().reset_index()
        
        # Plot a vs depth
        ax2.plot(depth_a['depth'], depth_a['a'], 'o-', color='purple', linewidth=2, markersize=10)
        
        # Add individual data points with jitter
        for depth in depths:
            depth_data = func_data[func_data['depth'] == depth]
            # Add small jitter to x for better visibility
            jitter = np.random.normal(0, 0.05, size=len(depth_data))
            ax2.scatter(depth_data['depth'] + jitter, depth_data['a'], 
                     alpha=0.6, color='plum', edgecolor='purple')
        
        # Set labels
        if i == 0:
            ax2.set_ylabel('Coefficient (a)', fontsize=12)
        ax2.set_xlabel('Depth', fontsize=12)
        ax2.set_xticks(depths)
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Add main title
    if isinstance(threshold_or_method, float):
        title = f'Power Law Parameters by Function and Depth (threshold={threshold_or_method})'
    else:
        title = f'Power Law Parameters by Function and Depth (saddle detection)'
    
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(threshold_or_method, float):
        output_path = os.path.join(output_dir, f"simple_grid_power_law_params_{threshold_or_method}.png")
    else:
        output_path = os.path.join(output_dir, "simple_grid_power_law_params_saddle.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved simple grid plot to {output_path}")
    
    # Also create a single row version with only n values
    fig = plt.figure(figsize=(n_functions*4, 4))
    
    for i, function in enumerate(functions):
        # Filter data for this function
        func_data = df[df['function'] == function]
        
        # Create subplot
        ax = fig.add_subplot(1, n_functions, i+1)
        
        # Group by depth and calculate mean n value
        depth_n = func_data.groupby('depth')['n'].mean().reset_index()
        
        # Plot n vs depth
        ax.plot(depth_n['depth'], depth_n['n'], 'o-', color='blue', linewidth=2, markersize=10)
        
        # Add individual data points with jitter
        depths = sorted(func_data['depth'].unique())
        for depth in depths:
            depth_data = func_data[func_data['depth'] == depth]
            # Add small jitter to x for better visibility
            jitter = np.random.normal(0, 0.05, size=len(depth_data))
            ax.scatter(depth_data['depth'] + jitter, depth_data['n'], 
                     alpha=0.6, color='skyblue', edgecolor='navy')
        
        # Set title and labels
        ax.set_title(f"{function}", fontsize=14)
        if i == 0:
            ax.set_ylabel('Exponent (n)', fontsize=12)
        ax.set_xlabel('Depth', fontsize=12)
        ax.set_xticks(depths)
        ax.set_ylim(0, max(df['n']) * 1.1)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add exact value annotations
        for j, (depth, n_val) in enumerate(zip(depth_n['depth'], depth_n['n'])):
            ax.annotate(f"{n_val:.2f}", (depth, n_val), 
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold')
    
    # Add main title
    if isinstance(threshold_or_method, float):
        title = f'Power Law Exponents (n) by Function and Depth (threshold={threshold_or_method})'
    else:
        title = f'Power Law Exponents (n) by Function and Depth (saddle detection)'
    
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    if isinstance(threshold_or_method, float):
        output_path = os.path.join(output_dir, f"simple_grid_n_values_{threshold_or_method}.png")
    else:
        output_path = os.path.join(output_dir, "simple_grid_n_values_saddle.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_example_loss_curves(results, output_dir, num_examples=20):
    """Plot example loss curves with saddle point detection to validate the method."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a sample of results
    unique_ids = list(results.keys())
    if len(unique_ids) > num_examples:
        sample_ids = np.random.choice(unique_ids, num_examples, replace=False)
    else:
        sample_ids = unique_ids
    
    for unique_id in sample_ids:
        result_data = results[unique_id]
        train_stats = result_data['train_stats']
        
        if train_stats is not None and len(train_stats) > 0:
            iterations = train_stats[:, 0]
            mean_losses = train_stats[:, 1]
            
            # Detect saddle point and create plot
            saddle_epoch, saddle_type = detect_saddle_point(
                iterations, mean_losses, plot=True, unique_id=unique_id
            )
            
            # Print detection results
            if saddle_epoch is not None:
                print(f"ID: {unique_id}, Saddle at epoch {saddle_epoch}, Type: {saddle_type}")
            else:
                print(f"ID: {unique_id}, No saddle detected")

def plot_saddle_points_with_power_law(results, output_dir):
    """
    Create plots for each function showing saddle point epochs vs dimension.
    Fits a power law to each series, excluding d=20.
    """
    # Group results by function name, batch size, and mode
    grouped_results = group_results_by_function_batch_mode(results)
    print(f"Found {len(grouped_results)} different function-batch-mode combinations")
    
    # Store power law parameters for all models
    power_law_params = []
    
    # Create debugging directory for saddle detection plots
    saddle_debug_dir = os.path.join(output_dir, "saddle_detection_plots")
    os.makedirs(saddle_debug_dir, exist_ok=True)
    
    # Process each function separately
    for (function_name, batch_size, mode), results_for_combo in grouped_results.items():
        print(f"Processing function: {function_name}, batch size: {batch_size}, mode: {mode}")
        
        # Create a new figure for this function
        plt.figure(figsize=(10, 6))
        
        # Extract architectures, learning rates, and dimensions
        architectures, learning_rates, dimensions, dim_to_arch_lr = get_hyperparameter_groups(results_for_combo)
        
        # Sort dimensions to ensure correct x-axis ordering
        dimensions = sorted(dimensions)
        
        # Find all unique depths and organize architectures by depth
        depth_to_architectures = defaultdict(list)
        for arch in architectures:
            hidden_size, depth = arch
            depth_to_architectures[depth].append(arch)
        
        # Create a colormap for depths
        depths = sorted(depth_to_architectures.keys())
        n_depths = len(depths)
        blues = plt.cm.Blues(np.linspace(0.4, 0.9, n_depths))
        
        # For each depth
        for depth_idx, depth in enumerate(depths):
            # For each architecture (hidden size) with this depth
            for arch_idx, arch in enumerate(depth_to_architectures[depth]):
                hidden_size, _ = arch
                
                # Collect data points for this architecture
                x_points = []  # dimensions
                y_points = []  # saddle epochs
                saddle_types = []  # type of saddle detected
                
                # Also keep track of cases where no saddle was detected
                no_saddle_dims = []
                
                for dim in dimensions:
                    # Skip d=20 as requested
                    if dim == 20:
                        continue
                        
                    # For each learning rate (we'll take the result with the most clear saddle)
                    best_saddle_epoch = None
                    best_saddle_type = None
                    best_saddle_clarity = -1  # Higher is better
                    
                    for lr in learning_rates:
                        if dim in dim_to_arch_lr and arch in dim_to_arch_lr[dim] and lr in dim_to_arch_lr[dim][arch]:
                            result_data = dim_to_arch_lr[dim][arch][lr]
                            train_stats = result_data['train_stats']
                            
                            if train_stats is not None and len(train_stats) > 0:
                                iterations = train_stats[:, 0]
                                mean_losses = train_stats[:, 1]
                                
                                # Save example saddle detection plots for debugging
                                plot_debug = (
                                    depth_idx == 0 and 
                                    arch_idx == 0 and 
                                    dim == dimensions[len(dimensions)//2]
                                )
                                
                                # Detect saddle point
                                saddle_epoch, saddle_type = detect_saddle_point(
                                    iterations, mean_losses, 
                                    plot=plot_debug,
                                    unique_id=f"{function_name}_d{dim}_h{hidden_size}_dep{depth}_lr{lr}"
                                )
                                
                                # Determine clarity score for this saddle detection
                                if saddle_type == "clear":
                                    clarity = 3
                                elif saddle_type == "weak":
                                    clarity = 2
                                elif saddle_type == "inflection":
                                    clarity = 1
                                else:
                                    clarity = 0
                                
                                # Update best saddle if this one is clearer
                                if clarity > best_saddle_clarity:
                                    best_saddle_epoch = saddle_epoch
                                    best_saddle_type = saddle_type
                                    best_saddle_clarity = clarity
                    
                    # Add to collection if a saddle was found
                    if best_saddle_epoch is not None:
                        x_points.append(dim)
                        y_points.append(best_saddle_epoch)
                        saddle_types.append(best_saddle_type)
                    else:
                        no_saddle_dims.append(dim)
                
                # Plot if we have points
                if x_points and len(x_points) >= 2:  # Need at least 2 points for fitting
                    # Convert to numpy arrays for fitting
                    x_array = np.array(x_points)
                    y_array = np.array(y_points)
                    
                    # Fit power law y = a * x^n
                    power_law_fit = fit_power_law(x_array, y_array)
                    
                    if power_law_fit is not None:
                        # Save parameters for later visualization
                        power_law_params.append({
                            'function': function_name,
                            'batch_size': batch_size,
                            'mode': mode,
                            'depth': depth,
                            'hidden_size': hidden_size,
                            'a': power_law_fit['a'],
                            'n': power_law_fit['n'],
                            'r_squared': power_law_fit['r_squared']
                        })
                        
                        # Generate fitted curve points for plotting
                        x_fit = np.logspace(np.log10(min(x_array)), np.log10(max(x_array)), 100)
                        y_fit = power_law_fit['a'] * x_fit ** power_law_fit['n']
                        
                        # Adjust color based on depth - darker blue for deeper networks
                        color = blues[depth_idx]
                        
                        # Use different markers for different depths (same marker for same depth)
                        marker = ['o', 's', '^', 'v', 'x', '*', 'D', 'p'][depth_idx % 8]
                        
                        label = f"depth={depth}, h={hidden_size} (n={power_law_fit['n']:.2f})"
                        
                        # Plot original data - use different markers for different saddle types
                        for i, (x, y, s_type) in enumerate(zip(x_array, y_array, saddle_types)):
                            marker_style = marker
                            if s_type == "clear":
                                ms = 10  # Clear saddles are larger markers
                                alpha = 1.0
                            elif s_type == "weak":
                                ms = 8
                                alpha = 0.8
                            else:  # inflection
                                ms = 6
                                alpha = 0.6
                                
                            plt.plot(x, y, marker=marker_style, linestyle='none', color=color, 
                                     markersize=ms, alpha=alpha)
                            
                        # Plot fitted power law
                        plt.plot(x_fit, y_fit, linestyle='-', color=color, alpha=0.7, linewidth=1.5, label=label)
                        
                        # Mark dimensions where no saddle was detected
                        for dim in no_saddle_dims:
                            plt.axvline(x=dim, color=color, linestyle=':', alpha=0.3)
        
        # Add labels and title
        plt.xlabel('Input Dimension (d)', fontsize=14)
        plt.ylabel('Saddle Point Epoch', fontsize=14)
        plt.title(f"{function_name} - Batch Size {batch_size} - Mode: {mode}", fontsize=16)
        
        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Use log-log scale for better visualization of power laws
        plt.xscale('log')
        plt.yscale('log')
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{function_name}_batch{batch_size}_saddle_power_law.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved saddle analysis plot for {function_name} to {output_path}")
    
    # Create dataframe from power law parameters
    df_power_law = pd.DataFrame(power_law_params)
    
    if not df_power_law.empty:
        # Create the simple grid visualization
        create_simple_grid_plot(df_power_law, output_dir, "saddle")
    
    return df_power_law

def main():
    # Hard-coded parameters - modify these as needed
    results_dir = "/home/goring/OnlineSGD/results_MSP/15_04_leap8_grid_nolr/all1"
    output_dir = "/home/goring/OnlineSGD/plots/plot_grid8_saddle_analysis"
    
    # Load all results
    results = load_results(results_dir)
    print(f"Loaded results for {len(results)} experiments")
    
    # Plot example loss curves with saddle detection (for validation)
    plot_example_loss_curves(results, os.path.join(output_dir, "example_curves"))
    
    # Analyze saddle points and fit power laws
    df_power_law = plot_saddle_points_with_power_law(results, output_dir)
    
    # Print summary of power law exponents by function
    if not df_power_law.empty:
        print("\nPower Law Exponent (n) Summary by Function:")
        func_summary = df_power_law.groupby('function')['n'].agg(['mean', 'std', 'min', 'max'])
        print(func_summary)
        
        print("\nPower Law Exponent (n) Summary by Depth:")
        depth_summary = df_power_law.groupby('depth')['n'].agg(['mean', 'std', 'min', 'max'])
        print(depth_summary)
        
        # Create a summary CSV for further analysis
        summary_path = os.path.join(output_dir, "saddle_power_law_params_summary.csv")
        df_power_law.to_csv(summary_path, index=False)
        print(f"Saved power law parameters to {summary_path}")
    
    print("All saddle analysis and plots completed")

if __name__ == "__main__":
    main()