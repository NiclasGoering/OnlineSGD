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

def load_results(results_dir):
    """Load experiment results from NPZ files in the specified directory."""
    npz_files = glob.glob(os.path.join(results_dir, "*.npz"))
    print(f"Found {len(npz_files)} result files in {results_dir}")
    
    results = {}
    error_count = 0
    error_files = []
    
    # Count monomials of different degrees for debugging
    monomial_counts = defaultdict(int)
    
    for npz_file in npz_files:
        try:
            # Extract function name from filename for debugging
            base_filename = os.path.basename(npz_file)
            function_prefix = base_filename.split('_')[0]
            monomial_counts[function_prefix] += 1
            
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
                    'train_stats': train_stats,
                    'filename': npz_file  # Store filename for debugging
                }
        except Exception as e:
            error_count += 1
            error_files.append(npz_file)
            print(f"Error loading {npz_file}: {str(e)}")
    
    print(f"Successfully loaded {len(results)} files, {error_count} files had errors")
    if error_count > 0:
        print(f"First 5 error files: {error_files[:5]}")
    
    # Print monomial counts
    print("Monomial function counts:")
    for func, count in sorted(monomial_counts.items()):
        print(f"  {func}: {count} files")
    
    return results

def extract_experiment_number(unique_id):
    """Extract experiment number from the unique ID."""
    parts = unique_id.split('_')
    for part in parts:
        if part.startswith('exp'):
            try:
                return int(part[3:])
            except ValueError:
                pass
    return None

def group_results_by_function_batch_width_depth(results):
    """
    Group results by function name, batch size, width, depth, and dimension.
    This facilitates calculating means and std across experiments.
    """
    grouped = defaultdict(list)
    
    for unique_id, result_data in results.items():
        metadata = result_data['metadata']
        function_name = metadata.get('function_name', 'unknown')
        batch_size = metadata.get('batch_size', 0)
        hidden_size = metadata.get('hidden_size', 0)
        depth = metadata.get('depth', 0)
        input_dim = metadata.get('input_dim', 0)
        mode = metadata.get('mode', 'standard')
        exp_num = extract_experiment_number(unique_id)
        
        # Group key includes all parameters except experiment number
        key = (function_name, batch_size, hidden_size, depth, input_dim, mode)
        grouped[key].append((exp_num, unique_id, result_data))
    
    return grouped

def find_epoch_below_threshold(result_data, threshold=0.001):
    """Find the first epoch where loss drops below the given threshold."""
    train_stats = result_data['train_stats']
    
    if train_stats is not None and len(train_stats) > 0:
        iterations = train_stats[:, 0]
        mean_losses = train_stats[:, 1]
        
        # Find the first index where loss is below threshold
        below_threshold_indices = np.where(mean_losses < threshold)[0]
        if len(below_threshold_indices) > 0:
            first_index = below_threshold_indices[0]
            return iterations[first_index]
    
    # No point below threshold
    return None

def calculate_mean_epochs_to_threshold(grouped_results, threshold=0.001):
    """
    Calculate mean and std of epochs to reach threshold across experiments.
    Returns a DataFrame with all results.
    """
    results_list = []
    
    for (function_name, batch_size, hidden_size, depth, input_dim, mode), experiment_data in grouped_results.items():
        # Extract epochs to threshold for each experiment
        epochs_list = []
        
        for exp_num, unique_id, result_data in experiment_data:
            epoch = find_epoch_below_threshold(result_data, threshold)
            if epoch is not None:
                epochs_list.append(epoch)
        
        # Calculate statistics if we have at least one valid result
        if epochs_list:
            mean_epoch = np.mean(epochs_list)
            std_epoch = np.std(epochs_list)
            
            results_list.append({
                'function': function_name,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'depth': depth,
                'input_dim': input_dim,
                'mode': mode,
                'mean_epochs': mean_epoch,
                'std_epochs': std_epoch,
                'num_experiments': len(epochs_list),
                'raw_epochs': epochs_list
            })
    
    # Convert to DataFrame for easier analysis
    return pd.DataFrame(results_list)

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
        'formula': f"y = {a:.4f} * x^{n:.4f}",
        'std_err': std_err
    }

def fit_power_laws_by_group(df):
    """
    Fit power laws for each function-width-depth-batch combination.
    Returns a DataFrame with fitted parameters.
    """
    power_law_params = []
    
    # Group by function, hidden_size, depth, and batch_size
    grouped = df.groupby(['function', 'hidden_size', 'depth', 'batch_size'])
    
    for (func, width, depth, batch), group_df in grouped:
        # Skip groups with less than 2 dimensions (need at least 2 points for fitting)
        if len(group_df) < 2:
            continue
        
        # Skip d=20 as requested
        fit_df = group_df[group_df['input_dim'] != 20]
        if len(fit_df) < 2:
            continue
        
        # Get dimensions and mean epochs
        x_values = np.array(fit_df['input_dim'])
        y_values = np.array(fit_df['mean_epochs'])
        
        # Fit power law
        power_law_fit = fit_power_law(x_values, y_values)
        
        if power_law_fit is not None:
            power_law_params.append({
                'function': func,
                'hidden_size': width,
                'depth': depth,
                'batch_size': batch,
                'a': power_law_fit['a'],
                'n': power_law_fit['n'],
                'r_squared': power_law_fit['r_squared'],
                'std_err': power_law_fit['std_err']
            })
    
    return pd.DataFrame(power_law_params)

def create_exponent_depth_plots(df_power_law, output_dir, threshold):
    """
    Create plots showing exponent (n) vs. depth for each function and width.
    Different colors for different widths, and variations of the same color for batch sizes.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique functions and widths
    functions = sorted(df_power_law['function'].unique())
    widths = sorted(df_power_law['hidden_size'].unique())
    
    # Define color scheme for widths
    width_colors = {
        512: 'blue',
        1024: 'green', 
        4096: 'red',
        8192: 'purple'
    }
    
    # Process each function separately
    for function in functions:
        # Filter data for this function
        func_data = df_power_law[df_power_law['function'] == function]
        if len(func_data) == 0:
            continue
            
        # Create a figure
        plt.figure(figsize=(10, 8))
        
        # Process each width
        for width in widths:
            width_data = func_data[func_data['hidden_size'] == width]
            if len(width_data) == 0:
                continue
            
            # Get batch sizes for this width
            batch_sizes = sorted(width_data['batch_size'].unique())
            
            # Create a color array for batch sizes (variations of the base color)
            base_color = width_colors.get(width, 'black')
            
            # For each batch size, create a lighter/darker version of the base color
            for j, batch in enumerate(batch_sizes):
                # Filter data for this batch size
                batch_data = width_data[width_data['batch_size'] == batch]
                
                # For color variations within the same width
                # Smaller batches: lighter colors, Larger batches: darker colors
                if len(batch_sizes) > 1:
                    alpha = 0.4 + 0.6 * (j / (len(batch_sizes) - 1))
                else:
                    alpha = 0.7
                
                # Group by depth to get mean n value for each depth
                depth_n = batch_data.groupby('depth').agg({
                    'n': 'mean',
                    'std_err': 'mean'  # Using the standard error from the fit
                }).reset_index()
                
                # Plot n vs depth with error bars
                plt.errorbar(
                    depth_n['depth'], 
                    depth_n['n'], 
                    yerr=depth_n['std_err'],
                    fmt='o-', 
                    color=base_color,
                    alpha=alpha,
                    linewidth=2, 
                    markersize=8, 
                    label=f"width={width}, batch={batch}"
                )
                
                # Add values as text annotations
                for _, row in depth_n.iterrows():
                    plt.annotate(
                        f"{row['n']:.2f}", 
                        (row['depth'], row['n']), 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=8, 
                        fontweight='bold'
                    )
        
        # Set title and labels
        plt.title(f"{function} - Power Law Exponent vs. Depth", fontsize=16)
        plt.xlabel('Depth', fontsize=14)
        plt.ylabel('Exponent (n)', fontsize=14)
        
        # Set limits and grid
        plt.ylim(0, max(func_data['n']) * 1.1)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Set x-ticks to be integer values
        depths = sorted(func_data['depth'].unique())
        plt.xticks(depths)
        
        # Add legend
        plt.legend(loc='best', fontsize=10)
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{function}_exponent_depth_plot_{threshold}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved exponent vs. depth plot for {function} to {output_path}")
    
    # Create separate plots for each width 
    for width in widths:
        # Filter data for this width
        width_data = df_power_law[df_power_law['hidden_size'] == width]
        if len(width_data) == 0:
            continue
            
        # Create a figure with subplots for each function
        n_functions = len(functions)
        fig, axes = plt.subplots(1, n_functions, figsize=(5*n_functions, 6), sharey=True)
        
        # Handle case of only one function
        if n_functions == 1:
            axes = [axes]
        
        # Process each function
        for i, function in enumerate(functions):
            # Filter data for this function
            func_width_data = width_data[width_data['function'] == function]
            if len(func_width_data) == 0:
                continue
                
            ax = axes[i]
            
            # Get batch sizes
            batch_sizes = sorted(func_width_data['batch_size'].unique())
            
            # Create color map for batch sizes (using colormap)
            cmap = plt.cm.viridis
            batch_colors = cmap(np.linspace(0, 1, len(batch_sizes)))
            
            # For each batch size
            for j, batch in enumerate(batch_sizes):
                # Filter data for this batch size
                batch_data = func_width_data[func_width_data['batch_size'] == batch]
                
                # Group by depth to get mean n value for each depth
                depth_n = batch_data.groupby('depth').agg({
                    'n': 'mean',
                    'std_err': 'mean'
                }).reset_index()
                
                # Plot n vs depth with error bars
                ax.errorbar(
                    depth_n['depth'], 
                    depth_n['n'], 
                    yerr=depth_n['std_err'],
                    fmt='o-', 
                    color=batch_colors[j],
                    linewidth=2, 
                    markersize=8, 
                    label=f"batch={batch}"
                )
                
                # Add values as text annotations
                for _, row in depth_n.iterrows():
                    ax.annotate(
                        f"{row['n']:.2f}", 
                        (row['depth'], row['n']), 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=8, 
                        fontweight='bold'
                    )
            
            # Set title and labels
            ax.set_title(f"{function}", fontsize=14)
            ax.set_xlabel('Depth', fontsize=12)
            if i == 0:
                ax.set_ylabel('Exponent (n)', fontsize=12)
            
            # Set limits and grid
            ax.set_ylim(0, max(width_data['n']) * 1.1)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Set x-ticks to be integer values
            depths = sorted(func_width_data['depth'].unique())
            ax.set_xticks(depths)
        
        # Add legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
                   ncol=min(6, len(handles)), fontsize=10)
        
        # Add main title
        fig.suptitle(f'Power Law Exponents (n) for Width={width} (threshold={threshold})', 
                    fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        output_path = os.path.join(output_dir, f"exponent_depth_width{width}_{threshold}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved exponent vs. depth plot for width={width} to {output_path}")
    
    return output_dir

def plot_scaling_with_dimension(df, output_dir, threshold):
    """
    Create plots showing epochs to threshold vs. dimension with fitted power laws.
    Shows means with standard deviation error bars.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique functions, widths, depths, and batch sizes
    functions = sorted(df['function'].unique())
    widths = sorted(df['hidden_size'].unique())
    depths = sorted(df['depth'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    
    # Process each function separately
    for function in functions:
        # Filter data for this function
        func_data = df[df['function'] == function]
        
        # For each depth, create a separate plot
        for depth in depths:
            depth_data = func_data[func_data['depth'] == depth]
            if len(depth_data) == 0:
                continue
            
            # Create a figure
            plt.figure(figsize=(12, 8))
            
            # Define color scheme for widths
            width_colors = {
                512: 'blue',
                1024: 'green', 
                4096: 'red',
                8192: 'purple'
            }
            
            # Process each width
            legend_entries = []
            
            for width in widths:
                width_data = depth_data[depth_data['hidden_size'] == width]
                if len(width_data) == 0:
                    continue
                
                # Get batch sizes for this width
                width_batch_sizes = sorted(width_data['batch_size'].unique())
                
                # Base color for this width
                base_color = width_colors.get(width, 'black')
                
                # For each batch size
                for j, batch in enumerate(width_batch_sizes):
                    batch_data = width_data[width_data['batch_size'] == batch]
                    
                    # Skip if less than 2 dimensions (needed for fitting)
                    if len(batch_data) < 2:
                        continue
                    
                    # Skip d=20 for fitting
                    fit_data = batch_data[batch_data['input_dim'] != 20]
                    if len(fit_data) < 2:
                        continue
                    
                    # For color variations within the same width
                    if len(width_batch_sizes) > 1:
                        alpha = 0.4 + 0.6 * (j / (len(width_batch_sizes) - 1))
                    else:
                        alpha = 0.7
                    
                    # Get dimensions and mean epochs
                    x_values = np.array(batch_data['input_dim'])
                    y_values = np.array(batch_data['mean_epochs'])
                    y_errors = np.array(batch_data['std_epochs'])
                    
                    # Sort by x_values
                    sort_idx = np.argsort(x_values)
                    x_sorted = x_values[sort_idx]
                    y_sorted = y_values[sort_idx]
                    y_errors_sorted = y_errors[sort_idx]
                    
                    # Plot data points with error bars
                    marker = ['o', 's', '^', 'v', 'x', '*'][j % 6]
                    plt.errorbar(
                        x_sorted, 
                        y_sorted, 
                        yerr=y_errors_sorted,
                        fmt=marker, 
                        color=base_color,
                        alpha=alpha,
                        markersize=8, 
                        label=f"width={width}, batch={batch}"
                    )
                    
                    # Fit power law using only non-d=20 points
                    x_fit = np.array(fit_data['input_dim'])
                    y_fit = np.array(fit_data['mean_epochs'])
                    
                    power_law_fit = fit_power_law(x_fit, y_fit)
                    
                    if power_law_fit is not None:
                        # Generate fitted curve points for plotting
                        x_curve = np.logspace(np.log10(min(x_fit)), np.log10(max(x_fit)), 100)
                        y_curve = power_law_fit['a'] * x_curve ** power_law_fit['n']
                        
                        # Plot fitted curve
                        plt.plot(
                            x_curve, 
                            y_curve, 
                            linestyle='-', 
                            color=base_color,
                            alpha=alpha,
                            linewidth=1.5
                        )
                        
                        # Add to legend
                        legend_entries.append(
                            f"width={width}, batch={batch}: n={power_law_fit['n']:.2f}"
                        )
            
            # Add labels and title
            plt.xlabel('Input Dimension (d)', fontsize=14)
            plt.ylabel(f'Epoch to reach loss < {threshold}', fontsize=14)
            plt.title(f"{function} - Depth {depth} - Power Law Scaling", fontsize=16)
            
            # Add grid and legend
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(legend_entries, loc='best')
            
            # Use log-log scale for better visualization
            plt.xscale('log')
            plt.yscale('log')
            
            # Save the plot
            output_path = os.path.join(output_dir, f"{function}_depth{depth}_scaling_plot_{threshold}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved scaling plot for {function}, depth={depth} to {output_path}")
    
    return output_dir

def create_all_functions_comparison_plot(df_power_law, output_dir, threshold):
    """
    Create a plot comparing exponent vs. depth across all functions.
    Each function gets its own line with a different color.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique widths and batch sizes
    widths = sorted(df_power_law['hidden_size'].unique())
    batch_sizes = sorted(df_power_law['batch_size'].unique())
    
    # For each width + batch combination
    for width in widths:
        # For a small batch and a large batch
        for batch in [min(batch_sizes), max(batch_sizes)]:
            # Filter data
            filtered_data = df_power_law[(df_power_law['hidden_size'] == width) & 
                                        (df_power_law['batch_size'] == batch)]
            
            if len(filtered_data) == 0:
                continue
                
            # Get unique functions
            functions = sorted(filtered_data['function'].unique())
            
            if len(functions) <= 1:
                continue
                
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Color map for functions
            cmap = plt.cm.tab10
            function_colors = cmap(np.linspace(0, 1, len(functions)))
            
            # Plot each function
            for i, function in enumerate(functions):
                func_data = filtered_data[filtered_data['function'] == function]
                
                # Group by depth
                depth_n = func_data.groupby('depth').agg({
                    'n': 'mean',
                    'std_err': 'mean'
                }).reset_index()
                
                if len(depth_n) < 2:
                    continue
                
                # Sort by depth
                depth_n = depth_n.sort_values('depth')
                
                # Plot
                plt.errorbar(
                    depth_n['depth'],
                    depth_n['n'],
                    yerr=depth_n['std_err'],
                    fmt='o-',
                    color=function_colors[i],
                    linewidth=2,
                    markersize=8,
                    label=function
                )
                
                # Add annotations
                for _, row in depth_n.iterrows():
                    plt.annotate(
                        f"{row['n']:.2f}",
                        (row['depth'], row['n']),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        fontweight='bold'
                    )
            
            # Add labels and title
            plt.xlabel('Depth', fontsize=14)
            plt.ylabel('Exponent (n)', fontsize=14)
            plt.title(f'Exponent vs. Depth - Width={width}, Batch={batch}', fontsize=16)
            
            # Add grid and legend
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            
            # Set integer x-ticks
            depths = sorted(filtered_data['depth'].unique())
            plt.xticks(depths)
            
            # Save the figure
            output_path = os.path.join(output_dir, f"all_functions_width{width}_batch{batch}_{threshold}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved function comparison plot for width={width}, batch={batch} to {output_path}")
    
    return output_dir

def create_dimension_epochs_grid(df, output_dir, threshold):
    """
    Create a grid of plots where:
    - Each grid cell is for a (width, batch_size) combination
    - X-axis of each plot is dimension
    - Y-axis of each plot is epochs to threshold
    - Different functions (k values) are shown within each plot
    - Grid is arranged with width increasing left to right (smaller widths on left)
    - Grid is arranged with batch_size increasing top to bottom
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique widths, batch sizes, and functions
    widths = sorted(df['hidden_size'].unique())  # Sorted normally (smaller on left)
    batch_sizes = sorted(df['batch_size'].unique())
    functions = sorted(df['function'].unique())
    depths = sorted(df['depth'].unique())
    
    # For each depth value, create a separate grid plot
    for depth in depths:
        # Filter data for this depth
        depth_data = df[df['depth'] == depth]
        if len(depth_data) == 0:
            continue
        
        # Create a grid of subplots
        n_rows = len(batch_sizes)
        n_cols = len(widths)
        
        # Skip if we don't have enough data
        if n_rows == 0 or n_cols == 0:
            print(f"Not enough data for grid plot at depth={depth}")
            continue
        
        # Create figure with appropriate size
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), 
                                 sharex=True, sharey=True)
        
        # Handle the case where we have only one row or column
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Color map for different functions
        cmap = plt.cm.tab10
        function_colors = {func: cmap(i % 10) for i, func in enumerate(functions)}
        
        # For each cell in the grid
        for row, batch in enumerate(batch_sizes):
            for col, width in enumerate(widths):
                # Get the corresponding axis
                ax = axes[row, col]
                
                # Filter data for this batch and width
                cell_data = depth_data[(depth_data['batch_size'] == batch) & 
                                      (depth_data['hidden_size'] == width)]
                
                # Skip if no data
                if len(cell_data) == 0:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.set_title(f"Width={width}, Batch={batch}")
                    continue
                
                # Plot each function
                for function in functions:
                    func_data = cell_data[cell_data['function'] == function]
                    
                    if len(func_data) == 0:
                        continue
                    
                    # Sort by dimension
                    func_data = func_data.sort_values('input_dim')
                    
                    # Get dimensions and mean epochs for plotting
                    x_values = np.array(func_data['input_dim'])
                    y_values = np.array(func_data['mean_epochs'])
                    y_errors = np.array(func_data['std_epochs'])
                    
                    # Plot
                    ax.errorbar(
                        x_values,
                        y_values,
                        yerr=y_errors,
                        fmt='o-',
                        color=function_colors[function],
                        label=function,
                        alpha=0.8
                    )
                    
                    # Fit power law (excluding d=20 if present)
                    fit_data = func_data[func_data['input_dim'] != 20]
                    if len(fit_data) >= 2:  # Need at least 2 points for fitting
                        x_fit = np.array(fit_data['input_dim'])
                        y_fit = np.array(fit_data['mean_epochs'])
                        
                        power_law_fit = fit_power_law(x_fit, y_fit)
                        
                        if power_law_fit is not None:
                            # Add exponent value as text
                            y_pos = max(y_values) * 1.1  # Position above the highest point
                            ax.text(
                                min(x_values) * 1.2,  # Position near left side
                                y_pos,
                                f"n = {power_law_fit['n']:.2f}",
                                color=function_colors[function],
                                fontweight='bold',
                                fontsize=9
                            )
                
                # Set title and labels
                ax.set_title(f"Width={width}, Batch={batch}")
                ax.set_xscale('log')
                ax.set_yscale('log')
                
                # Only add x-label for bottom row
                if row == n_rows - 1:
                    ax.set_xlabel('Dimension (d)')
                
                # Only add y-label for leftmost column
                if col == 0:
                    ax.set_ylabel('Epochs to threshold')
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a common legend
        # Find a subplot that has data to get legend handles
        handles = []
        labels = []
        for i in range(n_rows):
            for j in range(n_cols):
                h, l = axes[i, j].get_legend_handles_labels()
                if h:  # If this subplot has any legend handles
                    handles = h
                    labels = l
                    break
            if handles:  # If we found handles, break out of the outer loop
                break
        
        # If we found any legend handles
        if handles:
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), 
                      loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(5, len(by_label)))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Add overall title
        fig.suptitle(f'Dimension vs. Epochs - Depth={depth} (threshold={threshold})', 
                     fontsize=16, y=0.98)
        
        # Save the figure
        output_path = os.path.join(output_dir, f"dimension_epochs_grid_depth{depth}_{threshold}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved dimension vs. epochs grid plot for depth={depth} to {output_path}")
    
    return output_dir

def main():
    # Hard-coded parameters - modify as needed
    results_dir = "/home/goring/OnlineSGD/results_MSP/23_04_monomial_mup_biggrid_d2/complex_leap_exp_20250423_015524"
    output_dir = "/home/goring/OnlineSGD/plots/23_04_plots_monomial_mup"
    threshold = 0.3  # Loss threshold from config
    
    # Load all results
    results = load_results(results_dir)
    print(f"Loaded results for {len(results)} experiments")
    
    # Group results by function, batch, width, depth, and input_dim
    grouped_results = group_results_by_function_batch_width_depth(results)
    print(f"Found {len(grouped_results)} different parameter combinations")
    
    # Calculate mean epochs to threshold for each group
    df_epochs = calculate_mean_epochs_to_threshold(grouped_results, threshold)
    print(f"Calculated statistics for {len(df_epochs)} parameter combinations")
    
    # Fit power laws for each group
    df_power_law = fit_power_laws_by_group(df_epochs)
    print(f"Fitted power laws for {len(df_power_law)} parameter combinations")
    
    # Create detailed plots
    create_exponent_depth_plots(df_power_law, output_dir, threshold)
    
    # Create scaling plots
    plot_scaling_with_dimension(df_epochs, output_dir, threshold)
    
    # Create function comparison plots
    create_all_functions_comparison_plot(df_power_law, output_dir, threshold)
    
    # Create dimension vs epochs grid plot
    create_dimension_epochs_grid(df_epochs, output_dir, threshold)
    
    # Save power law parameters to CSV for further analysis
    power_law_csv = os.path.join(output_dir, f"power_law_params_{threshold}.csv")
    df_power_law.to_csv(power_law_csv, index=False)
    print(f"Saved power law parameters to {power_law_csv}")
    
    # Save epochs data to CSV as well
    epochs_csv = os.path.join(output_dir, f"epochs_to_threshold_{threshold}.csv")
    df_epochs.to_csv(epochs_csv, index=False)
    print(f"Saved epochs to threshold data to {epochs_csv}")
    
    print("All analysis and plots completed")

if __name__ == "__main__":
    main()