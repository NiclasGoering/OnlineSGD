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
    
    # Print a sample of filenames to check if mon9, mon10 files exist
    mon9_files = [f for f in npz_files if 'mon9' in f]
    mon10_files = [f for f in npz_files if 'mon10' in f]
    print(f"Found {len(mon9_files)} mon9 files and {len(mon10_files)} mon10 files")
    
    for npz_file in npz_files:
        try:
            with np.load(npz_file, allow_pickle=True) as data:
                # Extract metadata
                metadata_str = str(data['metadata'][0])
                
                # Define nan in the local scope before eval
                nan = float('nan')
                
                # Print metadata string for mon9/mon10 files to debug
                if 'mon9' in npz_file or 'mon10' in npz_file:
                    print(f"Loading file: {npz_file}")
                    print(f"Metadata: {metadata_str[:100]}...")  # Print first 100 chars
                
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
            error_count += 1
            error_files.append(npz_file)
            print(f"Error loading {npz_file}: {str(e)}")
    
    print(f"Successfully loaded {len(results)} files, {error_count} files had errors")
    if error_count > 0:
        print(f"First 5 error files: {error_files[:5]}")
    
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

# Function to find the epoch when loss first drops below threshold
def find_epoch_below_threshold(result_data, threshold=0.08):
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

def create_simple_grid_plot(df, output_dir, threshold):
    """
    Create a simple grid plot showing n and a values vs depth for each function.
    Functions are arranged side by side in a grid.
    Different widths (hidden_size) are shown in different colors.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.gridspec import GridSpec
    
    # Get unique functions
    functions = sorted(df['function'].unique())
    n_functions = len(functions)
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(n_functions*4, 8))
    gs = GridSpec(2, n_functions, figure=fig)
    
    # Get all unique hidden sizes (widths)
    all_widths = sorted(df['hidden_size'].unique())
    
    # Create colormap for different widths
    width_colors = plt.cm.tab10(np.linspace(0, 1, len(all_widths)))
    width_color_map = {width: width_colors[i] for i, width in enumerate(all_widths)}
    
    # Create a plot for each function
    for i, function in enumerate(functions):
        # Filter data for this function
        func_data = df[df['function'] == function]
        
        # Top row: n values vs depth
        ax1 = fig.add_subplot(gs[0, i])
        
        # Plot for each width
        for j, width in enumerate(all_widths):
            # Filter data for this width
            width_data = func_data[func_data['hidden_size'] == width]
            
            if len(width_data) > 0:
                # Group by depth and calculate mean n value for this width
                depth_n = width_data.groupby('depth')['n'].mean().reset_index()
                
                # Plot n vs depth for this width
                ax1.plot(depth_n['depth'], depth_n['n'], 'o-', 
                         color=width_color_map[width], 
                         linewidth=2, markersize=8, 
                         label=f"width={width}")
                
                # Add individual data points
                depths = sorted(width_data['depth'].unique())
                for depth in depths:
                    depth_width_data = width_data[width_data['depth'] == depth]
                    # Add small jitter to x for better visibility
                    jitter = np.random.normal(0, 0.05, size=len(depth_width_data))
                    ax1.scatter(depth_width_data['depth'] + jitter, depth_width_data['n'], 
                             alpha=0.6, color=width_color_map[width], edgecolor='black')
        
        # Set title and labels
        ax1.set_title(f"{function}", fontsize=14)
        if i == 0:
            ax1.set_ylabel('Exponent (n)', fontsize=12)
        ax1.set_xticks(sorted(func_data['depth'].unique()))
        ax1.set_ylim(0, max(df['n']) * 1.1)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Add legend only for the first function to avoid duplicates
        if i == 0:
            ax1.legend(title="Hidden Size", loc='best')
            
        # Bottom row: a values vs depth
        ax2 = fig.add_subplot(gs[1, i])
        
        # Plot for each width
        for j, width in enumerate(all_widths):
            # Filter data for this width
            width_data = func_data[func_data['hidden_size'] == width]
            
            if len(width_data) > 0:
                # Group by depth and calculate mean a value for this width
                depth_a = width_data.groupby('depth')['a'].mean().reset_index()
                
                # Plot a vs depth for this width
                ax2.plot(depth_a['depth'], depth_a['a'], 'o-', 
                         color=width_color_map[width], 
                         linewidth=2, markersize=8, 
                         label=f"width={width}")
                
                # Add individual data points
                depths = sorted(width_data['depth'].unique())
                for depth in depths:
                    depth_width_data = width_data[width_data['depth'] == depth]
                    # Add small jitter to x for better visibility
                    jitter = np.random.normal(0, 0.05, size=len(depth_width_data))
                    ax2.scatter(depth_width_data['depth'] + jitter, depth_width_data['a'], 
                             alpha=0.6, color=width_color_map[width], edgecolor='black')
        
        # Set labels
        if i == 0:
            ax2.set_ylabel('Coefficient (a)', fontsize=12)
        ax2.set_xlabel('Depth', fontsize=12)
        ax2.set_xticks(sorted(func_data['depth'].unique()))
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Add legend only for the first function to avoid duplicates
        if i == 0:
            ax2.legend(title="Hidden Size", loc='best')
    
    # Add main title
    fig.suptitle(f'Power Law Parameters by Function and Depth (threshold={threshold})', 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"simple_grid_power_law_params_{threshold}.png")
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
        
        # Plot for each width
        for j, width in enumerate(all_widths):
            # Filter data for this width
            width_data = func_data[func_data['hidden_size'] == width]
            
            if len(width_data) > 0:
                # Group by depth and calculate mean n value for this width
                depth_n = width_data.groupby('depth')['n'].mean().reset_index()
                
                # Plot n vs depth for this width
                ax.plot(depth_n['depth'], depth_n['n'], 'o-', 
                       color=width_color_map[width], 
                       linewidth=2, markersize=8, 
                       label=f"width={width}")
                
                # Add individual data points
                depths = sorted(width_data['depth'].unique())
                for depth in depths:
                    depth_width_data = width_data[width_data['depth'] == depth]
                    # Add small jitter to x for better visibility
                    jitter = np.random.normal(0, 0.05, size=len(depth_width_data))
                    ax.scatter(depth_width_data['depth'] + jitter, depth_width_data['n'], 
                             alpha=0.6, color=width_color_map[width], edgecolor='black')
                
                # Add value annotations for each point
                for _, row in depth_n.iterrows():
                    ax.annotate(f"{row['n']:.2f}", (row['depth'], row['n']), 
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', fontsize=8, fontweight='bold')
        
        # Set title and labels
        ax.set_title(f"{function}", fontsize=14)
        if i == 0:
            ax.set_ylabel('Exponent (n)', fontsize=12)
        ax.set_xlabel('Depth', fontsize=12)
        ax.set_xticks(sorted(func_data['depth'].unique()))
        ax.set_ylim(0, max(df['n']) * 1.1)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add legend only for the first function to avoid duplicates
        if i == 0:
            ax.legend(title="Hidden Size", loc='best')
    
    # Add main title
    fig.suptitle(f'Power Law Exponents (n) by Function and Depth (threshold={threshold})', 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, f"simple_grid_n_values_{threshold}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved simple grid plot of n values to {output_path}")
    
    return output_path

def plot_epochs_to_threshold_with_power_law(results, output_dir, threshold=0.08):
    """
    Create plots for each function showing epochs to reach threshold vs dimension.
    Each function gets its own plot, stacked vertically.
    Different network depths are shown in different blue shades.
    
    Fits a power law to each series, excluding d=20.
    """
    # Group results by function name, batch size, and mode
    grouped_results = group_results_by_function_batch_mode(results)
    print(f"Found {len(grouped_results)} different function-batch-mode combinations")
    
    # Store power law parameters for all models
    power_law_params = []
    
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
        
        # Create a colormap for depths - using blue shades as requested
        depths = sorted(depth_to_architectures.keys())
        n_depths = len(depths)
        blues = plt.cm.tab10(np.linspace(0.4, 0.9, n_depths))
        
        # For each depth
        for depth_idx, depth in enumerate(depths):
            # For each architecture (hidden size) with this depth
            for arch_idx, arch in enumerate(depth_to_architectures[depth]):
                hidden_size, _ = arch
                
                # Collect data points for this architecture
                x_points = []  # dimensions
                y_points = []  # epochs to threshold
                
                for dim in dimensions:
                    # Skip d=20 as requested
                    if dim == 20:
                        continue
                        
                    # For each learning rate (we'll take the best one)
                    min_epoch = float('inf')
                    reached_threshold = False
                    
                    for lr in learning_rates:
                        if dim in dim_to_arch_lr and arch in dim_to_arch_lr[dim] and lr in dim_to_arch_lr[dim][arch]:
                            result_data = dim_to_arch_lr[dim][arch][lr]
                            epoch = find_epoch_below_threshold(result_data, threshold)
                            
                            if epoch is not None:
                                min_epoch = min(min_epoch, epoch)
                                reached_threshold = True
                    
                    if reached_threshold:
                        x_points.append(dim)
                        y_points.append(min_epoch)
                
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
                        
                        # Plot original data
                        plt.plot(x_array, y_array, marker=marker, linestyle='none', color=color, 
                                 label=label, markersize=8)
                        
                        # Plot fitted power law
                        plt.plot(x_fit, y_fit, linestyle='-', color=color, alpha=0.7, linewidth=1.5)
        
        # Add labels and title
        plt.xlabel('Input Dimension (d)', fontsize=14)
        plt.ylabel(f'Epoch to reach loss < {threshold}', fontsize=14)
        plt.title(f"{function_name} - Batch Size {batch_size} - Mode: {mode}", fontsize=16)
        
        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Use log-log scale for better visualization of power laws
        plt.xscale('log')
        plt.yscale('log')
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{function_name}_batch{batch_size}_threshold_{threshold}_power_law.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved power law plot for {function_name} to {output_path}")
    
    # Create dataframe from power law parameters
    df_power_law = pd.DataFrame(power_law_params)
    
    if not df_power_law.empty:
        # Create the simple grid visualization
        create_simple_grid_plot(df_power_law, output_dir, threshold)
    
    return df_power_law

def main():
    # Hard-coded parameters - modify these as need2ed
    results_dir = "/home/goring/OnlineSGD/results_MSP/20_04_monomial_mup_nogc_grid_bs5/complex_leap_exp_20250420_034007"
    output_dir = "/home/goring/OnlineSGD/plots/20_04_plot_monomial_smallbatch"
    threshold = 0.003  # Loss threshold
    
    # Load all results
    results = load_results(results_dir)
    print(f"Loaded results for {len(results)} experiments")
    
    # Plot epochs to threshold with power law fits
    df_power_law = plot_epochs_to_threshold_with_power_law(results, output_dir, threshold)
    
    # Print summary of power law exponents by function
    if not df_power_law.empty:
        print("\nPower Law Exponent (n) Summary by Function:")
        func_summary = df_power_law.groupby('function')['n'].agg(['mean', 'std', 'min', 'max'])
        print(func_summary)
        
        print("\nPower Law Exponent (n) Summary by Depth:")
        depth_summary = df_power_law.groupby('depth')['n'].agg(['mean', 'std', 'min', 'max'])
        print(depth_summary)
    
    print("All analysis and plots completed")

if __name__ == "__main__":
    main()