#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import glob

def load_results(results_dir, force_reload=False):
    """Load results from a directory - either from summary file or by creating it from .npy files"""
    summary_file = os.path.join(results_dir, "lr_search_summary.json")
    
    # For partial results, always reload from npy files
    if not os.path.exists(summary_file) or force_reload:
        npy_files = glob.glob(os.path.join(results_dir, "lr_result_*.npy"))
        if npy_files:
            print(f"Creating summary from {len(npy_files)} .npy files...")
            data = create_summary_from_npy_files(results_dir)
            
            # Don't save the partial summary to avoid overwriting later
            return data, results_dir
        else:
            raise FileNotFoundError(f"No .npy files found in {results_dir}")
    else:
        with open(summary_file, 'r') as f:
            data = json.load(f)
        return data, results_dir

def create_summary_from_npy_files(results_dir):
    """Create a summary data structure from individual .npy files"""
    all_results = []
    
    # Load all result files
    for file in os.listdir(results_dir):
        if file.startswith("lr_result_") and file.endswith(".npy"):
            try:
                result = np.load(os.path.join(results_dir, file), allow_pickle=True).item()
                all_results.append(result)
                print(f"Loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    if not all_results:
        raise ValueError("No valid result files found")
    
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
            'search_info': result.get('search_info', {})
        })
    
    # Calculate statistics
    for key, config_data in summary['configurations'].items():
        optimal_lrs = [exp['optimal_lr'] for exp in config_data['experiments']]
        config_data['mean_optimal_lr'] = np.mean(optimal_lrs)
        config_data['std_optimal_lr'] = np.std(optimal_lrs) if len(optimal_lrs) > 1 else 0.0
    
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
            'std': np.std(values) if len(values) > 1 else 0.0,
            'count': len(values)
        }
    
    for width, values in summary['statistics']['mean_optimal_lr_by_width'].items():
        summary['statistics']['mean_optimal_lr_by_width'][width] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
            'count': len(values)
        }
    
    for depth, values in summary['statistics']['mean_optimal_lr_by_depth'].items():
        summary['statistics']['mean_optimal_lr_by_depth'][depth] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
            'count': len(values)
        }
    
    for mode, values in summary['statistics']['mean_optimal_lr_by_mode'].items():
        summary['statistics']['mean_optimal_lr_by_mode'][mode] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0.0,
            'count': len(values)
        }
    
    return summary

def extract_configuration_data(summary_data):
    """Extract and organize data from the summary for easier plotting"""
    organized_data = {}
    
    for config_key, config_data in summary_data['configurations'].items():
        # Parse the configuration key
        parts = config_key.split('_')
        
        # Extract values from the key
        k = int(parts[0][1:])  # k3 -> 3
        d = int(parts[1][1:])  # d20 -> 20
        h = int(parts[2][1:])  # h512 -> 512
        depth = int(parts[3][5:])  # depth1 -> 1
        mode = parts[4]  # standard or mup
        batch_size = int(parts[5][1:])  # b512 -> 512
        
        # Create nested structure: mode -> function (k,d) -> batch_size -> (width, depth) -> lr
        function_key = f"k{k}_d{d}"
        
        if mode not in organized_data:
            organized_data[mode] = {}
        if function_key not in organized_data[mode]:
            organized_data[mode][function_key] = {}
        if batch_size not in organized_data[mode][function_key]:
            organized_data[mode][function_key][batch_size] = {}
        
        # Store the mean optimal LR
        organized_data[mode][function_key][batch_size][(h, depth)] = config_data['mean_optimal_lr']
    
    return organized_data

def create_visualization(organized_data, save_path):
    """Create the visualization with separate grids for standard and mup"""
    # Get unique values
    all_function_keys = set()
    all_batch_sizes = set()
    all_widths = set()
    all_depths = set()
    
    for mode in organized_data:
        for function_key in organized_data[mode]:
            all_function_keys.add(function_key)
            for batch_size in organized_data[mode][function_key]:
                all_batch_sizes.add(batch_size)
                for (h, depth) in organized_data[mode][function_key][batch_size]:
                    all_widths.add(h)
                    all_depths.add(depth)
    
    if not all_function_keys:  # No data to plot
        print("No data found to plot")
        return
    
    all_function_keys = sorted(list(all_function_keys))
    all_batch_sizes = sorted(list(all_batch_sizes))
    all_widths = sorted(list(all_widths))
    all_depths = sorted(list(all_depths))
    
    print(f"Found data for functions: {all_function_keys}, batch_sizes: {all_batch_sizes}, widths: {all_widths}, depths: {all_depths}")
    
    # Create figure for each mode
    for mode in ['standard', 'mup']:
        if mode not in organized_data:
            print(f"No data for {mode} mode")
            continue
        
        # Create grid
        n_functions = len(all_function_keys)
        n_batch_sizes = len(all_batch_sizes)
        
        fig = plt.figure(figsize=(5 * n_batch_sizes, 4 * n_functions))
        gs = GridSpec(n_functions, n_batch_sizes, figure=fig)
        fig.suptitle(f'Maximum Learning Rate Heatmaps - {mode.upper()} Mode', fontsize=16, y=0.95)
        
        for i, function_key in enumerate(all_function_keys):
            for j, batch_size in enumerate(all_batch_sizes):
                ax = fig.add_subplot(gs[i, j])
                
                # Create matrix for heatmap
                lr_matrix = np.full((len(all_depths), len(all_widths)), np.nan)
                
                if (function_key in organized_data[mode] and 
                    batch_size in organized_data[mode][function_key]):
                    for (h, depth), lr in organized_data[mode][function_key][batch_size].items():
                        width_idx = all_widths.index(h)
                        depth_idx = all_depths.index(depth)
                        lr_matrix[depth_idx, width_idx] = lr
                
                # Create heatmap
                if not np.all(np.isnan(lr_matrix)):
                    im = ax.imshow(lr_matrix, aspect='auto', origin='lower', 
                                  cmap='viridis', interpolation='nearest')
                    
                    # Set ticks
                    ax.set_xticks(range(len(all_widths)))
                    ax.set_yticks(range(len(all_depths)))
                    ax.set_xticklabels(all_widths, rotation=45)
                    ax.set_yticklabels(all_depths)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Max LR', rotation=270, labelpad=15)
                else:
                    # If no data, add text
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # Labels
                ax.set_xlabel('Width')
                ax.set_ylabel('Depth')
                ax.set_title(f'{function_key}, batch_size={batch_size}')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, f'lr_heatmap_grid_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create individual plots for each function and batch size for detailed view
        for function_key in all_function_keys:
            if function_key not in organized_data[mode]:
                continue
                
            for batch_size in all_batch_sizes:
                if batch_size not in organized_data[mode][function_key]:
                    continue
                
                fig_individual = plt.figure(figsize=(8, 6))
                ax = fig_individual.add_subplot(111)
                
                # Create matrix for heatmap
                lr_matrix = np.full((len(all_depths), len(all_widths)), np.nan)
                
                for (h, depth), lr in organized_data[mode][function_key][batch_size].items():
                    width_idx = all_widths.index(h)
                    depth_idx = all_depths.index(depth)
                    lr_matrix[depth_idx, width_idx] = lr
                
                if not np.all(np.isnan(lr_matrix)):
                    im = ax.imshow(lr_matrix, aspect='auto', origin='lower', 
                                   cmap='viridis', interpolation='nearest')
                    
                    # Add text annotations
                    for depth_idx in range(len(all_depths)):
                        for width_idx in range(len(all_widths)):
                            if not np.isnan(lr_matrix[depth_idx, width_idx]):
                                text = ax.text(width_idx, depth_idx, 
                                               f'{lr_matrix[depth_idx, width_idx]:.1e}',
                                               ha="center", va="center", color="white",
                                               fontsize=8)
                    
                    ax.set_xticks(range(len(all_widths)))
                    ax.set_yticks(range(len(all_depths)))
                    ax.set_xticklabels(all_widths)
                    ax.set_yticklabels(all_depths)
                    
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Maximum Learning Rate', rotation=270, labelpad=15)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                
                ax.set_xlabel('Width')
                ax.set_ylabel('Depth')
                ax.set_title(f'Max Learning Rate - {mode.upper()}, {function_key}, batch_size={batch_size}')
                
                plt.tight_layout()
                individual_dir = os.path.join(save_path, 'individual_plots')
                os.makedirs(individual_dir, exist_ok=True)
                plt.savefig(os.path.join(individual_dir, f'lr_heatmap_{mode}_{function_key}_b{batch_size}.png'), 
                            dpi=300, bbox_inches='tight')
                plt.close()

def create_trend_plots(summary_data, save_path):
    """Create trend plots showing how optimal LR varies with width, depth, batch size"""
    # Extract statistics
    stats = summary_data['statistics']
    
    # Create figure for trends (only those that have data)
    n_plots = sum(1 for key in ['mean_optimal_lr_by_k', 'mean_optimal_lr_by_width', 
                               'mean_optimal_lr_by_depth', 'mean_optimal_lr_by_mode'] 
                  if stats.get(key))
    
    if n_plots == 0:
        print("No trend data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    fig.suptitle('Learning Rate Trends', fontsize=16, y=0.95)
    
    plot_idx = 0
    
    # Plot by k
    if 'mean_optimal_lr_by_k' in stats and stats['mean_optimal_lr_by_k']:
        ax = axes[plot_idx]
        k_data = stats['mean_optimal_lr_by_k']
        ks = []
        means = []
        stds = []
        
        for k, data in k_data.items():
            ks.append(k)
            means.append(data['mean'])
            stds.append(data['std'])
        
        ax.errorbar(ks, means, yerr=stds, fmt='o-', capsize=5)
        ax.set_xlabel('k (Parity Function)')
        ax.set_ylabel('Optimal Learning Rate')
        ax.set_title('Optimal LR by Parity Function')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot by width
    if 'mean_optimal_lr_by_width' in stats and stats['mean_optimal_lr_by_width']:
        ax = axes[plot_idx]
        width_data = stats['mean_optimal_lr_by_width']
        widths = []
        means = []
        stds = []
        
        for width, data in width_data.items():
            widths.append(width)
            means.append(data['mean'])
            stds.append(data['std'])
        
        # Sort by width
        sorted_data = sorted(zip(widths, means, stds))
        widths, means, stds = zip(*sorted_data)
        
        ax.errorbar(widths, means, yerr=stds, fmt='o-', capsize=5)
        ax.set_xlabel('Width')
        ax.set_ylabel('Optimal Learning Rate')
        ax.set_title('Optimal LR by Network Width')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot by depth
    if 'mean_optimal_lr_by_depth' in stats and stats['mean_optimal_lr_by_depth']:
        ax = axes[plot_idx]
        depth_data = stats['mean_optimal_lr_by_depth']
        depths = []
        means = []
        stds = []
        
        for depth, data in depth_data.items():
            depths.append(depth)
            means.append(data['mean'])
            stds.append(data['std'])
        
        # Sort by depth
        sorted_data = sorted(zip(depths, means, stds))
        depths, means, stds = zip(*sorted_data)
        
        ax.errorbar(depths, means, yerr=stds, fmt='o-', capsize=5)
        ax.set_xlabel('Depth')
        ax.set_ylabel('Optimal Learning Rate')
        ax.set_title('Optimal LR by Network Depth')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot by mode
    if 'mean_optimal_lr_by_mode' in stats and stats['mean_optimal_lr_by_mode']:
        ax = axes[plot_idx]
        mode_data = stats['mean_optimal_lr_by_mode']
        modes = []
        means = []
        stds = []
        
        for mode, data in mode_data.items():
            modes.append(mode)
            means.append(data['mean'])
            stds.append(data['std'])
        
        ax.bar(modes, means, yerr=stds, capsize=5)
        ax.set_ylabel('Optimal Learning Rate')
        ax.set_title('Optimal LR by Parameterization Mode')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for i in range(plot_idx + 1, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'lr_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set your paths directly here
    results_dir = '/home/goring/OnlineSGD/results_MSP/21_04_lr_finder_standard/lr_search_20250422_175544'
    output_dir = None  # Set to None to use the same directory as results
    
    # Load the results (force reload from npy files for partial results)
    try:
        summary_data, used_dir = load_results(results_dir, force_reload=True)
        print(f"Loaded {len(summary_data['configurations'])} configurations from: {used_dir}")
    except Exception as e:
        print(f"Error loading results: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(used_dir, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and organize data
    organized_data = extract_configuration_data(summary_data)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualization(organized_data, output_dir)
    create_trend_plots(summary_data, output_dir)
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()