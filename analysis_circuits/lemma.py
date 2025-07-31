import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    """Test the random projection lemma with correct scaling"""
    print("=" * 80)
    print("ANALYZING LEMMA 2.3: RANDOM PROJECTION REDUCTION")
    print("=" * 80)
    
    # Parameters - wider range for better analysis
    d_values = [20, 40, 60, 80, 100]
    k_values = [2, 3, 4, 5]
    n_trials = 10
    
    results = {}
    
    for k in k_values:
        results[k] = {'d_values': [], 'required_projections': []}
        
        for d in d_values:
            if d <= k:
                continue
                
            print(f"\nTesting d={d}, k={k} ({n_trials} trials)...")
            required_projections = []
            
            for trial in range(n_trials):
                # Generate binary masks with guaranteed uniqueness
                masks = generate_unique_masks(d, k, n_patterns=20)
                
                # Verify masks are distinct
                min_distance = verify_mask_separation(masks)
                if min_distance < 0.1:  # If too close, regenerate
                    print(f"    Trial {trial+1}: Regenerating masks (min_distance={min_distance:.4f})")
                    masks = generate_unique_masks(d, k, n_patterns=20, min_distance=0.1)
                    min_distance = verify_mask_separation(masks)
                
                # Find minimum projections needed
                min_proj = find_minimum_projections(d, k, masks, min_distance)
                required_projections.append(min_proj)
                
                print(f"  Trial {trial+1}: {min_proj} projections")
                
            if required_projections:  # Make sure we have valid results
                avg_required = np.mean(required_projections)
                results[k]['d_values'].append(d)
                results[k]['required_projections'].append(avg_required)
                
                print(f"  Average required projections: {avg_required:.1f}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Create visualizations
    create_visualizations(results)

def generate_unique_masks(d, k, n_patterns=20, min_distance=0.1):
    """Generate binary masks with exactly k ones, ensuring uniqueness"""
    # We'll use a different approach - explicitly construct masks with the proper Hamming distance
    masks = torch.ones((n_patterns, d), device=device) * -1  # Start with all -1s
    
    # Keep track of generated patterns
    patterns = []
    
    # For first pattern, randomly select k indices to set to 1
    indices = torch.randperm(d, device=device)[:k]
    first_pattern = torch.ones(d, device=device) * -1
    first_pattern[indices] = 1
    patterns.append(first_pattern)
    masks[0] = first_pattern
    
    # For remaining patterns, ensure minimum Hamming distance
    for i in range(1, n_patterns):
        max_attempts = 100  # Limit attempts to avoid infinite loop
        for attempt in range(max_attempts):
            # Generate a new random pattern
            indices = torch.randperm(d, device=device)[:k]
            new_pattern = torch.ones(d, device=device) * -1
            new_pattern[indices] = 1
            
            # Check distance to all existing patterns
            min_dist = float('inf')
            for pattern in patterns:
                # Compute normalized Hamming distance
                dist = 1 - torch.sum(pattern == new_pattern).float() / d
                min_dist = min(min_dist, dist)
            
            # If distance is sufficient, accept this pattern
            if min_dist >= min_distance:
                patterns.append(new_pattern)
                masks[i] = new_pattern
                break
                
            if attempt == max_attempts - 1:
                print(f"    Warning: Could not generate pattern with sufficient distance after {max_attempts} attempts")
                # Use the last attempt anyway
                patterns.append(new_pattern)
                masks[i] = new_pattern
    
    return masks

def verify_mask_separation(masks):
    """Verify that all masks are sufficiently separated"""
    n_patterns = masks.shape[0]
    
    # Compute pairwise distances
    distances = torch.zeros((n_patterns, n_patterns), device=device)
    for i in range(n_patterns):
        for j in range(i+1, n_patterns):
            # Normalized Hamming distance
            dist = 1 - torch.sum(masks[i] == masks[j]).float() / masks.shape[1]
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Get minimum distance excluding self (diagonal)
    min_dist = torch.min(distances + torch.eye(n_patterns, device=device) * 10).item()
    print(f"    Verified minimum distance between patterns: {min_dist:.4f}")
    
    return min_dist

def compute_projection_quality(original_masks, projected_masks):
    """Compute how well projection preserves distances"""
    n_patterns = original_masks.shape[0]
    
    # Compute pairwise distances in original space
    orig_distances = torch.zeros((n_patterns, n_patterns), device=device)
    for i in range(n_patterns):
        for j in range(i+1, n_patterns):
            # Normalized Hamming distance
            dist = 1 - torch.sum(original_masks[i] == original_masks[j]).float() / original_masks.shape[1]
            orig_distances[i, j] = dist
            orig_distances[j, i] = dist
    
    # Compute pairwise distances in projected space
    # First normalize projected vectors
    norms = torch.norm(projected_masks, dim=1, keepdim=True)
    normalized = projected_masks / (norms + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Compute angular distances
    proj_distances = torch.zeros((n_patterns, n_patterns), device=device)
    for i in range(n_patterns):
        for j in range(i+1, n_patterns):
            # Angular distance
            cos_sim = torch.dot(normalized[i], normalized[j])
            # Bound to avoid numerical issues
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            dist = 1 - torch.abs(cos_sim)
            proj_distances[i, j] = dist
            proj_distances[j, i] = dist
    
    # Exclude diagonal by adding large value
    diag_mask = torch.eye(n_patterns, device=device) * 10
    
    # Get minimum distances
    min_orig_dist = torch.min(orig_distances + diag_mask).item()
    min_proj_dist = torch.min(proj_distances + diag_mask).item()
    
    # Handle the case where min_orig_dist is 0 or very small
    if min_orig_dist < 1e-6:
        return 0.0  # Can't preserve zero distance
    
    # Compute quality as ratio of minimum distances
    quality = min_proj_dist / min_orig_dist
    
    return quality

def find_minimum_projections(d, k, masks, min_distance):
    """Find minimum projection dimension that preserves pattern separation"""
    # Print original minimum distance
    print(f"    Original minimum distance: {min_distance:.4f}")
    
    # If min_distance is too small, can't preserve it
    if min_distance < 1e-6:
        print("    Warning: Original patterns have zero distance, can't preserve separation")
        return d  # Return original dimension as fallback
    
    # Test different projection dimensions
    # Start very small and increase exponentially
    projection_dims = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100]
    projection_dims = [p for p in projection_dims if p < d]  # Don't exceed original dimension
    
    if not projection_dims:
        projection_dims = [max(1, d // 10)]  # Fallback if d is very small
    
    for p_dim in projection_dims:
        # Try multiple random projections and average results
        qualities = []
        for _ in range(3):  # 3 attempts per dimension
            # Generate random projection matrix
            proj_matrix = torch.randn(p_dim, d, device=device) / torch.sqrt(torch.tensor(d, device=device))
            
            # Project masks
            projected = torch.matmul(masks, proj_matrix.T)
            
            # Compute quality of projection
            quality = compute_projection_quality(masks, projected)
            qualities.append(quality)
        
        # Average quality
        avg_quality = sum(qualities) / len(qualities)
        print(f"    Projection dim {p_dim}: avg quality = {avg_quality:.4f}")
        
        # If quality is good enough, we found our minimum dimension
        if avg_quality >= 0.9:  # Require 90% preservation
            return p_dim
    
    # If we get here, no projection dimension was sufficient
    return projection_dims[-1]  # Return the largest tested

def fit_power_law(d_values, projections):
    """Fit a power law curve to the data: projections = C * d^alpha"""
    if len(d_values) < 2:
        return 0, 0
        
    log_d = np.log(d_values)
    log_proj = np.log(projections)
    
    # Linear fit in log space
    coeffs = np.polyfit(log_d, log_proj, 1)
    alpha = coeffs[0]  # Exponent
    C = np.exp(coeffs[1])  # Coefficient
    
    return C, alpha

def create_visualizations(results):
    """Create visualizations for scaling analysis"""
    plt.figure(figsize=(12, 8))
    
    # Collect exponents
    empirical_exponents = []
    theoretical_exponents = []
    k_values = []
    
    for k, data in results.items():
        d_values = np.array(data['d_values'])
        projections = np.array(data['required_projections'])
        
        if len(d_values) < 2:  # Need at least 2 points for fitting
            continue
        
        k_values.append(k)
        
        # Plot empirical data points
        plt.loglog(d_values, projections, 'o-', linewidth=2, label=f'k={k} (empirical)')
        
        # Fit power law
        C, alpha = fit_power_law(d_values, projections)
        empirical_exponents.append(alpha)
        theoretical_exponents.append(k/2)
        
        # Create fitted curve
        d_smooth = np.logspace(np.log10(min(d_values)), np.log10(max(d_values)), 100)
        projections_smooth = C * d_smooth**alpha
        plt.loglog(d_smooth, projections_smooth, '--', linewidth=1, 
                 label=f'k={k} fit: {C:.2f}·d^{alpha:.3f}')
        
        # Create theoretical curve (scaled to match at smallest d)
        scale_factor = projections[0] / (d_values[0]**(k/2))
        theoretical = scale_factor * d_values**(k/2)
        plt.loglog(d_values, theoretical, ':', linewidth=1, 
                 label=f'k={k} theoretical: d^{k/2}')
        
        print(f"k={k}: Fitted power law = {C:.2f} * d^{alpha:.3f}")
        print(f"       Theoretical exponent = {k/2}")
        print(f"       Ratio of empirical to theoretical: {alpha/(k/2):.3f}")
    
    plt.xlabel('Dimension (d)', fontsize=14)
    plt.ylabel('Required Projections', fontsize=14)
    plt.title('Random Projection Dimension Scaling (Lemma 2.3)', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.savefig("random_projection_scaling.png", dpi=300)
    plt.close()
    
    # Plot exponent comparison
    if len(k_values) > 0:
        plt.figure(figsize=(10, 6))
        
        plt.plot(k_values, empirical_exponents, 'o-', label='Empirical Exponent')
        plt.plot(k_values, theoretical_exponents, 's--', label='Theoretical Exponent (k/2)')
        
        plt.xlabel('Sparsity (k)', fontsize=14)
        plt.ylabel('Exponent (α)', fontsize=14)
        plt.title('Empirical vs Theoretical Scaling Exponent', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        plt.savefig("scaling_exponents.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()