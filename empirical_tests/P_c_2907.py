import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import os
import json # <-- Added for saving results

# --- 1.  Hyper-parameters & Model Setup ---
# Parameters are set to match the theoretical framework.
d, N, gamma = 40, 1024, 2.0
g_w = g_a = 1.0

# --- CORRECTED VARIANCE DEFINITIONS ---
# Variances are now scaled correctly according to the paper's theory.
sigma_w_sq = g_w / d
# Corrected: The prior variance for 'a' now uses N**gamma as per the paper's action.
sigma_a_sq = g_a / (N**gamma) 
sigma_w = np.sqrt(sigma_w_sq)
sigma_a = np.sqrt(sigma_a_sq)

# --- GPU/CPU Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU found. Using CUDA.")
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU.")

# --- Activation Function Definitions ---
def tanh(x):
    """Odd activation function."""
    return torch.tanh(x)

def relu(x):
    """Even activation function."""
    return torch.nn.functional.relu(x)

# --- 2.  Fast Boolean Sample Bank for Expectations (on GPU) ---
X_bank = (2 * torch.randint(0, 2, size=(200_000, d), device=device) - 1).float()

def Sigma(w, activation_fn):
    """Computes Sigma(w) = E[phi(w^T x)^2] using the static bank."""
    with torch.no_grad():
        z = X_bank @ w
        return torch.mean(activation_fn(z)**2)

def chi(A, X):
    """Computes the parity function chi_A(x) on a matrix of inputs."""
    return torch.prod(X[:, A], axis=1)

def J_A(w, A, activation_fn):
    """Computes J_A(w) = E[phi(w^T x) * chi_A(x)] using the static bank."""
    with torch.no_grad():
        z = X_bank @ w
        return torch.mean(activation_fn(z) * chi(A, X_bank))

# --- 3.  Analytically Integrated Distributions (PyTorch version) ---
def log_p_w(w, m_S, kappa, S, activation_fn):
   
    Sigma_w = Sigma(w, activation_fn)
    J_S_w = J_A(w, S, activation_fn)
    
    kappa_sq = kappa**2
    
    # This is the term inside the log() and the denominator of the original exponent term.
    # Add a small epsilon to kappa_sq for stability when kappa is exactly zero.
    denom_term = (N**gamma / sigma_a_sq) + (Sigma_w / (kappa_sq + 1e-20))
    
    # This is the numerator of the original exponent term's argument.
    J_sigma = J_S_w * (1 - m_S)
    
    # Calculate the exponent term in a stable way.
    # The original form was 0.25 * (J_sigma / kappa**2)**2 / denom_term.
    # This simplifies to the expression below, avoiding kappa**4 issues.
    stable_exp_term = 0.5 * (J_sigma**2 / (kappa_sq + 1e-20)) / denom_term

    log_prior_w = -0.5 * torch.dot(w, w) / sigma_w_sq
    log_det_term = -0.5 * torch.log(denom_term)
    
    return log_prior_w + log_det_term + stable_exp_term

def mh_step_w(w, m_S, kappa, step_size, S, activation_fn):
    """Performs a single Metropolis-Hastings step for w on the GPU."""
    w_proposal = w + step_size * torch.randn(d, device=device)
    log_acceptance_ratio = log_p_w(w_proposal, m_S, kappa, S, activation_fn) - log_p_w(w, m_S, kappa, S, activation_fn)
    
    if torch.log(torch.rand(1, device=device)) < log_acceptance_ratio:
        return w_proposal, True
    return w, False

@torch.no_grad()
def sample_w_then_a(m_S, kappa, n_samp, burn, step, S, activation_fn):
    """
    Samples w from its marginal p(w) via MCMC, then samples a from p(a|w).
    This version is refactored for numerical stability at low kappa.
    """
    w = torch.randn(d, device=device) * sigma_w
    accepted_count = 0
    W_samples, A_samples = [], []

    kappa_sq = kappa**2
    
    # Burn-in phase
    for t in range(burn):
        w, accepted = mh_step_w(w, m_S, kappa, step, S, activation_fn)

    # Sampling phase
    for t in range(n_samp):
        w, accepted = mh_step_w(w, m_S, kappa, step, S, activation_fn)
        if accepted:
            accepted_count += 1
            
        Sigma_w = Sigma(w, activation_fn)
        J_S_w = J_A(w, S, activation_fn)
        J_sigma = J_S_w * (1 - m_S)
        
        # --- Numerically Stable Calculation for mean and variance of 'a' ---
        stable_denom = (N**gamma / sigma_a_sq) * kappa_sq + Sigma_w
        var_a = kappa_sq / (stable_denom + 1e-12)
        mean_a = J_sigma / (stable_denom + 1e-12)
        
        a = torch.randn(1, device=device) * torch.sqrt(var_a) + mean_a
        
        W_samples.append(w.clone())
        A_samples.append(a)
            
    acceptance_rate = accepted_count / n_samp if n_samp > 0 else 0
    return torch.stack(W_samples), torch.stack(A_samples), acceptance_rate

# --- 4.  Self-Consistency Loop ---
@torch.no_grad()
def solve_kappa(kappa, S, activation_fn, m0=0.0, iters=220, damping=0.5):
    """Solves the self-consistency equation for m_S at a given kappa."""
    m_S = m0
    history = [m_S]
    
    # ✅ IMPROVED: Adaptive MCMC Step Size
    step_size = 0.1 # Start with a reasonable guess
    print(f"\n--- Solving for kappa = {kappa:.4e} (starting m_S = {m0:.4f}) ---")

    kappa_sq = kappa**2

    for i in range(iters):
        W_samples, _, acc_rate = sample_w_then_a(m_S, kappa, 4000, 1000, step_size, S, activation_fn)
        
        # Adapt step size to maintain a good acceptance rate (e.g., 20-40%)
        if acc_rate > 0.4 and step_size < 1.0: # Don't let it grow too large
            step_size *= 1.05
        elif acc_rate < 0.2:
            step_size /= 1.05
        
        J_S_w_all = torch.stack([J_A(w, S, activation_fn) for w in W_samples])
        Sigma_w_all = torch.stack([Sigma(w, activation_fn) for w in W_samples])

        J_sigma_all = J_S_w_all * (1 - m_S)
        
        # --- Numerically Stable Calculation for m_S update ---
        stable_denom_all = (N**gamma / sigma_a_sq) * kappa_sq + Sigma_w_all
        mean_a_all = J_sigma_all / (stable_denom_all + 1e-12)
        
        m_S_new = N * torch.mean(mean_a_all * J_S_w_all)
        
        m_S = (1 - damping) * m_S + damping * m_S_new.item()
        history.append(m_S)
        
        print(f"Iter {i+1:3d}: acc={acc_rate:.2f}, step={step_size:.4f}, m_S={m_S:.8f}")

        # --- UNIFIED CONVERGENCE CRITERION ---
        if i >= 100: # Wait for a while before checking
            last_15 = np.array(history[-15:])
            mean_val = np.mean(last_15)
            std_val = np.std(last_15)
            # Converged if standard deviation is tiny, or relative std is small
            is_stable = (std_val < 1e-7) or \
                        (abs(mean_val) > 1e-7 and (std_val / abs(mean_val)) < 0.005)
            
            if is_stable:
                print(f"CONVERGED: Value stabilized at m_S = {mean_val:.8f}")
                m_S = mean_val
                break
    
    # Get a final, high-quality sample set with the converged parameters
    W_final, A_final, _ = sample_w_then_a(m_S, kappa, 8000, 1000, step_size, S, activation_fn)
    return m_S, history, W_final, A_final

# --- 5.  Plotting Functions (Unchanged) ---
def create_and_save_individual_plot(kappa_0, kappa, m_S, history, W, A, S_k, act_name, output_dir):
    """Generates and saves a detailed plot for a single kappa value."""
    df = pd.DataFrame({
        f'Weight $w_{S[0]}$': W.cpu().numpy()[:, S[0]],
        'Amplitude $a$': A.cpu().numpy().flatten()
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Analysis for $\\kappa_0 = {kappa_0:.4e}$ ($\\kappa={kappa:.4e}$, {act_name.upper()}, |S|={S_k})", fontsize=18)

    # Convergence Plot
    ax1.plot(history, 'o-', label=f'Final $m_S = {m_S:.8f}$')
    ax1.set_title("Convergence of Order Parameter $m_S$")
    ax1.set_xlabel("Self-Consistency Iteration")
    ax1.set_ylabel("$m_S$")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Joint Distribution Plot
    sns.kdeplot(data=df, x=f'Weight $w_{S[0]}$', y='Amplitude $a$', 
                fill=True, cmap="plasma", thresh=0.05, ax=ax2)
    ax2.set_title(f"Joint Distribution $P(w_{{S[0]}}, a)$")
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"feature_learning_{act_name}_k{S_k}_kappa0_{kappa_0:.4e}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    print(f"Saved individual plot to {filepath}")
    plt.close(fig)

def create_summary_plot(kappa_0_values, m_S_values, S_k, act_name, output_dir):
    """Generates a summary plot of m_S vs. kappa_0."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Phase Transition: $m_S$ vs. $\\kappa_0$ ({act_name.upper()}, |S|={S_k})", fontsize=18)

    # Linear Scale
    ax1.plot(kappa_0_values, m_S_values, 'o-')
    ax1.set_title("Linear Scale")
    ax1.set_xlabel("Base Noise Parameter $\\kappa_0$")
    ax1.set_ylabel("Converged Order Parameter $m_S$")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Logarithmic Scale
    ax2.semilogx(kappa_0_values, m_S_values, 'o-')
    ax2.set_title("Logarithmic Scale")
    ax2.set_xlabel("Base Noise Parameter $\\kappa_0$")
    ax2.set_ylabel("Converged Order Parameter $m_S$")
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"summary_phase_transition_{act_name}_k{S_k}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200)
    print(f"\nSaved summary plot to {filepath}")
    plt.show()

# --- 6.  Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    output_dir = "/home/goring/OnlineSGD/empirical_tests/3007_MCMC8_1"     # Directory to save all plots
    activation_choice = 'relu'          # CHOOSE 'relu' or 'tanh'
    S = (0, 1,2,3)                          # CHOOSE the parity, e.g., (0, 1) for k=2
    
    # --- KAPPA_0 ITERATION SETUP ---
    # A log space is good for seeing transitions. We iterate from high kappa_0 (high noise) to low.
    kappa_0_values = np.logspace(5, -5, 30) # 30 steps from 1000 down to 0.00001

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    if activation_choice == 'tanh':
        activation_fn = tanh
    elif activation_choice == 'relu':
        activation_fn = relu
    else:
        raise ValueError("Invalid activation choice. Choose 'relu' or 'tanh'.")

    print(f"Using {activation_choice.upper()} activation for |S|={len(S)} parity task.")
    print(f"Saving all plots to directory: '{output_dir}'")
    
    # --- Main Loop ---
    converged_m_S_values = []
    # ✅ IMPROVED: Use annealing for the initial guess of m_S.
    # Start with m_S=0 for the highest noise level.
    m_S_guess = 1.0 
    
    for kappa_0 in kappa_0_values:
        # Calculate the effective kappa using the scaling from the paper
        kappa = kappa_0 * (N**(1 - gamma))
        
        m_S, history, W_final, A_final = solve_kappa(
            kappa, 
            S, 
            activation_fn, 
            m0=m_S_guess # Use the converged value from the previous kappa
        )
        converged_m_S_values.append(m_S)
        m_S_guess = m_S # Update the guess for the next, lower-noise run
        
        # Create and save the detailed plot for this kappa run
        create_and_save_individual_plot(
            kappa_0, kappa, m_S, history, W_final, A_final, 
            S_k=len(S), act_name=activation_choice, output_dir=output_dir
        )

    # --- 7. Save Results to JSON ---
    # This new section saves the final results to a structured JSON file.
    results_data = {
        "S_k": len(S),
        "activation": activation_choice,
        "gamma": gamma,
        "d": d,
        "N": N,
        # Convert numpy array to a standard Python list for JSON compatibility
        "kappa_0_values": kappa_0_values.tolist(), 
        # Reverse the m_S values to match the ascending order of kappa_0 for plotting
        "m_S_values": converged_m_S_values[::-1] 
    }
    
    json_filename = f"results_data_{activation_choice}_k{len(S)}.json"
    json_filepath = os.path.join(output_dir, json_filename)
    
    with open(json_filepath, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"\nSaved full results data to {json_filepath}")
    
    # --- Final Summary Plot ---
    # Reverse the lists for plotting from low kappa_0 to high
    create_summary_plot(sorted(kappa_0_values), converged_m_S_values[::-1], 
                        S_k=len(S), act_name=activation_choice, output_dir=output_dir)