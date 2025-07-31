import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import os
import json

# --- 1.  Hyper-parameters & Model Setup ---
# These parameters are defined according to the provided theoretical text.
d, N = 40, 1024
gamma = 2.0

# Fundamental scaling parameters from the theory.
# We set them to 1.0 as is common.
sigma_w = 1.0
sigma_a = 1.0

# --- Derived variances based on the paper's definitions ---
# Paper formula: g_w^2 = sigma_w / d
g_w_sq = sigma_w / d
g_w = np.sqrt(g_w_sq)

# Paper formula: g_a^2 = sigma_a / N^gamma
g_a_sq = sigma_a / (N**gamma)
g_a = np.sqrt(g_a_sq)

print(f"--- Parameters ---")
print(f"d={d}, N={N}, gamma={gamma}")
print(f"sigma_w={sigma_w:.2f}, sigma_a={sigma_a:.2f}")
print(f"g_w^2 = {g_w_sq:.4e}, g_a^2 = {g_a_sq:.4e}")
print(f"------------------")


# --- GPU/CPU Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU found. Using CUDA.")
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU.")

# --- Activation Function Definitions ---
def relu(x):
    """Even activation function."""
    return torch.nn.functional.relu(x)

# --- 2.  Fast Boolean Sample Bank for Expectations (on GPU) ---
# This bank is used to compute expectations over the input data x.
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
    """
    Calculates the log-probability of w, log p(w), which is -S_eff(w).
    This function implements the integrated-out action formula.
    """
    Sigma_w = Sigma(w, activation_fn)
    J_S_w = J_A(w, S, activation_fn)
    kappa_sq = kappa**2

    # --- Implementing S_eff(w) from the paper ---
    # S_eff(w) = S_prior(w) + 0.5*log(A) - B^2/(4A)
    # where A = (N^gamma/sigma_a + Sigma(w)/kappa^2) and B = (J_Y - m*J_S)/kappa^2.
    # We assume the target is y = chi_S(x), so J_Y = J_S.

    # Paper formula for the w prior term in the action: (d / (2 * sigma_w)) * ||w||^2
    s_prior_w = (d / (2.0 * sigma_w)) * torch.dot(w, w)

    # Paper formula for the coefficient of a^2 in the action: (N^gamma / sigma_a)
    prior_a_coeff = (N**gamma) / sigma_a
    
    # This is the term 'A' inside the log and in the denominator.
    denom_A = prior_a_coeff + (Sigma_w / (kappa_sq + 1e-20))

    # Paper formula for the log-determinant term in S_eff: 0.5 * log(N^gamma/sigma_a + Sigma(w)/kappa^2)
    log_det_term = 0.5 * torch.log(denom_A)

    # This is the term (J_Y - m_S*J_S) = J_S(w) * (1 - m_S)
    numerator_B_base = J_S_w * (1.0 - m_S)
    
    # Paper formula for the final term in S_eff: - ( (1/kappa^4) * (J_Y-m*J_S)^2 ) / (4 * A)
    exp_term_numerator_B_sq = (numerator_B_base**2) / (kappa_sq * kappa_sq + 1e-20)
    exp_term = exp_term_numerator_B_sq / (4.0 * denom_A)

    # The log probability is the negative of the action: log p(w) = -S_eff(w)
    # log p(w) = -s_prior_w - log_det_term + exp_term
    return -s_prior_w - log_det_term + exp_term


def mh_step_w(w, m_S, kappa, step_size, S, activation_fn):
    """Performs a single Metropolis-Hastings step for w on the GPU."""
    # Propose a new w from a Gaussian centered at the current w.
    w_proposal = w + step_size * torch.randn(d, device=device)
    
    # Acceptance probability depends on the ratio of probabilities (or difference of log-probs).
    log_acceptance_ratio = log_p_w(w_proposal, m_S, kappa, S, activation_fn) - log_p_w(w, m_S, kappa, S, activation_fn)
    
    if torch.log(torch.rand(1, device=device)) < log_acceptance_ratio:
        return w_proposal, True
    return w, False

@torch.no_grad()
def sample_w_then_a(m_S, kappa, n_samp, burn, step, S, activation_fn):
    """Samples w from its marginal p(w) via MCMC, then samples a from p(a|w)."""
    # Initialize w from its prior distribution: w ~ N(0, g_w^2 * I)
    w = torch.randn(d, device=device) * g_w
    
    accepted_count = 0
    W_samples, A_samples = [], []

    # Burn-in phase to let the chain reach the stationary distribution.
    for t in range(burn):
        w, _ = mh_step_w(w, m_S, kappa, step, S, activation_fn)

    # Sampling phase
    for t in range(n_samp):
        w, accepted = mh_step_w(w, m_S, kappa, step, S, activation_fn)
        if accepted:
            accepted_count += 1
            
        Sigma_w = Sigma(w, activation_fn)
        J_S_w = J_A(w, S, activation_fn)
        
        # --- Sample 'a' from the conditional posterior p(a|w) ---
        # Paper formula for inverse variance of a: sigma(w)^-2 = (N^gamma/sigma_a + Sigma(w)/kappa^2)
        prior_a_coeff = (N**gamma) / sigma_a
        var_a_inv = prior_a_coeff + (Sigma_w / (kappa**2 + 1e-12))
        var_a = 1.0 / var_a_inv
        
        # Paper formula for mean of a: mu(w) = var_a * (1/kappa^2) * (J_Y - m*J_S)
        mean_a_numerator = (J_S_w * (1.0 - m_S)) / (kappa**2 + 1e-12)
        mean_a = var_a * mean_a_numerator
        
        # Sample a from N(mean_a, var_a)
        a = torch.randn(1, device=device) * torch.sqrt(var_a) + mean_a
        
        W_samples.append(w.clone())
        A_samples.append(a)
            
    acceptance_rate = accepted_count / n_samp if n_samp > 0 else 0
    return torch.stack(W_samples), torch.stack(A_samples), acceptance_rate

# --- 4.  Self-Consistency Loop ---
@torch.no_grad()
def solve_kappa(kappa, S, activation_fn, m0=0.0, iters=220, damping=0.2):
    """
    Solves the self-consistency equation for m_S at a given kappa.
    Damping is reduced to 0.2 for better stability.
    """
    m_S = m0
    history = [m_S]
    
    step_size = 0.1 # Adaptive MCMC Step Size
    print(f"\n--- Solving for kappa = {kappa:.4e} (starting m_S = {m0:.4f}) ---")

    for i in range(iters):
        # 1. Sample (w, a) pairs from the posterior defined by the current m_S.
        # Increased number of samples for better statistical accuracy.
        W_samples, A_samples, acc_rate = sample_w_then_a(m_S, kappa, 5000, 1500, step_size, S, activation_fn)
        
        # Adapt MCMC step size for good mixing (acceptance rate ~25-40%).
        if acc_rate > 0.4 and step_size < 1.0:
            step_size *= 1.05
        elif acc_rate < 0.2:
            step_size /= 1.05
        
        # 2. Calculate the new order parameter m_S based on the samples.
        # Paper formula: m_S = N * <a * J_S(w)>
        J_S_w_all = torch.stack([J_A(w, S, activation_fn) for w in W_samples])
        m_S_new = N * torch.mean(A_samples * J_S_w_all)
        
        # 3. Update m_S with damping for stability.
        m_S = (1 - damping) * m_S + damping * m_S_new.item()
        history.append(m_S)
        
        print(f"Iter {i+1:3d}: acc={acc_rate:.2f}, step={step_size:.4f}, m_S={m_S:.8f}")

        # Convergence criterion: break if m_S has stabilized.
        if i >= 50: # Start checking for convergence after 50 iterations
            last_20 = np.array(history[-20:])
            mean_val = np.mean(last_20)
            std_val = np.std(last_20)
            
            # Refined convergence check for stability near zero
            converged_to_zero = abs(mean_val) < 1e-5 and std_val < 1e-5
            converged_to_nonzero = (abs(mean_val) > 1e-5 and (std_val / abs(mean_val)) < 0.01)

            if converged_to_zero or converged_to_nonzero:
                print(f"CONVERGED: Value stabilized at m_S = {mean_val:.8f}")
                m_S = mean_val
                break
    
    W_final, A_final, _ = sample_w_then_a(m_S, kappa, 8000, 1000, step_size, S, activation_fn)
    return m_S, history, W_final, A_final

# --- 5.  Plotting Functions (Unchanged) ---
def create_summary_plot(kappa_0_values, m_S_values, S_k, act_name, output_dir):
    """Generates a summary plot of m_S vs. kappa_0."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Phase Transition (Corrected): $m_S$ vs. $\\kappa_0$ ({act_name.upper()}, |S|={S_k})", fontsize=18)

    ax1.plot(kappa_0_values, m_S_values, 'o-')
    ax1.set_title("Linear Scale")
    ax1.set_xlabel("Base Noise Parameter $\\kappa_0$")
    ax1.set_ylabel("Converged Order Parameter $m_S$")
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.semilogx(kappa_0_values, m_S_values, 'o-')
    ax2.set_title("Logarithmic Scale")
    ax2.set_xlabel("Base Noise Parameter $\\kappa_0$")
    ax2.set_ylabel("Converged Order Parameter $m_S$")
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"summary_phase_transition_{act_name}_k{S_k}_THEORY_CORRECT.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.show()

# --- 6.  Main Execution ---
if __name__ == "__main__":
    output_dir = "/home/goring/OnlineSGD/empirical_tests/3007_MCMC8_corrected3" 
    activation_choice = 'relu'
    S = tuple(range(4)) # Target parity on first 4 features, |S|=4
    kappa_0_values = [1] #np.logspace(5, -5, 8) # Expanded range for clarity

    os.makedirs(output_dir, exist_ok=True)
    activation_fn = relu
    
    print(f"\nRunning with STRICTLY IMPLEMENTED theory for {activation_choice.upper()}, |S|={len(S)}")
    
    converged_m_S_values = []
    m_S_guess = 0.05 # Start with a neutral guess
    
    # We scan from high noise to low noise, using the previous result as a guess (continuation method).
    # Sorting ensures we start in the stable m_S=0 regime.
    for kappa_0 in sorted(kappa_0_values, reverse=True):
        # Paper formula: kappa = kappa_0 * N^(1-gamma)
        kappa = kappa_0 * (N**(1.0 - gamma))
        
        m_S, history, _, _ = solve_kappa(kappa, S, activation_fn, m0=m_S_guess)
        converged_m_S_values.append(m_S)
        m_S_guess = m_S # Use the converged value for the next iteration.
        
    # Reverse the results to match the original order of kappa_0_values for plotting
    final_m_S_values = list(reversed(converged_m_S_values))
    final_kappa_0s = sorted(kappa_0_values)

    results_data = {
        "S_k": len(S), "activation": activation_choice, "gamma": gamma, "d": d, "N": N,
        "sigma_w": sigma_w, "sigma_a": sigma_a,
        "kappa_0_values": final_kappa_0s, 
        "m_S_values": final_m_S_values
    }
    
    results_filename = f"results_data_{activation_choice}_k{len(S)}.json"
    with open(os.path.join(output_dir, results_filename), 'w') as f:
        json.dump(results_data, f, indent=4)
    
    create_summary_plot(final_kappa_0s, final_m_S_values, 
                        S_k=len(S), act_name=activation_choice, output_dir=output_dir)