import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json

# --- 1. Hyper-parameters ---
d, N, g_w, g_a = 40, 1024, 1.0, 1.0
gamma = 2.0
sigma_w_sq = g_w / d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Helper Functions ---
def relu(x): return torch.nn.functional.relu(x)

# A static bank of inputs for fast expectation calculations
X_bank = (2 * torch.randint(0, 2, size=(100_000, d), device=device) - 1).float()

@torch.no_grad()
def Sigma(w):
    """Computes Sigma(w) = E[phi(w^T x)^2]"""
    return torch.mean(relu(X_bank @ w)**2)

@torch.no_grad()
def J_A(w, A):
    """Computes J_A(w) = E[phi(w^T x) * chi_A(x)]"""
    chi_A_x = torch.prod(X_bank[:, A], axis=1)
    return torch.mean(relu(X_bank @ w) * chi_A_x)

# --- 3. Log-Probability and MALA Sampler ---
def log_p_w(w, m_S, kappa, S):
    """Calculates the log-probability of w. This is the objective function for MALA."""
    # These calculations need to be inside the function to be part of the gradient tape
    Sigma_w = torch.mean(relu(X_bank @ w)**2)
    chi_S_x = torch.prod(X_bank[:, S], axis=1)
    J_S_w = torch.mean(relu(X_bank @ w) * chi_S_x)
    
    kappa_sq = kappa**2
    
    # --- The Correct N^0 Prior ---
    # This prior matches the successful Langevin experiment.
    prior_a_term = 1.0 / g_a

    denom_term = prior_a_term + (Sigma_w / (kappa_sq + 1e-20))
    J_sigma = J_S_w * (1 - m_S)
    stable_exp_term = 0.25 * (J_sigma**2) / ((kappa_sq * kappa_sq + 1e-20) * denom_term)
    log_prior_w = -0.5 * d * torch.dot(w, w) / sigma_w_sq
    log_det_term = -0.5 * torch.log(denom_term)
    
    return log_prior_w + log_det_term + stable_exp_term

def mala_step_w(w, m_S, kappa, S, step_size):
    """Performs a single, stable MALA step using gradients."""
    w.requires_grad_(True)
    log_p_current = log_p_w(w, m_S, kappa, S)
    log_p_current.backward()
    grad_current = w.grad
    w.requires_grad_(False)
    
    noise = torch.randn_like(w) * (2 * step_size)**0.5
    w_proposal = w + step_size * grad_current + noise
    
    log_p_proposal = log_p_w(w_proposal, m_S, kappa, S)
    w_proposal.requires_grad_(True)
    log_p_temp_for_grad = log_p_w(w_proposal, m_S, kappa, S)
    log_p_temp_for_grad.backward()
    grad_proposal = w_proposal.grad
    w_proposal.requires_grad_(False)
    
    log_q_forward = -torch.sum((w_proposal - w - step_size * grad_current)**2) / (4 * step_size)
    log_q_reverse = -torch.sum((w - w_proposal - step_size * grad_proposal)**2) / (4 * step_size)
    log_alpha = log_p_proposal + log_q_reverse - log_p_current - log_q_forward
    
    if torch.log(torch.rand(1, device=device)) < log_alpha:
        return w_proposal, True
    return w, False

def sample_w_with_mala(m_S, kappa, S, n_samp=4000, burn=2000, initial_step=1e-7):
    """Runs the MALA sampler to get samples from p(w|m_S)."""
    w = torch.randn(d, device=device) * np.sqrt(sigma_w_sq)
    W_samples, step_size, accepted_count = [], initial_step, 0
    
    for t in range(burn + n_samp):
        w, accepted = mala_step_w(w, m_S, kappa, S, step_size)
        if t >= burn: 
            W_samples.append(w.clone())
        if t < burn: # Adapt step size during burn-in
            if accepted: accepted_count += 1
            if (t + 1) % 100 == 0:
                rate = accepted_count / 100
                if rate > 0.6: step_size *= 1.2
                if rate < 0.4: step_size *= 0.8
                accepted_count = 0
                
    return torch.stack(W_samples)

# --- 4. Self-Consistency Solver ---
def solve_kappa_mcmc(kappa, S, m0=0.0, outer_iters=100, damping=0.5, tolerance=1e-6):
    """Solves the self-consistency loop for m_S."""
    m_S = m0
    prior_a_term = 1.0 / g_a # Using the correct N^0 prior
    
    print(f"\n--- Solving for kappa = {kappa:.4e} (Initial m_S = {m0:.6f}) ---")
    
    for i in range(outer_iters):
        initial_step_size = min(1e-5, kappa**2 * 1e-2)
        W_samples = sample_w_with_mala(m_S, kappa, S, initial_step=initial_step_size)
        
        with torch.no_grad():
            J_S_sq_avg = torch.mean(torch.stack([J_A(w, S)**2 for w in W_samples]))
            Sigma_avg = torch.mean(torch.stack([Sigma(w) for w in W_samples]))
            integrand = J_S_sq_avg / (kappa**2 * prior_a_term + Sigma_avg)
            m_S_new = N * (1 - m_S) * integrand

        if i % 5 == 0: 
            print(f"  Outer iter {i:3d}: m_S = {m_S_new.item():.8f}")
            
        if abs(m_S - m_S_new.item()) < tolerance:
            print(f"  Converged after {i+1} outer iterations.")
            return m_S_new.item()
            
        m_S = (1 - damping) * m_S + damping * m_S_new.item()
        
    print(f"  Warning: Did not converge after {outer_iters} iterations.")
    return m_S

# --- 5. Main Execution ---
if __name__ == "__main__":
    output_dir, S = "/home/goring/OnlineSGD/empirical_tests/3007_MCMC8_mala", (0, 1, 2, 3)
    os.makedirs(output_dir, exist_ok=True)
    
    kappa_0_values = np.logspace(1.5, -1.5, 10)
    converged_m_S_values, m_S_guess = [], 0.05
    
    for kappa_0 in sorted(kappa_0_values, reverse=True):
        kappa = kappa_0 * (N**(1 - gamma))
        m_S = solve_kappa_mcmc(kappa, S, m0=m_S_guess)
        converged_m_S_values.append(m_S)
        m_S_guess = m_S
        
    converged_m_S_values.reverse()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(sorted(kappa_0_values), converged_m_S_values, 'o-')
    ax.set_title(f"Phase Transition (Corrected MCMC): $m_S$ vs. $\\kappa_0$", fontsize=16)
    ax.set_xlabel("Base Noise Parameter $\\kappa_0$")
    ax.set_ylabel("Converged Order Parameter $m_S$")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_phase_transition_final.png"))
    plt.show()