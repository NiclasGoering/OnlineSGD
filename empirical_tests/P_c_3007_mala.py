import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json

# --- 1.  Hyper-parameters & Model Setup ---
d, N = 40, 1024
g_w = g_a = 1.0
gamma = 2.0 # Using the required gamma=2 for mean-field theory
print(f"Using required gamma = {gamma}")

sigma_w_sq = g_w / d
sigma_a_sq = g_a / (N**gamma)

# --- GPU/CPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Activation Function & Helper Definitions ---
def relu(x):
    return torch.nn.functional.relu(x)

X_bank = (2 * torch.randint(0, 2, size=(20_000, d), device=device) - 1).float()


@torch.no_grad()
def Sigma(w):
    """Computes Sigma(w) = E[phi(w^T x)^2] using the static bank."""
    return torch.mean(relu(X_bank @ w)**2)

@torch.no_grad()
def J_A(w, A):
    """Computes J_A(w) = E[phi(w^T x) * chi_A(x)] using the static bank."""
    chi_A_x = torch.prod(X_bank[:, A], axis=1)
    return torch.mean(relu(X_bank @ w) * chi_A_x)

# --- 2. The Log-Probability Function ---
def log_p_w(w, m_S, kappa, S):
    """Calculates the log-probability of w, including all previous corrections."""
    Sigma_w = torch.mean(relu(X_bank @ w)**2)
    chi_S_x = torch.prod(X_bank[:, S], axis=1)
    J_S_w = torch.mean(relu(X_bank @ w) * chi_S_x)
    
    kappa_sq = kappa**2
    prior_a_term = (N**gamma) / sigma_a_sq
    denom_term = prior_a_term + (Sigma_w / (kappa_sq + 1e-20))
    J_sigma = J_S_w * (1 - m_S)
    
    stable_exp_term = 0.25 * (J_sigma**2) / ((kappa_sq * kappa_sq + 1e-20) * denom_term)
    log_prior_w = -0.5 * d * torch.dot(w, w) / sigma_w_sq
    log_det_term = -0.5 * torch.log(denom_term)
    
    return log_prior_w + log_det_term + stable_exp_term

# --- 3. Stable MCMC Solver: Metropolis-Adjusted Langevin Algorithm (MALA) ---


def mala_step_w(w, m_S, kappa, S, step_size):
    """Performs a single MALA step, using gradients for stable proposals."""
    w.requires_grad_(True)
    
    # Calculate log-probability and its gradient at the current point 'w'
    log_p_current = log_p_w(w, m_S, kappa, S)
    log_p_current.backward()
    grad_current = w.grad
    
    w.requires_grad_(False) # Detach from graph for the update
    
    # --- Proposal Step ---
    # Propose a new w' using the gradient information (Langevin diffusion)
    
    # --- THIS LINE IS THE FIX ---
    # Before: noise = torch.randn_like(w) * torch.sqrt(2 * step_size)
    # After: Use standard Python sqrt, which is **0.5
    noise = torch.randn_like(w) * (2 * step_size)**0.5
    
    w_proposal = w + step_size * grad_current + noise
    
    # --- Acceptance Step ---
    # Calculate log-probability at the proposed point w'
    log_p_proposal = log_p_w(w_proposal, m_S, kappa, S)
    
    # We need the gradient at the proposed point to calculate the reverse probability
    w_proposal.requires_grad_(True)
    log_p_temp_for_grad = log_p_w(w_proposal, m_S, kappa, S)
    log_p_temp_for_grad.backward()
    grad_proposal = w_proposal.grad
    w_proposal.requires_grad_(False)
    
    # Calculate log proposal probabilities (the q terms in the M-H ratio)
    # log q(w' | w)
    log_q_forward = -torch.sum((w_proposal - w - step_size * grad_current)**2) / (4 * step_size)
    # log q(w | w')
    log_q_reverse = -torch.sum((w - w_proposal - step_size * grad_proposal)**2) / (4 * step_size)
    
    # Full acceptance log-ratio
    log_alpha = log_p_proposal + log_q_reverse - log_p_current - log_q_forward
    
    if torch.log(torch.rand(1, device=device)) < log_alpha:
        return w_proposal, True
    return w, False


def sample_w_with_mala(m_S, kappa, S, n_samp=4000, burn=2000, initial_step=1e-7):
    """
    Samples w from its marginal p(w) using the stable MALA sampler.
    NOTE: This function must NOT have a @torch.no_grad() decorator.
    """
    w = torch.randn(d, device=device) * np.sqrt(sigma_w_sq)
    accepted_count = 0
    W_samples = []

    # Adaptive step size during burn-in
    step_size = initial_step

    # --- Burn-in phase (Requires gradients) ---
    for t in range(burn):
        w, accepted = mala_step_w(w, m_S, kappa, S, step_size)
        if accepted:
            accepted_count += 1
        # Simple step size adaptation
        if (t + 1) % 100 == 0:
            rate = accepted_count / 100
            if rate > 0.6: step_size *= 1.2
            if rate < 0.4: step_size *= 0.8
            accepted_count = 0

    # --- Sampling phase (Requires gradients) ---
    for t in range(n_samp):
        w, accepted = mala_step_w(w, m_S, kappa, S, step_size)
        W_samples.append(w.clone())

    return torch.stack(W_samples)


def solve_kappa_mcmc(kappa, S, m0=0.0, outer_iters=100, damping=0.5, tolerance=1e-6):
    """
    Solves the self-consistency equation for m_S using the robust MALA-MCMC.
    """
    m_S = m0
    prior_a_term = (N**gamma) / sigma_a_sq
    print(f"\n--- Solving for kappa = {kappa:.4e} (Initial m_S = {m0:.6f}) ---")

    for i in range(outer_iters):
        # --- Inner loop: Sample from p(w|m_S) using the stable MALA sampler ---
        # Gradients are enabled here by default, which is correct.
        initial_step_size = min(1e-5, kappa**2 * 1e-2)
        W_samples = sample_w_with_mala(m_S, kappa, S, initial_step=initial_step_size)

        # --- Outer loop: Update m_S using the new samples ---
        # Now that sampling is done, we can switch off gradients for the simple averaging.
        with torch.no_grad():
            J_S_sq_vals = []
            Sigma_vals = []
            for w_sample in W_samples:
                J_S_sq_vals.append(J_A(w_sample, S)**2)
                Sigma_vals.append(Sigma(w_sample))

            J_S_sq_avg = torch.mean(torch.stack(J_S_sq_vals))
            Sigma_avg = torch.mean(torch.stack(Sigma_vals))

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
    output_dir = "/home/goring/OnlineSGD/empirical_tests/3007_MCMC8_mala"
    os.makedirs(output_dir, exist_ok=True)
    
    S = (0, 1, 2, 3)
    kappa_0_values = np.logspace(1.5, -1.5, 20) # Focused range around the expected transition

    converged_m_S_values = []
    m_S_guess = 0.05

    print("\nStarting self-consistency solve with stable MALA-MCMC...")
    # "Cool" the system from high noise to low noise
    for kappa_0 in sorted(kappa_0_values, reverse=True):
        kappa = kappa_0 * (N**(1 - gamma))
        
        m_S = solve_kappa_mcmc(kappa, S, m0=m_S_guess)
        converged_m_S_values.append(m_S)
        m_S_guess = m_S

    converged_m_S_values.reverse()

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(sorted(kappa_0_values), converged_m_S_values, 'o-')
    ax.set_title(f"Phase Transition (MALA-MCMC): $m_S$ vs. $\\kappa_0$ ($\gamma=2$)", fontsize=16)
    ax.set_xlabel("Base Noise Parameter $\\kappa_0$")
    ax.set_ylabel("Converged Order Parameter $m_S$")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_phase_transition_mala.png"))
    plt.show()