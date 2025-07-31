import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd

# --- 1.  Hyper-parameters & Model Setup ---
# Using parameters from the critique that are known to show the transition clearly
d, N, gamma = 40, 1024, 2.0
g_w = g_a = 1.0

# Correctly scaled variances
sigma_w_sq = g_w / d
sigma_a_sq = g_a / N
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
    """Odd activation function, suitable for even parity tasks."""
    return torch.tanh(x)

def relu(x):
    """Even activation function, suitable for odd parity tasks."""
    return torch.nn.functional.relu(x)

# --- 2.  Fast Boolean Sample Bank for Expectations (on GPU) ---
# A static bank of inputs is created once on the selected device for efficiency.
X_bank = (2 * torch.randint(0, 2, size=(200_000, d), device=device) - 1).float()

def Sigma(w, activation_fn):
    """Computes Sigma(w) = E[phi(w^T x)^2] using the static bank."""
    z = X_bank @ w
    return torch.mean(activation_fn(z)**2)

def chi(A, X):
    """Computes the parity function chi_A(x) on a matrix of inputs."""
    return torch.prod(X[:, A], axis=1)

def J_A(w, A, activation_fn):
    """Computes J_A(w) = E[phi(w^T x) * chi_A(x)] using the static bank."""
    z = X_bank @ w
    return torch.mean(activation_fn(z) * chi(A, X_bank))

# --- 3.  Analytically Integrated Distributions (PyTorch version) ---

def log_p_w(w, m_S, kappa, S, activation_fn):
    """
    Calculates the marginal log-probability of the weights w, log p(w).
    """
    Sigma_w = Sigma(w, activation_fn)
    J_S_w = J_A(w, S, activation_fn)
    J_sigma = J_S_w - m_S * J_S_w

    coeff = (N**gamma / sigma_a_sq**2) + (Sigma_w / kappa**2)
    
    log_prior_w = -0.5 * d * torch.dot(w, w) / g_w
    log_det_term = -0.5 * torch.log(coeff)
    exp_term = 0.5 * (J_sigma / kappa**2)**2 / coeff
    
    return log_prior_w + log_det_term + exp_term

def mh_step_w(w, m_S, kappa, step_size, S, activation_fn):
    """Performs a single Metropolis-Hastings step for w on the GPU."""
    w_proposal = w + step_size * torch.randn(d, device=device)
    log_acceptance_ratio = log_p_w(w_proposal, m_S, kappa, S, activation_fn) - log_p_w(w, m_S, kappa, S, activation_fn)
    
    if torch.log(torch.rand(1, device=device)) < log_acceptance_ratio:
        return w_proposal, True
    return w, False

def sample_w_then_a(m_S, kappa, n_samp, burn, step, S, activation_fn):
    """
    Samples w from its marginal p(w) via MCMC, then samples a from p(a|w).
    """
    w = torch.randn(d, device=device) * sigma_w
    accepted_count = 0
    W_samples, A_samples = [], []

    for t in range(n_samp + burn):
        w, accepted = mh_step_w(w, m_S, kappa, step, S, activation_fn)
        if t >= burn:
            if accepted:
                accepted_count += 1
            
            # Now that we have a sample w, draw a from its conditional Gaussian
            Sigma_w = Sigma(w, activation_fn)
            J_sigma = J_A(w, S, activation_fn) - m_S * J_A(w, S, activation_fn)
            
            denom = (N**gamma / sigma_a_sq) + (Sigma_w / kappa**2)
            var_a = 1.0 / denom
            mean_a = var_a * J_sigma / kappa**2
            
            a = torch.randn(1, device=device) * torch.sqrt(var_a) + mean_a
            
            W_samples.append(w.clone())
            A_samples.append(a)
            
    acceptance_rate = accepted_count / n_samp if n_samp > 0 else 0
    return torch.stack(W_samples), torch.stack(A_samples), acceptance_rate

# --- 4.  Self-Consistency Loop ---

def solve_kappa(kappa, S, activation_fn, m0=0.0, iters=300, damping=0.5):
    """Solves the self-consistency equation for m_S at a given kappa."""
    m_S = m0
    history = [m_S]
    
    # MODIFIED: Adaptive MCMC step size that scales more gently with kappa.
    # This prevents the sampler from getting stuck at very low kappa values.
    step_size = min(0.1, np.sqrt(kappa) * 0.25)

    print(f"\n--- Solving for kappa = {kappa:.3f} (starting m_S = {m0:.2f}, step_size = {step_size:.4f}) ---")

    for i in range(iters):
        W_samples, _, acc_rate = sample_w_then_a(m_S, kappa, 4000, 1000, step_size, S, activation_fn)
        
        Sigma_w_all = torch.mean(activation_fn(X_bank @ W_samples.T)**2, axis=0)
        J_S_w_all = torch.mean(activation_fn(X_bank @ W_samples.T) * chi(S, X_bank).unsqueeze(1), axis=0)
        J_sigma_all = J_S_w_all - m_S * J_S_w_all
        
        denom_all = (N**gamma / sigma_a_sq) + (Sigma_w_all / kappa**2)
        var_a_all = 1.0 / denom_all
        mean_a_all = var_a_all * J_sigma_all / kappa**2
        
        m_S_new = N * torch.mean(mean_a_all * J_S_w_all)
        m_S = (1 - damping) * m_S + damping * m_S_new.item()
        history.append(m_S)
        print(f"Iter {i+1}: acc={acc_rate:.2f}, m_S={m_S:.8f}")
        
    W_final, A_final, _ = sample_w_then_a(m_S, kappa, 8000, 1000, step_size, S, activation_fn)
    return m_S, history, W_final, A_final

# --- 5.  Main Execution ---

if __name__ == "__main__":
    # --- Configuration ---
    output_path = "/home/goring/OnlineSGD/empirical_tests/feature_learning_gpu_plot_high_res_reluk0.05_7.png"
    activation_choice = 'relu'  # CHOOSE 'relu' or 'tanh'
    S = (0, 1, 2, 3)            # CHOOSE the parity freely, e.g., (0, 1, 2, 3) for k=4
    
    # --- Theoretical Note ---
    # ReLU (an even function) will struggle to learn even-parity tasks (|S| is even).
    # Tanh (an odd function) will struggle to learn odd-parity tasks (|S| is odd).
    # This simulation allows you to test these theoretical limits.
    
    if activation_choice == 'tanh':
        activation_fn = tanh
    elif activation_choice == 'relu':
        activation_fn = relu
    else:
        raise ValueError("Invalid activation choice. Choose 'relu' or 'tanh'.")

    print(f"Using {activation_choice.upper()} activation for |S|={len(S)} parity task.")

    kappa_unlearned = 5.0
    kappa_learned = 0.05
    
    mU, hU, WU, AU = solve_kappa(kappa_unlearned, S, activation_fn, m0=0.01)
    mL, hL, WL, AL = solve_kappa(kappa_learned, S, activation_fn, m0=0.1)

    # --- High-Resolution Plotting ---
    df_unlearned = pd.DataFrame({
        'Weight $w_1$': WU.cpu().numpy()[:, 0],
        'Amplitude $a$': AU.cpu().numpy().flatten()
    })
    df_learned = pd.DataFrame({
        'Weight $w_1$': WL.cpu().numpy()[:, 0],
        'Amplitude $a$': AL.cpu().numpy().flatten()
    })

    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(2, 2, height_ratios=(1, 1.5))
    ax_conv = fig.add_subplot(gs[0, :])
    
    fig.suptitle(f"Feature Learning with {activation_choice.upper()} for |S|={len(S)} (GPU Accelerated)", fontsize=20, y=0.98)

    ax_conv.plot(hU, 'o-', label=f'Unlearned ($\\kappa = {kappa_unlearned}$)')
    ax_conv.plot(hL, 'o-', label=f'Learned ($\\kappa = {kappa_learned}$)')
    ax_conv.set_title("Convergence of Order Parameter $m_S$")
    ax_conv.set_xlabel("Iteration"); ax_conv.set_ylabel("$m_S$")
    ax_conv.legend(); ax_conv.grid(True, linestyle='--', alpha=0.6)

    g_unlearned = sns.jointplot(data=df_unlearned, x='Weight $w_1$', y='Amplitude $a$', kind="kde",
                                fill=True, cmap="viridis", space=0, thresh=0.05,
                                marginal_kws=dict(fill=True, color='m'))
    g_unlearned.fig.suptitle(f"Unlearned Phase ($P(w_1, a)$ at $\\kappa={kappa_unlearned}$)", y=1.02)
    g_unlearned.savefig("unlearned_phase.png", dpi=300, bbox_inches='tight')
    plt.close(g_unlearned.fig)

    g_learned = sns.jointplot(data=df_learned, x='Weight $w_1$', y='Amplitude $a$', kind="kde",
                              fill=True, cmap="plasma", space=0, thresh=0.05,
                              marginal_kws=dict(fill=True, color='c'))
    g_learned.fig.suptitle(f"Learned Phase ($P(w_1, a)$ at $\\kappa={kappa_learned}$)", y=1.02)
    g_learned.savefig("learned_phase.png", dpi=300, bbox_inches='tight')
    plt.close(g_learned.fig)

    ax_unlearned_img = fig.add_subplot(gs[1, 0])
    ax_learned_img = fig.add_subplot(gs[1, 1])
    
    ax_unlearned_img.imshow(plt.imread("unlearned_phase.png"))
    ax_unlearned_img.axis('off')
    ax_unlearned_img.set_title("Unlearned Phase Distribution", y=-0.1)
    
    ax_learned_img.imshow(plt.imread("learned_phase.png"))
    ax_learned_img.axis('off')
    ax_learned_img.set_title("Learned Phase Distribution", y=-0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")
    plt.show()