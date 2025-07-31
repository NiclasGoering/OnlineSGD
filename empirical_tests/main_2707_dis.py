import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from filelock import FileLock

def generate_k_sparse_parity_data(P, d, k, device='cpu'):
    """
    Generates a dataset for the k-sparse parity problem on the specified device.
    """
    if k > d:
        raise ValueError("k (number of sparse features) cannot be greater than d (dimensionality).")
    X = torch.randint(2, size=(P, d), device=device, dtype=torch.float32) * 2 - 1
    relevant_features = X[:, :k]
    y = torch.prod(relevant_features, dim=1).unsqueeze(1)
    return X, y

class TwoLayerNet(nn.Module):
    """
    A two-layer neural network as specified by the user.
    f(x) = sum_i a_i * phi(w_i^T * x)
    """
    def __init__(self, d, N, g_w, g_a, gamma_scaling_exponent):
        super().__init__()
        self.d = d
        self.N = N
        sigma_w_sq = g_w / d
        sigma_a_sq = g_a / (N ** gamma_scaling_exponent)
        self.sigma_w = torch.tensor(sigma_w_sq**0.5)
        self.sigma_a = torch.tensor(sigma_a_sq**0.5)
        self.w = nn.Parameter(torch.randn(d, N) * self.sigma_w)
        self.a = nn.Parameter(torch.randn(N, 1) * self.sigma_a)
        self.phi = torch.relu

    def forward(self, x):
        pre_activation = x @ self.w
        post_activation = self.phi(pre_activation)
        output = post_activation @ self.a
        return output

def train_with_langevin(model, X_train, y_train, X_test, y_test, hyperparams, current_config, rank):
    """
    Trains the model using full-batch Langevin GD with mini-batch accumulation.
    """
    # Unpack hyperparameters
    eta = hyperparams['eta']
    epochs = hyperparams['epochs']
    log_interval = hyperparams['log_interval']
    batch_size = hyperparams['batch_size']
    early_stop_loss = hyperparams['early_stop_loss']
    early_stop_error = hyperparams['early_stop_error']
    
    P_train = X_train.shape[0]
    # Accessing attributes from the original model before compilation
    N = model._orig_mod.N if hasattr(model, '_orig_mod') else model.N
    sigma_a = model._orig_mod.sigma_a if hasattr(model, '_orig_mod') else model.sigma_a
    sigma_w = model._orig_mod.sigma_w if hasattr(model, '_orig_mod') else model.sigma_w

    kappa_0 = current_config['kappa_0']
    gamma_scaling_exponent = hyperparams['gamma_scaling_exponent']

    # Calculate temperature T based on kappa
    kappa = kappa_0 * (N ** (1 - gamma_scaling_exponent))
    T = 2 * (kappa ** 2)

    loss_fn = nn.MSELoss(reduction='sum')
    
    print(f"GPU {rank} | Starting job: P={P_train}, N={N}, kappa_0={kappa_0:.3f}, T={T:.4f}")
    start_time = time.time()

    for epoch in range(epochs + 1):
        model.train()
        
        # --- Gradient Accumulation for Full-Batch Gradient ---
        model.zero_grad(set_to_none=True) # More efficient
        for i in range(0, P_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            y_pred_batch = model(X_batch)
            batch_loss = loss_fn(y_pred_batch, y_batch)
            
            # Normalize loss for this batch to get correct gradient scaling for the full batch
            # and accumulate gradients
            (batch_loss / P_train).backward()

        # --- Manual Langevin Update using accumulated gradients ---
        with torch.no_grad():
            # Update for 'a'
            grad_a = model.a.grad # This is now sum(grad) / P_train
            noise_a = torch.randn_like(model.a) * (2 * T * eta)**0.5
            decay_a = (T / (sigma_a ** 2)) * model.a
            delta_a = -eta * (decay_a + grad_a) + noise_a
            model.a.add_(delta_a)

            # Update for 'w'
            grad_w = model.w.grad # This is now sum(grad) / P_train
            noise_w = torch.randn_like(model.w) * (2 * T * eta)**0.5
            decay_w = (T / (sigma_w ** 2)) * model.w
            delta_w = -eta * (decay_w + grad_w) + noise_w
            model.w.add_(delta_w)

        # --- Logging and Early Stopping Check ---
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                # Full evaluation on train set for accurate error
                # We can do this in batches if P_train is very large
                train_mse_accum = 0.0
                train_correct_accum = 0
                for i in range(0, P_train, batch_size):
                    X_batch = X_train[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]
                    y_pred_batch = model(X_batch)
                    train_mse_accum += loss_fn(y_pred_batch, y_batch).item()
                    train_correct_accum += (torch.sign(y_pred_batch) == y_batch).float().sum().item()
                
                train_mse = train_mse_accum / P_train
                train_error_01 = 1.0 - (train_correct_accum / P_train)

                # Evaluation on test set
                y_pred_test = model(X_test)
                test_mse = loss_fn(y_pred_test, y_test) / X_test.shape[0]
                test_preds_class = torch.sign(y_pred_test)
                test_error_01 = (test_preds_class != y_test).float().mean()
                
                elapsed_time = time.time() - start_time
                print(
                    f"GPU {rank} | P={P_train}, k0={kappa_0:.3f} | Ep {epoch:6d} | "
                    f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse.item():.4f} | "
                    f"Train Err: {train_error_01:.4f} | Test Err: {test_error_01.item():.4f} | "
                    f"Time: {elapsed_time:.2f}s"
                )

                # Early stopping conditions
                if train_mse < early_stop_loss or train_error_01 < early_stop_error:
                    print(f"GPU {rank} | Early stopping triggered at epoch {epoch}.")
                    break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_train_mse = train_mse
        final_train_error_01 = train_error_01
        
        y_pred_test = model(X_test)
        final_test_mse = (loss_fn(y_pred_test, y_test) / X_test.shape[0]).item()
        final_test_error_01 = (torch.sign(y_pred_test) != y_test).float().mean().item()
    
    print(f"GPU {rank} | Finished job: P={P_train}, kappa_0={kappa_0:.3f}")

    return {
        "train_mse": final_train_mse, "test_mse": final_test_mse,
        "train_error_01": final_train_error_01, "test_error_01": final_test_error_01,
    }

def save_result(result, json_path):
    """Safely appends a result to the JSON file using a file lock."""
    lock = FileLock(str(json_path) + ".lock")
    with lock:
        try:
            if json_path.exists() and json_path.stat().st_size > 0:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
        except json.JSONDecodeError:
            data = [] # If file is corrupted, start fresh
        data.append(result)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

def worker(rank, world_size, job_queue, hyperparams, save_dir):
    """The main worker function for each GPU process."""
    device = f'cuda:{rank}'
    print(f"Worker on GPU {rank} started.")

    # Generate a fixed test set for this worker
    X_test, y_test = generate_k_sparse_parity_data(
        hyperparams['P_test'], hyperparams['d'], hyperparams['k'], device=device
    )

    while not job_queue.empty():
        try:
            current_config = job_queue.get(timeout=1)
        except mp.queues.Empty:
            break

        P_train = current_config['P']
        
        # Generate training data
        X_train, y_train = generate_k_sparse_parity_data(
            P_train, hyperparams['d'], hyperparams['k'], device=device
        )

        # Instantiate and compile model
        model = TwoLayerNet(
            d=hyperparams['d'], N=hyperparams['N'], g_w=hyperparams['g_w'],
            g_a=hyperparams['g_a'], gamma_scaling_exponent=hyperparams['gamma_scaling_exponent']
        ).to(device)
        model = torch.compile(model)

        # Run training
        final_errors = train_with_langevin(
            model, X_train, y_train, X_test, y_test, hyperparams, current_config, rank
        )
        
        # Combine config and results, and save
        full_result = {**current_config, **final_errors, "gpu": rank}
        save_result(full_result, save_dir / "training_results.json")

    print(f"Worker on GPU {rank} finished.")

def main():
    """Main function to set up and run the distributed experiment."""
    # --- Performance Setting ---
    # This enables the use of TF32 cores on Ampere GPUs (like H100) for faster matmul.
    torch.set_float32_matmul_precision('high')

    hyperparams = {
        "d": 60, "k": 4, "N": 1000, "g_w": 1.0, "g_a": 1.0,
        "gamma_scaling_exponent": 2.0, "eta": 0.001, "epochs": 10000000,
        "log_interval": 10000, "P_test": 100000,
        "batch_size": 200000, # For gradient accumulation
        "early_stop_loss": 0.08, "early_stop_error": 0.01,
    }
    
    save_dir = Path("/home/goring/OnlineSGD/empirical_tests/results_2707_d60k4_3")
    save_dir.mkdir(exist_ok=True)

    # --- Experiment Grid ---
    P_values = [100,1000,2500,5000,7500, 10000,20000,30000,40000,50000,60000,70000,80000,90000, 100000, 200000,300000]
    kappa_0_values = [0.01, 0.1, 1.0,3.16]
    # --- Job Setup ---
    json_path = save_dir / "training_results.json"
    
    # Load existing results to avoid re-running completed jobs
    completed_jobs = set()
    if json_path.exists() and json_path.stat().st_size > 0:
        with open(json_path, 'r') as f:
            try:
                results = json.load(f)
                for r in results:
                    completed_jobs.add((r['P'], r['kappa_0']))
            except json.JSONDecodeError:
                print("Warning: Could not decode existing results.json. Starting fresh.")


    # Create job queue
    job_queue = mp.Queue()
    job_count = 0
    for P in P_values:
        for k0 in kappa_0_values:
            if (P, k0) not in completed_jobs:
                job_queue.put({"P": P, "kappa_0": k0})
                job_count += 1

    if job_count == 0:
        print("All experiments already completed. Exiting.")
        return

    print(f"Total jobs to run: {job_count}")
    
    # Save full hyperparams to a separate file
    with open(save_dir / "hyperparameters.json", 'w') as f:
        json.dump(hyperparams, f, indent=4)

    # --- Start Workers ---
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA devices found. This script requires a GPU.")
        return
    
    print(f"Found {world_size} GPUs. Starting workers...")
    mp.spawn(worker, args=(world_size, job_queue, hyperparams, save_dir), nprocs=world_size, join=True)
    
    print("\n--- All jobs completed ---")

if __name__ == '__main__':
    # Set start method for multiprocessing
    # 'fork' is default on Linux, 'spawn' is required on macOS/Windows
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
