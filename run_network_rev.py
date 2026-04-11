import numpy as np
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

# =====================================================================
# [CONTROL PANEL] - Adjust research parameters here
# =====================================================================
CONFIG = {
    'tau_max': 7,            # Maximum dynamic delay (days)
    'num_surrogates': 100,  # Number of Null Model shuffles (Recommended: 1000)
    'percentile': 95,        # Significance level (95th or 99th percentile)
    'total_time_steps': 8906, # Total time steps (JJAS days)
    'n_jobs' : -1            # Use all available CPU cores
}

# =====================================================================
# 1. Event Synchronization Function
# =====================================================================
def calculate_event_synchronization_exact(t_i, t_j, tau_max):
    """
    Calculate Event Synchronization (ES) between two time series of extreme events.
    Based on the dynamic time delay (tau) method.
    """
    n_i, n_j = len(t_i), len(t_j)
    
    if n_i < 2 or n_j < 2:
        return 0.0, 0.0

    # 1. Calculate dynamic tau for both time series
    tau_i = np.empty(n_i)
    tau_i[0] = t_i[1] - t_i[0]
    tau_i[-1] = t_i[-1] - t_i[-2]
    tau_i[1:-1] = np.minimum(t_i[1:-1] - t_i[:-2], t_i[2:] - t_i[1:-1])
    
    tau_j = np.empty(n_j)
    tau_j[0] = t_j[1] - t_j[0]
    tau_j[-1] = t_j[-1] - t_j[-2]
    tau_j[1:-1] = np.minimum(t_j[1:-1] - t_j[:-2], t_j[2:] - t_j[1:-1])

    # 2. Compute time difference matrix (diff = t_j - t_i)
    diff_matrix = t_j[None, :] - t_i[:, None] 
    tau_matrix = 0.5 * np.minimum(tau_i[:, None], tau_j[None, :])
    
    # 3. Apply J_ij and J_ji conditions
    # Condition 1: Event j occurs before event i (t_i > t_j, hence diff < 0)
    # Adds to c(i|j)
    mask_ij = (diff_matrix < 0) & (-diff_matrix <= tau_max) & (-diff_matrix <= tau_matrix)
    c_ij = np.sum(mask_ij)
    
    # Condition 2: Event i occurs before event j (t_j > t_i, hence diff > 0)
    # Adds to c(j|i)
    mask_ji = (diff_matrix > 0) & (diff_matrix <= tau_max) & (diff_matrix <= tau_matrix)
    c_ji = np.sum(mask_ji)
    
    # Condition 3: Simultaneous occurrence (t_i == t_j, hence diff == 0)
    # Distribute equally (0.5 to both directions)
    sync_same_day = np.sum(diff_matrix == 0)
    c_ij += 0.5 * sync_same_day
    c_ji += 0.5 * sync_same_day
    
    return c_ij, c_ji
# =====================================================================
# 2. Worker Functions for Parallelism
# =====================================================================
def compute_c_pair(i, j, times_i, times_j, tau_max):
    c_ij, c_ji = calculate_event_synchronization_exact(times_i, times_j, tau_max)
    return i, j, c_ij, c_ji

def worker_function(s_pair, time_steps, n_surr, p_level, t_max):
    """
    Generates surrogate distributions and returns the significance threshold.
    """
    s1, s2 = s_pair
    fake_scores_12 = np.zeros(n_surr)
    fake_scores_21 = np.zeros(n_surr)
    
    rng = np.random.default_rng()
    for k in range(n_surr):
        # Randomly shuffle events within the valid time frame
        fake_t1 = np.sort(rng.choice(time_steps, s1, replace=False))
        fake_t2 = np.sort(rng.choice(time_steps, s2, replace=False))
        
        c_12, c_21 = calculate_event_synchronization_exact(fake_t1, fake_t2, tau_max=t_max)
        fake_scores_12[k] = c_12
        fake_scores_21[k] = c_21
        
    return {
        (s1, s2): np.percentile(fake_scores_12, p_level),
        (s2, s1): np.percentile(fake_scores_21, p_level)
    }

# =====================================================================
# 3. Main Execution Block
# =====================================================================
if __name__ == '__main__':
    print(f"Initializing Network Construction with CONFIG: {CONFIG}")
    
    # Load Event Times
    with open('epe_times.pkl', 'rb') as f:
        epe_times = pickle.load(f)
    
    num_grids = len(epe_times)
    s_array = np.array([len(times) for times in epe_times])

    # [Step 1] Parallel C_matrix Calculation
    print(f"Computing C_matrix based on tau_max = {CONFIG['tau_max']}...")
    pairs = [(i, j) for i in range(num_grids) for j in range(i)]
    c_results = Parallel(n_jobs=CONFIG['n_jobs'])(
        delayed(compute_c_pair)(p[0], p[1], epe_times[p[0]], epe_times[p[1]], CONFIG['tau_max']) 
        for p in tqdm(pairs)
    )
    C_matrix = np.zeros((num_grids, num_grids))
    for i, j, c_ij, c_ji in c_results:
        C_matrix[i, j], C_matrix[j, i] = c_ij, c_ji

    # [Step 2] Identify Unique Combinations for Null Model
    required_s_pairs = set()
    i_idx, j_idx = np.nonzero(C_matrix)

    for i, j in zip(i_idx, j_idx):
        if i != j and s_array[i] >= 2 and s_array[j] >= 2:
            required_s_pairs.add(tuple(sorted((s_array[i], s_array[j]))))

    # [Step 3] Parallel Null Model (Lookup Table)
    print(f"Parallel Null Model (Surrogates={CONFIG['num_surrogates']}, Unique Pairs={len(required_s_pairs)})...")
    null_results = Parallel(n_jobs=CONFIG['n_jobs'])(
        delayed(worker_function)(
            pair, CONFIG['total_time_steps'], CONFIG['num_surrogates'], CONFIG['percentile'], CONFIG['tau_max']
        ) for pair in tqdm(list(required_s_pairs), desc="Null Model")
    )

    lookup_table = {}
    for res in null_results: lookup_table.update(res)

    # [Step 4] Adjacency Matrix & Metrics
    print("Building Adjacency Matrix (A_matrix) and calculating Divergence...")
    A_matrix = np.zeros((num_grids, num_grids), dtype=np.uint8)
    for i, j in zip(i_idx, j_idx):
        if i != j and s_array[i] >= 2 and s_array[j] >= 2:
            # Significant edges: Observed ES strength > Null Model threshold
            if C_matrix[i, j] > lookup_table[(s_array[i], s_array[j])]:
                A_matrix[i, j] = 1

    # Calculate Network Metrics
    k_out = np.sum(A_matrix, axis=1).astype(int)
    k_in = np.sum(A_matrix, axis=0).astype(int)
    network_divergence = k_out - k_in

    # [Step 5] Save Results
    np.save(f"C_matrix_tau{CONFIG['tau_max']}.npy", C_matrix)
    np.save(f"A_matrix_tau{CONFIG['tau_max']}_surr{CONFIG['num_surrogates']}.npy", A_matrix)
    np.save(f"network_divergence_tau{CONFIG['tau_max']}.npy", network_divergence)
    
    print(f"!! Analysis complete !! Config used: {CONFIG}")
    print("Files saved with dynamic naming based on CONFIG.")