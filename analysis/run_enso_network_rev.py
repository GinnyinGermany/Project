import pandas as pd
import numpy as np
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
import os

# =====================================================================
# [CONTROL PANEL] - Adjust research parameters here
# =====================================================================
CONFIG = {
    'tau_max': 5,            # Maximum dynamic delay (days)
    'num_surrogates': 100,  # Number of Null Model shuffles (Standard: 1000)
    'percentile': 95,        # Significance level (95th or 99th percentile)
    'n_jobs' : -1            # Use all available CPU cores
}

# =====================================================================
# 1. Core Event Synchronization Function
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
# 2. Parallel Computing Wrappers
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
        fake_t1 = np.sort(rng.choice(time_steps, size=s1, replace=False))
        fake_t2 = np.sort(rng.choice(time_steps, size=s2, replace=False))
        
        c_12, c_21 = calculate_event_synchronization_exact(fake_t1, fake_t2, tau_max=t_max)
        fake_scores_12[k] = c_12
        fake_scores_21[k] = c_21
        
    return {
        (s1, s2): np.percentile(fake_scores_12, p_level),
        (s2, s1): np.percentile(fake_scores_21, p_level)
    }

# =====================================================================
# 3. Network Construction Pipeline
# =====================================================================
def build_enso_network(subset_epe_times, valid_days, phase_name):
    """
    Constructs the optimized network for a specific ENSO phase.
    Includes parallelized C_matrix and significance testing via Lookup Table.
    """
    num_grids = len(subset_epe_times)
    s_array = np.array([len(times) for times in subset_epe_times])
    C_mat = np.zeros((num_grids, num_grids))
    
    # Step 1: Parallel C_matrix Calculation
    print(f"[{phase_name}] Phase: Computing C_matrix...")
    pairs = [(i, j) for i in range(num_grids) for j in range(i)]
    c_results = Parallel(n_jobs=CONFIG['n_jobs'])(
        delayed(compute_c_pair)(p[0], p[1], subset_epe_times[p[0]], subset_epe_times[p[1]], CONFIG['tau_max']) 
        for p in tqdm(pairs, desc=f"{phase_name} ES Strength")
    )
    for i, j, c_ij, c_ji in c_results:
        C_mat[i, j], C_mat[j, i] = c_ij, c_ji
            
    # Step 2: Identify unique event frequency pairs for the Lookup Table
    i_indices, j_indices = np.nonzero(C_mat)
    required_s_pairs = set()
    for i, j in zip(i_indices, j_indices):
        if i != j and s_array[i] >= 2 and s_array[j] >= 2:
            required_s_pairs.add(tuple(sorted((s_array[i], s_array[j]))))
                
    # Step 3: Parallel Null Model Simulation
    print(f"[{phase_name}] Phase: Parallel Null Model (Pairs: {len(required_s_pairs)})...")
    null_results = Parallel(n_jobs=CONFIG['n_jobs'])(
        delayed(worker_function)(pair, valid_days, CONFIG['num_surrogates'], CONFIG['percentile'], CONFIG['tau_max']) 
        for pair in tqdm(list(required_s_pairs), desc=f"{phase_name} Significance")
    )
    
    lookup_table = {}
    for res in null_results: lookup_table.update(res)
        
    # Step 4: Filter significant edges and calculate Divergence
    A_mat = np.zeros((num_grids, num_grids), dtype=np.uint8)
    for i, j in zip(i_indices, j_indices):
        if i != j and s_array[i] >= 2 and s_array[j] >= 2:
            if C_mat[i, j] > lookup_table[(s_array[i], s_array[j])]:
                A_mat[i, j] = 1
                
    divergence = np.sum(A_mat, axis=1).astype(int) - np.sum(A_mat, axis=0).astype(int)
    
    # Save phase-specific results
    suffix = f"{phase_name}_tau{CONFIG['tau_max']}_surr{CONFIG['num_surrogates']}"
    np.save(f'A_matrix_{suffix}.npy', A_mat)
    np.save(f'divergence_{suffix}.npy', divergence)
    
    return A_mat, divergence

# =====================================================================
# 4. Main Execution Logic
# =====================================================================
if __name__ == '__main__':
    # 1. Data Initialization
    with open('epe_times.pkl', 'rb') as f: epe_times = pickle.load(f)
    with open('jjas_columns.pkl', 'rb') as f: jjas_columns = pickle.load(f)
    enso_df = pd.read_csv('/Users/kim-wonjin/Documents/CLEWS/Python/Nonlinear/Project/data/Nina34_anom.csv')
    
    # 2. ENSO Phase Classification
    enso_df.columns = [col.strip() for col in enso_df.columns]
    enso_df['Date'] = pd.to_datetime(enso_df['Date'])
    jjas_enso = enso_df[enso_df['Date'].dt.month.isin([6, 7, 8, 9])]
    yearly_enso = jjas_enso.groupby(jjas_enso['Date'].dt.year)['Nino Anom 3.4 Index'].mean().reset_index()
    
    el_nino_years = yearly_enso[yearly_enso['Nino Anom 3.4 Index'] >= 0.5]['Date'].values
    la_nina_years = yearly_enso[yearly_enso['Nino Anom 3.4 Index'] <= -0.5]['Date'].values
    
    jjas_dates = pd.to_datetime(jjas_columns)
    valid_days_elnino = np.where(jjas_dates.year.isin(el_nino_years))[0]
    valid_days_lanina = np.where(jjas_dates.year.isin(la_nina_years))[0]
    
    # 3. Filtering Event Series based on Phase-Specific Temporal Domain
    epe_elnino = [t[np.isin(t, valid_days_elnino)] for t in epe_times]
    epe_lanina = [t[np.isin(t, valid_days_lanina)] for t in epe_times]
    
    # 4. Execute Analysis Pipeline for both phases
    print("Starting Comparative ENSO Network Analysis...")
    A_el, div_el = build_enso_network(epe_elnino, valid_days_elnino, "ElNino")
    A_la, div_la = build_enso_network(epe_lanina, valid_days_lanina, "LaNina")
    
    print(f"\nAll ENSO phases analyzed. Config applied: {CONFIG}")