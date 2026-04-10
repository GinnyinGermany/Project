# run_enso_network.py 내용
import pandas as pd
import numpy as np
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
import os

# ---------------------------------------------------------
# 1. 필수 함수 정의 (Event Synchronization)
# ---------------------------------------------------------
def calculate_event_synchronization_exact(t_i, t_j, tau_max=8):
    n_i, n_j = len(t_i), len(t_j)
    if n_i < 2 or n_j < 2: return 0.0, 0.0
    
    tau_i = np.empty(n_i)
    tau_i[0] = t_i[1] - t_i[0]
    tau_i[-1] = t_i[-1] - t_i[-2]
    tau_i[1:-1] = np.minimum(t_i[1:-1] - t_i[:-2], t_i[2:] - t_i[1:-1])

    tau_j = np.empty(n_j)
    tau_j[0] = t_j[1] - t_j[0]
    tau_j[-1] = t_j[-1] - t_j[-2]
    tau_j[1:-1] = np.minimum(t_j[1:-1] - t_j[:-2], t_j[2:] - t_j[1:-1])

    diff_matrix = t_j[None, :] - t_i[:, None] 
    tau_matrix = 0.5 * np.minimum(tau_i[:, None], tau_j[None, :])
    
    mask_ij = (diff_matrix < 0) & (-diff_matrix <= tau_max) & (-diff_matrix <= tau_matrix)
    c_ij = np.sum(mask_ij)
    
    mask_ji = (diff_matrix > 0) & (diff_matrix <= tau_max) & (diff_matrix <= tau_matrix)
    c_ji = np.sum(mask_ji)
    
    sync_same_day = np.sum(diff_matrix == 0)
    c_ij += 0.5 * sync_same_day
    c_ji += 0.5 * sync_same_day
    
    return c_ij, c_ji

# ---------------------------------------------------------
# 2. 널 모델 워커 함수
# ---------------------------------------------------------
def worker_function(s_pair, valid_days, num_surrogates=20, percentile=95):
    s1, s2 = s_pair
    fake_scores_12 = np.zeros(num_surrogates)
    fake_scores_21 = np.zeros(num_surrogates)
    
    for k in range(num_surrogates):
        fake_t1 = np.sort(np.random.choice(valid_days, s1, replace=False))
        fake_t2 = np.sort(np.random.choice(valid_days, s2, replace=False))
        
        c_12, c_21 = calculate_event_synchronization_exact(fake_t1, fake_t2, tau_max=8)
        fake_scores_12[k] = c_12
        fake_scores_21[k] = c_21
        
    return {(s1, s2): np.percentile(fake_scores_12, percentile),
            (s2, s1): np.percentile(fake_scores_21, percentile)}

# ---------------------------------------------------------
# 3. 네트워크 구축 자동화 함수
# ---------------------------------------------------------
def build_network(subset_epe_times, valid_days, phase_name):
    num_grids = len(subset_epe_times)
    s_array = np.array([len(times) for times in subset_epe_times])
    C_mat = np.zeros((num_grids, num_grids))
    
    print(f"[{phase_name}] C_matrix 계산 중...")
    for i in tqdm(range(num_grids), desc=f"{phase_name} C_mat"):
        for j in range(i):
            c_ij, c_ji = calculate_event_synchronization_exact(subset_epe_times[i], subset_epe_times[j], tau_max=8)
            C_mat[i, j] = c_ij
            C_mat[j, i] = c_ji 
            
    i_indices, j_indices = np.nonzero(C_mat)
    required_s_pairs = set()
    for i, j in zip(i_indices, j_indices):
        if i != j and s_array[i] >= 2 and s_array[j] >= 2:
            required_s_pairs.add(tuple(sorted((s_array[i], s_array[j]))))
                
    print(f"[{phase_name}] 널 모델 병렬 계산 시작 (조합: {len(required_s_pairs)}개)...")
    results = Parallel(n_jobs=-1)(
        delayed(worker_function)(pair, valid_days, 20, 95) 
        for pair in tqdm(list(required_s_pairs), desc=f"{phase_name} Null")
    )
        
    lookup_table = {}
    for res in results: lookup_table.update(res)
        
    A_mat = np.zeros((num_grids, num_grids), dtype=np.uint8)
    for i, j in zip(i_indices, j_indices):
        if i != j and s_array[i] >= 2 and s_array[j] >= 2:
            if C_mat[i, j] > lookup_table[(s_array[i], s_array[j])]:
                A_mat[i, j] = 1
                
    divergence = np.sum(A_mat, axis=1).astype(int) - np.sum(A_mat, axis=0).astype(int)
    return A_mat, divergence

# ---------------------------------------------------------
# 4. 메인 실행부
# ---------------------------------------------------------
if __name__ == '__main__':
    # 데이터 로드
    with open('epe_times.pkl', 'rb') as f: epe_times = pickle.load(f)
    with open('jjas_columns.pkl', 'rb') as f: jjas_columns = pickle.load(f)
    enso_df = pd.read_csv('Nina34_anom.csv')
    
    # ENSO 분류 로직
    enso_df.columns = [col.strip() for col in enso_df.columns]
    enso_df['Date'] = pd.to_datetime(enso_df['Date'])
    jjas_enso = enso_df[enso_df['Date'].dt.month.isin([6, 7, 8, 9])]
    yearly_enso = jjas_enso.groupby(jjas_enso['Date'].dt.year)['Nino Anom 3.4 Index'].mean().reset_index()
    
    el_nino_years = yearly_enso[yearly_enso['Nino Anom 3.4 Index'] >= 0.5]['Date'].values
    la_nina_years = yearly_enso[yearly_enso['Nino Anom 3.4 Index'] <= -0.5]['Date'].values
    
    jjas_dates = pd.to_datetime(jjas_columns)
    valid_days_elnino = np.where(jjas_dates.year.isin(el_nino_years))[0]
    valid_days_lanina = np.where(jjas_dates.year.isin(la_nina_years))[0]
    
    # 위상별 데이터 필터링
    epe_elnino = [t[np.isin(t, valid_days_elnino)] for t in epe_times]
    epe_lanina = [t[np.isin(t, valid_days_lanina)] for t in epe_times]
    
    # 연산 실행
    A_el, div_el = build_network(epe_elnino, valid_days_elnino, "El Nino")
    A_la, div_la = build_network(epe_lanina, valid_days_lanina, "La Nina")
    
    # 결과 저장
    np.save('A_matrix_elnino.npy', A_el)
    np.save('div_elnino.npy', div_el)
    np.save('A_matrix_lanina.npy', A_la)
    np.save('div_lanina.npy', div_la)
    
    print("\n✅ 모든 ENSO 분석 완료 및 파일 저장 성공!")