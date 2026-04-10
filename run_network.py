# run_network.py 파일 내용
import numpy as np
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

# ---------------------------------------------------------
# 1. 필수 함수 정의 (원진 님이 쓰시던 함수 그대로 복붙하세요)
# ---------------------------------------------------------
def calculate_event_synchronization_exact(t_i, t_j, tau_max=8):
    """
    Event Synchronization calculating
    """
    n_i, n_j = len(t_i), len(t_j)
    
    if n_i < 2 or n_j < 2:
        return 0.0, 0.0

    # 1. 동적 타우(tau) 계산
    tau_i = np.empty(n_i)
    tau_i[0] = t_i[1] - t_i[0]
    tau_i[-1] = t_i[-1] - t_i[-2]
    tau_i[1:-1] = np.minimum(t_i[1:-1] - t_i[:-2], t_i[2:] - t_i[1:-1])

    tau_j = np.empty(n_j)
    tau_j[0] = t_j[1] - t_j[0]
    tau_j[-1] = t_j[-1] - t_j[-2]
    tau_j[1:-1] = np.minimum(t_j[1:-1] - t_j[:-2], t_j[2:] - t_j[1:-1])

    # 2. 시차 행렬 계산 (diff = t_j - t_i)
    diff_matrix = t_j[None, :] - t_i[:, None] 
    tau_matrix = 0.5 * np.minimum(tau_i[:, None], tau_j[None, :])
    
    # 3. 논문의 J_ij, J_ji 수식 완벽 적용
    
    # 수식 1: j가 먼저 발생하고 i가 나중에 발생 (t_i > t_j 즉, diff < 0)
    # 이미지의 J_ij 수식: 0 < t_i - t_j <= tau 
    # c(i|j)에 합산
    mask_ij = (diff_matrix < 0) & (-diff_matrix <= tau_max) & (-diff_matrix <= tau_matrix)
    c_ij = np.sum(mask_ij)
    
    # 수식 2: i가 먼저 발생하고 j가 나중에 발생 (t_j > t_i 즉, diff > 0)
    # 이미지의 J_ji 수식: 0 < t_j - t_i <= tau
    # c(j|i)에 합산
    mask_ji = (diff_matrix > 0) & (diff_matrix <= tau_max) & (diff_matrix <= tau_matrix)
    c_ji = np.sum(mask_ji)
    
    # 수식 3: 동시 발생 (t_i == t_j 즉, diff == 0)
    # J_ij, J_ji 모두 1/2 부여
    sync_same_day = np.sum(diff_matrix == 0)
    c_ij += 0.5 * sync_same_day
    c_ji += 0.5 * sync_same_day
    
    return c_ij, c_ji

def worker_function(s_pair, total_time_steps, num_surrogates=20, percentile=95):
    s1, s2 = s_pair
    fake_scores_12 = np.zeros(num_surrogates)
    fake_scores_21 = np.zeros(num_surrogates)
    
    for k in range(num_surrogates):
        fake_t1 = np.sort(np.random.choice(total_time_steps, s1, replace=False))
        fake_t2 = np.sort(np.random.choice(total_time_steps, s2, replace=False))
        
        c_12, c_21 = calculate_event_synchronization_exact(fake_t1, fake_t2, tau_max=8)
        fake_scores_12[k] = c_12
        fake_scores_21[k] = c_21
        
    return {
        (s1, s2): np.percentile(fake_scores_12, percentile),
        (s2, s1): np.percentile(fake_scores_21, percentile)
    }

# ---------------------------------------------------------
# 2. 메인 실행부
# ---------------------------------------------------------
if __name__ == '__main__':
    print("데이터 로딩 중...")
    
    # 1단계에서 저장한 재료들 불러오기
    C_matrix = np.load('C_matrix.npy')
    with open('epe_times.pkl', 'rb') as f:
        epe_times = pickle.load(f)
    
    # ★ 1단계에서 출력된 total_time_steps 숫자를 직접 적어주세요. (예: 8906)
    total_time_steps = 8906 # <--- 반드시 본인의 JJAS 전체 일수로 수정!
    
    num_grids = len(epe_times)
    s_array = np.array([len(times) for times in epe_times])

    required_s_pairs = set()
    i_indices, j_indices = np.nonzero(C_matrix)

    for i, j in zip(i_indices, j_indices):
        if i != j and s_array[i] >= 2 and s_array[j] >= 2:
            required_s_pairs.add(tuple(sorted((s_array[i], s_array[j]))))

    print(f"병렬 연산 시작 (총 {len(required_s_pairs)}개 조합)...")
    
    results = Parallel(n_jobs=-1)(
        delayed(worker_function)(pair, total_time_steps, 20, 95) 
        for pair in tqdm(list(required_s_pairs))
    )

    lookup_table = {}
    for res in results:
        lookup_table.update(res)

    print("A_matrix 구축 및 Divergence 계산 중...")
    A_matrix = np.zeros((num_grids, num_grids), dtype=np.uint8)
    for i, j in zip(i_indices, j_indices):
        if i != j and s_array[i] >= 2 and s_array[j] >= 2:
            if C_matrix[i, j] > lookup_table[(s_array[i], s_array[j])]:
                A_matrix[i, j] = 1

    k_out = np.sum(A_matrix, axis=1).astype(int)
    k_in = np.sum(A_matrix, axis=0).astype(int)
    network_divergence = k_out - k_in

    # 최종 결과물 저장
    np.save('A_matrix.npy', A_matrix)
    np.save('network_divergence.npy', network_divergence)
    print("완료! 결과가 A_matrix.npy 와 network_divergence.npy 로 저장되었습니다.")