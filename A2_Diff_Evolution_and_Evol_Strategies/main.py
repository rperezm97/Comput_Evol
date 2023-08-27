from multiprocessing import Pool
import numpy as np

def evaluate_ea(config_str, n_eval=5):
    EA_instance = Evolutive_algorithm(identifier_id=config_str)  # Assume the constructor sets the parameters
    tester = Test(EA=EA_instance, n_exe=n_eval, initial_exe=0, t_n=1.96)  # 95% confidence level, t_n=1.96
    tester.run_test()
    return tester.MBF[-1], tester.SR, tester.MT

def aggregate_results_into_tensor(results, param_grid_shape):
    MBF_tensor = np.zeros(param_grid_shape)
    SR_tensor = np.zeros(param_grid_shape)
    MT_tensor = np.zeros(param_grid_shape)
    
    idx = 0
    for i in range(param_grid_shape[0]):
        for j in range(param_grid_shape[1]):
            for k in range(param_grid_shape[2]):
                MBF, SR, MT = results[idx]
                MBF_tensor[i, j, k] = MBF
                SR_tensor[i, j, k] = SR
                MT_tensor[i, j, k] = MT
                idx += 1
                
    return MBF_tensor, SR_tensor, MT_tensor
def perform_grid_search(param_grid):
    param_combinations = [f"p1{p1}_p2{p2}_p3{p3}" for p1 in param_grid['p1'] for p2 in param_grid['p2'] for p3 in param_grid['p3']]

    with Pool() as pool:
        results = pool.map(evaluate_ea, param_combinations)

    return results


if __name__ == "__main__":
    param_grid = {'p1': [0.3, 0.6], 'p2': [0.1, 0.5], 'p3': [0.7, 0.9]}
    param_grid_shape = (len(param_grid['p1']), len(param_grid['p2']), len(param_grid['p3']))
    
    results = perform_grid_search(param_grid)
    MBF_tensor, SR_tensor, MT_tensor = aggregate_results_into_tensor(results, param_grid_shape)
    
    print(f"MBF Tensor: {MBF_tensor}")
    print(f"SR Tensor: {SR_tensor}")
    print(f"MT Tensor: {MT_tensor}")
