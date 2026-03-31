import time

from env.task_env import TaskEnv
import numpy as np
import pickle
import pandas as pd
import glob
from natsort import natsorted


if __name__ == '__main__':
    folder = 'RALTestSet'
    method = 'greedy10'
    files = natsorted(glob.glob(f'../{folder}/env_*.pkl'), key=lambda y: y.lower())
    perf_metrics = {'success_rate':[], 'makespan': [], 'time_cost':[], 'waiting_time': [], 'travel_dist': [], 'efficiency': []}
    b = []
    for i in files:
        print(i)
        env = pickle.load(open(i, 'rb'))
        a = time.time()
        env.init_state()
        env.execute_greedy_action(i.replace('.pkl', '/'), method, False)
        reward, finished_tasks = env.get_episode_reward(100)
        if np.sum(finished_tasks) / len(finished_tasks) < 1:
            perf_metrics['success_rate'].append(np.sum(finished_tasks) / len(finished_tasks))
            perf_metrics['makespan'].append(np.nan)
            perf_metrics['time_cost'].append(np.nan)
            perf_metrics['waiting_time'].append(np.nan)
            perf_metrics['travel_dist'].append(np.nan)
            perf_metrics['efficiency'].append(np.nan)
        else:
            perf_metrics['success_rate'].append(np.sum(finished_tasks) / len(finished_tasks))
            perf_metrics['makespan'].append(env.current_time)
            perf_metrics['time_cost'].append(np.nanmean(np.nan_to_num(env.get_matrix(env.task_dic, 'time_start'), nan=100)))
            perf_metrics['waiting_time'].append(np.mean(env.get_matrix(env.agent_dic, 'sum_waiting_time')))
            perf_metrics['travel_dist'].append(np.sum(env.get_matrix(env.agent_dic, 'travel_dist')))
            perf_metrics['efficiency'].append(np.mean(env.get_efficiency()))
        b.append(time.time() - a)
    df = pd.DataFrame(perf_metrics)
    print(np.mean(b))
    df.to_csv(f'../{folder}/{method}.csv')
