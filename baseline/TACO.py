import copy
import os.path
from env.task_env import TaskEnv
import math
import numpy as np
import pickle


def baseline(env, path):
    if os.path.exists(path):
        routes = pickle.load(open(path, 'rb'))
        if routes is None:
            return None
        for agent_id, route in enumerate(routes):
            env.pre_set_route(route[1:], agent_id)
        return True


if __name__ == '__main__':
    import pickle
    import pandas as pd
    import glob
    from natsort import natsorted
    time = []
    folder = 'RALTestSet'
    method = 'sas'  # 'sas'
    files = natsorted(glob.glob(f'../{folder}/env_*.pkl'), key=lambda y: y.lower())
    perf_metrics = {'success_rate':[], 'makespan': [], 'time_cost':[], 'waiting_time': [], 'travel_dist': [], 'efficiency': []}
    for i in files:
        print(i)
        env = pickle.load(open(i, 'rb'))
        env.init_state()
        re = baseline(env, i.replace('.pkl', '/') + method + '.solution')
        if re is None:
            perf_metrics['success_rate'].append(np.nan)
            perf_metrics['makespan'].append(np.nan)
            perf_metrics['time_cost'].append(np.nan)
            perf_metrics['waiting_time'].append(np.nan)
            perf_metrics['travel_dist'].append(np.nan)
            perf_metrics['efficiency'].append(np.nan)
            continue
        env.force_wait = True
        env.execute_by_route(i.replace('.pkl', '/'), method, False)
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
    df = pd.DataFrame(perf_metrics)
    df.to_csv(f'../{folder}/{method}.csv')