import os.path
import random

from env.task_env import TaskEnv
import numpy as np
import yaml
import pickle
import math
from itertools import permutations
from collections import defaultdict

def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = math.hypot((from_node[0] - to_node[0]),(from_node[1] - to_node[1]))
    return distances


agent_yaml = dict()
task_yaml = dict()
planner_param = dict()
graph_yaml = dict()
folder = 'RALTestSet'
planner = 'TEAMPLANNER_CONDET'
solver_time = 600.0
for i in range(50):
    print(f'GENERATING THE  param file no {i+1}')
    env = pickle.load(open(f'{folder}/env_{i}.pkl', 'rb'))
    # env = TaskEnv((3, 5), (3, 5), (20, 30), 5, 10, seed=i)
    # if os.path.exists(f'{folder}/env_{i}.pkl'):
    if not os.path.exists(f'{folder}/env_{i}'):
        os.mkdir(f'{folder}/env_{i}')
        # if os.path.exists(f'{folder}/env_{i}/*.yaml'):
        #     continue
    # env.force_waiting = True
    coords = env.get_matrix(env.task_dic, 'location')
    dist_matrix = compute_euclidean_distance_matrix(coords)
    depot_distance = []
    if planner == 'TEAMPLANNER_CONDET':
        new_depots = env.get_matrix(env.depot_dic, 'location')
        for index in range(len(new_depots)):
            distance_list = []

            for j in range(len(coords)):
                distance_list.append(math.hypot((new_depots[index][0] - coords[j][0]), (new_depots[index][1] - coords[j][1])))
            depot_distance.append(distance_list)
    elif planner == 'TEAMPLANNER_DET':

        new_depots = env.get_matrix(env.agent_dic, 'depot')
        for index in range(len(new_depots)):
            distance_list = []

            for j in range(len(coords)):
                distance_list.append(math.hypot((new_depots[index][0] - coords[j][0]), (new_depots[index][1] - coords[j][1])))
            depot_distance.append(distance_list)

    p = list(permutations(range(len(env.task_dic)), 2))
    if planner == 'TEAMPLANNER_CONDET':
        for a in range(len(env.depot_dic)):
            capVec = env.species_dict['abilities'][a].tolist()

            agent_yaml.update({f'vehicle{a}': {'engCap': 1e6, 'engCost': 0., 'capVector': capVec, 'capVar': [0.0]*env.traits_dim}})
            graph_yaml.update({f'vehicle{a}': {
                f'edge{i}': [t[0], t[1], 0, dist_matrix[t[0]][t[1]], 0, float(dist_matrix[t[0]][t[1]]/0.2)] for i, t in enumerate(p)
            }
            })
            for j in range(len(env.task_dic)):
                graph_yaml[f'vehicle{a}'][f'edge{2*j+len(p)}'] = [int(env.tasks_num)+a, j, 0, depot_distance[a][j], 0, depot_distance[a][j]/0.2]
                graph_yaml[f'vehicle{a}'][f'edge{2*j+len(p)+1}'] = [j, int(env.tasks_num) + int(len(env.depot_dic)) + a, 0, depot_distance[a][j], 0, depot_distance[a][j]/0.2]
            for id, task in env.task_dic.items():
                graph_yaml[f'vehicle{a}'][f'node{id}'] = float(task['time'])
    elif planner == 'TEAMPLANNER_DET':
        for a in range(len(env.agent_dic)):
            capVec = [0.0]*(env.species_num)
            capVec[env.agent_dic[a]['species']] = 1.0
            agent_yaml.update({f'vehicle{a}': {'engCap': 1e6, 'engCost': 1., 'capVector': capVec, 'capVar': [0.0]*env.species_num}})
            graph_yaml.update({f'vehicle{a}': {
                f'edge{i}': [t[0], t[1], 0, dist_matrix[t[0]][t[1]], 0, float(dist_matrix[t[0]][t[1]]/0.2)] for i, t in enumerate(p)
            }
            })
            for j in range(len(env.task_dic)):
                graph_yaml[f'vehicle{a}'][f'edge{2*j+len(p)}'] = [int(env.tasks_num) + a, j, 0, depot_distance[a][j], 0, depot_distance[a][j]/0.2]
                graph_yaml[f'vehicle{a}'][f'edge{2*j+len(p)+1}'] = [j, int(env.tasks_num) + int(env.agents_num) + a, 0, depot_distance[a][j], 0, depot_distance[a][j]/0.2]
            for id, task in env.task_dic.items():
                graph_yaml[f'vehicle{a}'][f'node{id}'] = float(task['time'])

    for task in env.task_dic.values():
        task_yaml[f'task{task["ID"]}'] = {}
        k = 0
        for m in range(env.traits_dim):
            if task['requirements'][m] != 0:
                and_dict = {'or0': {'geq': True, 'capId': int(m), 'capReq': float(task['requirements'][m]), 'capVar': 0.}}

                task_yaml[f'task{task["ID"]}'][f'and{k}'] = and_dict
                k += 1
    with open(f'{folder}/env_{i}/vehicle_param.yaml', 'w') as f:
        yaml.dump(agent_yaml, f, sort_keys=False)
    with open(f'{folder}/env_{i}/task_param.yaml', 'w') as f:
        yaml.dump(task_yaml, f, default_flow_style=False)
    planner_param = {
                    'flagOptimizeCost': True,
                    'flagTaskComplete': True,
                    'flagSprAddCutToSameType': True,
                    'taskCompleteReward': 10000,
                    'timePenalty': 100,
                    'recoursePenalty': 1.0,
                    'taskRiskPenalty': 0.0,
                    'LARGETIME': 10000.0,
                    'MAXTIME': 1000.0,
                    'MAXENG': 1E8,
                    'flagSolver': planner,
                    'CcpBeta': 0.95,
                    'taskBeta': 0.95,
                    'solverMaxTime': solver_time,
                    'solverIterMaxTime': 50.0,
                    'flagNotUseUnralavant': True,
                    'MAXALPHA': 20.0,
                    'taskNum': int(env.tasks_num),
                    'vehNum': env.species_num if planner == 'TEAMPLANNER_CONDET' else int(env.agents_num),
                    'capNum': env.traits_dim,
                    'vehTypeNum': env.species_num,
                    'vehNumPerType': [int(i) for i in env.species_dict['number']],
                    'sampleNum': 500,
                    'randomType': 0,
                    'capType': [0]*env.traits_dim,
                    'vehicleParamFile': f'./{folder}/env_{i}/vehicle_param.yaml',
                    'taskParamFile': f'./{folder}/env_{i}/task_param.yaml',
                    'graphFile': f'./{folder}/env_{i}/graph.yaml'
    }
    with open(f'{folder}/env_{i}/planner_param.yaml', 'w') as f:
        yaml.dump(planner_param, f, sort_keys=False)
    with open(f'{folder}/env_{i}/graph.yaml', 'w') as f:
        yaml.dump(graph_yaml, f, sort_keys=False)
    print(f'COMPLETED GENERATING THE  param file no {i + 1}')
