import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from itertools import combinations, product
import copy


class TaskEnv:
    def __init__(self, per_species_range=(10, 10), species_range=(5, 5), tasks_range=(30, 30), traits_dim=5,
                 decision_dim=10, max_task_size=2, duration_scale=5, seed=None, plot_figure=False):
        """
        :param traits_dim: number of capabilities in this problem, e.g. 3 traits
        :param seed: seed to generate pseudo random problem instance
        """
        # 随机数生成器（传入 seed 时使用可复现实验）
        self.rng = None
        # 环境规模与任务生成参数
        self.per_species_range = per_species_range
        self.species_range = species_range
        self.tasks_range = tasks_range
        self.max_task_size = max_task_size
        self.duration_scale = duration_scale
        # 是否启用轨迹动画绘制
        self.plot_figure = plot_figure
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # 能力维度与决策特征维度
        self.traits_dim = traits_dim
        self.decision_dim = decision_dim

        # 生成任务/智能体/仓库/物种信息与距离邻接结构
        self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()
        self.species_distance_matrix, self.species_neighbor_matrix = self.generate_distance_matrix()
        # self.species_mask = self.calculate_optimized_ability()
        # 记录当前环境规模
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict['number'])
        # 联盟分配矩阵：行对应 agent，列对应 task
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        # self.best_route = self.calculate_tsp_route()

        # 仿真时钟与运行状态
        self.current_time = 0
        self.dt = 0.1
        self.max_waiting_time = 200
        self.depot_waiting_time = 0
        self.finished = False
        self.reactive_planning = False

    def random_int(self, low, high, size=None):
        # 优先使用实例级 RNG（可复现）；否则回退到全局 numpy 随机数
        if self.rng is not None:
            integer = self.rng.integers(low, high, size)
        else:
            integer = np.random.randint(low, high, size)
        return integer

    def random_value(self, row, col):
        # 优先使用实例级 RNG（可复现）；否则回退到全局 numpy 随机数
        if self.rng is not None:
            value = self.rng.random((row, col))
        else:
            value = np.random.rand(row, col)
        return value

    def random_choice(self, a, size=None, replace=True):
        # 优先使用实例级 RNG（可复现）；否则回退到全局 numpy 随机数
        if self.rng is not None:
            choice = self.rng.choice(a, size, replace)
        else:
            choice = np.random.choice(a, size, replace)
        return choice

    def generate_task(self, tasks_num):
        # 随机生成任务需求矩阵：(任务数, 能力维度)
        tasks_ini = self.random_int(0, self.max_task_size, (tasks_num, self.traits_dim))

        # 若存在全零任务，则重新生成
        while not np.all(np.sum(tasks_ini, axis=1) != 0):
            tasks_ini = self.random_int(0, self.max_task_size, (tasks_num, self.traits_dim))

        # 固定任务需求示例（已注释）
        # tasks_ini = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [2, 1, 0, 0, 0],
        #                       [1, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 1, 0, 0, 0]])

        # 返回任务需求
        return tasks_ini

    def generate_agent(self, species_num):
        # agents_ini = self.random_value(species_num, self.traits_dim) > 0.8
        # while not np.all(np.sum(agents_ini, axis=1) != 0):
        #     agents_ini = self.random_value(species_num, self.traits_dim) > 0.8

        # 随机生成各物种的能力矩阵（元素取 0 或 1）
        agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))

        # 若存在全零能力物种，或物种能力重复，则重新生成
        while not np.all(np.sum(agents_ini, axis=1) != 0) or np.unique(agents_ini, axis=0).shape[0] != species_num:
            agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))

        # 固定能力矩阵示例（已注释）
        # agents_ini = np.diag(np.ones(self.traits_dim))
        # agents_ini = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])

        # 返回生成的物种能力矩阵
        return agents_ini

    def generate_env(self):
        # 随机生成任务数量
        tasks_num = self.random_int(self.tasks_range[0], self.tasks_range[1] + 1)

        # 随机生成物种数量
        species_num = self.random_int(self.species_range[0], self.species_range[1] + 1)

        # 随机生成每个物种的智能体数量
        agents_species_num = [
            self.random_int(self.per_species_range[0], self.per_species_range[1] + 1)
            for _ in range(species_num)
        ]

        # 生成物种能力矩阵和任务需求矩阵
        agents_ini = self.generate_agent(species_num)
        tasks_ini = self.generate_task(tasks_num)

        # 若当前所有物种总能力无法覆盖任务需求，则重新生成
        while not np.all(np.matmul(agents_species_num, agents_ini) >= tasks_ini):
            agents_ini = self.generate_agent(species_num)
            tasks_ini = self.generate_task(tasks_num)

        # 随机生成各物种仓库位置
        depot_loc = self.random_value(species_num, 2)

        # 随机生成各物种执行成本
        cost_ini = [self.random_value(1, 1) for _ in range(species_num)]

        # 随机生成任务位置
        tasks_loc = self.random_value(tasks_num, 2)

        # 随机生成任务持续时间
        tasks_time = self.random_value(tasks_num, 1) * self.duration_scale

        # 初始化任务、智能体、仓库和物种信息字典
        task_dic = dict()
        agent_dic = dict()
        depot_dic = dict()
        species_dict = dict()

        # 记录物种能力矩阵和各物种数量
        species_dict['abilities'] = agents_ini
        species_dict['number'] = agents_species_num

        # 任务字典结构：{任务ID: {需求、位置、持续时间、状态等信息}}
        for i in range(tasks_num):
            task_dic[i] = {'ID': i,
                           'requirements': tasks_ini[i, :],  # requirements of the task
                           'members': [],  # members of the task
                           'cost': [],  # cost of each agent
                           'location': tasks_loc[i, :],  # location of the task
                           'feasible_assignment': False,  # whether the task assignment is feasible
                           'finished': False,
                           'time_start': 0,
                           'time_finish': 0,
                           'status': tasks_ini[i, :],
                           'time': float(tasks_time[i, :]),
                           'sum_waiting_time': 0,
                           'efficiency': 0,
                           'abandoned_agent': [],
                           'optimized_ability': None,
                           'optimized_species': []}

        # agent字典结构：{智能体ID: {物种、能力、位置、路径、状态等信息}}
        i = 0
        for s, n in enumerate(agents_species_num):
            species_dict[s] = []
            for j in range(n):
                agent_dic[i] = {'ID': i,
                                'species': s,
                                'abilities': agents_ini[s, :],
                                'location': depot_loc[s, :],
                                'route': [- s - 1],
                                'current_task': - s - 1,
                                'contributed': False,
                                'arrival_time': [0.],
                                'cost': cost_ini[s],
                                'travel_time': 0,
                                'velocity': 0.2,
                                'next_decision': 0,
                                'depot': depot_loc[s, :],
                                'travel_dist': 0,
                                'sum_waiting_time': 0,
                                'current_action_index': 0,
                                'decision_step': 0,
                                'task_waiting_ratio': 1,
                                'trajectory': [],
                                'angle': 0,
                                'returned': False,
                                'assigned': False,
                                'pre_set_route': None,
                                'no_choice': False}
                species_dict[s].append(i)
                i += 1

        # 仓库字典结构：{仓库ID: {位置、成员等信息}}
        for s in range(species_num):
            depot_dic[s] = {'location': depot_loc[s, :],
                            'members': species_dict[s],
                            'ID': - s - 1}

        return task_dic, agent_dic, depot_dic, species_dict

    def generate_distance_matrix(self):
        # 初始化各物种的距离矩阵和邻居排序矩阵
        species_distance_matrix = {}
        species_neighbor_matrix = {}

        # 遍历每个物种
        for species in range(len(self.species_dict['number'])):
            # 将该物种的仓库和所有任务合并为节点集合
            tmp_dic = {-1: self.depot_dic[species], **self.task_dic}
            distances = {}

            # 计算任意两个节点之间的距离
            for from_counter, from_node in tmp_dic.items():
                distances[from_counter] = {}
                for to_counter, to_node in tmp_dic.items():
                    if from_counter == to_counter:
                        distances[from_counter][to_counter] = 0
                    else:
                        distances[from_counter][to_counter] = self.calculate_eulidean_distance(from_node, to_node)

            # 按距离从近到远对每个节点的邻居进行排序
            sorted_distance_matrix = {k: sorted(dist, key=lambda x: dist[x]) for k, dist in distances.items()}

            # 保存当前物种的距离矩阵和邻居排序结果
            species_distance_matrix[species] = distances
            species_neighbor_matrix[species] = sorted_distance_matrix

        # 返回所有物种的距离矩阵和邻居矩阵
        return species_distance_matrix, species_neighbor_matrix

    def reset(self, test_env=None, seed=None):
        # 若提供随机种子，则重新初始化随机数生成器
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = None

        # 若提供测试环境，则直接加载；否则重新生成环境
        if test_env is not None:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = test_env
        else:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()

        # 更新任务数、智能体数和物种数
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict['number'])

        # 重置联盟分配矩阵
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))

        # 重置当前时间和结束标志
        self.current_time = 0
        self.finished = False

    def init_state(self):
        # 初始化所有任务状态
        for task in self.task_dic.values():
            task.update(
                members=[],                    # 当前参与该任务的智能体
                cost=[],                       # 执行该任务的成本
                finished=False,                # 任务是否完成
                status=task['requirements'],   # 当前剩余需求，初始等于任务需求
                feasible_assignment=False,     # 当前分配是否可行
                time_start=0,                  # 任务开始时间
                time_finish=0,                 # 任务完成时间
                sum_waiting_time=0,            # 累计等待时间
                efficiency=0,                  # 任务执行效率
                abandoned_agent=[]             # 因等待超时而放弃的智能体
            )

        # 初始化所有智能体状态
        for agent in self.agent_dic.values():
            agent.update(
                route=[-agent['species'] - 1],   # 初始路径为所属仓库
                location=agent['depot'],         # 初始位置为仓库
                contributed=False,               # 是否已对任务作出贡献
                next_decision=0,                 # 下一次决策时间
                travel_time=0,                   # 当前移动时间
                travel_dist=0,                   # 累计移动距离
                arrival_time=[0.],               # 到达各节点的时间记录
                assigned=False,                  # 是否已被有效分配
                sum_waiting_time=0,              # 累计等待时间
                current_action_index=0,          # 当前动作索引
                decision_step=0,                 # 决策步数
                trajectory=[],                   # 运动轨迹
                angle=0,                         # 当前朝向角度
                returned=False,                  # 是否已返回仓库
                pre_set_route=None,              # 预设路径
                current_task=-1,                 # 当前任务编号，-1 表示未执行任务
                task_waiting_ratio=1,            # 任务等待比例
                no_choice=False,                 # 是否无可选动作
                next_action=0                    # 下一步动作
            )

        # 初始化各仓库成员
        for depot in self.depot_dic.values():
            depot.update(members=self.species_dict[-depot['ID'] - 1])

        # 重置全局状态
        self.current_time = 0
        self.max_waiting_time = 200
        self.finished = False

    @staticmethod
    def find_by_key(data, target):
        # 遍历字典中的键和值
        for key, value in data.items():
            # 如果当前值仍是字典，则递归继续查找
            if isinstance(value, dict):
                yield from TaskEnv.find_by_key(value, target)
            # 如果当前键等于目标键，则返回对应的值
            elif key == target:
                yield value

    @staticmethod
    def get_matrix(dictionary, key):
        """
        根据指定键，从字典中提取对应的值列表

        :param key: 要提取的键
        :param dictionary: 目标字典
        """
        key_matrix = []

        # 遍历字典中每个元素的值
        for value in dictionary.values():
            # 提取指定键对应的值
            key_matrix.append(value[key])

        # 返回提取结果列表
        return key_matrix

    @staticmethod
    def calculate_eulidean_distance(agent, task):
        # 计算两个位置之间的欧几里得距离
        return np.linalg.norm(agent['location'] - task['location'])

    def calculate_optimized_ability(self):
        # 遍历每个任务，计算其最优能力组合
        for task in self.task_dic.values():
            task_status = task['status']  # 当前任务还需满足的能力需求

            # 获取各物种数量和能力矩阵
            in_species_num = self.species_dict['number']
            species_ability = self.species_dict['abilities']

            # 为每个物种生成可能的数量取值范围
            num_set = [list(range(0, self.max_task_size + 1)) for _ in in_species_num]

            # 枚举所有可能的物种组合
            group_combinations = list(product(*num_set))

            abilities = []
            contained_spe = []

            # 计算每种组合对应的总能力
            for sample in group_combinations:
                ability = np.zeros((1, self.traits_dim))
                for spe, num in enumerate(sample):
                    ability += sample[spe] * species_ability[spe]
                contained_spe.append(np.array(sample) > 0)  # 记录该组合包含哪些物种
                abilities.append(ability)

            # 计算每种组合对当前任务的有效能力
            effective_ability = np.maximum(np.minimum(task_status, np.vstack(abilities)), 0)

            # 计算每种组合的评分
            score = np.divide(
                effective_ability,
                np.vstack(abilities),
                where=np.vstack(abilities) > 0,
                out=np.zeros_like(np.vstack(abilities), dtype=float)
            ) * effective_ability
            score = np.sum(score, axis=1)

            # 选出评分最高的组合作为最优能力组合
            action_index = np.argmax(score)

            # 取评分最高的两个组合
            group_sort = np.argsort(score)[-2:]

            task['optimized_ability'] = abilities[action_index]

            # 合并前两个高分组合中涉及到的物种
            optimized_species = []
            for ind in group_sort:
                optimized_species.append(contained_spe[ind])
            task['optimized_species'] = np.logical_or(*optimized_species)

        # 汇总所有任务的最优物种掩码
        species_mask = np.vstack(self.get_matrix(self.task_dic, 'optimized_species'))
        return species_mask

    def get_current_agent_status(self, agent):
        # 用于保存所有智能体的当前状态
        status = []

        # 遍历环境中的每个智能体
        for a in self.agent_dic.values():
            if a['current_task'] >= 0:
                # 当前智能体正在执行某个任务
                current_task = a['current_task']

                # 获取该智能体到达当前任务的时间
                arrival_time = self.get_arrival_time(a['ID'], current_task)

                # 计算剩余行驶时间
                travel_time = np.clip(arrival_time - self.current_time, a_min=0, a_max=None)

                if self.current_time <= self.task_dic[current_task]['time_start']:
                    # 若任务尚未开始，则计算当前等待时间和剩余工作时间
                    current_waiting_time = np.clip(self.current_time - arrival_time, a_min=0, a_max=None)
                    remaining_working_time = np.clip(
                        self.task_dic[current_task]['time_start'] + self.task_dic[current_task]['time'] - self.current_time,
                        a_min=0,
                        a_max=None
                    )
                else:
                    # 若任务已经开始，则不再等待，也不计算剩余工作时间
                    current_waiting_time = 0
                    remaining_working_time = 0
            else:
                # 若当前没有任务，则相关时间均为 0
                travel_time = 0
                current_waiting_time = 0
                remaining_working_time = 0

            # 拼接当前智能体的状态向量
            temp_status = np.hstack([
                a['abilities'],                    # 智能体能力
                travel_time,                       # 剩余行驶时间
                remaining_working_time,            # 剩余工作时间
                current_waiting_time,              # 当前等待时间
                agent['location'] - a['location'], # 相对当前位置差
                a['assigned']                      # 是否已分配
            ])
            status.append(temp_status)

        # 将所有智能体状态堆叠成矩阵
        current_agents = np.vstack(status)
        return current_agents

    def get_current_task_status(self, agent):
        # 保存所有任务的状态信息
        status = []

        # 遍历所有任务
        for t in self.task_dic.values():
            # 计算当前智能体到该任务的行驶时间
            travel_time = self.calculate_eulidean_distance(agent, t) / agent['velocity']

            # 拼接任务状态向量
            temp_status = np.hstack([
                t['status'],                         # 当前剩余需求
                t['requirements'],                   # 原始任务需求
                t['time'],                           # 任务执行时间
                travel_time,                         # 到达该任务的时间
                agent['location'] - t['location'],   # 智能体与任务的相对位置
                t['feasible_assignment']             # 任务分配是否可行
            ])
            status.append(temp_status)

        # 在最前面加入仓库状态，作为一个特殊节点
        status = [np.hstack([
            np.zeros(self.traits_dim),                                   # 仓库当前状态置零
            -np.ones(self.traits_dim),                                   # 仓库无任务需求，用 -1 标记
            0,                                                           # 仓库执行时间为 0
            self.calculate_eulidean_distance(agent, self.depot_dic[agent['species']]) / agent['velocity'],  # 回仓库时间
            agent['location'] - agent['depot'],                          # 智能体与仓库的相对位置
            1                                                            # 仓库默认可达
        ])] + status

        # 将所有任务状态堆叠成矩阵
        current_tasks = np.vstack(status)
        return current_tasks

    def get_unfinished_task_mask(self):
        # 对未完成任务标记取反，生成掩码
        mask = np.logical_not(self.get_unfinished_tasks())
        return mask

    def get_unfinished_tasks(self):
        # 记录所有未完成任务的状态
        unfinished_tasks = []

        # 遍历所有任务
        for task in self.task_dic.values():
            # 若任务分配不可行且仍有未满足需求，则视为未完成任务
            unfinished_tasks.append(task['feasible_assignment'] is False and np.any(task['status'] > 0))

        # 返回未完成任务列表
        return unfinished_tasks

    def get_arrival_time(self, agent_id, task_id):
        # 获取指定智能体的到达时间记录
        arrival_time = self.agent_dic[agent_id]['arrival_time']

        # 在该智能体路径中找到目标任务对应的位置索引
        arrival_for_task = np.where(np.array(self.agent_dic[agent_id]['route']) == task_id)[0][-1]

        # 返回该智能体到达目标任务的时间
        return float(arrival_time[arrival_for_task])

    def get_abilities(self, members):
        # 若没有成员，则返回全 0 能力向量
        if len(members) == 0:
            return np.zeros(self.traits_dim)
        else:
            # 统计所有成员能力之和
            return np.sum(np.array([self.agent_dic[member]['abilities'] for member in members]), axis=0)
    
    def get_contributable_task_mask(self, agent_id):
        # 获取当前智能体信息
        agent = self.agent_dic[agent_id]

        # 初始化任务掩码，默认所有任务都不可贡献
        contributable_task_mask = np.ones(self.tasks_num, dtype=bool)

        # 遍历所有任务
        for task in self.task_dic.values():
            # 只考虑当前尚未可行分配的任务
            if not task['feasible_assignment']:
                # 计算当前智能体对任务剩余需求的可贡献能力
                ability = np.maximum(np.minimum(task['status'], agent['abilities']), 0.)

                # 若可贡献能力大于 0，则将该任务标记为可贡献
                if ability.sum() > 0:
                    contributable_task_mask[task['ID']] = False

        # 返回可贡献任务掩码
        return contributable_task_mask

    def get_waiting_tasks(self):
        # 初始化任务等待掩码，默认所有任务都不在等待
        waiting_tasks = np.ones(self.tasks_num, dtype=bool)

        # 用于记录正在等待中的智能体
        waiting_agents = []

        # 遍历所有任务
        for task in self.task_dic.values():
            # 若任务尚未可行分配，但已有成员到达，则认为该任务处于等待状态
            if not task['feasible_assignment'] and len(task['members']) > 0:
                waiting_tasks[task['ID']] = False
                waiting_agents += task['members']

        # 返回等待任务掩码和等待中的智能体列表
        return waiting_tasks, waiting_agents

    def agent_update(self):
        # 遍历所有智能体，更新其下一次决策时间
        for agent in self.agent_dic.values():
            if agent['current_task'] < 0:
                # 若当前不在任务上
                if np.all(self.get_matrix(self.task_dic, 'feasible_assignment')):
                    # 如果所有任务都已完成可行分配，则不再需要决策
                    agent['next_decision'] = np.nan
                elif not np.isnan(agent['next_decision']):
                    # 否则将下一次决策时间设为无穷大，表示暂时阻塞
                    agent['next_decision'] = np.inf
                else:
                    pass
            else:
                # 当前智能体正在某个任务上
                current_task = self.task_dic[agent['current_task']]

                if current_task['feasible_assignment']:
                    # 若当前任务已形成可行分配
                    if agent['ID'] in current_task['members']:
                        # 若该智能体是任务成员，则其下次决策时间为任务完成时刻
                        agent['next_decision'] = float(current_task['time_finish'])

                        # 若任务已经开始执行，则标记该智能体已被有效分配
                        if self.current_time >= float(current_task['time_start']):
                            agent['assigned'] = True
                    else:
                        # 若该智能体不是有效成员，则等待超时后重新决策
                        agent['next_decision'] = self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time
                        agent['assigned'] = False
                else:
                    # 若当前任务尚不可行，则等待到最大等待时间后重新决策
                    agent['next_decision'] = self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time
                    agent['assigned'] = False

    def task_update(self):
        # 记录本次更新中新变为可行分配的任务
        f_task = []

        # 遍历所有任务，检查任务状态和完成情况
        for task in self.task_dic.values():
            if not task['feasible_assignment']:
                # 计算当前任务成员的总能力
                abilities = self.get_abilities(task['members'])

                # 获取所有成员到达该任务的时间
                arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])

                # 更新任务剩余需求状态
                task['status'] = task['requirements'] - abilities

                # 如果任务需求已全部满足
                if (task['status'] <= 0).all():
                    # 若所有成员到达时间差不超过最大等待时间，则任务可执行
                    if np.max(arrival) - np.min(arrival) <= self.max_waiting_time:
                        task['time_start'] = float(np.max(arrival, keepdims=True))             # 任务开始时间
                        task['time_finish'] = float(np.max(arrival, keepdims=True) + task['time'])  # 任务完成时间
                        task['feasible_assignment'] = True
                        f_task.append(task['ID'])
                    else:
                        # 若等待时间过长，则移除等待超时的成员
                        task['feasible_assignment'] = False
                        infeasible_members = arrival <= np.max(arrival, keepdims=True) - self.max_waiting_time
                        for member in np.array(task['members'])[infeasible_members]:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
                else:
                    # 若任务需求仍未满足，则检查是否有成员等待超时
                    task['feasible_assignment'] = False
                    for member in np.array(task['members']):
                        if self.current_time - self.get_arrival_time(member, task['ID']) >= self.max_waiting_time:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
            else:
                # 若任务已可行分配，检查是否已完成
                if self.current_time >= task['time_finish']:
                    task['finished'] = True

        # 检查各仓库中的智能体是否已经返回
        for depot in self.depot_dic.values():
            for member in depot['members']:
                # 若当前时间已到达该智能体返回仓库的时间，且所有任务都已完成可行分配
                if self.current_time >= self.get_arrival_time(member, depot['ID']) and np.all(self.get_matrix(self.task_dic, 'feasible_assignment')):
                    self.agent_dic[member]['returned'] = True

        # 返回本次新变为可行分配的任务列表
        return f_task

    def next_decision(self):
        # 取出所有智能体的下一次决策时间
        decision_time = np.array(self.get_matrix(self.agent_dic, 'next_decision'))

        # 若所有智能体都不再需要决策，则返回空列表和最晚到达时间
        if np.all(np.isnan(decision_time)):
            return ([], []), max(map(lambda x: max(x) if x else 0, self.get_matrix(self.agent_dic, 'arrival_time')))

        # 取出无可选动作的标记
        no_choice = self.get_matrix(self.agent_dic, 'no_choice')

        # 对无可选动作的智能体，将其决策时间设为无穷大
        decision_time = np.where(no_choice, np.inf, decision_time)

        # 找到下一次最早的决策时刻
        next_decision = np.nanmin(decision_time)

        # 若最早决策时刻仍为无穷大，则改为按最后到达时间推进
        if np.isinf(next_decision):
            arrival_time = np.array([agent['arrival_time'][-1] for agent in self.agent_dic.values()])
            decision_time = np.where(no_choice, np.inf, arrival_time)
            next_decision = np.nanmin(decision_time)

        # 找出在该时刻需要释放的已完成智能体
        finished_agents = np.where(decision_time == next_decision)[0].tolist()

        # 记录被阻塞但可以重新释放的智能体
        blocked_agents = []
        for agent_id in np.where(np.isinf(decision_time))[0].tolist():
            if next_decision >= self.agent_dic[agent_id]['arrival_time'][-1]:
                blocked_agents.append(agent_id)

        # 返回可释放的智能体和下一次决策时刻
        release_agents = (finished_agents, blocked_agents)
        return release_agents, next_decision

    def agent_step(self, agent_id, task_id, decision_step):
        """
        :param agent_id: 智能体编号
        :param task_id: 任务编号（注意：传入后会减 1，0 表示返回仓库）
        :param decision_step: 智能体当前决策步
        :return: 执行结果标记、是否执行成功、新变为可行分配的任务列表
        """
        # 将外部动作编号转换为内部任务编号
        task_id = task_id - 1

        if task_id != -1:
            # 如果目标是任务，则获取对应智能体和任务信息
            agent = self.agent_dic[agent_id]
            task = self.task_dic[task_id]

            # 如果该任务已完成可行分配，则不能重复加入
            if task['feasible_assignment']:
                return -1, False, []
        else:
            # 如果目标为 -1，则表示返回所属仓库
            agent = self.agent_dic[agent_id]
            task = self.depot_dic[agent['species']]

        # 将目标节点加入智能体路径
        agent['route'].append(task['ID'])

        # 记录切换前的任务
        previous_task = agent['current_task']

        # 更新当前任务编号
        agent['current_task'] = task_id

        # 计算前往目标的移动时间
        travel_time = self.calculate_eulidean_distance(agent, task) / agent['velocity']
        agent['travel_time'] = travel_time

        # 累加移动距离
        agent['travel_dist'] += self.calculate_eulidean_distance(agent, task)

        # 如果上一个任务已在当前时刻前完成，则从任务完成时刻出发；否则从当前时刻出发
        if previous_task >= 0 and self.task_dic[previous_task]['time_finish'] < self.current_time:
            current_time = self.task_dic[previous_task]['time_finish']
        else:
            current_time = self.current_time

        # 记录到达目标的时间
        agent['arrival_time'] += [current_time + travel_time]

        # 更新智能体当前位置
        agent['location'] = task['location']

        # 更新决策步和可选状态
        agent['decision_step'] = decision_step
        agent['no_choice'] = False

        # 若该智能体尚未加入任务成员列表，则加入
        if agent_id not in task['members']:
            task['members'].append(agent_id)

        # 更新任务状态，并记录新变为可行分配的任务
        f_t = self.task_update()

        # 更新所有智能体状态
        self.agent_update()

        # 返回执行成功标记和任务更新结果
        return 0, True, f_t

    def agent_observe(self, agent_id, max_waiting=False):
        # 获取当前智能体信息
        agent = self.agent_dic[agent_id]

        # 获取未完成任务掩码
        mask = self.get_unfinished_task_mask()

        # 获取当前智能体可贡献任务掩码
        contributable_mask = self.get_contributable_task_mask(agent_id)

        # 合并两个掩码，过滤不可选任务
        mask = np.logical_or(mask, contributable_mask)

        # 若启用最大等待约束，则进一步处理等待中的任务
        if max_waiting:
            waiting_tasks_mask, waiting_agents = self.get_waiting_tasks()
            waiting_len = np.sum(waiting_tasks_mask == 0)

            # 若等待中的任务过多，则将其也纳入掩码限制
            if waiting_len > 5:
                mask = np.logical_or(mask, waiting_tasks_mask)

        # 在最前面插入仓库动作的掩码，False 表示仓库可选
        mask = np.insert(mask, 0, False)

        # if mask.all():
        #     mask = np.insert(mask, 0, False)
        # else:
        #     mask = np.insert(mask, 0, True)

        # 获取当前所有智能体状态，并扩展 batch 维度
        agents_info = np.expand_dims(self.get_current_agent_status(agent), axis=0)

        # 获取当前所有任务状态，并扩展 batch 维度
        tasks_info = np.expand_dims(self.get_current_task_status(agent), axis=0)

        # 为掩码增加 batch 维度
        mask = np.expand_dims(mask, axis=0)

        # 返回任务信息、智能体信息和动作掩码
        return tasks_info, agents_info, mask
    
    
    def calculate_waiting_time(self):
        # 先将所有智能体的累计等待时间清零
        for agent in self.agent_dic.values():
            agent['sum_waiting_time'] = 0

        # 遍历所有任务，统计任务和智能体的等待时间
        for task in self.task_dic.values():
            # 获取该任务所有成员的到达时间
            arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])

            if len(arrival) != 0:
                if task['feasible_assignment']:
                    # 若任务已形成可行分配，则等待时间为最晚到达者与其他成员的到达时间差之和
                    task['sum_waiting_time'] = np.sum(np.max(arrival) - arrival) \
                                            + len(task['abandoned_agent']) * self.max_waiting_time
                else:
                    # 若任务尚未形成可行分配，则等待时间按当前时刻累计
                    task['sum_waiting_time'] = np.sum(self.current_time - arrival) \
                                            + len(task['abandoned_agent']) * self.max_waiting_time
            else:
                # 若任务当前没有成员，仅统计放弃任务的智能体等待时间
                task['sum_waiting_time'] = len(task['abandoned_agent']) * self.max_waiting_time

            # 统计仍在任务中的成员等待时间
            for member in task['members']:
                if task['feasible_assignment']:
                    self.agent_dic[member]['sum_waiting_time'] += np.max(arrival) - self.get_arrival_time(member, task['ID'])
                else:
                    self.agent_dic[member]['sum_waiting_time'] += (
                        self.current_time - self.get_arrival_time(member, task['ID'])
                        if self.current_time - self.get_arrival_time(member, task['ID']) > 0 else 0
                    )

            # 放弃该任务的智能体记为最大等待时间
            for member in task['abandoned_agent']:
                self.agent_dic[member]['sum_waiting_time'] += self.max_waiting_time

    def check_finished(self):
        # 先更新任务状态
        self.task_update()

        # 获取下一次决策的智能体和对应时间
        decision_agents, current_time = self.next_decision()

        # dead_lock = self.check_dead_lock()

        # 若当前没有需要继续决策的智能体，则判断是否全部结束
        if len(decision_agents[0]) + len(decision_agents[1]) == 0:
            self.current_time = current_time

            # 所有智能体都已返回仓库，且所有任务都已完成，则环境结束
            finished = np.all(self.get_matrix(self.agent_dic, 'returned')) and np.all(self.get_matrix(self.task_dic, 'finished'))
        else:
            finished = False

        # 返回是否结束
        return finished

    def generate_traj(self):
        # 为每个智能体生成运动轨迹
        for agent in self.agent_dic.values():
            # 按时间步记录智能体位置
            time_step = 0
            angle = 0

            # 遍历路径中的每一段移动
            for i in range(1, len(agent['route'])):
                # 获取当前节点和下一个节点（任务点或仓库）
                current_task = self.task_dic[agent['route'][i - 1]] if agent['route'][i - 1] >= 0 else self.depot_dic[agent['species']]
                next_task = self.task_dic[agent['route'][i]] if agent['route'][i] >= 0 else self.depot_dic[agent['species']]

                # 计算移动方向角
                angle = np.arctan2(next_task['location'][1] - current_task['location'][1],
                                next_task['location'][0] - current_task['location'][0])

                # 计算两点间距离和移动总时间
                distance = self.calculate_eulidean_distance(next_task, current_task)
                total_time = distance / agent['velocity']

                # 获取当前节点和下一个节点的到达时间
                arrival_time_next = agent['arrival_time'][i]
                arrival_time_current = agent['arrival_time'][i - 1]

                # 计算到达下一个节点后的下一次决策时间
                if next_task['ID'] >= 0 and agent['ID'] in next_task['members'] \
                        and next_task['feasible_assignment']:
                    if next_task['time_start'] - arrival_time_next <= self.max_waiting_time:
                        next_decision = next_task['time_finish']
                    else:
                        next_decision = arrival_time_next + self.max_waiting_time
                elif next_task['ID'] < 0 and i != len(agent['route']) - 1:
                    next_decision = arrival_time_next + self.depot_waiting_time
                else:
                    next_decision = arrival_time_next + self.max_waiting_time

                # 计算离开当前节点的时间
                if current_task['ID'] < 0 and i == 1:
                    current_decision = 0
                elif current_task['ID'] < 0:
                    current_decision = arrival_time_current + self.depot_waiting_time
                else:
                    if agent['ID'] in current_task['members'] \
                            and current_task['time_start'] - arrival_time_current <= self.max_waiting_time \
                            and current_task['feasible_assignment']:
                        current_decision = current_task['time_finish']
                    else:
                        current_decision = arrival_time_current + self.max_waiting_time

                # 按时间步生成从当前节点到下一个节点的轨迹
                while time_step < next_decision:
                    time_step += self.dt
                    if time_step < arrival_time_next:
                        fraction_of_time = (time_step - current_decision) / total_time
                        if fraction_of_time <= 1:
                            x = current_task['location'][0] + fraction_of_time * (
                                        next_task['location'][0] - current_task['location'][0])
                            y = current_task['location'][1] + fraction_of_time * (
                                        next_task['location'][1] - current_task['location'][1])
                            agent['trajectory'].append(np.hstack([x, y, angle]))
                        else:
                            agent['trajectory'].append(
                                np.hstack([next_task['location'][0], next_task['location'][1], angle])
                            )
                    else:
                        # 到达目标后保持在目标位置
                        agent['trajectory'].append(
                            np.array([next_task['location'][0], next_task['location'][1], angle])
                        )

            # 若轨迹时间不足当前全局时间，则补齐到仓库位置
            while time_step < self.current_time:
                time_step += self.dt
                agent['trajectory'].append(
                    np.array([
                        self.depot_dic[agent['species']]['location'][0],
                        self.depot_dic[agent['species']]['location'][1],
                        angle
                    ])
                )

    def get_episode_reward(self, max_time=100):
        # 先计算任务和智能体的累计等待时间
        self.calculate_waiting_time()

        # 计算当前任务分配效率
        eff = self.get_efficiency()

        # 获取所有任务的完成状态
        finished_tasks = self.get_matrix(self.task_dic, 'finished')

        # 统计所有智能体的总移动距离
        dist = np.sum(self.get_matrix(self.agent_dic, 'travel_dist'))

        # 若任务已完成，则奖励与当前时间和效率有关；否则使用最大时间惩罚
        reward = - self.current_time - eff * 10 if self.finished else - max_time - eff * 10

        # 返回回合奖励和任务完成状态
        return reward, finished_tasks

    def get_efficiency(self):
        for task in self.task_dic.values():
            if task['feasible_assignment']:
                task['efficiency'] = abs(np.sum(task['requirements'] - task['status'])) / task['requirements'].sum()
            else:
                task['efficiency'] = 10
        efficiency = np.mean(self.get_matrix(self.task_dic, 'efficiency'))
        return efficiency

    def stack_trajectory(self):
        for agent in self.agent_dic.values():
            agent['trajectory'] = np.vstack(agent['trajectory'])

    def plot_animation(self, path, n):
        self.generate_traj()
        plot_robot_icon = False
        if plot_robot_icon:
            drone = plt.imread('env/drone.png')
            drone_oi = OffsetImage(drone, zoom=0.05)

        def get_cmap(n, name='Dark2'):
            '''
            Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name.
            '''
            return plt.cm.get_cmap(name, n)

        cmap = get_cmap(self.species_num)
        # Set up the plot
        self.stack_trajectory()
        finished_tasks = self.get_matrix(self.task_dic, 'finished')
        finished_rate = np.sum(finished_tasks) / len(finished_tasks)
        gif_len = int(self.current_time/self.dt)
        fig, ax = plt.subplots(dpi=100)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0, right=0.85, top=0.87, bottom=0.02)
        lines = [ax.plot([], [], color=cmap(a['species']), zorder=0)[0] for a in self.agent_dic.values()]
        ax.set_title(f'Agents finish {finished_rate * 100}% tasks within {self.current_time:.2f}min.'
                     f'\nCurrent time is {0:.2f}min')
        color_map = []
        for i in range(self.species_num):
            color_map.append(patches.Patch(color=cmap(i), label='Agent species ' + str(i)))
        color_map.append(patches.Patch(color='g', label='Finished task'))
        color_map.append(patches.Patch(color='b', label='Unfinished task'))
        # red_patch = patches.Patch(color='r', label='Single agent')
        # yellow_patch = patches.Patch(color='y', label='Two agents')
        # cyan_patch = patches.Patch(color='c', label='Three agents')
        # magenta_patch = patches.Patch(color='m', label='>= Four agents')
        if plot_robot_icon:
            ax.legend(handles=color_map, bbox_to_anchor=(0.99, 0.7))
        else:
            ax.legend(handles=color_map,
                      bbox_to_anchor=(0.99, 0.7))
        task_squares = [ax.add_patch(patches.RegularPolygon(xy=(task['location'][0] * 10,
                                                             task['location'][1] * 10),
                                                            numVertices=int(task['requirements'].sum()) + 3,
                                                            radius=0.3, color='b')) for task in self.task_dic.values()]
        depot_tri = [ax.add_patch(patches.Circle((depot['location'][0] * 10,
                                                  depot['location'][1] * 10),
                                                 0.2, color='r')) for depot in self.depot_dic.values()]
        agent_group = [ax.text(agent['location'][0] * 10, agent['location'][1] * 10, str(agent['ID']),
                               horizontalalignment='center', verticalalignment='center', fontsize=8) for agent in self.agent_dic.values()]
        if plot_robot_icon:
            agent_triangles = []
            for a in self.agent_dic.values():
                agent_triangles.append(ax.add_artist(AnnotationBbox(drone_oi, (self.depot_dic[a['species']]['location'][0] * 10,
                                                     self.depot_dic[a['species']]['location'][1] * 10),
                                       frameon=False)))
        else:
            agent_triangles = [ax.add_patch(patches.RegularPolygon(xy=(self.depot_dic[a['species']]['location'][0] * 10,
                                                                       self.depot_dic[a['species']]['location'][1] * 10), numVertices=3,
                                                                   radius=0.2, color=cmap(a['species'])))
                               for a in self.agent_dic.values()]

        # Define the update function for the animation
        def update(frame):
            ax.set_title(f'Agents finish {finished_rate * 100}% tasks within {self.current_time:.2f}min.'
                         f'\nCurrent time is {frame * self.dt:.2f}min')
            pos = np.round([agent['trajectory'][frame, 0:2] for agent in self.agent_dic.values()], 4)
            unq, count = np.unique(pos, axis=0, return_counts=True)
            for agent in self.agent_dic.values():
                repeats = int(count[np.argwhere(np.all(unq == np.round(agent['trajectory'][frame, 0:2], 4), axis=1))])
                agent_triangles[agent['ID']].xy = tuple(agent['trajectory'][frame, 0:2] * 10)
                agent_group[agent['ID']].set_position(tuple(agent['trajectory'][frame, 0:2] * 10))
                agent_group[agent['ID']].set_text(str(repeats))
                if plot_robot_icon:
                    agent_triangles[agent['ID']].xyann = tuple(agent['trajectory'][frame, 0:2] * 10)
                    agent_triangles[agent['ID']].xybox = tuple(agent['trajectory'][frame, 0:2] * 10)
                # else:
                #     agent_triangles[agent['ID']].set_color('m' if repeats >= 4 else 'c' if repeats == 3
                #                                            else 'y' if repeats == 2 else 'r')
                agent_triangles[agent['ID']].orientation = agent['trajectory'][frame, 2] - np.pi / 2
                # Add the current frame's data point to the plot for each trajectory
                if frame > 40:
                    lines[agent['ID']].set_data(agent['trajectory'][frame - 40:frame + 1, 0] * 10,
                                                agent['trajectory'][frame - 40:frame + 1, 1] * 10)
                else:
                    lines[agent['ID']].set_data(agent['trajectory'][:frame + 1, 0] * 10,
                                                agent['trajectory'][:frame + 1, 1] * 10)

            for task in self.task_dic.values():
                if self.reactive_planning:
                    if task['ID'] > np.clip(frame * self.dt//10 * 20 + 20, 20, 100):
                        task_squares[task['ID']].set_color('w')
                        task_squares[task['ID']].set_zorder(0)
                    else:
                        task_squares[task['ID']].set_color('b')
                        task_squares[task['ID']].set_zorder(1)
                if frame * self.dt >= task['time_finish'] > 0:
                    task_squares[task['ID']].set_color('g')
            return lines

        # Set up the animation
        ani = FuncAnimation(fig, update, frames=gif_len, interval=100, blit=True)
        ani.save(f'{path}/episode_{n}_{self.current_time:.1f}.gif')

    def execute_by_route(self, path='./', method=0, plot_figure=False):
        self.plot_figure = plot_figure
        self.max_waiting_time = 200
        while not self.finished and self.current_time < 200:
            decision_agents, current_time = self.next_decision()
            self.current_time = current_time
            decision_agents = decision_agents[0] + decision_agents[1]
            for agent in decision_agents:
                if self.agent_dic[agent]['pre_set_route'] is None or not self.agent_dic[agent]['pre_set_route']:
                    self.agent_step(agent, 0, 0)
                    self.agent_dic[agent]['next_decision'] = np.nan
                    continue
                self.agent_step(agent, self.agent_dic[agent]['pre_set_route'].pop(0), 0)
            self.finished = self.check_finished()
        if self.plot_figure:
            self.plot_animation(path, method)
        print(self.current_time)
        return self.current_time

    def execute_greedy_action(self, path='./', method=0, plot_figure=False):
        self.plot_figure = plot_figure
        while not self.finished and self.current_time < 200:
            release_agents, current_time = self.next_decision()
            self.current_time = current_time
            while release_agents[0] or release_agents[1]:
                agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                agent = self.agent_dic[agent_id]
                tasks_info, agents_info, mask = self.agent_observe(agent_id, max_waiting=True)
                dist = np.inf
                action = None
                for task_id, masked in enumerate(mask[0, :]):
                    if not masked:
                        dist_ = self.calculate_eulidean_distance(agent, self.task_dic[
                            task_id - 1]) if task_id - 1 >= 0 else self.calculate_eulidean_distance(agent,
                                                                                          self.depot_dic[agent['species']])
                        if dist_ < dist:
                            action = task_id
                self.agent_step(agent_id, action, 0)
            self.finished = self.check_finished()
        if self.plot_figure:
            self.plot_animation(path, method)
        print(self.current_time)
        return self.current_time

    def pre_set_route(self, routes, agent_id):
        if not self.agent_dic[agent_id]['pre_set_route']:
            self.agent_dic[agent_id]['pre_set_route'] = routes
        else:
            self.agent_dic[agent_id]['pre_set_route'] += routes

    def process_map(self, path):
        import pandas as pd
        grouped_tasks = dict()
        groups = list(set(np.array(self.get_matrix(self.task_dic, 'requirements')).squeeze(1).tolist()))
        for task_requirement in groups:
            grouped_tasks[task_requirement] = dict()
        index = np.zeros_like(groups)
        for i, task in self.task_dic.items():
            requirement = int(task['requirements'])
            ind = index[groups.index(requirement)]
            grouped_tasks[requirement].update({ind: task})
            index[groups.index(requirement)] += 1
        grouped_tasks = {key: value for key, value in grouped_tasks.items() if len(value) > 0}
        time_finished = [self.get_matrix(dic, 'time_finish') for dic in grouped_tasks.values()]
        t = 0
        time_tick_stamp = dict()
        while t <= self.current_time:
            time_tick_stamp[t] = [np.sum(np.array(ratio) < t)/len(ratio) for ratio in time_finished]
            t += 0.1
            t = np.round(t, 1)
        pd = pd.DataFrame(time_tick_stamp)
        pd.to_csv(f'{path}time_RL.csv')


if __name__ == '__main__':
    import pickle
    testSet = 'RALTestSet'
    os.mkdir(f'../{testSet}')
    for i in range(50):
        env = TaskEnv((3, 3), (5, 5), (20, 20), 5, seed=i)
        pickle.dump(env, open(f'../{testSet}/env_{i}.pkl', 'wb'))
    env.init_state()
