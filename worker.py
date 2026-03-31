import pickle
import time
import torch
import numpy as np
import random
from env.task_env import TaskEnv
from attention import AttentionNet
import scipy.signal as signal
from parameters import *
import copy
from torch.nn import functional as F
from torch.distributions import Categorical


def discount(x, gamma):
    # 计算折扣累计和（Discounted Return）：
    # y[t] = x[t] + gamma * x[t+1] + gamma^2 * x[t+2] + ...
    # 通过“序列反转 + 一阶线性滤波 + 再反转”高效实现
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def zero_padding(x, padding_size, length):
    # 在张量的“高度/行”维度底部补 0，使长度从 length 补到 padding_size
    # ZeroPad2d 参数顺序是 (left, right, top, bottom)，这里只补 bottom
    pad = torch.nn.ZeroPad2d((0, 0, 0, padding_size - length))
    x = pad(x)
    return x


class Worker:
    def __init__(self, mete_agent_id, local_network, local_baseline, global_step,
                 device='cuda', save_image=False, seed=None, env_params=None):

        # 运行设备与基础上下文信息
        self.device = device
        self.metaAgentID = mete_agent_id
        self.global_step = global_step
        self.save_image = save_image

        # 未显式传入环境参数时，使用全局配置中的默认范围
        if env_params is None:
            env_params = [EnvParams.SPECIES_AGENTS_RANGE, EnvParams.SPECIES_RANGE, EnvParams.TASKS_RANGE]

        # 构建主环境，并复制一份用于 baseline 对照评估
        self.env = TaskEnv(*env_params, EnvParams.TRAIT_DIM, EnvParams.DECISION_DIM, seed=seed, plot_figure=save_image)
        self.baseline_env = copy.deepcopy(self.env)

        # 本地策略网络与 baseline 网络
        self.local_baseline = local_baseline
        self.local_net = local_network

        # 轨迹缓存、回合编号与性能统计
        self.experience = {idx:[] for idx in range(7)}
        self.episode_number = None
        self.perf_metrics = {}

        # 预留 RNN 状态容器与回合最大时间限制
        self.p_rnn_state = {}
        self.max_time = EnvParams.MAX_TIME

    def run_episode(self, training=True, sample=False, max_waiting=False):
        # 经验缓存：0~6 分别存储 agents/task/action/mask/reward/index/advantage
        buffer_dict = {idx:[] for idx in range(7)}
        perf_metrics = {}
        current_action_index = 0
        decision_step = 0

        # 主循环：直到任务完成、超时或动作步数达到上限
        while not self.env.finished and self.env.current_time < EnvParams.MAX_TIME and current_action_index < 300:
            with torch.no_grad():
                # 获取本轮可决策 agent，并推进环境时钟
                release_agents, current_time = self.env.next_decision()
                self.env.current_time = current_time
                random.shuffle(release_agents[0])
                finished_task = []

                # 依次为可决策 agent 选择动作并执行
                while release_agents[0] or release_agents[1]:
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.env.agent_dic[agent_id]
                    task_info, total_agents, mask = self.convert_torch(self.env.agent_observe(agent_id, max_waiting))

                    # 若无可行动作，根据可行分配状态决定跳过或标记 no_choice
                    block_flag = mask[0, 1:].all().item()
                    if block_flag and not np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')):
                        agent['no_choice'] = block_flag
                        continue
                    elif block_flag and np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue

                    # 训练阶段对观测进行固定长度补齐，便于批处理
                    if training:
                        task_info, total_agents, mask = self.obs_padding(task_info, total_agents, mask)
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    probs, _ = self.local_net(task_info, total_agents, mask, index)

                    # 训练时按分布采样；测试时可采样或贪心
                    if training:
                        action = Categorical(probs).sample()
                        # 防止采样到超过任务数量的无效动作
                        while action.item() > self.env.tasks_num:
                            action = Categorical(probs).sample()
                    else:
                        if sample:
                            action = Categorical(probs).sample()
                        else:
                            action = torch.argmax(probs, dim=1)
                    r, doable, f_t = self.env.agent_step(agent_id, action.item(), decision_step)
                    agent['current_action_index'] = current_action_index
                    finished_task.append(f_t)

                    # 仅记录可执行动作对应的训练样本
                    if training and doable:
                        buffer_dict[0] += total_agents
                        buffer_dict[1] += task_info
                        buffer_dict[2] += action.unsqueeze(0)
                        buffer_dict[3] += mask
                        buffer_dict[4] += torch.FloatTensor([[0]]).to(self.device)  # reward
                        buffer_dict[5] += index
                        buffer_dict[6] += torch.FloatTensor([[0]]).to(self.device)
                        current_action_index += 1

                    # 一轮决策结束后检查终止条件
                self.env.finished = self.env.check_finished()
                decision_step += 1

                # 回合结束后统一计算终局奖励与任务完成情况
        terminal_reward, finished_tasks = self.env.get_episode_reward(self.max_time)

                # 汇总关键性能指标
        perf_metrics['success_rate'] = [np.sum(finished_tasks)/len(finished_tasks)]
        perf_metrics['makespan'] = [self.env.current_time]
        perf_metrics['time_cost'] = [np.nanmean(self.env.get_matrix(self.env.task_dic, 'time_start'))]
        perf_metrics['waiting_time'] = [np.mean(self.env.get_matrix(self.env.agent_dic, 'sum_waiting_time'))]
        perf_metrics['travel_dist'] = [np.sum(self.env.get_matrix(self.env.agent_dic, 'travel_dist'))]
        perf_metrics['efficiency'] = [self.env.get_efficiency()]
        return terminal_reward, buffer_dict, perf_metrics

    def baseline_test(self):
        # baseline评估时关闭绘图，避免额外开销影响测试速度
        self.baseline_env.plot_figure = False
        # 记录评估指标（当前函数暂未返回该字典，保留用于后续扩展）
        perf_metrics = {}
        # 统计已执行动作次数，用于防止陷入过长决策循环
        current_action_index = 0
        # 墙钟计时起点，用于超时保护（30秒）
        start = time.time()
        # baseline评估主循环：任务未完成、未超过最大仿真时间、且动作数未超上限
        while not self.baseline_env.finished and self.baseline_env.current_time < self.max_time and current_action_index < 300:
            with torch.no_grad():
                # 推进到下一决策时刻，并取出当前可决策agent集合
                release_agents, current_time = self.baseline_env.next_decision()
                # 对第一优先队列随机打乱，减少固定顺序带来的偏置
                random.shuffle(release_agents[0])
                self.baseline_env.current_time = current_time
                # 墙钟超时保护，避免极端情况下评估卡住
                if time.time() - start > 30:
                    break
                # 两个待决策队列都处理完后，再进入下一时刻
                while release_agents[0] or release_agents[1]:
                    # 优先从主队列取agent，空了再取次队列
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.baseline_env.agent_dic[agent_id]
                    # 获取该agent观测并转成torch张量
                    task_info, total_agents, mask = self.convert_torch(self.baseline_env.agent_observe(agent_id, False))
                    # 除“返回仓库”外其余动作均被mask时，视为无可行动作
                    return_flag = mask[0, 1:].all().item()
                    # 若仍存在可行任务分配，则标记no_choice并跳过当前agent
                    if return_flag and not np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')):
                        self.baseline_env.agent_dic[agent_id]['no_choice'] = return_flag
                        continue
                    # 若所有任务都不可分配且agent已在空闲状态，则直接跳过
                    elif return_flag and np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                    # 补齐观测长度，匹配网络输入维度
                    task_info, total_agents, mask = self.obs_padding(task_info, total_agents, mask)
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    probs, _ = self.local_baseline(task_info, total_agents, mask, index)
                    # baseline策略使用贪心选择（最大概率动作）
                    action = torch.argmax(probs, 1)
                    self.baseline_env.agent_step(agent_id, action.item(), None)
                    # 统计本回合已执行动作数
                    current_action_index += 1
                # 当前决策时刻结束后，检查是否满足整体完成条件
                self.baseline_env.finished = self.baseline_env.check_finished()

        # 回合结束后获取终局奖励（finished_tasks在此函数中未进一步使用）
        reward, finished_tasks = self.baseline_env.get_episode_reward(self.max_time)
        return reward

    def work(self, episode_number):
        """
        Interacts with the environment. The agent gets either gradients or experience buffer
        """
        baseline_rewards = []
        buffers = []
        metrics = []
        # 是否强制将“开启任务上限”作为等待策略的控制开关
        max_waiting = TrainParams.FORCE_MAX_OPEN_TASK
        for _ in range(TrainParams.POMO_SIZE):
            # 每次 POMO 采样前重置环境，得到一条独立轨迹
            self.env.init_state()
            terminal_reward, buffer, perf_metrics = self.run_episode(episode_number,True, max_waiting)
            # 若该次回合奖励无效（NaN），放宽等待策略后重试下一条轨迹
            if terminal_reward is np.nan:
                max_waiting = True
                continue
            # 收集有效轨迹的终局奖励、经验缓存和评估指标
            baseline_rewards.append(terminal_reward)
            buffers.append(buffer)
            metrics.append(perf_metrics)
        # 以 POMO 轨迹奖励均值作为 baseline，用于后续优势估计
        baseline_reward = np.nanmean(baseline_rewards)

        for idx, buffer in enumerate(buffers):
            for key in buffer.keys():
                if key == 6:
                    # key=6 存的是 advantage：A = R_i - mean(R)
                    for i in range(len(buffer[key])):
                        buffer[key][i] += baseline_rewards[idx] - baseline_reward
                # 将每条轨迹的同类数据按 key 追加到总经验池
                if key not in self.experience.keys():
                    self.experience[key] = buffer[key]
                else:
                    self.experience[key] += buffer[key]

        for metric in metrics:
            # 汇总每条轨迹产出的性能指标字典
            for key in metric.keys():
                # 按指标名拼接列表，便于后续统一统计均值/方差
                if key not in self.perf_metrics.keys():
                    self.perf_metrics[key] = metric[key]
                else:
                    self.perf_metrics[key] += metric[key]

        if self.save_image:
            try:
                # 按回合导出动画，便于训练过程可视化
                self.env.plot_animation(SaverParams.GIFS_PATH, episode_number)
            except:
                # 可视化失败不影响训练主流程
                pass
        # 记录最近一次完成的回合编号
        self.episode_number = episode_number

    def convert_torch(self, args):
        data = []
        for arg in args:
            # 将环境返回的 numpy/list 观测统一转为 float Tensor
            data.append(torch.tensor(arg, dtype=torch.float).to(self.device))
        # 返回与输入顺序一致的张量列表，供网络前向使用
        return data

    @staticmethod
    def obs_padding(task_info, agents, mask):
        # 任务特征补齐到固定任务数上限（含“返回”动作位）
        task_info = F.pad(task_info, (0, 0, 0, EnvParams.TASKS_RANGE[1] + 1 - task_info.shape[1]), 'constant', 0)
        # agent 特征补齐到固定智能体数量上限
        agents = F.pad(agents, (0, 0, 0, EnvParams.SPECIES_AGENTS_RANGE[1] * EnvParams.SPECIES_RANGE[1] - agents.shape[1]), 'constant', 0)
        # mask 补 1 表示补齐位置不可选，避免被策略采样到
        mask = F.pad(mask, (0, EnvParams.TASKS_RANGE[1] + 1 - mask.shape[1]), 'constant', 1)
        return task_info, agents, mask


if __name__ == '__main__':
    # 指定默认运行设备（需确保本机可用 CUDA）
    device = torch.device('cuda')
    # torch.manual_seed(9)
    # checkpoint = torch.load(SaverParams.MODEL_PATH + '/checkpoint.pth')
    # 构建策略网络；此处未加载 checkpoint，使用当前随机初始化参数
    localNetwork = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    # localNetwork.load_state_dict(checkpoint['best_model'])
    # 以不同随机种子重复运行，便于做稳定性/均值表现观察
    for i in range(100):
        worker = Worker(1, localNetwork, localNetwork, 0, device=device, seed=i, save_image=False)
        worker.work(i)
        print(i)
