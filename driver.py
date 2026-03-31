""" 
初始化配置环境，定义日志记录器，加载/保存模型，生成环境参数和测试集种子，启动训练过程并进行评估。

"""
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random

# 导入自己的配置文件和模块
from attention import AttentionNet
from runner import RLRunner
from parameters import *
from env.task_env import TaskEnv
"""判断变化是否真实存在"""
from scipy.stats import ttest_rel    # 配对样本 t 检验
from torch.distributions import Categorical     # 表示多个离散动作的概率分布


class Logger(object):
    def __init__(self):
        # 当前训练中的全局策略网络（用于更新）
        self.global_net = None
        # 基线网络（用于评估/对照，降低训练方差）
        self.baseline_net = None
        # 全局优化器
        self.optimizer = None
        # 学习率衰减调度器
        self.lr_decay = None

        # TensorBoard 日志写入器，训练指标会写到 TRAIN_PATH
        self.writer = SummaryWriter(SaverParams.TRAIN_PATH)

        # 如果开启保存功能，则创建模型保存目录
        if SaverParams.SAVE:
            os.makedirs(SaverParams.MODEL_PATH, exist_ok=True)

        # 如果开启保存功能，则创建 GIF 可视化结果目录
        if SaverParams.SAVE:
            os.makedirs(SaverParams.GIFS_PATH, exist_ok=True)

    def set(self, global_net, baseline_net, optimizer, lr_decay):
        # 注入训练过程中要用到的核心对象，便于 Logger 在其他方法中统一访问
        self.global_net = global_net          # 当前用于训练更新的策略网络
        self.baseline_net = baseline_net      # 作为对照/评估的基线网络
        self.optimizer = optimizer            # 训练优化器（如 Adam）
        self.lr_decay = lr_decay              # 学习率调度器（用于衰减学习率）

    def write_to_board(self, tensorboard_data, curr_episode):
        # 将多个 step 收集到的指标转为 numpy，便于做批量统计
        tensorboard_data = np.array(tensorboard_data)
        # 对每一列指标做均值聚合（忽略 NaN），得到当前写盘窗口的统计结果
        tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    
        # 依次解包训练与性能指标
        reward, p_l, entropy, grad_norm, success_rate, time, time_cost, waiting, distance, effi = tensorboard_data
    
        # 组织成 TensorBoard 的 tag -> value 映射
        metrics = {'Loss/Learning Rate': self.lr_decay.get_last_lr()[0],  # 当前学习率
                   'Loss/Policy Loss': p_l,                                # 策略损失
                   'Loss/Entropy': entropy,                                # 策略熵（探索程度）
                   'Loss/Grad Norm': grad_norm,                            # 梯度范数
                   'Loss/Reward': reward,                                  # 平均奖励
                   'Perf/Makespan': time,                                  # 完工时间
                   'Perf/Success rate': success_rate,                      # 成功率
                   'Perf/Time cost': time_cost,                            # 时间成本
                   'Perf/Waiting time': waiting,                           # 等待时间
                   'Perf/Traveling distance': distance,                    # 总行驶距离
                   'Perf/Waiting Efficiency': effi                         # 等待效率
                   }
    
        # 将当前 episode 的各项指标写入 TensorBoard
        for k, v in metrics.items():
            self.writer.add_scalar(tag=k, scalar_value=v, global_step=curr_episode)

    def load_saved_model(self):
        # 提示开始加载断点模型
        print('Loading Model...')
    
        # 从指定目录读取训练断点（包含网络参数、优化器状态、训练进度等）
        checkpoint = torch.load(SaverParams.MODEL_PATH + '/checkpoint.pth')
    
        # 根据配置决定加载“最佳模型”还是“当前模型”
        if SaverParams.LOAD_FROM == 'best':
            model = 'best_model'
        else:
            model = 'model'
    
        # 恢复全局网络和基线网络参数
        self.global_net.load_state_dict(checkpoint[model])
        self.baseline_net.load_state_dict(checkpoint[model])
    
        # 恢复优化器与学习率调度器状态，保证训练可无缝继续
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_decay.load_state_dict(checkpoint['lr_decay'])
    
        # 读取训练进度信息
        curr_episode = checkpoint['episode']
        curr_level = checkpoint['level']
        best_perf = checkpoint['best_perf']
    
        # 打印恢复后的关键信息，便于检查
        print("curr_episode set to ", curr_episode)
        print("best_perf so far is ", best_perf)
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])
    
        # 可选：重置优化器和学习率调度器（常用于迁移训练或重新热启动）
        if TrainParams.RESET_OPT:
            self.optimizer = optim.Adam(self.global_net.parameters(), lr=TrainParams.LR)
            self.lr_decay = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=TrainParams.DECAY_STEP,
                gamma=0.98
            )
    
        # 返回恢复后的训练状态
        return curr_episode, curr_level, best_perf

    def save_model(self, curr_episode, curr_level, best_perf):
        # 提示开始保存模型
        print('Saving model', end='\n')
    
        # 组装断点字典：包含模型参数、优化器状态、训练进度和历史最优表现
        checkpoint = {
            "model": self.global_net.state_dict(),              # 当前训练模型参数
            "best_model": self.baseline_net.state_dict(),       # 基线/最优对照模型参数
            "best_optimizer": self.optimizer.state_dict(),      # 预留字段：最优模型对应优化器状态
            "optimizer": self.optimizer.state_dict(),           # 当前优化器状态（恢复训练必需）
            "episode": curr_episode,                            # 当前训练轮次
            "lr_decay": self.lr_decay.state_dict(),             # 学习率调度器状态
            "level": curr_level,                                # 当前难度等级
            "best_perf": best_perf                              # 历史最佳性能指标
        }
    
        # 构造断点文件路径并写入磁盘
        path_checkpoint = "./" + SaverParams.MODEL_PATH + "/checkpoint.pth"
        torch.save(checkpoint, path_checkpoint)
    
        # 提示保存完成
        print('Saved model', end='\n')

    @staticmethod
    def generate_env_params(curr_level=None):
        # 随机生成“每个物种的智能体数量”（上下界相同，表示本次环境固定该值）
        per_species_num = np.random.randint(
            EnvParams.SPECIES_AGENTS_RANGE[0],
            EnvParams.SPECIES_AGENTS_RANGE[1] + 1
        )
    
        # 随机生成“物种数量”
        species_num = np.random.randint(
            EnvParams.SPECIES_RANGE[0],
            EnvParams.SPECIES_RANGE[1] + 1
        )
    
        # 随机生成“任务数量”
        tasks_num = np.random.randint(
            EnvParams.TASKS_RANGE[0],
            EnvParams.TASKS_RANGE[1] + 1
        )
    
        # 按环境构造所需格式返回参数区间列表（这里每项都固定为同一个值）
        params = [
            (per_species_num, per_species_num),
            (species_num, species_num),
            (tasks_num, tasks_num)
        ]
        return params

    @staticmethod
    def generate_test_set_seed():
        # 为评估阶段生成固定数量的随机种子，保证不同模型在同一批测试环境上对比
        test_seed = np.random.randint(
            low=0,                          # 种子下界（包含）
            high=1e8,                       # 种子上界（不包含）
            size=TrainParams.EVALUATION_SAMPLES  # 需要生成的评估样本数
        ).tolist()                          # 转成 Python 列表，便于后续传递给 worker
        return test_seed


def fuse_two_dicts(ini_dictionary1, ini_dictionary2):
    # 只有当第二个字典存在时才执行融合
    if ini_dictionary2 is not None:
        # 先做键级别合并，得到统一的键集合
        merged_dict = {**ini_dictionary1, **ini_dictionary2}
        final_dict = {}

        # 对每个键执行“列表拼接”式融合
        # 这里默认两个字典同名键对应的值都支持 +（通常是 list）
        for k, v in merged_dict.items():
            final_dict[k] = ini_dictionary1[k] + v

        # 返回融合后的新字典，不修改原字典引用
        return final_dict
    else:
        # 若第二个字典为空，直接返回第一个字典
        return ini_dictionary1


def main():
    # 初始化日志管理器（负责 TensorBoard 记录、模型保存/加载等）
    logger = Logger()
    
    # 启动 Ray 分布式运行时，用于并行采样和训练 worker 调度
    ray.init()
    
    # 全局模型所在设备（通常用于参数更新）
    device = torch.device('cuda') if TrainParams.USE_GPU_GLOBAL else torch.device('cpu')
    
    # 本地/worker 使用设备（可能与全局设备不同，用于推理或采样）
    local_device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')
    
    # 创建主训练网络（策略网络）并放到全局设备
    global_network = AttentionNet(
        TrainParams.AGENT_INPUT_DIM,
        TrainParams.TASK_INPUT_DIM,
        TrainParams.EMBEDDING_DIM
    ).to(device)
    
    # 创建基线网络（用于对照评估、稳定训练）并放到全局设备
    baseline_network = AttentionNet(
        TrainParams.AGENT_INPUT_DIM,    # Agent输入：基础特征6 + trait维度
        TrainParams.TASK_INPUT_DIM,     # Task输入：基础特征5 + 2trait维度
        TrainParams.EMBEDDING_DIM       # 隐表示维度（128）与主网络一致
    ).to(device)
    
    # 为主训练网络构建优化器
    global_optimizer = optim.Adam(global_network.parameters(), lr=TrainParams.LR)
    
    # 学习率衰减策略：每隔 DECAY_STEP 步按 0.98 衰减
    lr_decay = optim.lr_scheduler.StepLR(
        global_optimizer,
        step_size=TrainParams.DECAY_STEP,
        gamma=0.98
    )
    
    # 将网络、优化器、调度器注入到 logger，供后续统一管理
    logger.set(global_network, baseline_network, global_optimizer, lr_decay)
    
    # 训练状态初始化：当前 episode、难度等级、历史最佳性能
    curr_episode = 0                    
    curr_level = 0
    best_perf = -200
    
    # 若配置为加载断点，则恢复训练状态
    if SaverParams.LOAD_MODEL:
        curr_episode, curr_level, best_perf = logger.load_saved_model()
    
    # 启动多个并行的 meta agent（Ray 远程 actor）
    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]

    # 准备初始权重（考虑全局训练设备与采样设备可能不同）
    if device != local_device:
        # 为了让远程 worker 在 local_device 上推理，先临时搬运到 local_device 再导出参数
        weights = global_network.to(local_device).state_dict()
        baseline_weights = baseline_network.to(local_device).state_dict()
        # 导出后将网络搬回训练设备，避免影响后续反向传播与更新
        global_network.to(device)
        baseline_network.to(device)
    else:
        # 设备一致时可直接读取参数
        weights = global_network.state_dict()
        baseline_weights = baseline_network.state_dict()
    
    # 将权重放入 Ray 对象存储，避免在进程间重复拷贝
    weights_memory = ray.put(weights)
    baseline_weights_memory = ray.put(baseline_weights)
    
    # 启动每个远程 runner 的首个训练任务
    jobs = []
    
    # 生成当前难度下的环境参数，并分发给各个 meta agent
    env_params = logger.generate_env_params(curr_level)
    for i, meta_agent in enumerate(meta_agents):
        jobs.append(meta_agent.training.remote(
            weights_memory,
            baseline_weights_memory,
            curr_episode,
            env_params
        ))
        curr_episode += 1  # 每提交一个任务，episode 计数递增
    
    # 生成评估用随机种子集合（用于后续公平对比）
    test_set = logger.generate_test_set_seed()
    
    # 基线评估值（首次评估前为空）
    baseline_value = None
    
    # 经验回放缓存：键 0~6 对应不同训练字段（状态、动作、回报等）
    experience_buffer = {idx: [] for idx in range(7)}
    
    # 训练性能统计缓存（用于窗口平均后写入 TensorBoard）
    perf_metrics = {
        'success_rate': [],
        'makespan': [],
        'time_cost': [],
        'waiting_time': [],
        'travel_dist': [],
        'efficiency': []
    }
    
    # 汇总后的训练数据窗口（达到 SUMMARY_WINDOW 后写日志）
    training_data = []

    try:
        while True:
            # 等待任意一个远程采样任务完成（异步并行中的“先完成先处理”）
            done_id, jobs = ray.wait(jobs)
            done_job = ray.get(done_id)[0]
        
            # 拿到该任务返回的：采样数据、性能指标、以及该 worker 的信息
            buffer, metrics, info = done_job
        
            # 将新返回的数据拼接进全局缓存
            experience_buffer = fuse_two_dicts(experience_buffer, buffer)
            perf_metrics = fuse_two_dicts(perf_metrics, metrics)
        
            # 标记本轮是否发生了参数更新（用于决定是否广播新权重）
            update_done = False
        
            # 当缓存样本数达到一个 batch 时，执行训练
            if len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                train_metrics = []
        
                # 只要缓存还够一个 batch，就持续训练（可能一次循环内更新多次）
                while len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                    rollouts = {}
        
                    # 从每个字段中切出一个 batch
                    for k, v in experience_buffer.items():
                        rollouts[k] = v[:TrainParams.BATCH_SIZE]
        
                    # 从缓存中删除已取走的数据
                    for k in experience_buffer.keys():
                        experience_buffer[k] = experience_buffer[k][TrainParams.BATCH_SIZE:]
        
                    # 如果剩余不足一个 batch，标记更新完成并清空残余，保证下轮重新累计
                    if len(experience_buffer[0]) < TrainParams.BATCH_SIZE:
                        update_done = True
                    if update_done:
                        for v in experience_buffer.values():
                            del v[:]
        
                    # 将 batch 数据堆叠并搬到训练设备
                    agent_inputs = torch.stack(rollouts[0], dim=0).to(device)             # (batch, sample_size, 2)
                    task_inputs = torch.stack(rollouts[1], dim=0).to(device)              # (batch, sample_size, k_size)
                    action_batch = torch.stack(rollouts[2], dim=0).unsqueeze(1).to(device)    # (batch, 1, 1)
                    global_mask_batch = torch.stack(rollouts[3], dim=0).to(device)        # (batch, 1, 1)
                    reward_batch = torch.stack(rollouts[4], dim=0).unsqueeze(1).to(device)    # (batch, 1, 1)
                    index = torch.stack(rollouts[5]).to(device)
                    advantage_batch = torch.stack(rollouts[6], dim=0).to(device)          # (batch, 1, 1)
        
                    # REINFORCE：前向得到动作概率分布
                    probs, _ = global_network(task_inputs, agent_inputs, global_mask_batch, index)
                    dist = Categorical(probs)
        
                    # 当前动作的对数概率与熵（熵可作为探索程度监控）
                    logp = dist.log_prob(action_batch.flatten())
                    entropy = dist.entropy().mean()
        
                    # 策略梯度损失：-logπ(a|s) * advantage
                    policy_loss = -logp * advantage_batch.flatten().detach()
                    policy_loss = policy_loss.mean()
        
                    loss = policy_loss
                    global_optimizer.zero_grad()
        
                    # 反向传播 + 梯度裁剪 + 参数更新
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=100, norm_type=2)
                    global_optimizer.step()
        
                    # 记录本次更新的核心训练指标
                    train_metrics.append([
                        reward_batch.mean().item(),
                        policy_loss.item(),
                        entropy.item(),
                        grad_norm.item()
                    ])
        
                # 每轮训练后更新学习率调度器
                lr_decay.step()
        
                # 聚合性能指标并清空缓存
                perf_data = []
                for k, v in perf_metrics.items():
                    perf_data.append(np.nanmean(perf_metrics[k]))
                    del v[:]
        
                # 聚合训练指标并与性能指标拼成一条日志数据
                train_metrics = np.nanmean(train_metrics, axis=0)
                for v in perf_metrics.values():
                    del v[:]
                data = [*train_metrics, *perf_data]
                training_data.append(data)
        
            # 到达日志窗口大小后写入 TensorBoard
            if len(training_data) >= TrainParams.SUMMARY_WINDOW:
                logger.write_to_board(training_data, curr_episode)
                training_data = []
        
            # 如果本轮发生更新，则将新权重广播到 Ray 对象存储
            if update_done:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    baseline_weights = baseline_network.to(local_device).state_dict()
                    global_network.to(device)
                    baseline_network.to(device)
                else:
                    weights = global_network.state_dict()
                    baseline_weights = baseline_network.state_dict()
                weights_memory = ray.put(weights)
                baseline_weights_memory = ray.put(baseline_weights)
        
            # 给刚完成任务的那个 worker 立即派发下一次采样任务（保持 worker 满负载）
            env_params = logger.generate_env_params(curr_level)
            jobs.append(
                meta_agents[info['id']].training.remote(
                    weights_memory,
                    baseline_weights_memory,
                    curr_episode,
                    env_params
                )
            )
            curr_episode += 1
        
            # 课程学习：按设定节奏提升任务难度（上限 level 10）
            if curr_episode // (TrainParams.INCREASE_DIFFICULTY * (curr_level + 1)) == 1 and curr_level < 10:
                curr_level += 1
                print('increase difficulty to level', curr_level)
        
            # 周期性保存训练断点
            if curr_episode % 512 == 0:
                logger.save_model(curr_episode, curr_level, best_perf)
        
            # 是否开启评估流程
            if TrainParams.EVALUATE:
                # 每隔固定步数进行一次完整评估
                if curr_episode % 1024 == 0:
                    # 先停止当前训练 worker，避免训练与评估互相干扰
                    ray.wait(jobs, num_returns=TrainParams.NUM_META_AGENT)
                    for a in meta_agents:
                        ray.kill(a)
                    print('Evaluate baseline model at ', curr_episode)
        
                    # 若 baseline_value 为空，先评估基线模型并缓存结果
                    if baseline_value is None:
                        test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(TrainParams.NUM_META_AGENT)]
                        for _, test_agent in enumerate(test_agent_list):
                            ray.get(test_agent.set_baseline_weights.remote(baseline_weights_memory))
        
                        rewards = dict()
                        seed_list = copy.deepcopy(test_set)
                        evaluate_jobs = [
                            test_agent_list[i].testing.remote(seed=seed_list.pop())
                            for i in range(TrainParams.NUM_META_AGENT)
                        ]
        
                        # 异步拉取评估结果，并持续给空闲 worker 补任务，直到跑完所有测试种子
                        while True:
                            test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                            test_result = ray.get(test_done_id)[0]
                            reward, seed, meta_id = test_result
                            rewards[seed] = reward
                            if seed_list:
                                evaluate_jobs.append(test_agent_list[meta_id].testing.remote(seed=seed_list.pop()))
                            if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                                break
        
                        rewards = dict(sorted(rewards.items()))
                        baseline_value = np.stack(list(rewards.values()))
        
                        for a in test_agent_list:
                            ray.kill(a)
        
                    # 评估当前模型
                    test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(TrainParams.NUM_META_AGENT)]
                    for _, test_agent in enumerate(test_agent_list):
                        ray.get(test_agent.set_baseline_weights.remote(weights_memory))
        
                    rewards = dict()
                    seed_list = copy.deepcopy(test_set)
                    evaluate_jobs = [
                        test_agent_list[i].testing.remote(seed=seed_list.pop())
                        for i in range(TrainParams.NUM_META_AGENT)
                    ]
        
                    while True:
                        # 等待任意一个评估任务完成，返回完成任务ID和剩余任务列表
                        test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                    
                        # 取回该任务结果（格式：reward, seed, meta_id）
                        test_result = ray.get(test_done_id)[0]
                        reward, seed, meta_id = test_result
                    
                        # 按 seed 记录回报，便于后续排序对齐
                        rewards[seed] = reward
                    
                        # 如果还有未评估的 seed，就把新任务派发给刚空闲的那个 worker（meta_id）
                        if seed_list:
                            evaluate_jobs.append(test_agent_list[meta_id].testing.remote(seed=seed_list.pop()))
                    
                        # 当收集到预设数量的评估样本后结束循环
                        if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                            break
        
                    rewards = dict(sorted(rewards.items()))
                    test_value = np.stack(list(rewards.values()))
        
                    for a in test_agent_list:
                        ray.kill(a)
        
                    # 评估结束后重建训练 worker
                    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]
        
                    # 若当前模型优于基线，则做配对 t 检验判断显著性，再决定是否更新基线
                    print('test value', test_value.mean())
                    print('baseline value', baseline_value.mean())
                    if test_value.mean() > baseline_value.mean():
                        _, p = ttest_rel(test_value, baseline_value)
                        print('p value', p)
        
                        if p < 0.05:
                            print('update baseline model at ', curr_episode)
        
                            # 提取当前权重并同步到 baseline 网络
                            if device != local_device:
                                weights = global_network.to(local_device).state_dict()
                                global_network.to(device)
                            else:
                                weights = global_network.state_dict()
        
                            baseline_weights = copy.deepcopy(weights)
                            baseline_network.load_state_dict(baseline_weights)
        
                            # 更新 Ray 共享对象中的权重
                            weights_memory = ray.put(weights)
                            baseline_weights_memory = ray.put(baseline_weights)
        
                            # 更新测试集，避免过拟合固定测试样本
                            test_set = logger.generate_test_set_seed()
                            print('update test set')
        
                            # 清空 baseline 缓存，下一次重新评估
                            baseline_value = None
                            best_perf = test_value.mean()
        
                            # 保存最新“最佳”检查点
                            logger.save_model(curr_episode, None, best_perf)
        
                    # 重新派发训练任务，恢复训练循环
                    jobs = []
                    for i, meta_agent in enumerate(meta_agents):
                        jobs.append(
                            meta_agent.training.remote(
                                weights_memory,
                                baseline_weights_memory,
                                curr_episode,
                                env_params
                            )
                        )
                        curr_episode += 1

    except KeyboardInterrupt:
        # 捕获用户的 Ctrl+C 中断信号，准备安全退出训练流程
        print("CTRL_C pressed. Killing remote workers")
        # 逐个关闭远程 worker，避免 Ray 进程残留
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
