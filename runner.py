import torch
import numpy as np
import ray
import os
from attention import AttentionNet
from worker import Worker
from parameters import *
from env.task_env import TaskEnv


class Runner(object):
    """Actor object to start running simulation on workers.
    Gradient computation is also executed on this object."""

    def __init__(self, metaAgentID):
        # 当前 Runner 的唯一编号（对应一个 Ray actor）
        self.metaAgentID = metaAgentID
    
        # 根据训练配置选择运行设备：USE_GPU=True 用 CUDA，否则用 CPU
        self.device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')
    
        # 本地策略网络（用于采样/执行当前策略）
        self.localNetwork = AttentionNet(
            TrainParams.AGENT_INPUT_DIM,
            TrainParams.TASK_INPUT_DIM,
            TrainParams.EMBEDDING_DIM
        )
        # 将策略网络放到指定设备
        self.localNetwork.to(self.device)
    
        # 本地基线网络（用于评估或 baseline 对比）
        self.localBaseline = AttentionNet(
            TrainParams.AGENT_INPUT_DIM,
            TrainParams.TASK_INPUT_DIM,
            TrainParams.EMBEDDING_DIM
        )
        # 将基线网络放到指定设备
        self.localBaseline.to(self.device)

    def get_weights(self):
        # 导出本地策略网络参数（state_dict），通常用于回传或同步
        return self.localNetwork.state_dict()
    
    def set_weights(self, weights):
        # 用外部传入参数覆盖本地策略网络（从全局网络同步）
        self.localNetwork.load_state_dict(weights)
    
    def set_baseline_weights(self, weights):
        # 用外部传入参数覆盖本地基线网络（评估/对照模型同步）
        self.localBaseline.load_state_dict(weights)

    def training(self, global_weights, baseline_weights, curr_episode, env_params):
        # 打印当前由哪个 meta agent 执行第几轮任务
        print("starting episode {} on metaAgent {}".format(curr_episode, self.metaAgentID))
    
        # 将本地网络参数同步为主进程下发的最新参数
        self.set_weights(global_weights)
        self.set_baseline_weights(baseline_weights)
    
        # 默认不保存可视化图片
        save_img = False
        # 若开启保存图片，并且到达保存间隔，则本轮保存
        if SaverParams.SAVE_IMG:
            if curr_episode % SaverParams.SAVE_IMG_GAP == 0:
                save_img = True
    
        # 创建 worker 执行一次完整训练采样/交互流程
        worker = Worker(
            self.metaAgentID,      # worker/agent 编号
            self.localNetwork,     # 当前策略网络
            self.localBaseline,    # 基线网络
            curr_episode,          # 当前轮次
            self.device,           # 运行设备
            save_img,              # 是否保存图像
            None,                  # 测试 seed（训练时为空）
            env_params             # 环境参数配置
        )
    
        # 执行本轮工作，内部会与环境交互并收集经验
        worker.work(curr_episode)
    
        # 取回采样经验与性能指标，回传给主进程做聚合和更新
        buffer = worker.experience
        perf_metrics = worker.perf_metrics
    
        # 额外返回元信息，便于主进程识别是哪个 runner 完成了任务
        info = {
            "id": self.metaAgentID,
            "episode_number": curr_episode,
        }
    
        # 返回训练结果：经验缓存、性能指标、任务信息
        return buffer, perf_metrics, info

    def testing(self, seed=None):
        # 构造评估用 worker：
        # episode 设为 0、不保存图像、使用给定随机种子保证可复现
        worker = Worker(
            self.metaAgentID,    # 当前 runner 编号
            self.localNetwork,   # 当前待评估策略网络
            self.localBaseline,  # 基线网络（供对照/内部使用）
            0,                   # 测试模式下不依赖训练轮次
            self.device,         # 运行设备
            False,               # 测试时不保存图像
            seed                 # 指定测试种子
        )
    
        # 执行基线测试流程，返回该 seed 下的 reward
        reward = worker.baseline_test()
    
        # 返回 reward + seed + metaAgentID，便于主进程汇总与追踪来源
        return reward, seed, self.metaAgentID


# 将 Runner 声明为 Ray 远程 Actor
# num_cpus=1：每个 actor 占用 1 个 CPU
# num_gpus=TrainParams.NUM_GPU / TrainParams.NUM_META_AGENT：
# 将总 GPU 资源按 meta agent 数量均分给每个 actor（可为小数）
@ray.remote(num_cpus=1, num_gpus=TrainParams.NUM_GPU / TrainParams.NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        # 复用父类初始化逻辑（设备选择、网络构建等）
        super().__init__(metaAgentID)


if __name__ == '__main__':
    # 仅在直接运行 [runner.py](http://_vscodecontentref_/0) 时执行下面的调试代码（被 import 时不会执行）
    ray.init()

    # 创建一个远程 RLRunner actor，metaAgentID=0
    runner = RLRunner.remote(0)

    # 向远程 actor 提交一个任务，并拿到异步任务句柄
    # 注意：singleThreadedJob 需在类中已定义，否则这里会报方法不存在
    job_id = runner.singleThreadedJob.remote(1)

    # 阻塞等待远程任务执行完成并获取返回结果
    out = ray.get(job_id)

    # 打印返回结果中的第 2 个元素（索引 1）
    print(out[1])
