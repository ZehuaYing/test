driver.py
  ├─ 初始化全局策略网络 / baseline 网络 / optimizer / lr scheduler
  ├─ 启动多个 Ray Runner
  ├─ 分发当前权重给各个 Runner
  ├─ 收集 Worker 产生的 rollout buffer
  ├─ 用 REINFORCE 更新 global_network
  ├─ 周期性评估并决定是否更新 baseline_network
  └─ 保存 checkpoint

runner.py
  └─ 每个 Ray actor 持有一个本地网络副本
      └─ 创建 Worker 执行一次 episode 采样

worker.py
  └─ 驱动 TaskEnv 跑完整局仿真
      ├─ 从环境取 observation
      ├─ 调用 AttentionNet 选 action
      ├─ 执行动作 agent_step
      ├─ 收集 experience
      └─ 输出 perf metrics

env/task_env.py
  └─ 维护任务、机器人、仓库、时间推进、约束、奖励

attention.py
  └─ 定义 AttentionNet
      ├─ task encoder
      ├─ agent encoder
      ├─ 双向 cross decoder
      ├─ global decoder
      └─ pointer 输出任务选择概率

训练完成后的输出：
  模型文件：model/{}
  TensorBoard日志：train/{}

model/{FOLDER_NAME}/checkpoint.pth保存模型
  存在model和best-model两个权重（评估best_model）
  使用自带的离线测试集进行测试
  test.py进行评估