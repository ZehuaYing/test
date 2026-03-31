class EnvParams:
    # 每个物种的智能体数量范围 (min, max)
    # 现在是 (3, 3)，表示每个物种固定 3 个智能体
    SPECIES_AGENTS_RANGE = (3, 3)

    # 物种数量范围 (min, max)
    # 训练时会在该区间内随机采样物种数
    SPECIES_RANGE = (3, 5)

    # 任务数量范围 (min, max)
    # 训练时环境任务数会在该区间随机变化
    TASKS_RANGE = (15, 50)

    # 单回合最大仿真时间步（超时判失败或截断）
    MAX_TIME = 200

    # 智能体/任务能力特征维度（trait 向量长度）
    TRAIT_DIM = 5

    # 决策特征维度（用于构造决策状态的长度上限）
    DECISION_DIM = 30


class TrainParams:
    # worker 侧是否使用 GPU（一般是远程采样/推理）
    USE_GPU = False

    # 全局训练器是否使用 GPU（主网络反向传播）
    USE_GPU_GLOBAL = True

    # Ray 可用 GPU 数量上限（调度资源时使用）
    NUM_GPU = 1

    # 并行的远程采样器数量（Ray actor 数）
    NUM_META_AGENT = 4

    # 初始学习率
    LR = 1e-5

    # 折扣因子 gamma（回报衰减系数）
    # 设为 1 表示不做时间折扣
    GAMMA = 1

    # 学习率衰减步长（StepLR 的 step_size）
    # 每 2000 次 step 执行一次衰减
    DECAY_STEP = 2e3

    # 加载模型后是否重置优化器与学习率调度器状态
    RESET_OPT = False

    # 是否定期进入评估流程（与 baseline 比较）
    EVALUATE = True

    # 每次评估的测试样本数（随机种子数量）
    EVALUATION_SAMPLES = 256

    # 是否重置 Ray（当前代码里基本未实际使用）
    RESET_RAY = False

    # 难度提升间隔系数（课程学习节奏）
    INCREASE_DIFFICULTY = 20000

    # TensorBoard 统计窗口大小（累计多少条后写一次）
    SUMMARY_WINDOW = 8

    # 模仿学习/示教采样比例（若代码启用 demon 逻辑会用到）
    DEMON_RATE = 0.5

    # 模仿学习衰减率（注释里给了经验值）
    IL_DECAY = -1e-5  # -1e-6 700k decay 0.5, -1e-5 70k decay 0.5, -1e-4 7k decay 0.5

    # 策略更新 batch 大小（经验池凑够后再训练）
    BATCH_SIZE = 512

    # 智能体输入维度：6 个基础特征 + trait 维度
    AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM

    # 任务输入维度：5 个基础特征 + 2 倍 trait 维度
    TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM

    # 网络嵌入维度（AttentionNet 隐表示大小）
    EMBEDDING_DIM = 128

    # 采样规模（每回合候选样本/节点数量上限）
    SAMPLE_SIZE = 200

    # padding 长度（对齐批次张量）
    PADDING_SIZE = 50

    # POMO 并行 rollout 数（多起点并行策略）
    POMO_SIZE = 10

    # 是否强制最大开放任务数（约束任务开放策略）
    FORCE_MAX_OPEN_TASK = False


class SaverParams:
    # 保存目录名标签（一次实验一个名字）
    FOLDER_NAME = 'save_1'

    # 模型 checkpoint 保存目录
    MODEL_PATH = f'model/{FOLDER_NAME}'

    # TensorBoard 日志目录
    TRAIN_PATH = f'train/{FOLDER_NAME}'

    # 可视化 gif 保存目录
    GIFS_PATH = f'gifs/{FOLDER_NAME}'

    # 启动时是否加载已有模型
    LOAD_MODEL = False

    # 加载哪个权重：current 或 best
    LOAD_FROM = 'current'  # 'best'

    # 是否启用模型保存
    SAVE = True

    # 是否保存可视化图像
    SAVE_IMG = True

    # 图像保存间隔（按 episode 间隔）
    SAVE_IMG_GAP = 1000