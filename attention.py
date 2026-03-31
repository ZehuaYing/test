import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp.autocast_mode import autocast
from parameters import *


def get_attn_pad_mask(seq_q, seq_k):
    # 提取 query/key 的批大小与序列长度（输入形状通常为 B x L x D）
    batch_size, len_q = seq_q.sum(dim=2).size()
    batch_size, len_k = seq_k.sum(dim=2).size()
    # eq(zero) is PAD token
    # 找出全 0 的位置作为 PAD（True 表示该位置需要被 mask）
    pad_attn_mask_k = seq_q.eq(0).all(2).data.eq(1).unsqueeze(1)  # batch_size x 1 x len_q, one is masking
    pad_attn_mask_q = seq_k.eq(0).all(2).data.eq(1).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    # 扩展到注意力矩阵形状 B x len_q x len_k，分别对齐 q 轴与 k 轴
    pad_attn_mask_k = pad_attn_mask_k.expand(batch_size, len_k, len_q).permute(0, 2, 1)
    pad_attn_mask_q = pad_attn_mask_q.expand(batch_size, len_q, len_k)
    # 只要 q 或 k 任一侧是 PAD，就在该注意力位置置 True（不可见）
    return ~torch.logical_and(~pad_attn_mask_k, ~pad_attn_mask_q)  # batch_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    # 构建自注意力掩码形状：B x L x L
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 生成严格下三角可见矩阵（当前位置可见历史位置，不可见未来位置）
    subsequent_mask = np.logical_not(np.triu(np.ones(attn_shape), k=0)).astype(int)
    # 转为 Torch ByteTensor，供后续注意力计算使用
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        # 单头注意力中输入/输出特征维度（本实现保持同维）
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        # 对打分矩阵做 tanh 截断，抑制极端 logits
        self.tanh_clipping = 10
        # 缩放点积注意力的系数：1/sqrt(d_k)
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        # 可学习的 query/key 线性投影参数
        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        # 参数初始化
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            # 按最后一维大小计算初始化范围，控制不同层的参数尺度
            stdv = 1. / math.sqrt(param.size(-1))
            # 在 [-stdv, stdv] 上做均匀初始化
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        # 未传入上下文时，退化为自注意力（h=q）
        if h is None:
            h = q

        # h: B x target_size x input_dim, q: B x n_query x input_dim
        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        # 将 batch 维与序列维展平，便于一次矩阵乘法完成线性投影
        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        # 投影后再还原为三维张量
        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        # 线性映射得到 Q/K
        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # batch_size*targets_size*key_dim

        # 缩放点积注意力打分，并做 tanh clipping 抑制极值
        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            # 将掩码扩展到与打分矩阵同形状；被 mask 位置置为极小值
            mask = mask.view(batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            U[mask.bool()] = -1e8
        # 概率分布与对数概率（训练时可用于策略梯度）
        attention = torch.softmax(U, dim=-1)  # batch_size*n_query*targets_size
        logp_list = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        # 单头注意力直接将注意力权重作为输出概率
        probs = attention

        return probs, logp_list


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        # 多头数量与输入/输出总维度
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        # 每个 head 的 value/key 子维度
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        # 缩放点积注意力系数：1/sqrt(d_k)
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        # 各 head 的 Q/K/V 投影参数，以及拼接后的输出投影参数
        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        # 参数初始化
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            # 根据最后一维大小设置初始化范围，避免不同矩阵尺度差异过大
            stdv = 1. / math.sqrt(param.size(-1))
            # 采用均匀分布初始化到 [-stdv, stdv]
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        # 未提供上下文时退化为自注意力
        if h is None:
            h = q

        # h: B x target_size x D, q: B x n_query x D
        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        # 展平后统一做各 head 的线性投影
        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        # 约定 Q/K/V 的目标形状：H x B x L x d
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        # 计算每个 head 的 Q/K/V
        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        # 缩放点积注意力分数
        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            # 将 mask 扩展到所有 head，对不可见位置置 -inf
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            # U[mask.bool()] = -np.inf
            U[mask.bool()] = -np.inf
        # 在 key 维归一化得到注意力权重
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            # 数值安全：被 mask 的位置强制回写为 0
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            attention = attnc
        # print(attention)

        # 用注意力权重对 V 加权求和，得到各 head 输出
        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # 拼接多头结果并做线性映射回 embedding_dim
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class GateFFNDense(nn.Module):
    def __init__(self, model_dim, hidden_unit=512):
        super(GateFFNDense, self).__init__()
        # 门控分支：生成 0~1 的门值
        self.W = nn.Linear(model_dim, hidden_unit, bias=False)
        # 候选分支：提供被门控的线性特征
        self.V = nn.Linear(model_dim, hidden_unit, bias=False)
        # 输出映射：将隐藏维映射回模型维度
        self.W2 = nn.Linear(hidden_unit, model_dim, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, hidden_states):
        # 计算门值（Sigmoid）
        hidden_act = self.act(self.W(hidden_states))
        # 计算候选特征
        hidden_linear = self.V(hidden_states)
        # 按元素门控融合
        hidden_states = hidden_act * hidden_linear
        # 投影回原始特征维度
        hidden_states = self.W2(hidden_states)
        return hidden_states


class GateFFNLayer(nn.Module):
    def __init__(self, model_dim):
        super(GateFFNLayer, self).__init__()
        # 门控前馈子层
        self.DenseReluDense = GateFFNDense(model_dim)
        # 预归一化（Pre-Norm）
        self.layer_norm = Normalization(model_dim)

    def forward(self, hidden_states):
        # 先做层归一化，再进入门控前馈网络
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        return forwarded_states


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        # 对最后一维特征做层归一化
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        # 先展平前置维度以复用 LayerNorm，再还原回原始形状
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        # 多头自注意力子层
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        # 注意力前的归一化
        self.normalization1 = Normalization(embedding_dim)
        # 门控前馈子层
        self.feedForward = GateFFNLayer(embedding_dim)

    def forward(self, src, mask=None):
        # 第一条残差支路：注意力块
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        # 第二条残差支路：前馈块
        h1 = h
        h = self.feedForward(h)
        h = h + h1
        return h


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        # 预留的解码器自注意力模块（当前 forward 中未使用）
        self.dec_self_attn = MultiHeadAttention(embedding_dim, n_head)
        # 编码器-解码器交叉注意力
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        # 门控前馈子层
        self.feedForward = GateFFNLayer(embedding_dim)
        # 分别用于 tgt 与 memory 的归一化
        self.normalization1 = Normalization(embedding_dim)
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, dec_self_attn_mask, dec_enc_attn_mask):
        # 第一条残差支路：交叉注意力（query 来自 tgt，key/value 来自 memory）
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization2(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=dec_enc_attn_mask)
        h = h + h0
        # 第二条残差支路：前馈网络
        h1 = h
        h = self.feedForward(h)
        h = h + h1
        return h


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=4, n_layer=2):
        super(Encoder, self).__init__()
        # 按给定层数堆叠 EncoderLayer
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, mask=None):
        # 依次通过每一层编码器
        for layer in self.layers:
            src = layer(src, mask)
        # 返回最终编码表示
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=4, n_layer=2):
        super(Decoder, self).__init__()
        # 按给定层数堆叠 DecoderLayer
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        # 依次通过每一层解码器
        for layer in self.layers:
            tgt = layer(tgt, memory, dec_self_attn_mask, dec_enc_attn_mask)
        # 返回最终解码表示
        return tgt


class AttentionNet(nn.Module):
    def __init__(self, agent_input_dim, task_input_dim, embedding_dim):
        super(AttentionNet, self).__init__()
        # 将 agent/task 原始特征映射到统一的 embedding 空间
        self.agent_embedding = nn.Linear(agent_input_dim, embedding_dim)
        self.task_embedding = nn.Linear(task_input_dim, embedding_dim)  # layer for input information
        # 融合当前 agent 状态、任务全局信息、agent 全局信息
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

        # 任务与 agent 各自的编码器
        self.taskEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        # 双向交叉解码：task<-agent 与 agent<-task
        self.crossDecoder1 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.crossDecoder2 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.agentEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        # 基于融合状态的全局解码器
        self.globalDecoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        # 指针网络头：输出对任务动作的选择概率
        self.pointer = SingleHeadAttention(embedding_dim)
        # self.LSTM = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)

    def encoding_tasks(self, task_inputs, mask=None):
        # 任务原始特征 -> embedding
        task_embedding = self.task_embedding(task_inputs)
        # 经任务编码器提取上下文表示
        task_encoding = self.taskEncoder(task_embedding, mask)
        embedding_dim = task_encoding.size(-1)
        # 扩展 mask 到特征维，用于后续按位置过滤 PAD
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)
        # 将 PAD 位置置为 NaN，使均值聚合时自动忽略
        compressed_task = torch.where(mean_mask, torch.nan, task_embedding)
        # 对任务维做聚合，得到任务全局摘要向量（B x 1 x D）
        aggregated_tasks = torch.nanmean(compressed_task, dim=1).unsqueeze(1)
        return aggregated_tasks, task_encoding

    def encoding_agents(self, agents_inputs, mask=None):
        # agent 原始特征 -> embedding
        agents_embedding = self.agent_embedding(agents_inputs)
        # 经 agent 编码器提取上下文表示
        agents_encoding = self.agentEncoder(agents_embedding, mask)
        embedding_dim = agents_encoding.size(-1)
        # 扩展 mask 到特征维，用于后续按位置过滤 PAD
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)
        # 将 PAD 位置置为 NaN，使均值聚合时自动忽略
        compressed_task = torch.where(mean_mask, torch.nan, agents_embedding)
        # 对 agent 维做聚合，得到 agent 全局摘要向量（B x 1 x D）
        aggregated_agents = torch.nanmean(compressed_task, dim=1).unsqueeze(1)
        return aggregated_agents, agents_encoding

    def forward(self, tasks, agents, global_mask, index):
        # 构建 task/agent 自注意力与交叉注意力所需掩码
        task_mask = get_attn_pad_mask(tasks, tasks)
        agent_mask = get_attn_pad_mask(agents, agents)
        task_agent_mask = get_attn_pad_mask(tasks, agents)
        agent_task_mask = get_attn_pad_mask(agents, tasks)
        # 分别编码任务与 agent，并得到全局摘要向量
        aggregated_task, task_encoding = self.encoding_tasks(tasks, mask=task_mask)
        aggregated_agents, agents_encoding = self.encoding_agents(agents, mask=agent_mask)
        # 双向交叉解码，建模 task-agent 相互影响
        task_agent_feature = self.crossDecoder1(task_encoding, agents_encoding, None, task_agent_mask)
        agent_task_feature = self.crossDecoder2(agents_encoding, task_encoding, None, agent_task_mask)
        # 根据 index 取出当前决策 agent 的上下文状态
        current_state1 = torch.gather(agent_task_feature, 1, index.repeat(1, 1, agent_task_feature.size(2)))
        # 融合当前状态与全局摘要，形成决策查询向量
        current_state = self.fusion(torch.cat((current_state1, aggregated_task, aggregated_agents), dim=-1))
        # 在任务特征上做全局解码，得到用于指针选择的状态
        current_state_prime = self.globalDecoder(current_state, task_agent_feature, None, global_mask)
        # pointer 输出动作概率与对数概率
        probs, logps = self.pointer(current_state_prime, task_agent_feature, mask=global_mask)
        # 去掉 n_query 维（此处通常为 1）
        logps = logps.squeeze(1)
        probs = probs.squeeze(1)
        return probs, logps


def padding_inputs(inputs):
    # 按序列长度做补齐，padding_value=1（与下方 mask 规则保持一致）
    seq = pad_sequence(inputs, batch_first=False, padding_value=1)
    # 调整为 (batch, length, feature) 对应的轴顺序
    seq = seq.permute(2, 1, 0)
    # 初始化 mask：0 表示有效位置，1 表示 padding 位置
    mask = torch.zeros_like(seq, dtype=torch.int64)
    ones = torch.ones_like(seq, dtype=torch.int64)
    # 将值为 1 的补齐位置标记为 1
    mask = torch.where(seq != 1, mask, ones)
    return seq, mask

