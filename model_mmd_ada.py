import torch
import numpy as np
from torch import nn
from .encoders import Bert_Encoder, FC_Layers, User_Encoder, ADD, CAT, MLP_Layers
from torch.nn.init import xavier_normal_
from tllib.modules.kernels import GaussianKernel
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
import torch.nn.functional as F
import copy
from info_nce import InfoNCE, info_nce
from typing import Optional
# 新导入（添加 KAN 版本）：
from .encoders import (
    Bert_Encoder, FC_Layers, User_Encoder, ADD, CAT, MLP_Layers,
    # 新增 KAN 版本
    KAN_Layers, FC_Layers_MLP_KAN, FC_Layers_KAN, FC_Layers_KAN_Feature_MLP, FC_Layers_Lightweight_KAN
)


# 线性核函数 - 计算输入特征间的内积
class LinearKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        """
        初始化线性核函数

        参数:
            sigma: 未使用但保留参数，保持API一致性
            track_running_stats: 未使用但保留参数，保持API一致性
            alpha: 未使用但保留参数，保持API一致性
        """
        super(LinearKernel, self).__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        计算特征矩阵X的线性核矩阵

        参数:
            X: 形状为[batch_size, feature_dim]的特征矩阵

        返回:
            形状为[batch_size, batch_size]的核矩阵，表示每对样本间的线性相似度
        """
        return X.mm(X.t())  # 计算内积矩阵


# 余弦相似度核函数 - 计算特征向量间余弦距离
class CosKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 eps: Optional[float] = 1e-8):
        """
        初始化余弦相似度核函数

        参数:
            sigma: 未使用但保留参数，保持API一致性
            track_running_stats: 未使用但保留参数，保持API一致性
            eps: 防止除零的小常数
        """
        super(CosKernel, self).__init__()
        self.eps = eps  # 防止数值不稳定的极小值

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        计算特征矩阵X的余弦相似度核矩阵

        参数:
            X: 形状为[batch_size, feature_dim]的特征矩阵

        返回:
            形状为[batch_size, batch_size]的核矩阵，表示每对样本间的余弦距离
            注意返回的是1减去余弦相似度，所以相似样本值较小
        """
        # 计算每个向量的L2范数
        d = X.norm(dim=1)[:, None]
        # 对特征向量进行归一化，防止除零错误
        e = X / torch.clamp(d, min=self.eps)
        # 返回1减去余弦相似度矩阵(余弦距离)
        return 1 - e.mm(e.t())


# 拉普拉斯核函数 - 基于曼哈顿距离的核函数
class LapKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        """
        初始化拉普拉斯核函数

        参数:
            sigma: 核宽度参数
            track_running_stats: 是否跟踪运行时统计量以自适应调整sigma
            alpha: 调整sigma时的比例因子
        """
        super(LapKernel, self).__init__()
        assert track_running_stats or sigma is not None
        # 初始化sigma平方或设为None等待运行时更新
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        计算特征矩阵X的拉普拉斯核矩阵

        参数:
            X: 形状为[batch_size, feature_dim]的特征矩阵

        返回:
            形状为[batch_size, batch_size]的核矩阵
        """
        # 计算每对样本之间的曼哈顿距离平方
        man_distance_square = (torch.abs(X.unsqueeze(0) - X.unsqueeze(1))).sum(2)

        # 如果跟踪运行时统计量，则自适应更新sigma_square
        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(man_distance_square.detach())

        # 返回核矩阵: exp(-distance²/sigma²)
        return torch.exp(-man_distance_square / (self.sigma_square))


# 基础序列推荐模型 - 只使用内容特征
class Model2(torch.nn.Module):
    """
    基于LLM嵌入的序列推荐模型

    该模型不使用物品ID嵌入，直接使用物品文本特征进行推荐
    """

    def __init__(self, args, item_num, use_modal):
        """
        初始化模型

        参数:
            args: 包含模型参数的对象
            item_num: 物品总数
            use_modal: 是否使用多模态特征(文本等)
        """
        super(Model2, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1  # 序列长度+1用于下一物品预测
        self.dnn_layers = args.dnn_layers  # 深度网络层数
        self.mo_dnn_layers = args.mo_dnn_layers  # 多模态转换网络层数
        self.gamma = args.gamma  # MMD损失权重系数

        # 用户编码器 - 基于Transformer处理用户行为序列
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        # 文本特征转换层 - 将LLM嵌入转换为适合推荐的维度
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                   item_embedding_dim=args.embedding_dim,
                                   dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        # 根据配置选择特征融合方法
        if 'add' in args.item_tower:
            self.fc = ADD()  # 简单相加特征
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim * 2,  # 拼接特征后降维
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        # 多层感知机 - 进一步处理特征
        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                                     dnn_layers=self.dnn_layers,
                                     drop_rate=args.drop_rate)

        # 物品ID嵌入层 - 在此模型中未实际使用
        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)  # 初始化嵌入权重

        # 损失函数 - 二元交叉熵
        self.criterion = nn.BCEWithLogitsLoss()

        # MMD损失 - 用于特征空间对齐，此模型中未使用
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])

    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):
        """
        模型前向传播

        参数:
            sample_items_id: 物品ID张量
            input_embs_content: 物品文本嵌入
            log_mask: 序列掩码，标记有效位置
            local_rank: 当前进程的本地排名(分布式训练)

        返回:
            loss: 训练损失
        """
        # 此模型没有使用ID嵌入
        # input_embs_id = self.id_embedding(sample_items_id)

        if self.use_modal:
            # 转换文本嵌入到适合的维度
            input_embs_content = self.turn_dim1(input_embs_content)
            # 直接使用转换后的文本嵌入
            input_embs_all = self.mlp_layers(input_embs_content)

            # 以下MMD损失在此模型中被注释掉未使用
            # mkmmd_loss = self.gamma*self.mkmmd_loss(input_embs_content, input_embs_id)
            # mkmmd_loss = self.gamma*(self.mkmmd_loss(input_embs_id, input_embs_content)+self.mkmmd_loss(input_embs_content, input_embs_id))
        else:
            input_embs_all = input_embs_id  # 这一行实际无效，因为input_embs_id未定义

        # 重塑为序列形式，包含正样本和负样本两个通道
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]  # 正样本
        neg_items_embs = input_embs[:, :, 1]  # 负样本

        # 准备输入序列、目标正样本和负样本
        input_logs_embs = pos_items_embs[:, :-1, :]  # 输入序列(不含最后一个)
        target_pos_embs = pos_items_embs[:, 1:, :]  # 目标正样本(不含第一个)
        target_neg_embs = neg_items_embs[:, :-1, :]  # 目标负样本(不含最后一个)

        # 通过用户编码器处理序列
        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        # 计算预测分数 - 点积相似度
        pos_score = (prec_vec * target_pos_embs).sum(-1)  # 正样本分数
        neg_score = (prec_vec * target_neg_embs).sum(-1)  # 负样本分数

        # 生成标签 - 正样本为1，负样本为0
        pos_labels = torch.ones(pos_score.shape).to(local_rank)
        neg_labels = torch.zeros(neg_score.shape).to(local_rank)

        # 找出有效的位置(通过掩码)
        indices = torch.where(log_mask != 0)
        # 计算二元交叉熵损失
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])

        # 此处没有加MMD损失
        # if self.use_modal:
        #    loss+=mkmmd_loss
        return loss


# 特征对齐推荐模型 - 使用InfoNCE损失对齐特征
class Model2_align(torch.nn.Module):
    """
    使用InfoNCE损失进行特征对齐的序列推荐模型

    同时使用ID特征和文本特征，并通过对比学习使两种特征空间对齐
    """

    def __init__(self, args, item_num, use_modal):
        """
        初始化模型

        参数:
            args: 包含模型参数的对象
            item_num: 物品总数
            use_modal: 是否使用多模态特征(文本等)
        """
        super(Model2_align, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma

        # 用户编码器
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        # 文本特征转换层
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                   item_embedding_dim=args.embedding_dim,
                                   dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        # 特征融合方法
        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim * 2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        # 多层感知机
        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                                     dnn_layers=self.dnn_layers,
                                     drop_rate=args.drop_rate)

        # 物品ID嵌入层
        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)

        # 主损失函数 - 二元交叉熵
        self.criterion = nn.BCEWithLogitsLoss()

        # MMD损失 - 用于特征空间对齐
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])

        # InfoNCE损失 - 用于对比学习
        self.loss = InfoNCE()

    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):
        """
        模型前向传播

        参数:
            sample_items_id: 物品ID张量
            input_embs_content: 物品文本嵌入
            log_mask: 序列掩码，标记有效位置
            local_rank: 当前进程的本地排名(分布式训练)

        返回:
            loss: 训练损失(包括主损失和InfoNCE对齐损失)
        """
        # 获取物品ID嵌入
        input_embs_id = self.id_embedding(sample_items_id)

        if self.use_modal:
            # 转换文本嵌入
            input_embs_content = self.turn_dim1(input_embs_content)
            # 融合ID嵌入和文本嵌入
            input_embs_all = self.mlp_layers(self.fc(input_embs_id, input_embs_content))

            # 使用InfoNCE损失代替MMD损失，权重为0.01
            mkmmd_loss = 0.01 * self.loss(input_embs_content, input_embs_id)
        else:
            input_embs_all = input_embs_id

        # 重塑为序列形式
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        # 准备输入序列、目标正样本和负样本
        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        # 通过用户编码器处理序列
        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        # 计算预测分数
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)

        # 生成标签
        pos_labels = torch.ones(pos_score.shape).to(local_rank)
        neg_labels = torch.zeros(neg_score.shape).to(local_rank)

        # 计算主损失 - 二元交叉熵
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])

        # 加上InfoNCE对齐损失
        loss += mkmmd_loss

        return loss


# 纯ID推荐模型 - 只使用ID特征，支持知识迁移
class Model2_id(torch.nn.Module):
    """
    基于ID的序列推荐模型，支持知识迁移

    该模型主要使用物品ID特征进行推荐，忽略文本特征
    """

    def __init__(self, args, item_num, use_modal):
        """
        初始化模型

        参数:
            args: 包含模型参数的对象
            item_num: 物品总数
            use_modal: 是否使用多模态特征(在此模型中实际未使用)
        """
        super(Model2_id, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma

        # 主用户编码器
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        # 备用用户编码器 - 用于知识迁移
        self.user_enc_2 = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        # 文本特征转换层 - 此模型中实际未使用    FC_Layers
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                   item_embedding_dim=args.embedding_dim,
                                   dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        # 特征融合方法 - 此模型中实际未使用
        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim * 2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        # 多层感知机 - 此模型中实际未使用
        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                                     dnn_layers=self.dnn_layers,
                                     drop_rate=args.drop_rate)

        # 主物品ID嵌入层
        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        # 备用物品ID嵌入层 - 用于知识迁移
        self.other_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)

        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()

        # MMD损失 - 此模型中实际未使用
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])

    def freeze(self):
        """
        知识迁移函数 - 从备用编码器复制到主编码器
        此处与其他模型不同，是反向迁移
        """
        self.user_encoder = copy.deepcopy(self.user_enc_2)
        self.id_embedding.weight.data.copy_(self.other_embedding.weight.data)

    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):
        """
        模型前向传播

        参数:
            sample_items_id: 物品ID张量
            input_embs_content: 物品文本嵌入(此模型中未使用)
            log_mask: 序列掩码，标记有效位置
            local_rank: 当前进程的本地排名(分布式训练)

        返回:
            loss: 训练损失
        """
        # 获取物品ID嵌入
        input_embs_id = self.id_embedding(sample_items_id)

        if self.use_modal:
            # 此模型忽略文本特征，直接使用ID嵌入
            input_embs_all = input_embs_id
        else:
            input_embs_all = input_embs_id

        # 重塑为序列形式
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        # 准备输入序列、目标正样本和负样本
        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        # 通过用户编码器处理序列
        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        # 计算预测分数
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)

        # 生成标签
        pos_labels = torch.ones(pos_score.shape).to(local_rank)
        neg_labels = torch.zeros(neg_score.shape).to(local_rank)

        # 计算损失
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])

        # 不使用MMD损失
        return loss


# 知识迁移推荐模型 - 融合ID和文本特征，支持知识迁移
class Model2_transfer(torch.nn.Module):
    """
    支持知识迁移的多模态序列推荐模型

    融合ID特征和文本特征，并通过MMD损失对齐特征空间，同时支持模型间知识迁移
    """

    def __init__(self, args, item_num, use_modal):
        """
        初始化模型

        参数:
            args: 包含模型参数的对象
            item_num: 物品总数
            use_modal: 是否使用多模态特征(文本等)
        """
        super(Model2_transfer, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma

        # 主用户编码器
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        # 备用用户编码器 - 用于知识迁移
        self.user_enc_2 = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        # 主文本特征转换层
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                   item_embedding_dim=args.embedding_dim,
                                   dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        # 备用文本特征转换层
        self.turn_dim_random = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                         item_embedding_dim=args.embedding_dim,
                                         dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        # 特征融合方法
        if 'add' in args.item_tower:
            self.fc = ADD()  # 加法融合
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim * 2,  # 拼接融合
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        # 多层感知机
        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                                     dnn_layers=self.dnn_layers,
                                     drop_rate=args.drop_rate)

        # 主物品ID嵌入层
        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        # 备用物品ID嵌入层 - 用于知识迁移
        self.other_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)

        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()

        # MMD损失 - 用于特征空间对齐
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])

    def freeze(self):
        """
        知识迁移函数 - 从主模型复制到备用模型
        通常在预训练后调用，为下游任务准备
        """
        # 复制ID嵌入权重
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        # 复制用户编码器
        self.user_enc_2 = copy.deepcopy(self.user_encoder)

    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):
        """
        模型前向传播

        参数:
            sample_items_id: 物品ID张量
            input_embs_content: 物品文本嵌入
            log_mask: 序列掩码，标记有效位置
            local_rank: 当前进程的本地排名(分布式训练)

        返回:
            loss: 训练损失(包括主损失和MMD对齐损失)
        """
        # 获取物品ID嵌入(使用备用嵌入层)
        input_embs_id = self.other_embedding(sample_items_id)

        if self.use_modal:
            # 转换文本嵌入
            input_embs_content = self.turn_dim1(input_embs_content)
            # 融合ID嵌入和文本嵌入
            input_embs_all = self.mlp_layers(self.fc(input_embs_id, input_embs_content))
            # 计算MMD损失，权重为0.2
            mkmmd_loss = 0.2 * self.mkmmd_loss(input_embs_content, input_embs_id)
        else:
            input_embs_all = input_embs_id

        # 重塑为序列形式
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        # 准备输入序列、目标正样本和负样本
        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        # 通过备用用户编码器处理序列(知识迁移后使用)
        prec_vec = self.user_enc_2(input_logs_embs, log_mask, local_rank)
        # 计算预测分数
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)

        # 生成标签
        pos_labels = torch.ones(pos_score.shape).to(local_rank)
        neg_labels = torch.zeros(neg_score.shape).to(local_rank)

        # 计算主损失
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])

        # 加上MMD损失
        if self.use_modal:
            loss += mkmmd_loss

        return loss


# 多编码器混合推荐模型 - 使用三个不同编码器和自适应融合
class Model_new3_3(torch.nn.Module):
    """
    多编码器混合推荐模型，支持自适应权重融合

    使用三个不同的编码器分别处理不同特征，并通过学习的权重融合预测结果
    """

    def __init__(self, args, item_num, use_modal):
        """
        初始化模型

        参数:
            args: 包含模型参数的对象
            item_num: 物品总数
            use_modal: 是否使用多模态特征
        """
        super(Model_new3_3, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma
        self.gamma2 = args.gamma2
        self.gamma3 = args.gamma3

        # 三个独立的用户编码器
        # 编码器1 - 处理ID特征
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        # 编码器2 - 处理融合特征
        self.user_enc_2 = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        # 编码器3 - 处理纯文本特征
        self.user_enc_3 = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        # 两个文本特征转换层，用于不同编码器
        # 转换层1 - 用于融合特征FC_Layers  FC_Layers_Lightweight_KAN
        self.turn_dim1 = FC_Layers_Lightweight_KAN(word_embedding_dim=args.word_embedding_dim,
                                                   item_embedding_dim=args.embedding_dim,
                                                   dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)
        # 转换层3 - 用于纯文本特征
        self.turn_dim3 = FC_Layers_Lightweight_KAN(word_embedding_dim=args.word_embedding_dim,
                                                   item_embedding_dim=args.embedding_dim,
                                                   dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        # 特征融合方法
        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim * 2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        # 两个多层感知机，用于不同特征处理
        # MLP1 - 处理融合特征
        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                                     dnn_layers=self.dnn_layers,
                                     drop_rate=args.drop_rate)
        # MLP3 - 处理纯文本特征
        self.mlp_layer_3 = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                                      dnn_layers=self.dnn_layers,
                                      drop_rate=args.drop_rate)

        # 方案1: 直接替换，使用相同的层配置
        # self.mlp_layers = KAN_Layers(
        #     layers=[args.embedding_dim] * (self.dnn_layers + 1),
        #     dnn_layers=self.dnn_layers,
        #     drop_rate=args.drop_rate,
        # )
        #
        # # MLP3 - 处理纯文本特征
        # self.mlp_layer_3 = KAN_Layers(
        #     layers=[args.embedding_dim] * (self.dnn_layers + 1),
        #     dnn_layers=self.dnn_layers,
        #     drop_rate=args.drop_rate,
        # )

        # 物品ID嵌入层
        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        self.other_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)

        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        self.act = nn.Sigmoid()  # 用于激活自适应权重

        # 三组自适应权重参数 - 为不同位置的物品分配不同权重
        a = np.linspace(1, 0, 11)  # 权重a从1递减到0
        self.alpha = torch.nn.Parameter(torch.tensor(a))

        b = np.linspace(0, 1, 11)  # 权重b从0递增到1
        self.alpha3 = torch.nn.Parameter(torch.tensor(b))

        c = np.linspace(self.gamma2, self.gamma2 - 1, 11)  # 权重c从gamma2递减
        self.alpha2 = torch.nn.Parameter(torch.tensor(c))

        # MMD损失 - 此模型中未使用
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])

    # 下面提供了10种不同的freeze方法，用于不同的迁移学习和微调场景
    def freeze(self):
        """允许所有参数更新，并复制ID嵌入"""
        for param in self.parameters():
            param.requires_grad = True
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)

    def freeze1(self):
        """冻结ID嵌入和主用户编码器"""
        for param in self.parameters():
            param.requires_grad = True
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        self.user_enc_2 = copy.deepcopy(self.user_encoder)
        for name, param in self.named_parameters():
            if "id_embedding" in name or "user_encoder" in name:
                param.requires_grad = False

    def freeze2(self):
        """仅冻结ID嵌入"""
        for param in self.parameters():
            param.requires_grad = True
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        self.user_enc_2 = copy.deepcopy(self.user_encoder)
        for name, param in self.named_parameters():
            if "id_embedding" in name:
                param.requires_grad = False

    def freeze3(self):
        """仅冻结ID嵌入，不复制用户编码器"""
        for param in self.parameters():
            param.requires_grad = True
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        for name, param in self.named_parameters():
            if "id_embedding" in name:
                param.requires_grad = False

    def freeze4(self):
        """仅冻结主用户编码器"""
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "user_encoder" in name:
                param.requires_grad = False

    def freeze5(self):
        """冻结所有编码器和特征处理模块"""
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "user_encoder" in name or "enc_2" in name or "turn_dim1" in name or "fc" in name or "mlp_layers" in name:
                param.requires_grad = False

    def freeze6(self):
        """允许所有参数更新"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze7(self):
        """冻结第二编码器和特征处理模块"""
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "enc_2" in name or "turn_dim1" in name or "fc" in name or "mlp_layers" in name:
                param.requires_grad = False

    def freeze8(self):
        """冻结主用户编码器"""
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "user_encoder" in name:
                param.requires_grad = False

    def freeze9(self):
        """冻结第二用户编码器"""
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "user_enc_2" in name:
                param.requires_grad = False

    def freeze10(self):
        """冻结主用户编码器和第二用户编码器"""
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "user_encoder" in name or "user_enc_2" in name:
                param.requires_grad = False

    def forward(self, sample_items_id, input_embs_content, log_mask, bin_pos_item, bin_neg_item, local_rank):
        """
        模型前向传播

        参数:
            sample_items_id: 物品ID张量
            input_embs_content: 物品文本嵌入
            log_mask: 序列掩码，标记有效位置
            bin_pos_item: 正样本的位置类别索引
            bin_neg_item: 负样本的位置类别索引
            local_rank: 当前进程的本地排名(分布式训练)

        返回:
            loss: 训练损失
        """
        # 获取物品ID嵌入 - 两种嵌入分别用于不同流程
        input_embs_id = self.id_embedding(sample_items_id)
        input_other = self.other_embedding(sample_items_id)

        if self.use_modal:
            # 流程2: 融合ID和文本特征
            input_embs_content1 = self.turn_dim1(input_embs_content)
            input_embs_all = self.mlp_layers(self.fc(input_other, input_embs_content1))

            # 流程3: 纯文本特征
            input_embs_content3 = self.turn_dim3(input_embs_content)
            input_embs_all3 = self.mlp_layer_3(input_embs_content3)
        else:
            input_embs_all = input_embs_id

        # 流程2: 处理融合特征的序列
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        # 使用第二编码器获取预测向量
        prec_vec_2 = self.user_enc_2(input_logs_embs, log_mask, local_rank)
        # 计算预测分数 - 路径21222222core_2 = (prec_vec_2 * target_pos_embs).sum(-1)

        pos_score_2 = (prec_vec_2 * target_pos_embs).sum(-1)
        neg_score_2 = (prec_vec_2 * target_neg_embs).sum(-1)

        # 流程1: 处理ID特征的序列
        input_embs1 = input_embs_id.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs1[:, :, 0]
        neg_items_embs = input_embs1[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        # 使用主编码器获取预测向量
        prec_vec_1 = self.user_encoder(input_logs_embs, log_mask, local_rank)
        # 计算预测分数 - 路径1
        pos_score_1 = (prec_vec_1 * target_pos_embs).sum(-1)
        neg_score_1 = (prec_vec_1 * target_neg_embs).sum(-1)

        # 流程3: 处理纯文本特征的序列
        input_embs3 = input_embs_all3.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs3[:, :, 0]
        neg_items_embs = input_embs3[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        # 使用第三编码器获取预测向量
        prec_vec_3 = self.user_enc_3(input_logs_embs, log_mask, local_rank)
        # 计算预测分数 - 路径3
        pos_score_3 = (prec_vec_3 * target_pos_embs).sum(-1)
        neg_score_3 = (prec_vec_3 * target_neg_embs).sum(-1)

        # 获取每个位置的融合权重并激活
        alpha = self.act(self.alpha[bin_pos_item])  # 路径1权重
        alpha2 = self.act(self.alpha2[bin_pos_item])  # 路径2权重
        alpha3 = self.act(self.alpha3[bin_pos_item])  # 路径3权重
        beta = self.act(self.alpha[bin_neg_item])  # 负样本路径1权重
        beta2 = self.act(self.alpha2[bin_neg_item])  # 负样本路径2权重
        beta3 = self.act(self.alpha3[bin_neg_item])  # 负样本路径3权重

        # 计算权重和
        alpha_s = alpha + alpha2 + alpha3
        beta_s = beta + beta2 + beta3

        # 加权融合三条路径的预测分数
        pos_score = alpha / alpha_s * pos_score_1 + alpha2 / alpha_s * pos_score_2 + alpha3 / alpha_s * pos_score_3
        neg_score = beta / beta_s * neg_score_1 + beta2 / beta_s * neg_score_2 + beta3 / beta_s * neg_score_3

        # 生成标签
        pos_labels = torch.ones(pos_score.shape).to(local_rank)
        neg_labels = torch.zeros(neg_score.shape).to(local_rank)

        # 计算损失
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])

        return loss
