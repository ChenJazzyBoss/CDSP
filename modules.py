import torch.nn as nn
import torch


class PositionwiseFeedForward(nn.Module):
    """
    前馈神经网络模块，用于Transformer中的位置感知前馈处理。

    该模块包含两个线性变换层，中间有一个ReLU激活函数，
    还应用了残差连接和层归一化。

    参数:
        d_model (int): 输入特征的维度。
        d_inner (int): 内部前馈层的维度，通常比d_model大几倍。
        dropout (float): Dropout比率，用于防止过拟合。
    """

    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_inner)  # 第一个线性变换，将维度从d_model扩展到d_inner
        self.w_2 = nn.Linear(d_inner, d_model)  # 第二个线性变换，将维度从d_inner还原到d_model
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化，用于稳定训练
        self.dropout = nn.Dropout(dropout)  # Dropout，防止过拟合
        self.activate = nn.ReLU()  # ReLU激活函数

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]

        返回:
            Tensor: 经过前馈网络处理后的张量，形状为 [batch_size, seq_len, d_model]
        """
        residual = x  # 保存残差连接的输入
        # 应用两个线性变换和激活函数，并使用dropout
        x = self.dropout(self.w_2(self.activate(self.w_1(x))))
        return self.layer_norm(residual + x)  # 应用残差连接和层归一化


class SelfAttention(nn.Module):
    """
    实现自注意力机制的核心计算部分。

    计算查询(query)、键(key)和值(value)之间的注意力分数，
    并返回加权的值向量和注意力权重。

    参数:
        temperature (float): 缩放因子，用于缩放点积，通常设置为 sqrt(d_k)
        dropout (float): Dropout比率，应用于注意力权重。
    """

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature  # 缩放因子，用于缩放点积，防止softmax梯度消失
        self.dropout = nn.Dropout(dropout)  # 应用于注意力权重的dropout
        self.softmax = nn.Softmax(dim=-1)  # 在最后一个维度上应用softmax

    def forward(self, query, key, value, mask):
        """
        计算自注意力。

        参数:
            query (Tensor): 查询张量，形状为 [batch_size, n_heads, len_q, d_k]
            key (Tensor): 键张量，形状为 [batch_size, n_heads, len_k, d_k]
            value (Tensor): 值张量，形状为 [batch_size, n_heads, len_v, d_v]
            mask (Tensor): 掩码张量，用于屏蔽某些位置的注意力

        返回:
            tuple: (加权值, 注意力权重)
        """
        # 计算注意力分数：Q和K的矩阵乘法，然后除以温度（缩放）
        attn = torch.matmul(query, key.transpose(-2, -1)) / self.temperature

        # 应用掩码（通常用于屏蔽填充标记或未来标记）
        attn = attn + mask

        # 应用softmax得到概率分布，然后应用dropout
        p_attn = self.dropout(self.softmax(attn))

        # 将注意力权重与值相乘，得到上下文向量
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制实现。

    将注意力机制分成多个"头"并行计算，然后合并结果。
    这允许模型关注不同位置的不同表示子空间的信息。

    参数:
        n_heads (int): 注意力头的数量。
        d_model (int): 模型的维度，必须能被n_heads整除。
        dropout (float): Dropout比率。
    """

    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        assert d_model % n_heads == 0  # 确保d_model可以被n_heads整除
        self.d_model = d_model  # 模型维度
        self.d_k = d_model // n_heads  # 每个头的维度
        self.n_heads = n_heads  # 头的数量
        self.d_v = self.d_k  # 值的维度，通常等于键的维度

        # 线性变换层，用于生成查询、键和值
        self.w_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)  # 查询的线性变换
        self.w_K = nn.Linear(d_model, n_heads * self.d_k, bias=False)  # 键的线性变换
        self.w_V = nn.Linear(d_model, n_heads * self.d_v, bias=False)  # 值的线性变换
        self.fc = nn.Linear(n_heads * self.d_v, d_model, bias=False)  # 输出的线性变换

        # 自注意力层
        self.self_attention = SelfAttention(temperature=self.d_k ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)  # 输出的dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化

    def forward(self, query, key, value, mask):
        """
        多头注意力的前向传播。

        参数:
            query (Tensor): 查询张量，形状为 [batch_size, len_q, d_model]
            key (Tensor): 键张量，形状为 [batch_size, len_k, d_model]
            value (Tensor): 值张量，形状为 [batch_size, len_v, d_model]
            mask (Tensor): 掩码张量，用于屏蔽某些位置的注意力

        返回:
            Tensor: 经过多头注意力处理后的张量，形状为 [batch_size, len_q, d_model]
        """
        # 获取批次大小和序列长度
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query  # 保存用于残差连接

        # 线性变换并重塑形状以便多头处理
        # 形状变化: [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        q = self.w_Q(query).view(sz_b, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_K(key).view(sz_b, len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_V(value).view(sz_b, len_v, self.n_heads, self.d_v).transpose(1, 2)

        # 应用自注意力计算
        x, attn = self.self_attention(q, k, v, mask=mask)

        # 重塑回原始形状并合并多头结果
        # 形状变化: [batch_size, n_heads, seq_len, d_v] -> [batch_size, seq_len, n_heads, d_v] -> [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, self.d_model)

        # 应用输出线性变换和dropout
        x = self.dropout(self.fc(x))

        # 应用残差连接和层归一化
        return self.layer_norm(residual + x)


class TransformerBlock(nn.Module):
    """
    Transformer块，包含一个多头自注意力层和一个前馈神经网络。

    这是Transformer编码器的基本构建块，由自注意力层和前馈网络组成，
    两者都有残差连接和层归一化。

    参数:
        d_model (int): 模型的维度。
        n_heads (int): 注意力头的数量。
        d_inner (int): 前馈网络内部层的维度。
        dropout (float): Dropout比率。
    """

    def __init__(self, d_model, n_heads, d_inner, dropout):
        super().__init__()
        # 多头自注意力层
        self.multi_head_attention = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        # 前馈神经网络层
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, block_input, mask):
        """
        Transformer块的前向传播。

        参数:
            block_input (Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]
            mask (Tensor): 掩码张量，用于屏蔽某些位置的注意力

        返回:
            Tensor: 经过Transformer块处理后的张量，形状为 [batch_size, seq_len, d_model]
        """
        # 应用多头自注意力（注意这里query, key, value都是相同的，这是自注意力机制）
        output = self.multi_head_attention(block_input, block_input, block_input, mask)
        # 应用前馈网络
        return self.feed_forward(output)


class TransformerEncoder(torch.nn.Module):
    """
    Transformer编码器，由多个Transformer块堆叠而成。

    完整的Transformer编码器实现，包括位置编码和多层Transformer块。
    注意：此实现假设输入嵌入已经准备好，因此没有包含词嵌入层。

    参数:
        n_vocab (int): 词汇表大小（此实现未使用，因为假设输入已经是嵌入）。
        n_position (int): 位置编码的最大位置数。
        d_model (int): 模型的维度。
        n_heads (int): 注意力头的数量。
        dropout (float): Dropout比率。
        n_layers (int): Transformer块的层数。
    """

    def __init__(self, n_vocab, n_position, d_model, n_heads, dropout, n_layers):
        super(TransformerEncoder, self).__init__()
        # 注释掉的代码是词嵌入层，当前实现假设输入已经是嵌入向量
        # self.word_embedding = nn.Embedding(n_vocab + 1, d_model, padding_idx=0)

        # 位置编码，用于给序列添加位置信息
        self.position_embedding = nn.Embedding(n_position, d_model)

        self.dropout = nn.Dropout(p=dropout)  # 应用于嵌入的dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 应用于嵌入的层归一化

        # 创建多层Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_inner=d_model * 4,  # 通常前馈网络内部维度是模型维度的4倍
                dropout=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, input_embs, log_mask, att_mask):
        """
        Transformer编码器的前向传播。

        参数:
            input_embs (Tensor): 输入嵌入张量，形状为 [batch_size, seq_len, d_model]
            log_mask (Tensor): 日志掩码，用于指示哪些位置是有效的（非填充）
            att_mask (Tensor): 注意力掩码，用于屏蔽自注意力中的某些连接

        返回:
            Tensor: 经过Transformer编码器处理后的张量，形状为 [batch_size, seq_len, d_model]
        """
        # 生成位置索引，从0到seq_len-1
        position_ids = torch.arange(log_mask.size(1), dtype=torch.long, device=log_mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(log_mask)  # 扩展到与log_mask相同的形状

        # 将输入嵌入与位置嵌入相加，并应用层归一化
        output = self.layer_norm(input_embs + self.position_embedding(position_ids))

        # 应用dropout
        output = self.dropout(output)

        # 依次通过每个Transformer块
        for transformer in self.transformer_blocks:
            output = transformer.forward(output, att_mask)

        return output
