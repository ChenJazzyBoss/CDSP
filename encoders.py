import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from .modules import TransformerEncoder

# PCA支持检查
try:
    from sklearn.decomposition import PCA

    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False


# ========== KAN 库导入检查 ==========
def try_import_efficient_kan():
    """智能导入 efficient-kan"""
    try:
        from efficient_kan import KANLinear
        print("✅ efficient-kan 导入成功")
        return KANLinear, True
    except ImportError:
        print("❌ efficient-kan 未找到")
        return None, False


def try_import_pykan():
    """智能导入 pykan (官方版本)"""
    try:
        from kan import KAN
        print("✅ pykan (官方) 导入成功")
        return KAN, True
    except ImportError:
        print("❌ pykan 未找到")
        return None, False


# 执行导入检查
KANLinear, EFFICIENT_KAN_AVAILABLE = try_import_efficient_kan()
PyKAN, PYKAN_AVAILABLE = try_import_pykan()


# ========== 原始 MLP_Layers（保持不变） ==========
class MLP_Layers(torch.nn.Module):
    """原始的 MLP_Layers - 完全不变，保证稳定性"""

    def __init__(self, layers, dnn_layers, drop_rate):
        super(MLP_Layers, self).__init__()
        self.layers = layers
        self.dnn_layers = dnn_layers
        if self.dnn_layers > 0:
            mlp_modules = []
            for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x):
        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        else:
            return x


# ========== 可指定版本的 KAN_Layers ==========
class KAN_Layers(torch.nn.Module):
    """
    可指定版本的KAN_Layers，通过参数控制使用哪个版本

    🎯 版本控制：
    - version='auto': 自动选择最佳版本
    - version='efficient': 强制使用 Efficient-KAN
    - version='pykan': 强制使用 PyKAN
    - version='mlp': 强制使用标准 MLP

    使用方式：
    model = KAN_Layers(layers, dnn_layers, drop_rate, version='efficient')
    """

    def __init__(self, layers, dnn_layers, drop_rate, version='auto'):
        """
        参数:
            layers: 层维度列表
            dnn_layers: 网络层数
            drop_rate: dropout率
            version: 'auto', 'efficient', 'pykan', 'mlp'
        """
        super(KAN_Layers, self).__init__()
        self.layers = layers
        self.dnn_layers = dnn_layers
        self.version = version

        # 根据指定版本初始化
        if version == 'auto':
            self._auto_select_version(drop_rate)
        elif version == 'efficient':
            self._use_efficient_version(drop_rate)
        elif version == 'pykan':
            self._use_pykan_version(drop_rate)
        elif version == 'mlp':
            self._use_mlp_version(drop_rate)
        else:
            print(f"⚠️ 未知版本 '{version}'，使用自动选择")
            self._auto_select_version(drop_rate)

    def _auto_select_version(self, drop_rate):
        """自动选择最佳版本"""
        if EFFICIENT_KAN_AVAILABLE:
            print("🎯 自动选择: Efficient-KAN")
            self._use_efficient_kan(drop_rate)
        elif PYKAN_AVAILABLE:
            print("🎯 自动选择: PyKAN")
            self._use_pykan(drop_rate)
        else:
            print("🎯 自动选择: 标准 MLP")
            self._use_mlp_fallback(drop_rate)

    def _use_efficient_version(self, drop_rate):
        """使用 Efficient-KAN 版本"""
        if EFFICIENT_KAN_AVAILABLE:
            print("✅ 指定版本: Efficient-KAN")
            self._use_efficient_kan(drop_rate)
        else:
            print("❌ Efficient-KAN 不可用，回退到标准 MLP")
            self._use_mlp_fallback(drop_rate)

    def _use_pykan_version(self, drop_rate):
        """使用 PyKAN 版本"""
        if PYKAN_AVAILABLE:
            print("✅ 指定版本: PyKAN")
            self._use_pykan(drop_rate)
        else:
            print("❌ PyKAN 不可用，回退到标准 MLP")
            self._use_mlp_fallback(drop_rate)

    def _use_mlp_version(self, drop_rate):
        """使用标准 MLP 版本"""
        print("✅ 指定版本: 标准 MLP")
        self._use_mlp_fallback(drop_rate)

    def _use_efficient_kan(self, drop_rate):
        """使用 Efficient-KAN"""
        print("🔧 使用 Efficient-KAN")
        if self.dnn_layers > 0:
            kan_modules = []
            for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
                kan_modules.append(nn.Dropout(p=drop_rate))
                kan_modules.append(
                    KANLinear(
                        in_features=input_size,
                        out_features=output_size,
                        grid_size=3,  # 推荐系统优化参数
                        spline_order=2,  # 平衡复杂度和性能
                        scale_noise=0.05  # 保持稳定性
                    )
                )
            self.mlp_layers = nn.Sequential(*kan_modules)
        self.implementation = 'efficient_kan'

    def _use_pykan(self, drop_rate):
        """使用 PyKAN"""
        print("🔧 使用 PyKAN")
        if self.dnn_layers > 0:
            self.kan_model = PyKAN(
                width=self.layers,
                grid=3,
                k=2,
                noise_scale=0.05,
                base_fun='silu',
                symbolic_enabled=True,
                seed=42,
                device='cpu'
            )

            if drop_rate > 0:
                self.dropout = nn.Dropout(p=drop_rate)
            else:
                self.dropout = None
        self.implementation = 'pykan'

    def _use_mlp_fallback(self, drop_rate):
        """回退到标准MLP"""
        print("⚠️ KAN库不可用，使用标准MLP")
        if self.dnn_layers > 0:
            mlp_modules = []
            for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)
            self.apply(self._init_weights)
        self.implementation = 'mlp'

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x):
        """前向传播 - 与MLP_Layers完全相同的接口"""
        if self.dnn_layers > 0:
            if self.implementation == 'pykan':
                if hasattr(self, 'dropout') and self.dropout is not None:
                    x = self.dropout(x)
                return self.kan_model(x)
            else:
                return self.mlp_layers(x)
        else:
            return x

    def to(self, device):
        """设备转移"""
        super().to(device)
        if self.implementation == 'pykan' and hasattr(self, 'kan_model'):
            self.kan_model.to(device)
        return self

    # KAN特有功能（可选使用）
    def get_kan_regularization_loss(self, regularize_activation=0.01, regularize_entropy=0.01):
        """获取KAN正则化损失"""
        if self.implementation == 'efficient_kan' and self.dnn_layers > 0:
            reg_loss = 0
            for layer in self.mlp_layers:
                if hasattr(layer, 'regularization_loss'):
                    reg_loss += layer.regularization_loss(regularize_activation, regularize_entropy)
            return reg_loss
        return torch.tensor(0.0)

    def plot_kan(self, **kwargs):
        """KAN可视化（仅PyKAN）"""
        if self.implementation == 'pykan' and hasattr(self, 'kan_model'):
            return self.kan_model.plot(**kwargs)
        print(f"ℹ️ plot_kan 仅在 PyKAN 下可用，当前使用: {self.implementation}")
        return None

    def auto_symbolic(self, **kwargs):
        """自动符号回归（仅PyKAN）"""
        if self.implementation == 'pykan' and hasattr(self, 'kan_model'):
            return self.kan_model.auto_symbolic(**kwargs)
        print(f"ℹ️ auto_symbolic 仅在 PyKAN 下可用，当前使用: {self.implementation}")
        return None


# class MLP_Layers(torch.nn.Module):
#     def __init__(self, layers, dnn_layers, drop_rate):
#         super(MLP_Layers, self).__init__()
#         self.layers = layers
#         self.dnn_layers = dnn_layers
#         if self.dnn_layers > 0:
#             mlp_modules = []
#             for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
#                 mlp_modules.append(nn.Dropout(p=drop_rate))
#                 mlp_modules.append(nn.Linear(input_size, output_size))
#                 mlp_modules.append(nn.GELU())
#             self.mlp_layers = nn.Sequential(*mlp_modules)
#             self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.Embedding):
#             xavier_normal_(module.weight.data)
#         elif isinstance(module, nn.Linear):
#             xavier_normal_(module.weight.data)
#             if module.bias is not None:
#                 constant_(module.bias.data, 0)
#
#     def forward(self, x):
#         if self.dnn_layers > 0:
#             return self.mlp_layers(x)
#         else:
#             return x


class ADD(torch.nn.Module):
    def __init__(self, ):
        super(ADD, self).__init__()

    def forward(self, x, y):
        return x + y


class CAT(torch.nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(CAT, self).__init__()
        mlp_modules = []
        mlp_modules.append(nn.Dropout(p=drop_rate))
        mlp_modules.append(nn.Linear(input_dim, output_dim))
        mlp_modules.append(nn.GELU())
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x, y):
        con_cat = torch.cat([x, y], 1)
        return self.mlp_layers(con_cat)


class FC_Layers(torch.nn.Module):
    def __init__(self, word_embedding_dim, item_embedding_dim, dnn_layers, drop_rate):
        super(FC_Layers, self).__init__()
        self.dnn_layers = dnn_layers
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()

        if self.dnn_layers > 0:
            self.mlp_layers = MLP_Layers(layers=[item_embedding_dim] * (self.dnn_layers + 1),
                                         dnn_layers=self.dnn_layers,
                                         drop_rate=drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        x = self.activate(self.fc(sample_items))
        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        else:
            return x


class FC_Layers_MLP_KAN(torch.nn.Module):
    """
    🎯 MLP→KAN 两阶段降维层 (简化版)

    **完全兼容FC_Layers接口，可直接替换！**

    使用方式：
    # 原来：fc = FC_Layers(4096, 128, 2, 0.1)
    # 现在：fc = FC_Layers_MLP_KAN(4096, 128, 2, 0.1)
    """

    def __init__(self, word_embedding_dim, item_embedding_dim, dnn_layers, drop_rate):
        super(FC_Layers_MLP_KAN, self).__init__()

        self.dnn_layers = dnn_layers

        # 计算中间维度
        intermediate_dim = int((word_embedding_dim * item_embedding_dim) ** 0.5)
        intermediate_dim = ((intermediate_dim + 63) // 64) * 64
        intermediate_dim = max(intermediate_dim, item_embedding_dim * 2)

        # 阶段1：MLP降维
        self.mlp_stage = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(word_embedding_dim, intermediate_dim),
            nn.GELU()
        )

        # 阶段2：KAN降维 (自动回退)
        if EFFICIENT_KAN_AVAILABLE:
            self.kan_stage = nn.Sequential(
                nn.Dropout(drop_rate),
                KANLinear(intermediate_dim, item_embedding_dim, grid_size=3, spline_order=2)
            )
            self.use_pykan = False
            print(f"✅ 使用 Efficient-KAN: {word_embedding_dim}→{intermediate_dim}→{item_embedding_dim}")
        elif PYKAN_AVAILABLE:
            self.kan_model = KAN(width=[intermediate_dim, item_embedding_dim], grid=3, k=2)
            self.kan_dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
            self.use_pykan = True
            print(f"✅ 使用 PyKAN: {word_embedding_dim}→{intermediate_dim}→{item_embedding_dim}")
        else:
            self.kan_stage = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(intermediate_dim, item_embedding_dim),
                nn.GELU()
            )
            self.use_pykan = False
            print(f"⚠️ KAN不可用，使用MLP: {word_embedding_dim}→{intermediate_dim}→{item_embedding_dim}")

        # 阶段3：后续MLP层
        if self.dnn_layers > 0:
            mlp_modules = []
            layers = [item_embedding_dim] * (self.dnn_layers + 1)
            for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.post_mlp = nn.Sequential(*mlp_modules)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        # 阶段1：MLP降维
        x = self.mlp_stage(sample_items)

        # 阶段2：KAN降维
        if self.use_pykan:
            if hasattr(self, 'kan_dropout') and self.kan_dropout is not None:
                x = self.kan_dropout(x)
            x = self.kan_model(x)
        else:
            x = self.kan_stage(x)

        # 阶段3：后续处理
        if self.dnn_layers > 0:
            x = self.post_mlp(x)

        return x

    def to(self, device):
        super().to(device)
        if hasattr(self, 'use_pykan') and self.use_pykan and hasattr(self, 'kan_model'):
            self.kan_model.to(device)
        return self


class FC_Layers_KAN(torch.nn.Module):
    """
    🎯 纯KAN降维层 (简化版)

    **完全兼容FC_Layers接口，直接KAN降维4096→128！**

    使用方式：
    # 原来：fc = FC_Layers(4096, 128, 2, 0.1)
    # 现在：fc = FC_Layers_KAN(4096, 128, 2, 0.1)
    """

    def __init__(self, word_embedding_dim, item_embedding_dim, dnn_layers, drop_rate):
        super(FC_Layers_KAN, self).__init__()

        self.dnn_layers = dnn_layers

        # 直接KAN降维：4096→128
        if EFFICIENT_KAN_AVAILABLE:
            self.fc = nn.Sequential(
                nn.Dropout(drop_rate),
                KANLinear(
                    in_features=word_embedding_dim,
                    out_features=item_embedding_dim,
                    grid_size=5,  # 大降维比例，用更大的grid
                    spline_order=3,  # 更高阶样条，增强表达能力
                    scale_noise=0.1,
                    scale_base=1.0,
                    scale_spline=1.0
                )
            )
            self.use_pykan = False
            print(f"✅ 使用 Efficient-KAN 直接降维: {word_embedding_dim}→{item_embedding_dim}")

        elif PYKAN_AVAILABLE:
            self.kan_model = KAN(
                width=[word_embedding_dim, item_embedding_dim],
                grid=5,  # 大降维比例，用更大的grid
                k=3,  # 更高阶样条
                noise_scale=0.1,
                base_fun='silu'
            )
            self.kan_dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
            self.use_pykan = True
            print(f"✅ 使用 PyKAN 直接降维: {word_embedding_dim}→{item_embedding_dim}")

        else:
            # 回退到标准MLP（与原FC_Layers相同）
            self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
            self.activate = nn.GELU()
            self.use_pykan = False
            print(f"⚠️ KAN不可用，使用标准MLP: {word_embedding_dim}→{item_embedding_dim}")

        # 后续MLP层（与原FC_Layers完全相同）
        if self.dnn_layers > 0:
            mlp_modules = []
            layers = [item_embedding_dim] * (self.dnn_layers + 1)
            for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        # KAN直接降维
        if self.use_pykan:
            if hasattr(self, 'kan_dropout') and self.kan_dropout is not None:
                x = self.kan_dropout(sample_items)
            else:
                x = sample_items
            x = self.kan_model(x)
        else:
            if hasattr(self, 'activate'):  # 标准MLP回退
                x = self.activate(self.fc(sample_items))
            else:  # Efficient-KAN
                x = self.fc(sample_items)

        # 后续MLP层
        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        else:
            return x

    def to(self, device):
        super().to(device)
        if hasattr(self, 'use_pykan') and self.use_pykan and hasattr(self, 'kan_model'):
            self.kan_model.to(device)
        return self


# 🎯 方案1: 纯KAN直接降维 (已有方案)
class FC_Layers_KAN_Direct(torch.nn.Module):
    """纯KAN直接降维: 4096→128 一步到位"""

    def __init__(self, word_embedding_dim, item_embedding_dim, dnn_layers, drop_rate):
        super(FC_Layers_KAN_Direct, self).__init__()
        self.dnn_layers = dnn_layers

        if EFFICIENT_KAN_AVAILABLE:
            self.fc = nn.Sequential(
                nn.Dropout(drop_rate),
                KANLinear(word_embedding_dim, item_embedding_dim, grid_size=5, spline_order=3)
            )
            self.use_pykan = False
            print(f"✅ [方案1] Efficient-KAN直接降维: {word_embedding_dim}→{item_embedding_dim}")
        elif PYKAN_AVAILABLE:
            self.kan_model = KAN(width=[word_embedding_dim, item_embedding_dim], grid=5, k=3)
            self.kan_dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
            self.use_pykan = True
            print(f"✅ [方案1] PyKAN直接降维: {word_embedding_dim}→{item_embedding_dim}")
        else:
            self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
            self.activate = nn.GELU()
            self.use_pykan = False
            print(f"⚠️ [方案1] KAN不可用，使用MLP: {word_embedding_dim}→{item_embedding_dim}")

        if self.dnn_layers > 0:
            mlp_modules = []
            layers = [item_embedding_dim] * (self.dnn_layers + 1)
            for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        if self.use_pykan:
            if hasattr(self, 'kan_dropout') and self.kan_dropout is not None:
                x = self.kan_dropout(sample_items)
            else:
                x = sample_items
            x = self.kan_model(x)
        else:
            if hasattr(self, 'activate'):
                x = self.activate(self.fc(sample_items))
            else:
                x = self.fc(sample_items)

        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        return x


# 🎯 方案2: 渐进式MLP降维 (稳定保守)
class FC_Layers_Progressive_MLP(torch.nn.Module):
    """渐进式MLP降维: 4096→2048→1024→512→128 逐步降维"""

    def __init__(self, word_embedding_dim, item_embedding_dim, dnn_layers, drop_rate):
        super(FC_Layers_Progressive_MLP, self).__init__()
        self.dnn_layers = dnn_layers

        # 计算渐进降维的中间维度
        num_stages = 4  # 4个阶段
        dims = [word_embedding_dim]
        for i in range(1, num_stages):
            dim = int(word_embedding_dim * (0.5 ** i))  # 每阶段减半
            dim = max(dim, item_embedding_dim)  # 不小于目标维度
            dims.append(dim)
        dims.append(item_embedding_dim)

        # 去重并保持递减
        dims = sorted(list(set(dims)), reverse=True)
        if dims[-1] != item_embedding_dim:
            dims.append(item_embedding_dim)

        # 构建渐进MLP
        progressive_modules = []
        for i in range(len(dims) - 1):
            progressive_modules.extend([
                nn.Dropout(drop_rate),
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU()
            ])

        self.progressive_mlp = nn.Sequential(*progressive_modules)

        if self.dnn_layers > 0:
            mlp_modules = []
            layers = [item_embedding_dim] * (self.dnn_layers + 1)
            for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)

        print(f"✅ [方案2] 渐进式MLP: {word_embedding_dim}→{dims[1:-1]}→{item_embedding_dim}")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        x = self.progressive_mlp(sample_items)
        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        return x


# 🎯 方案3: KAN特征提取 + MLP降维
class FC_Layers_KAN_Feature_MLP(torch.nn.Module):
    """KAN特征提取+MLP降维: 4096→[KAN]→512→[MLP]→128"""

    def __init__(self, word_embedding_dim, item_embedding_dim, dnn_layers, drop_rate):
        super(FC_Layers_KAN_Feature_MLP, self).__init__()
        self.dnn_layers = dnn_layers

        # 特征提取维度 (保持较高维度进行特征提取)
        feature_dim = max(word_embedding_dim // 8, 512)  # 通常512维

        # 阶段1: KAN特征提取
        if EFFICIENT_KAN_AVAILABLE:
            self.feature_kan = nn.Sequential(
                nn.Dropout(drop_rate),
                KANLinear(word_embedding_dim, feature_dim, grid_size=3, spline_order=2)
            )
            self.use_pykan = False
            print(f"✅ [方案3] Efficient-KAN特征提取: {word_embedding_dim}→{feature_dim}")
        elif PYKAN_AVAILABLE:
            self.kan_model = KAN(width=[word_embedding_dim, feature_dim], grid=3, k=2)
            self.kan_dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
            self.use_pykan = True
            print(f"✅ [方案3] PyKAN特征提取: {word_embedding_dim}→{feature_dim}")
        else:
            self.feature_kan = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(word_embedding_dim, feature_dim),
                nn.GELU()
            )
            self.use_pykan = False
            print(f"⚠️ [方案3] MLP特征提取: {word_embedding_dim}→{feature_dim}")

        # 阶段2: MLP降维
        self.dimension_mlp = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, item_embedding_dim),
            nn.GELU()
        )

        if self.dnn_layers > 0:
            mlp_modules = []
            layers = [item_embedding_dim] * (self.dnn_layers + 1)
            for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)

        print(f"       MLP降维: {feature_dim}→{item_embedding_dim}")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        # 阶段1: KAN特征提取
        if self.use_pykan:
            if hasattr(self, 'kan_dropout') and self.kan_dropout is not None:
                x = self.kan_dropout(sample_items)
            else:
                x = sample_items
            x = self.kan_model(x)
        else:
            x = self.feature_kan(sample_items)

        # 阶段2: MLP降维
        x = self.dimension_mlp(x)

        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        return x


# 🎯 方案4: 双KAN串联 (分离式)
class FC_Layers_Dual_KAN(torch.nn.Module):
    """双KAN串联: 4096→[KAN1]→1024→[KAN2]→128 (真正的两个独立KAN)"""

    def __init__(self, word_embedding_dim, item_embedding_dim, dnn_layers, drop_rate):
        super(FC_Layers_Dual_KAN, self).__init__()
        self.dnn_layers = dnn_layers

        # 中间维度
        intermediate_dim = max(word_embedding_dim // 4, 1024)  # 通常1024维

        # 🔧 方式1: Efficient-KAN双层
        if EFFICIENT_KAN_AVAILABLE:
            # 第一个KAN: 粗降维 (4096→1024)
            self.kan1 = nn.Sequential(
                nn.Dropout(drop_rate),
                KANLinear(
                    word_embedding_dim, intermediate_dim,
                    grid_size=3,  # 粗降维用较小网格
                    spline_order=2,  # 二次样条
                    scale_noise=0.1
                )
            )
            # 第二个KAN: 精细降维 (1024→128)
            self.kan2 = nn.Sequential(
                nn.Dropout(drop_rate),
                KANLinear(
                    intermediate_dim, item_embedding_dim,
                    grid_size=5,  # 精细降维用较大网格
                    spline_order=3,  # 三次样条，更强表达力
                    scale_noise=0.1
                )
            )
            self.kan_type = 'efficient_dual'
            print(f"✅ [方案4] 双Efficient-KAN分离: {word_embedding_dim}→{intermediate_dim}→{item_embedding_dim}")

        # 🔧 方式2: PyKAN双层 (分别创建)
        elif PYKAN_AVAILABLE:
            # 第一个PyKAN: 粗降维
            self.kan1_model = KAN(
                width=[word_embedding_dim, intermediate_dim],
                grid=3, k=2,
                noise_scale=0.1,
                base_fun='silu'
            )
            # 第二个PyKAN: 精细降维
            self.kan2_model = KAN(
                width=[intermediate_dim, item_embedding_dim],
                grid=5, k=3,
                noise_scale=0.1,
                base_fun='silu'
            )
            self.kan1_dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
            self.kan2_dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
            self.kan_type = 'pykan_dual'
            print(f"✅ [方案4] 双PyKAN分离: {word_embedding_dim}→{intermediate_dim}→{item_embedding_dim}")

        # 🔧 方式3: 混合方式 (一个用KAN，一个用MLP)
        elif EFFICIENT_KAN_AVAILABLE or PYKAN_AVAILABLE:
            if EFFICIENT_KAN_AVAILABLE:
                # KAN做特征提取，MLP做降维
                self.kan1 = nn.Sequential(
                    nn.Dropout(drop_rate),
                    KANLinear(word_embedding_dim, intermediate_dim, grid_size=3, spline_order=2)
                )
                self.kan_type = 'efficient_mlp_hybrid'
            else:
                # PyKAN做特征提取
                self.kan1_model = KAN(
                    width=[word_embedding_dim, intermediate_dim],
                    grid=3, k=2
                )
                self.kan1_dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
                self.kan_type = 'pykan_mlp_hybrid'

            # MLP做最终降维
            self.kan2 = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(intermediate_dim, item_embedding_dim),
                nn.GELU()
            )
            print(f"✅ [方案4] KAN+MLP混合: {word_embedding_dim}→{intermediate_dim}→{item_embedding_dim}")

        # 🔧 方式4: 双MLP回退
        else:
            self.kan1 = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(word_embedding_dim, intermediate_dim),
                nn.GELU()
            )
            self.kan2 = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(intermediate_dim, item_embedding_dim),
                nn.GELU()
            )
            self.kan_type = 'mlp_dual'
            print(f"⚠️ [方案4] 双MLP回退: {word_embedding_dim}→{intermediate_dim}→{item_embedding_dim}")

        if self.dnn_layers > 0:
            mlp_modules = []
            layers = [item_embedding_dim] * (self.dnn_layers + 1)
            for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        # 根据KAN类型选择前向路径
        if self.kan_type == 'efficient_dual':
            # 双Efficient-KAN
            x = self.kan1(sample_items)
            x = self.kan2(x)

        elif self.kan_type == 'pykan_dual':
            # 双PyKAN分离
            if self.kan1_dropout is not None:
                x = self.kan1_dropout(sample_items)
            else:
                x = sample_items
            x = self.kan1_model(x)

            if self.kan2_dropout is not None:
                x = self.kan2_dropout(x)
            x = self.kan2_model(x)

        elif self.kan_type == 'efficient_mlp_hybrid':
            # Efficient-KAN + MLP
            x = self.kan1(sample_items)
            x = self.kan2(x)

        elif self.kan_type == 'pykan_mlp_hybrid':
            # PyKAN + MLP
            if self.kan1_dropout is not None:
                x = self.kan1_dropout(sample_items)
            else:
                x = sample_items
            x = self.kan1_model(x)
            x = self.kan2(x)

        else:  # mlp_dual
            # 双MLP回退
            x = self.kan1(sample_items)
            x = self.kan2(x)

        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        return x

    def to(self, device):
        super().to(device)
        if hasattr(self, 'kan1_model'):
            self.kan1_model.to(device)
        if hasattr(self, 'kan2_model'):
            self.kan2_model.to(device)
        return self


# 🎯 方案5: 轻量级KAN (快速训练)
class FC_Layers_Lightweight_KAN(torch.nn.Module):
    """轻量级KAN: 参数少、训练快"""

    def __init__(self, word_embedding_dim, item_embedding_dim, dnn_layers, drop_rate):
        super(FC_Layers_Lightweight_KAN, self).__init__()
        self.dnn_layers = dnn_layers

        if EFFICIENT_KAN_AVAILABLE:
            self.fc = nn.Sequential(
                nn.Dropout(drop_rate),
                KANLinear(
                    word_embedding_dim, item_embedding_dim,
                    grid_size=2,  # 更小网格，减少参数
                    spline_order=1,  # 线性样条，训练更快
                    scale_noise=0.05  # 较少噪声
                )
            )
            self.use_pykan = False
            print(f"✅ [方案5] 轻量级Efficient-KAN: {word_embedding_dim}→{item_embedding_dim}")
        elif PYKAN_AVAILABLE:
            self.kan_model = KAN(
                width=[word_embedding_dim, item_embedding_dim],
                grid=2,  # 更小网格
                k=1,  # 线性样条
                noise_scale=0.05
            )
            self.kan_dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
            self.use_pykan = True
            print(f"✅ [方案5] 轻量级PyKAN: {word_embedding_dim}→{item_embedding_dim}")
        else:
            self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
            self.activate = nn.GELU()
            self.use_pykan = False
            print(f"⚠️ [方案5] 标准MLP: {word_embedding_dim}→{item_embedding_dim}")

        if self.dnn_layers > 0:
            mlp_modules = []
            layers = [item_embedding_dim] * (self.dnn_layers + 1)
            for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        if self.use_pykan:
            if hasattr(self, 'kan_dropout') and self.kan_dropout is not None:
                x = self.kan_dropout(sample_items)
            else:
                x = sample_items
            x = self.kan_model(x)
        else:
            if hasattr(self, 'activate'):
                x = self.activate(self.fc(sample_items))
            else:
                x = self.fc(sample_items)

        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        return x


class User_Encoder(torch.nn.Module):
    def __init__(self, item_num, max_seq_len, item_dim, num_attention_heads, dropout, n_layers):
        super(User_Encoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_vocab=item_num, n_position=max_seq_len,
                                                      d_model=item_dim, n_heads=num_attention_heads,
                                                      dropout=dropout, n_layers=n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)


class Text_Encoder_mean(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder_mean, self).__init__()
        self.bert_model = bert_model
        # self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        # self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        input_mask_expanded = text_attmask.unsqueeze(-1).expand(hidden_states.size()).float()
        mean_output = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)
        return mean_output
        # mean_output = self.fc(mean_output)
        # return self.activate(mean_output)


class Text_Encoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder, self).__init__()
        self.bert_model = bert_model
        # self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        # self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        return hidden_states[:, 0]
        # cls = self.fc(hidden_states[:, 0])
        # return self.activate(cls)


class Bert_Encoder(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(Bert_Encoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title * 2,
            'abstract': args.num_words_abstract * 2,
            'body': args.num_words_body * 2
        }
        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())
                [:list(self.attributes2length.keys()).index(key)]
            )
            for key in self.attributes2length.keys()
        }

        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title', 'abstract', 'body']

        if 'opt' in args.bert_model_load:
            self.text_encoders = nn.ModuleDict({
                'title': Text_Encoder_mean(bert_model, args.embedding_dim, args.word_embedding_dim)
            })
        else:
            self.text_encoders = nn.ModuleDict({
                'title': Text_Encoder(bert_model, args.embedding_dim, args.word_embedding_dim)
            })

        self.newsname = [name for name in set(args.news_attributes) & set(text_encoders_candidates)]

    def forward(self, news):
        text_vectors = [
            self.text_encoders['title'](
                torch.narrow(news, 1, self.attributes2start[name], self.attributes2length[name]))
            for name in self.newsname
        ]
        if len(text_vectors) == 1:
            final_news_vector = text_vectors[0]
        else:
            final_news_vector = torch.mean(torch.stack(text_vectors, dim=1), dim=1)
        return final_news_vector

