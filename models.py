import torch
import torch.nn as nn
import torch.fft
import math
import torch.nn.functional as F

# 检测可用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 频率通道注意力模块
class FrequencyChannelAttention(nn.Module):
    def __init__(self, embed_size, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size // reduction),
            nn.ReLU(),
            nn.Linear(embed_size // reduction, embed_size),
        ).to(device)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        return x * torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)


# 频率空间注意力模块
class FrequencySpatialAttention(nn.Module):
    def __init__(self, kernel_size=7): #影响卷积核的感受野大小
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2) # 保证输入输出图像尺寸相同
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, # 计算 x 在通道维度上的均值，生成 [B, 1, H, W]
                             keepdim=True) # 保持输出张量维度不变
        max_out, _ = torch.max(x, dim=1, keepdim=True) # torch.max 返回两个值（最大值和索引），这里 max_out, _ 只取最大值，输出形状为[B,1,H,W]
        combined = torch.cat([avg_out, max_out], dim=1) # [B, 2, H, W]
        attn = self.conv(combined) # 得到注意力图，形状是[B,1,H,W],这个注意力图包含了空间上的关注信息
        return x * self.sigmoid(attn) # 来调整输入的特征图，突出注意力区域
        # 这个操作是基于元素的乘法（即在每个像素位置上乘以对应的注意力权重），将注意力应用到输入特征图上


"""提取高频特征（DCT IDCT 高通滤波器）"""


class HighDctFrequencyExtractor(nn.Module):
    def __init__(self, embed_size, alpha=0.05):
        super().__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1(exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None
        self.embed_size = embed_size
        self.channel_attn = FrequencyChannelAttention(embed_size)  # 添加通道注意力

    def create_dct_matrix(self, N:int) -> torch.Tensor:
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    def high_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        mask[:alpha_h, :alpha_w] = 0 # 掩码方向
        return x * mask

    def forward(self, x):
        xq = self.dct_2d(x)
        xq_high = self.high_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        eps = 1e-6
        xh = (xh - min_vals) / (max_vals - min_vals + eps)
        xh = self.channel_attn(xh)  # 应用通道注意力
        # 高频通常反映的是图像的频域信息（不是位置域信息）,空间注意力无法有效增强频域中的特征响应
        return xh


"""提取低频特征（DCT IDCT 低频滤波器）"""


class LowDctFrequencyExtractor(nn.Module):
    def __init__(self, embed_size, alpha=0.95):
        super().__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1(exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None
        self.embed_size = embed_size
        self.conv = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(embed_size)
        self.channel_attn = FrequencyChannelAttention(embed_size)  # 添加通道注意力
        self.spatial_attn = FrequencySpatialAttention()  # 添加空间注意力 # 未传入参数，使用默认卷积核，用于计算空间注意力的生成

    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    def low_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        mask[-alpha_h:, -alpha_w:] = 0
        return x * mask

    def forward(self, x):
        xq = self.dct_2d(x)
        xq_low = self.low_pass_filter(xq, self.alpha)
        xl = self.idct_2d(xq_low)
        xl = F.relu(self.batch_norm(self.conv(xl)))
        xl = self.channel_attn(xl)  # 应用通道注意力
        xl = self.spatial_attn(xl)  # 应用空间注意力
        return xl


"""细节增强（CNN + 自注意力机制）"""


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, embed_size, num_heads):
        super().__init__()
        self.dim_in = dim_in
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, embed_size)
        self.linear_k = nn.Linear(dim_in, embed_size)
        self.linear_v = nn.Linear(dim_in, embed_size)
        self.scale = 1 / math.sqrt(self.embed_size // self.num_heads)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        if x.dim() == 4:
            batch_size, C, H, W = x.shape
            x = x.view(batch_size, C, -1).transpose(1, 2)

        batch_size, seq_len, dim_in = x.shape
        q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.embed_size // self.num_heads).transpose(1,
                                                                                                                    2)
        k = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.embed_size // self.num_heads).transpose(1,
                                                                                                                    2)
        v = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.embed_size // self.num_heads).transpose(1,
                                                                                                                    2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
        return self.fc(out)


class CNNWithAttention(nn.Module):
    def __init__(self, embed_size, num_heads, d_model):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1)
        self.attention = MultiHeadSelfAttention(dim_in=embed_size, embed_size=embed_size, num_heads=num_heads)
        self.fc = nn.Linear(embed_size, embed_size)
        self.freq_spatial_attn = FrequencySpatialAttention()  # 空间注意力
        self.freq_channel_attn = FrequencyChannelAttention(embed_size)  # 通道注意力

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.freq_spatial_attn(x)
        x = self.freq_channel_attn(x)
        x = self.attention(x)  # 输出形状 [B, seq_len, embed_size]
        # 恢复为 4D 张量 [B, embed_size, H, W]
        B, seq_len, embed_size = x.shape
        H = W = int(math.sqrt(seq_len))  # 假设 H 和 W 相等
        x = x.view(B, embed_size, H, W)
        return self.fc(x.transpose(1, 3)).transpose(1, 3)  # 保持形状


"""扩散先验引导注意力"""


class DiffusionAttention(nn.Module):
    def __init__(self, input_channels, feature_dim, attention_dim, alpha=0.2, dropout=0.1):
        super().__init__()
        # 修正参数定义
        self.diffusion_conv = nn.Conv2d(input_channels, feature_dim, 3, padding=1)
        self.attention_conv = nn.Conv2d(feature_dim, attention_dim, 3, padding=1)
        self.fc = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.freq_attn = FrequencyChannelAttention(feature_dim)
        self.to(device)

    def forward(self, x):
        noise = torch.randn_like(x) * 0.1
        x_noise = x + noise

        diffusion_features = F.relu(self.diffusion_conv(x_noise))
        diffusion_features = self.freq_attn(diffusion_features)  # 应用通道注意力
        attention_features = self.attention_conv(diffusion_features)

        batch_size, C, H, W = attention_features.shape
        attention_map = attention_features.view(batch_size, C, H * W).transpose(1, 2)

        attention_weights = self.fc(attention_map)
        attention_weights = self.softmax(attention_weights).transpose(1, 2)
        attention_weights = self.dropout(attention_weights)
        attention_weights = attention_weights.view(batch_size, 1, H, W)

        x_attention = x * attention_weights
        return x_attention, attention_weights

"""伪造检测与定位"""


class DiffuseNetWithForgeryDetection(nn.Module):
    def __init__(self, embed_size=256, num_heads=8, dropout=0.1, d_model=512):
        super().__init__()
        # 输入适配层
        self.input_adapter = nn.Sequential(
            nn.Conv2d(3, embed_size, 3, padding=1),
            nn.BatchNorm2d(embed_size),
            nn.ReLU()
        )

        # 频率特征提取
        self.high_freq = HighDctFrequencyExtractor(embed_size)
        self.low_freq = LowDctFrequencyExtractor(embed_size)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_size * 2, embed_size, 1),
            FrequencyChannelAttention(embed_size),
            FrequencySpatialAttention()
        )

        # 注意力机制
        self.cnn_attention = CNNWithAttention(embed_size, num_heads, d_model)
        self.diff_attention = DiffusionAttention(
            input_channels=embed_size,
            feature_dim=d_model,
            attention_dim=d_model,  # 添加缺失的attention_dim参数
            alpha=0.2,
            dropout=dropout
        )

        # 输出头
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_size, 1),
            nn.Sigmoid()
        )

        self.localization_head = nn.Sequential(
            nn.Conv2d(embed_size, 4, 3, padding=1),
            nn.Tanh()  # 坐标归一化到[-1,1]
        )

    def forward(self, x):
        # 特征提取
        x = self.input_adapter(x)
        x_high = self.high_freq(x)
        x_low = self.low_freq(x)

        # 特征对齐
        if x_high.size() != x_low.size():
            x_low = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear')

        # 特征融合
        x_fused = self.fusion(torch.cat([x_high, x_low], dim=1))

        # 注意力处理
        x_attn = self.cnn_attention(x_fused)
        x_attn, _ = self.diff_attention(x_attn)

        # 输出
        detection = self.detection_head(x_attn)
        localization = self.localization_head(x_attn)

        return detection, localization