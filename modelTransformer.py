import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnedIRConvolver(nn.Module):
    """可学习的脉冲响应卷积（基于时域快速实现，训练友好）
    输入: (B, 1, T)
    输出: (B, 1, T) 与输入对齐（截断/填充）
    """
    def __init__(self, ir_length=44100, wet=0.5):
        super().__init__()
        self.ir_length = ir_length
        # 初始化为近似单位冲激，以免训练初期破音
        ir = torch.zeros(ir_length)
        ir[0] = 1.0
        self.ir = nn.Parameter(ir)  # (L,)
        # 可训练的湿度参数
        self.wet_param = nn.Parameter(torch.tensor(float(wet)).atanh())  # 通过sigmoid映射到(0,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        assert c == 1
        # 使用 group conv1d 实现 1D 卷积（时间反转的核）
        # 构造 (out_channels=1, in_channels=1, kernel=L)
        kernel = self.ir.flip(0).view(1, 1, -1)
        y = F.conv1d(x, kernel, padding=self.ir_length - 1)
        y = y[:, :, :t]
        wet = torch.sigmoid(self.wet_param)
        return (1 - wet) * x + wet * y


class TrainableFastConvolver(nn.Module):
    """
    可训练的高性能频域卷积器 (专为训练优化)
    
    这个模块使用全块FFT卷积，是训练长卷积核最快、最直接的方法。
    它在数学上等价于时域卷积，并且完全可微分。
    """
    def __init__(self, ir_length, train_seq_len, wet=0.25):
        """
        Args:
            ir_length (int): 可学习脉冲响应的长度
            train_seq_len (int): 训练时输入音频的固定长度
            wet (float): 初始干湿比
        """
        super().__init__()
        self.ir_length = ir_length
        self.train_seq_len = train_seq_len

        # --- 核心优化：在初始化时就计算好FFT尺寸 ---
        # 卷积结果的长度为 train_seq_len + ir_length - 1
        required_len = self.train_seq_len + self.ir_length - 1
        # 找到比它大的最小的2的次幂，以获得最高FFT效率
        self.fft_size = 1 << (required_len - 1).bit_length()

        # --- 可学习参数 ---
        # 初始化为近似单位冲激，避免训练初期数值爆炸或静音
        ir = torch.zeros(ir_length)
        ir[0] = 1.0
        self.ir = nn.Parameter(ir)  # (L,)

        # 可学习的干湿比参数
        self.wet_param = nn.Parameter(torch.tensor(float(wet)).atanh())

        print(f"TrainableFastConvolver initialized:")
        print(f"  Train Seq Length: {self.train_seq_len}")
        print(f"  IR Length: {self.ir_length}")
        print(f"  Optimized FFT Size: {self.fft_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入音频块 (B, 1, train_seq_len)
        Returns:
            y (torch.Tensor): 输出音频块 (B, 1, train_seq_len)
        """
        batch_size, channels, seq_len = x.shape
        
        # 安全检查，确保输入长度符合预期
        if seq_len != self.train_seq_len:
            raise ValueError(f"Input seq_len ({seq_len}) must match configured train_seq_len ({self.train_seq_len})")

        # 1. 准备IR的FFT
        #    因为 self.ir 是可训练参数，它的FFT必须在每次前向传播时重新计算
        padded_ir = F.pad(self.ir, (0, self.fft_size - self.ir_length))
        ir_fft = torch.fft.rfft(padded_ir) # (fft_size//2 + 1,)

        # 2. 准备输入的FFT
        padded_x = F.pad(x, (0, self.fft_size - seq_len))
        x_fft = torch.fft.rfft(padded_x) # (B, 1, fft_size//2 + 1)

        # 3. 频域相乘 (核心)
        #    需要将 ir_fft 的维度扩展以匹配 x_fft 的 batch 维度
        y_fft = x_fft * ir_fft.view(1, 1, -1)

        # 4. 逆FFT回到时域
        y_conv = torch.fft.irfft(y_fft, n=self.fft_size)

        # 5. 裁剪到原始输入长度
        y_conv = y_conv[:, :, :seq_len]

        # 6. 应用干湿比混合
        wet = torch.sigmoid(self.wet_param)
        return (1 - wet) * x + wet * y_conv

class AudioPositionalEncoding(nn.Module):
    """专为吉他效果器设计的多尺度可学习位置编码"""
    
    def __init__(self, d_model, max_seq_len=4410, dropout=0.05, num_scales=4):
        """
        Args:
            d_model: 模型维度 (建议256+)
            max_seq_len: 最大序列长度 (prepare.py生成的4410)
            num_scales: 多尺度数量 (关键参数)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_len = max_seq_len
        self.num_scales = num_scales
        self.d_model = d_model
        
        # 确保 d_model 能被 num_scales 整除
        assert d_model % num_scales == 0, f"d_model ({d_model}) must be divisible by num_scales ({num_scales})"
        self.scale_dim = d_model // num_scales
        
        # 多尺度位置编码 (核心创新)
        # 每个尺度捕获不同时间分辨率的特征
        self.scales = nn.ParameterList([
            nn.Parameter(torch.randn(max_seq_len, self.scale_dim) * 0.05)
            for _ in range(num_scales)
        ])
        
        # 音频感知初始化 (关键!)
        self._audio_aware_init()
        
        # 尺度权重 (让模型学习各尺度重要性)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # 频率感知掩码 (增强高频敏感性)
        freq_mask = self._create_frequency_mask(max_seq_len)
        self.register_buffer('freq_mask', freq_mask)
    
    def _audio_aware_init(self):
        """基于音频信号特性的专业初始化"""
        for i, scale in enumerate(self.scales):
            # 尺度i对应的时间分辨率: 2^i 毫秒
            time_scale = 2 ** i  # 1ms, 2ms, 4ms, 8ms
            
            # 创建与吉他信号匹配的初始模式
            pos = torch.arange(self.max_seq_len, dtype=torch.float)
            
            # 生成基础模式 (对数间隔频率)
            base_freq = 0.5 * (0.2 ** i)  # 递减频率
            # 为每个维度生成不同的模式
            pattern_matrix = torch.zeros(self.max_seq_len, self.scale_dim)
            for j in range(self.scale_dim):
                pattern = torch.sin(pos * base_freq * (j + 1))  # 不同频率
                noise = torch.randn_like(pattern) * 0.02
                pattern_matrix[:, j] = (pattern + noise) * 0.1
            
            # 应用到参数
            with torch.no_grad():
                scale.data = pattern_matrix
    
    def _create_frequency_mask(self, seq_len):
        """增强高频细节的掩码 (针对吉他瞬态)"""
        mask = torch.ones(seq_len, 1)
        # 增强前200ms的高频权重 (关键瞬态区域)
        mask[:int(0.2 * seq_len)] = 1.5
        # 渐变衰减
        for i in range(int(0.2 * seq_len), seq_len):
            mask[i] = 1.5 - 0.5 * (i - 0.2*seq_len) / (0.8*seq_len)
        return mask
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)

        # 多尺度位置编码组合（正确方式）
        pos_enc_list = []
        weights = torch.softmax(self.scale_weights, dim=0)  # shape: [num_scales]

        for i, scale in enumerate(self.scales):
            scale_enc = scale[:seq_len, :]  # [seq_len, scale_dim]
            weight = weights[i]
            weighted_scale_enc = scale_enc * weight
            pos_enc_list.append(weighted_scale_enc)

        # 拼接所有尺度的位置编码
        pos_enc = torch.cat(pos_enc_list, dim=1)  # [seq_len, d_model]
        
        # 验证维度
        assert pos_enc.shape == (seq_len, self.d_model), f"pos_enc shape {pos_enc.shape} != ({seq_len}, {self.d_model})"

        # 应用频率掩码
        freq_mask = self.freq_mask[:seq_len]  # [seq_len, 1]
        pos_enc = pos_enc * freq_mask  # [seq_len, d_model] * [seq_len, 1]

        # 增加 batch 维度并 dropout
        pos_enc = pos_enc.unsqueeze(0)  # [1, seq_len, d_model]
        return self.dropout(x + pos_enc)

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


def _conv_stack(dilations, in_channels, out_channels, kernel_size):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList(
        [
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
            )
            for i, d in enumerate(dilations)
        ]
    )

class WaveNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, output_channels=None):
        super(WaveNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        internal_channels = int(num_channels * 2)
        self.hidden = _conv_stack(dilations, num_channels, internal_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)
        self.input_layer = CausalConv1d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=1,
        )

        # 修改这里：支持自定义输出通道数
        self.output_channels = output_channels or num_channels
        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=self.output_channels,
            kernel_size=1,
        )
        self.num_channels = num_channels

    def forward(self, x):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            # gated activation
            out_hidden_split = torch.split(out_hidden, self.num_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2) :]

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out

class CausalTransformer(nn.Module):
    """专为吉他效果器设计的因果Transformer模型
    完美匹配prepare.py生成的数据格式 (batch, 1, seq_len)
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_seq_len=4410):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len  # 支持prepare.py生成的4410长度
        
        # ✅ 修改1: 输入投影层 - 每个时间点的1维特征 -> d_model
        self.input_proj = nn.Linear(1, d_model)

        self.pos_encoding = AudioPositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_scales=4  # 捕获1ms/2ms/4ms/8ms多尺度特征
        )
        
        # Transformer编码器 (使用batch_first=True提升性能)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True  # ✅ 新增: 提升性能
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # ✅ 修改2: 输出投影层 - 从d_model -> 1
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)  # 输出每个时间点的1维预测
        )
        
        # 创建因果掩码 (max_seq_len, max_seq_len)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
        
        self._init_weights()
        print(f"Causal Transformer initialized for prepare.py data (max_seq_len={max_seq_len}, d_model={d_model})")
        print("Accepts input shape: (batch_size, 1, seq_len) where seq_len <= max_seq_len")

    def _init_weights(self):
        """专业音频模型的特殊初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if isinstance(p, nn.Linear) and p.out_features == 1:  # ✅ 修改: 检查输出维度为1
                nn.init.uniform_(p.weight, -0.01, 0.01)
                nn.init.zeros_(p.bias)

    def forward(self, x):
        """
        前向传播 (严格因果)
        Args:
            x: (batch_size, 1, seq_len) 音频输入 - prepare.py生成的格式
        Returns:
            y: (batch_size, 1, seq_len) 处理后的音频
        """
        batch_size, channels, seq_len = x.shape
        assert channels == 1, f"Expected 1 channel, got {channels}"
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        # 1. 转置为 (batch_size, seq_len, 1) 以匹配线性层输入
        x = x.transpose(1, 2)  # (B, 1, T) -> (B, T, 1)
        
        # 2. ✅ 输入投影 - 每个时间点从1维映射到d_model
        x = self.input_proj(x)  # (B, T, 1) -> (B, T, D)

        # 3. 应用位置编码
        x = self.pos_encoding(x, seq_len=seq_len)
        
        # 4. ✅ 创建适当大小的因果掩码 (使用batch_first时mask需要调整)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # 5. 添加数值稳定性措施
        # 保存原始值以便检查NaN
        x_check = x.clone()
        
        # 5. Transformer处理
        try:
            x = self.transformer_encoder(x, mask=causal_mask)
        except Exception as e:
            print(f"Transformer error: {e}")
            print(f"Input stats - mean: {x_check.mean()}, std: {x_check.std()}, max: {x_check.max()}, min: {x_check.min()}")
            # 返回一个安全的输出
            return torch.zeros_like(x_check).transpose(1, 2)
        
        # 6. 输出投影 - 从d_model映射回1维
        y = self.output_proj(x)  # (B, T, D) -> (B, T, 1)
        
        # 7. 裁剪回原始序列长度（如果需要）
        y = y[:, :seq_len, :]
        
        # 8. 转置回 (batch_size, 1, seq_len) 匹配输入格式
        y = y.transpose(1, 2)
        
        # 9. 限制输出范围
        return torch.tanh(y)


class HybridAudioModel(nn.Module):
    def __init__(self, wavenet_channels=32, transformer_d_model=32, num_layers=2, 
                 dilation_depth=6, num_repeat=1, kernel_size=2,
                 transformer_nhead=4, transformer_dim_feedforward=128):
        super().__init__()
        
        # 前置 WaveNet 层 (局部特征提取)
        self.wavenet_front = WaveNet(
            num_channels=wavenet_channels,
            dilation_depth=dilation_depth,
            num_repeat=num_repeat,
            kernel_size=kernel_size,
            output_channels=wavenet_channels  # 让WaveNet输出多通道
        )

        # 投影层：把 wavenet_channels -> 1，适配 CausalTransformer 输入
        self.channel_projection = nn.Conv1d(wavenet_channels, 1, kernel_size=1)
        
        # Transformer (全局特征建模)
        self.transformer = CausalTransformer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=num_layers,
            dim_feedforward=transformer_dim_feedforward
        )
        
    def forward(self, x):
        # 输入: x = [B, 1, 4410]
        #print(f"Input shape: {x.shape}")
        
        # 先用 WaveNet 提取局部特征
        local_features = self.wavenet_front(x)  # [B, wavenet_channels, 4410]
        #print(f"WaveNet output shape: {local_features.shape}")
        
        # 使用投影层将多通道压缩为单通道
        projected_features = self.channel_projection(local_features)  # [B, 1, 4410]
        #print(f"After projection: {projected_features.shape}")
        
        # 用 Transformer 建模全局依赖
        output = self.transformer(projected_features)  # [B, 1, 4410]
        #print(f"Transformer output shape: {output.shape}")
        
        return output

class EncoderBlock(nn.Module):
    """编码器模块：步长卷积降采样 + 普通卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.conv_down = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv_block = nn.Sequential(
            CausalConv1d(out_channels, out_channels, kernel_size=kernel_size), nn.GELU(),
            CausalConv1d(out_channels, out_channels, kernel_size=kernel_size)
        )
        self.residual_proj = CausalConv1d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        res = self.residual_proj(x)
        out = self.conv_down(x)
        out = self.conv_block(out)
        out_len, res_len = out.shape[2], res.shape[2]
        if res_len > out_len:
            res = res[:, :, :out_len]
        return F.gelu(out + res)

class DecoderBlock(nn.Module):
    """解码器模块：转置卷积上采样 + 跳跃连接 + 普通卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, output_padding=stride-1)
        self.conv_block = nn.Sequential(
            CausalConv1d(out_channels * 2, out_channels, kernel_size=kernel_size), nn.GELU(),
            CausalConv1d(out_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x, skip_connection):
        up_x = self.upsample(x)
        
        # 动态创建skip connection适配器（如果需要）
        if skip_connection.size(1) != up_x.size(1):
            adapter = nn.Conv1d(skip_connection.size(1), up_x.size(1), kernel_size=1).to(x.device)
            skip_connection = adapter(skip_connection)
        
        # 长度对齐
        if skip_connection.size(2) > up_x.size(2):
            skip_connection = skip_connection[:, :, :up_x.size(2)]
        if up_x.size(2) > skip_connection.size(2):
            up_x = up_x[:, :, :skip_connection.size(2)]
            
        merged = torch.cat([up_x, skip_connection], dim=1)
        return self.conv_block(merged)

class ReverbLongModel(nn.Module):
    """
    能处理长序列的深度 U-Net 混合模型 - 完全动态版本
    """
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=512,
                 encoder_channels=[1, 32, 64, 128], 
                 decoder_channels=[128, 64, 32, 16],
                 encoder_strides=[4, 4, 4,],
                 decoder_strides=[4, 4, 4]):
        super().__init__()
        
        self.input_length = 44100
        # 确保编码器和解码器通道数匹配
        assert len(encoder_channels) == len(decoder_channels), "Encoder and decoder must have same number of layers"
        
        # --- 动态编码器 ---
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            in_ch = encoder_channels[i]
            out_ch = encoder_channels[i + 1]
            stride = encoder_strides[i] if i < len(encoder_strides) else 2
            kernel_size = 7 if i < 2 else 5
            self.encoders.append(EncoderBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride))
        
        # --- 瓶颈层 (简化Transformer) ---
        self.proj_to_transformer = nn.Linear(encoder_channels[-1], d_model)

        self.pos_encoding = AudioPositionalEncoding(
            d_model=d_model,
            max_seq_len=44100,  # 因为你处理的是 88200 长度的片段
            dropout=0.05,
            num_scales=4
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # 新增：LayerNorm 放在前面，有助于训练稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj_from_transformer = nn.Linear(d_model, encoder_channels[-1])

        # --- 动态解码器 ---
        self.decoders = nn.ModuleList()
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            stride = decoder_strides[i] if i < len(decoder_strides) else 2
            kernel_size = 7 if i >= len(decoder_channels) - 3 else 5
            self.decoders.append(DecoderBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride))

        # 最终输出层
        self.output_conv = CausalConv1d(decoder_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        input_length = x.shape[2]
        
        # 验证输入长度
        if input_length != self.input_length:
            raise ValueError(f"Expected input length {self.input_length}, got {input_length}")
        
        # 1. 编码器路径 - 正确的skip connection收集
        skip_connections = [x]  # 原始输入
        current = x
        
        for encoder in self.encoders:
            current = encoder(current)
            skip_connections.append(current)  # 保存每个编码器的输出
        
        # skip_connections现在是: [x, enc1_out, enc2_out, enc3_out, enc4_out]
        
        # 2. 瓶颈层 - Transformer处理
        bottleneck = current.transpose(1, 2)  # [B, channels, seq_len] -> [B, seq_len, channels]
        bottleneck = self.proj_to_transformer(bottleneck)  # [B, seq_len, channels] -> [B, seq_len, d_model]

        # 动态创建因果掩码
        seq_len = bottleneck.shape[1]
        bottleneck = self.pos_encoding(bottleneck, seq_len=seq_len)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=bottleneck.device), diagonal=1).bool()
        
        # 应用因果Transformer
        bottleneck = self.transformer(bottleneck, mask=causal_mask)  # [B, seq_len, d_model]
        bottleneck = self.proj_from_transformer(bottleneck)  # [B, seq_len, d_model] -> [B, seq_len, channels]
        bottleneck = bottleneck.transpose(1, 2)  # [B, seq_len, channels] -> [B, channels, seq_len]

        # 3. 解码器路径 - 正确的skip connection对应关系
        current = bottleneck
        
        # 从后往前使用skip connections (但保持正确的对应关系)
        # dec0使用enc4_out, dec1使用enc3_out, dec2使用enc2_out, dec3使用enc1_out, dec4使用原始输入x
        for i, decoder in enumerate(self.decoders):
            # 获取对应的skip connection (从后往前数)
            skip_idx = -(i + 2)  # -2, -3, -4, -5
            skip_conn = skip_connections[skip_idx]
            current = decoder(current, skip_conn)

        # 4. 最终输出
        output = self.output_conv(current)
        
        # 确保输出长度严格等于输入长度
        if output.shape[2] > input_length:
            output = output[:, :, :input_length]
        elif output.shape[2] < input_length:
            pad_len = input_length - output.shape[2]
            output = F.pad(output, (0, pad_len))
            
        return torch.tanh(output)

class HybridWaveNetUNet(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=512,
                 encoder_channels=[1, 32, 64, 128], 
                 decoder_channels=[128, 64, 32, 16],
                 encoder_strides=[4, 4, 4],
                 decoder_strides=[4, 4, 4],
                 # WaveNet 后处理参数
                 wavenet_channels=16,
                 wavenet_dilation_depth=4,
                 wavenet_num_repeat=1,
                 # IR 分支
                 use_ir=False,
                 ir_length=32768,
                 ir_wet=0.25):
        super().__init__()
        
        self.input_length = 44100
        self.use_ir = use_ir
        
        # 原有的 U-Net + Transformer 模型
        self.main_model = ReverbLongModel(
            d_model, nhead, num_layers, dim_feedforward,
            encoder_channels, decoder_channels,
            encoder_strides, decoder_strides
        )
        
        # 后处理 WaveNet - 使用你之前定义的 WaveNet 类
        self.post_wavenet = WaveNet(
            num_channels=wavenet_channels,
            dilation_depth=wavenet_dilation_depth,
            num_repeat=wavenet_num_repeat,
            kernel_size=2,
            output_channels=1  # 输出单通道音频
        )
        
        # 残差门控参数（初始化为 0，经 sigmoid 后约等于 0.5）
        self.gate_param = nn.Parameter(torch.tensor(0.0))

                # --- 核心修改：替换卷积模块 ---
        if self.use_ir:
            self.ir_branch = TrainableFastConvolver(
                ir_length=ir_length, 
                train_seq_len=self.input_length, # 传入关键参数
                wet=ir_wet
            )

    @property
    def ir(self):
        """暴露可学习的脉冲响应，便于调试、可视化、导出"""
        if hasattr(self, 'ir_branch') and self.ir_branch is not None:
            return self.ir_branch.ir
        return None
        
    def forward(self, x):
        input_length = x.shape[2]
        
        # 主模型处理
        main_output = self.main_model(x)
        
        # 确保主模型输出长度正确
        if main_output.shape[2] > input_length:
            main_output = main_output[:, :, :input_length]
        elif main_output.shape[2] < input_length:
            pad_len = input_length - main_output.shape[2]
            main_output = F.pad(main_output, (0, pad_len))
        
        # WaveNet 后处理
        refined_output = self.post_wavenet(main_output)

        # 可选 IR 卷积分支（针对长尾）
        if self.use_ir:
            ir_out = self.ir_branch(main_output)
            refined_output = 0.5 * refined_output + 0.5 * ir_out
        
        k = torch.sigmoid(self.gate_param)

        # 融合方式：凸组合 (convex combination)
        final_output = (1 - k) * main_output + k * refined_output
        
        # 确保最终输出长度正确
        if final_output.shape[2] > input_length:
            final_output = final_output[:, :, :input_length]
        elif final_output.shape[2] < input_length:
            pad_len = input_length - final_output.shape[2]
            final_output = F.pad(final_output, (0, pad_len))
            
        return torch.tanh(final_output)