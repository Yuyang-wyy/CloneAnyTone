"""
The Wavenet-related part is adapted from the open-source repository: GuitarML/PedalNetRT, available at: https://github.com/GuitarML/PedalNetRT. 
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnedIRConvolver(nn.Module):
    """Trainable Impulse Response Convolver (optimized for time-domain training)
    Input:  (B, 1, T)
    Output: (B, 1, T) aligned with input (truncated/padded)
    """
    def __init__(self, ir_length=44100, wet=0.5):
        super().__init__()
        self.ir_length = ir_length
        # Initialize as approximate unit impulse to avoid clipping during early training
        ir = torch.zeros(ir_length)
        ir[0] = 1.0
        self.ir = nn.Parameter(ir)  # (L,)
         # Trainable wet/dry mix parameter
        self.wet_param = nn.Parameter(torch.tensor(float(wet)).atanh())  # mapped to (0,1) via sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        assert c == 1
        # Use group conv1d to implement 1D convolution (with time-reversed kernel)
        # Construct kernel: (out_channels=1, in_channels=1, kernel_size=L)
        kernel = self.ir.flip(0).view(1, 1, -1)
        y = F.conv1d(x, kernel, padding=self.ir_length - 1)
        y = y[:, :, :t]
        wet = torch.sigmoid(self.wet_param)
        return (1 - wet) * x + wet * y


class TrainableFastConvolver(nn.Module):
    """
    Trainable High-Performance Frequency-Domain Convolver (optimized for training long kernels)
    
    This module uses full-block FFT convolution — the fastest and most straightforward method 
    for training long convolution kernels. It is mathematically equivalent to time-domain 
    convolution and fully differentiable.
    """
    def __init__(self, ir_length, train_seq_len, wet=0.25):
        """
        Args:
            ir_length (int): Length of trainable impulse response
            train_seq_len (int): Fixed input audio length during training
            wet (float): Initial wet/dry mix ratio
        """
        super().__init__()
        self.ir_length = ir_length
        self.train_seq_len = train_seq_len

        # Convolution output length = train_seq_len + ir_length - 1
        required_len = self.train_seq_len + self.ir_length - 1
        # Find smallest power of 2 greater than required_len for optimal FFT efficiency
        self.fft_size = 1 << (required_len - 1).bit_length()

        # --- Trainable Parameters ---
        # Initialize as approximate unit impulse to avoid numerical instability or silence early in training
        ir = torch.zeros(ir_length)
        ir[0] = 1.0
        self.ir = nn.Parameter(ir)  # (L,)

        # Trainable wet/dry mix parameter
        self.wet_param = nn.Parameter(torch.tensor(float(wet)).atanh())

        print(f"TrainableFastConvolver initialized:")
        print(f"  Train Seq Length: {self.train_seq_len}")
        print(f"  IR Length: {self.ir_length}")
        print(f"  Optimized FFT Size: {self.fft_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input audio block (B, 1, train_seq_len)
        Returns:
            y (torch.Tensor): Output audio block (B, 1, train_seq_len)
        """
        batch_size, channels, seq_len = x.shape
        
        # Safety check: ensure input length matches expected training length
        if seq_len != self.train_seq_len:
            raise ValueError(f"Input seq_len ({seq_len}) must match configured train_seq_len ({self.train_seq_len})")

        # 1. Prepare IR FFT
        # Since self.ir is a trainable parameter, its FFT must be recomputed at every forward pass
        padded_ir = F.pad(self.ir, (0, self.fft_size - self.ir_length))
        ir_fft = torch.fft.rfft(padded_ir) # (fft_size//2 + 1,)

        # 2. Prepare input FFT
        padded_x = F.pad(x, (0, self.fft_size - seq_len))
        x_fft = torch.fft.rfft(padded_x) # (B, 1, fft_size//2 + 1)

        # 3. Multiply in frequency domain (core operation)
        # Expand ir_fft dimensions to match batch dimension of x_fft
        y_fft = x_fft * ir_fft.view(1, 1, -1)

        # 4. Inverse FFT back to time domain
        y_conv = torch.fft.irfft(y_fft, n=self.fft_size)

        # 5. Truncate to original input length
        y_conv = y_conv[:, :, :seq_len]

        # 6. Apply wet/dry mix
        wet = torch.sigmoid(self.wet_param)
        return (1 - wet) * x + wet * y_conv

class AudioPositionalEncoding(nn.Module):
    """Multi-scale learnable positional encoding designed for guitar audio effects"""
    
    def __init__(self, d_model, max_seq_len=4410, dropout=0.05, num_scales=4):
        """
        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length 
            num_scales: Number of temporal scales (key parameter)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_len = max_seq_len
        self.num_scales = num_scales
        self.d_model = d_model
        
        # Ensure d_model is divisible by num_scales
        assert d_model % num_scales == 0, f"d_model ({d_model}) must be divisible by num_scales ({num_scales})"
        self.scale_dim = d_model // num_scales
        
        # Multi-scale positional encoding (core innovation)
        # Each scale captures features at different temporal resolutions
        self.scales = nn.ParameterList([
            nn.Parameter(torch.randn(max_seq_len, self.scale_dim) * 0.05)
            for _ in range(num_scales)
        ])
        
        # Audio-aware initialization (critical!))
        self._audio_aware_init()
        
        # Scale weights (allow model to learn importance of each scale)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # Frequency-aware mask (enhance sensitivity to high-frequency transients)
        freq_mask = self._create_frequency_mask(max_seq_len)
        self.register_buffer('freq_mask', freq_mask)
    
    def _audio_aware_init(self):
        """Professional initialization based on audio signal characteristics"""
        for i, scale in enumerate(self.scales):
            # Time resolution for scale i: 2^i milliseconds
            time_scale = 2 ** i  # 1ms, 2ms, 4ms, 8ms
            
            # Create patterns matching guitar signal characteristics
            pos = torch.arange(self.max_seq_len, dtype=torch.float)
            
            # Generate base pattern (logarithmically spaced frequencies)
            base_freq = 0.5 * (0.2 ** i)   # decreasing frequency
            # Generate different patterns for each dimension
            pattern_matrix = torch.zeros(self.max_seq_len, self.scale_dim)
            for j in range(self.scale_dim):
                pattern = torch.sin(pos * base_freq * (j + 1))  # varying frequencies
                noise = torch.randn_like(pattern) * 0.02
                pattern_matrix[:, j] = (pattern + noise) * 0.1
            
            # Apply to parameter
            with torch.no_grad():
                scale.data = pattern_matrix
    
    def _create_frequency_mask(self, seq_len):
        """Mask to enhance high-frequency detail (targeting guitar transients)"""
        mask = torch.ones(seq_len, 1)
        # Boost weight for first 200ms (critical transient region))
        mask[:int(0.2 * seq_len)] = 1.5
        # Gradual decay
        for i in range(int(0.2 * seq_len), seq_len):
            mask[i] = 1.5 - 0.5 * (i - 0.2*seq_len) / (0.8*seq_len)
        return mask
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)

        # Combine multi-scale positional encodings
        pos_enc_list = []
        weights = torch.softmax(self.scale_weights, dim=0)  # shape: [num_scales]

        for i, scale in enumerate(self.scales):
            scale_enc = scale[:seq_len, :]  # [seq_len, scale_dim]
            weight = weights[i]
            weighted_scale_enc = scale_enc * weight
            pos_enc_list.append(weighted_scale_enc)

        # Concatenate all scales
        pos_enc = torch.cat(pos_enc_list, dim=1)  # [seq_len, d_model]
        
        # Validate dimensions
        assert pos_enc.shape == (seq_len, self.d_model), f"pos_enc shape {pos_enc.shape} != ({seq_len}, {self.d_model})"

        # Apply frequency mask
        freq_mask = self.freq_mask[:seq_len]  # [seq_len, 1]
        pos_enc = pos_enc * freq_mask  # [seq_len, d_model] * [seq_len, 1]

        # Add batch dimension and apply dropout
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

        # Support custom output channel count
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

            # Gated activation
            out_hidden_split = torch.split(out_hidden, self.num_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2) :]

        # Modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out

class CausalTransformer(nn.Module):
    """Causal Transformer model designed for guitar audio effects
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_seq_len=4410):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len 
        
        # Input projection: 1D feature per timestep → d_model
        self.input_proj = nn.Linear(1, d_model)

        self.pos_encoding = AudioPositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_scales=4  # Capture 1ms/2ms/4ms/8ms multi-scale features
        )
        
        # Transformer encoder (using batch_first=True for performance)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True  # Improved performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection: d_model → 1D prediction per timestep
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)  # Output 1-D prediction per timestep
        )
        
        # Create causal mask (max_seq_len, max_seq_len)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
        
        self._init_weights()
        print(f"Causal Transformer initialized for prepare.py data (max_seq_len={max_seq_len}, d_model={d_model})")
        print("Accepts input shape: (batch_size, 1, seq_len) where seq_len <= max_seq_len")

    def _init_weights(self):
        """Specialized initialization for professional audio models"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if isinstance(p, nn.Linear) and p.out_features == 1:
                nn.init.uniform_(p.weight, -0.01, 0.01)
                nn.init.zeros_(p.bias)

    def forward(self, x):
        """
        Forward pass (strictly causal)
        Args:
            x: (batch_size, 1, seq_len) audio input — format generated by prepare.py
        Returns:
            y: (batch_size, 1, seq_len) processed audio
        """
        batch_size, channels, seq_len = x.shape
        assert channels == 1, f"Expected 1 channel, got {channels}"
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        # 1. Transpose to (batch_size, seq_len, 1) for linear layer
        x = x.transpose(1, 2)  # (B, 1, T) -> (B, T, 1)
        
        # 2. Input projection — map 1D → d_model per timestep
        x = self.input_proj(x)  # (B, T, 1) -> (B, T, D)

        # 3. Apply positional encoding
        x = self.pos_encoding(x, seq_len=seq_len)
        
        # 4. Create appropriately sized causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # 5. Add numerical stability measures
        x_check = x.clone()
        
        # 6. Transformer processing
        try:
            x = self.transformer_encoder(x, mask=causal_mask)
        except Exception as e:
            print(f"Transformer error: {e}")
            print(f"Input stats - mean: {x_check.mean()}, std: {x_check.std()}, max: {x_check.max()}, min: {x_check.min()}")
            # Return safe output
            return torch.zeros_like(x_check).transpose(1, 2)
        
        # 7. Output projection — map d_model → 1D
        y = self.output_proj(x)  # (B, T, D) -> (B, T, 1)
        
        # 8. Truncate to original sequence length if needed
        y = y[:, :seq_len, :]
        
        # 9. Transpose back to (batch_size, 1, seq_len) to match input format
        y = y.transpose(1, 2)
        
        # 10. Clamp output range
        return torch.tanh(y)


class HybridAudioModel(nn.Module):
    def __init__(self, wavenet_channels=32, transformer_d_model=32, num_layers=2, 
                 dilation_depth=6, num_repeat=1, kernel_size=2,
                 transformer_nhead=4, transformer_dim_feedforward=128):
        super().__init__()
        
        # Front-end WaveNet layer (local feature extraction)
        self.wavenet_front = WaveNet(
            num_channels=wavenet_channels,
            dilation_depth=dilation_depth,
            num_repeat=num_repeat,
            kernel_size=kernel_size,
            output_channels=wavenet_channels 
        )

        # Projection layer: reduce wavenet_channels → 1 to match CausalTransformer input
        self.channel_projection = nn.Conv1d(wavenet_channels, 1, kernel_size=1)
        
        # Transformer (global dependency modeling)
        self.transformer = CausalTransformer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=num_layers,
            dim_feedforward=transformer_dim_feedforward
        )
        
    def forward(self, x):
        # Input: x = [B, 1, 4410]
        #print(f"Input shape: {x.shape}")
        
        local_features = self.wavenet_front(x)  # [B, wavenet_channels, 4410]
        #print(f"WaveNet output shape: {local_features.shape}")
        
        projected_features = self.channel_projection(local_features)  # [B, 1, 4410]
        #print(f"After projection: {projected_features.shape}")
        
        output = self.transformer(projected_features)  # [B, 1, 4410]
        #print(f"Transformer output shape: {output.shape}")
        
        return output

class EncoderBlock(nn.Module):
    """Encoder block: strided convolution downsampling + standard convolution block"""
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
    """Decoder block: transposed convolution upsampling + skip connection + standard convolution block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, output_padding=stride-1)
        self.conv_block = nn.Sequential(
            CausalConv1d(out_channels * 2, out_channels, kernel_size=kernel_size), nn.GELU(),
            CausalConv1d(out_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x, skip_connection):
        up_x = self.upsample(x)
        
        # Dynamically create skip connection adapter if needed
        if skip_connection.size(1) != up_x.size(1):
            adapter = nn.Conv1d(skip_connection.size(1), up_x.size(1), kernel_size=1).to(x.device)
            skip_connection = adapter(skip_connection)
        
        # Length alignment
        if skip_connection.size(2) > up_x.size(2):
            skip_connection = skip_connection[:, :, :up_x.size(2)]
        if up_x.size(2) > skip_connection.size(2):
            up_x = up_x[:, :, :skip_connection.size(2)]
            
        merged = torch.cat([up_x, skip_connection], dim=1)
        return self.conv_block(merged)

class ReverbLongModel(nn.Module):
    """
    Deep U-Net Hybrid Model for Long Sequences — Fully Dynamic Version
    """
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=512,
                 encoder_channels=[1, 32, 64, 128], 
                 decoder_channels=[128, 64, 32, 16],
                 encoder_strides=[2, 4, 4],
                 decoder_strides=[4, 4, 2]):
        super().__init__()
        
        self.input_length = 88200
        # Ensure encoder and decoder have matching number of layers
        assert len(encoder_channels) == len(decoder_channels), "Encoder and decoder must have same number of layers"
        
        # --- Dynamic Encoder ---
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            in_ch = encoder_channels[i]
            out_ch = encoder_channels[i + 1]
            stride = encoder_strides[i] if i < len(encoder_strides) else 2
            kernel_size = 7 if i < 2 else 5
            self.encoders.append(EncoderBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride))
        
        # --- Bottleneck (Simplified Transformer) ---
        self.proj_to_transformer = nn.Linear(encoder_channels[-1], d_model)

        self.pos_encoding = AudioPositionalEncoding(
            d_model=d_model,
            max_seq_len=44100,
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
            norm_first=True  # Added: LayerNorm before sublayers, improves training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj_from_transformer = nn.Linear(d_model, encoder_channels[-1])

        # --- Dynamic Decoder ---
        self.decoders = nn.ModuleList()
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            stride = decoder_strides[i] if i < len(decoder_strides) else 2
            kernel_size = 7 if i >= len(decoder_channels) - 3 else 5
            self.decoders.append(DecoderBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride))

        # Final output layer
        self.output_conv = CausalConv1d(decoder_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        input_length = x.shape[2]
        
        # Validate input length
        if input_length != self.input_length:
            raise ValueError(f"Expected input length {self.input_length}, got {input_length}")
        
        # 1. Encoder path — collect skip connections correctly
        skip_connections = [x]  # Original input
        current = x
        
        for encoder in self.encoders:
            current = encoder(current)
            skip_connections.append(current)  # Save output of each encoder
        
        # skip_connections [x, enc1_out, enc2_out, enc3_out, enc4_out]
        
        # 2. Bottleneck — Transformer processing
        bottleneck = current.transpose(1, 2)  # [B, channels, seq_len] -> [B, seq_len, channels]
        bottleneck = self.proj_to_transformer(bottleneck)  # [B, seq_len, channels] -> [B, seq_len, d_model]

        # Dynamically create causal mask
        seq_len = bottleneck.shape[1]
        bottleneck = self.pos_encoding(bottleneck, seq_len=seq_len)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=bottleneck.device), diagonal=1).bool()
        
        # Apply causal Transformer
        bottleneck = self.transformer(bottleneck, mask=causal_mask)  # [B, seq_len, d_model]
        bottleneck = self.proj_from_transformer(bottleneck)  # [B, seq_len, d_model] -> [B, seq_len, channels]
        bottleneck = bottleneck.transpose(1, 2)  # [B, seq_len, channels] -> [B, channels, seq_len]

        # 3. Decoder path — correct skip connection correspondence
        current = bottleneck
        
        # Use skip connections in reverse order (dec0 uses enc4_out, dec1 uses enc3_out, etc.)
        for i, decoder in enumerate(self.decoders):
            skip_idx = -(i + 2)  # -2, -3, -4, -5
            skip_conn = skip_connections[skip_idx]
            current = decoder(current, skip_conn)

        # 4. Final output
        output = self.output_conv(current)
        
        # Ensure output length strictly equals input length
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
                 encoder_strides=[2, 4, 4],
                 decoder_strides=[4, 4, 2],
                 # WaveNet post-processing parameters
                 wavenet_channels=16,
                 wavenet_dilation_depth=4,
                 wavenet_num_repeat=1,
                 # IR branch
                 use_ir=False,
                 ir_length=32768,
                 ir_wet=0.25):
        super().__init__()
        
        self.input_length = 88200
        self.use_ir = use_ir
        
        # Original U-Net + Transformer model
        self.main_model = ReverbLongModel(
            d_model, nhead, num_layers, dim_feedforward,
            encoder_channels, decoder_channels,
            encoder_strides, decoder_strides
        )
        
        # Post-processing WaveNet — using previously defined WaveNet class
        self.post_wavenet = WaveNet(
            num_channels=wavenet_channels,
            dilation_depth=wavenet_dilation_depth,
            num_repeat=wavenet_num_repeat,
            kernel_size=2,
            output_channels=1  
        )
        
        # Residual gating parameter (initialized to 0, sigmoid yields ~0.5)
        self.gate_param = nn.Parameter(torch.tensor(0.0))

        # IR module
        if self.use_ir:
            self.ir_branch = TrainableFastConvolver(
                ir_length=ir_length, 
                train_seq_len=self.input_length, # Pass critical parameter
                wet=ir_wet
            )

    @property
    def ir(self):
        """Expose trainable impulse response for debugging, visualization, or export"""
        if hasattr(self, 'ir_branch') and self.ir_branch is not None:
            return self.ir_branch.ir
        return None
        
    def forward(self, x):
        input_length = x.shape[2]
        
        # Main model processing
        main_output = self.main_model(x)
        
        # Ensure main model output length is correct
        if main_output.shape[2] > input_length:
            main_output = main_output[:, :, :input_length]
        elif main_output.shape[2] < input_length:
            pad_len = input_length - main_output.shape[2]
            main_output = F.pad(main_output, (0, pad_len))
        
        # WaveNet post-processing
        refined_output = self.post_wavenet(main_output)

        # Optional IR convolution branch
        if self.use_ir:
            ir_out = self.ir_branch(main_output)
            refined_output = 0.5 * refined_output + 0.5 * ir_out
        
        k = torch.sigmoid(self.gate_param)

        # Fusion method: convex combination
        final_output = (1 - k) * main_output + k * refined_output
        
        # Ensure final output length is correct
        if final_output.shape[2] > input_length:
            final_output = final_output[:, :, :input_length]
        elif final_output.shape[2] < input_length:
            pad_len = input_length - final_output.shape[2]
            final_output = F.pad(final_output, (0, pad_len))
            
        return torch.tanh(final_output)