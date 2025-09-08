import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
import numpy as np
from modelTransformer import CausalTransformer, HybridAudioModel, ReverbLongModel, HybridWaveNetUNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main(args):
    torch.autograd.set_detect_anomaly(True)
    # 加载数据（更健壮的路径解析）
    model_dir = os.path.dirname(args.model) if os.path.dirname(args.model) else "."
    base_name = os.path.splitext(os.path.basename(args.model))[0]
    candidate_paths = []
    if getattr(args, "data", None):
        candidate_paths.append(args.data)
    candidate_paths.extend([
        os.path.join(model_dir, f"{base_name}_dataDelayOnly.pickle"),
        os.path.join("preparedData", "DelayOnly.pickle"),
        os.path.join("preparedData", "Overdrive+Reverb+Delay.pickle"),
        os.path.join("preparedData", "Overdrive+Reverb.pickle"),
    ])
    data_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
    if data_path is None:
        raise FileNotFoundError(f"Could not find prepared data. Tried: {candidate_paths}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded data from: {data_path}")
    
    # 确认数据形状 (应该是 (num_samples, 1, 4410))
    # 动态获取序列长度
    seq_len = data['x_train'].shape[2]
    print(f"Data shapes: x_train={data['x_train'].shape}, y_train={data['y_train'].shape}")
    print(f"Expected format: (num_samples, 1, sample_size={seq_len})")
    
    # 创建数据集 (保持prepare.py的维度顺序)
    train_dataset = TensorDataset(
        torch.from_numpy(data["x_train"]),  # (num_samples, 1, seq_len)
        torch.from_numpy(data["y_train"])
    )
    
    # 验证数据形状
    sample_x, sample_y = train_dataset[0]
    print(f"Dataset sample shapes: x={sample_x.shape}, y={sample_y.shape}")
    print("Expected model input shape: (batch_size, 1, seq_len)")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # Windows兼容
        pin_memory=True if torch.cuda.is_available() and not args.cpu else False
    )
    
    # 检查第一个batch
    for x, y in train_loader:
        print(f"Loader batch shapes: x={x.shape}, y={y.shape}")
        break
    
    # 初始化模型 - 使用动态序列长度
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # 根据模型类型创建不同模型
    if args.model_type == "transformer":
        model = CausalTransformer(
            d_model=args.transformer_d_model,
            nhead=args.transformer_nhead,
            num_layers=args.transformer_layers,
            dim_feedforward=args.transformer_dim_feedforward,
            max_seq_len=seq_len  # 使用动态序列长度
        ).to(device)
    elif args.model_type == "hybrid":
        model = HybridAudioModel(
            wavenet_channels=args.wavenet_channels,
            transformer_d_model=args.transformer_d_model,
            num_layers=args.transformer_layers,
            transformer_nhead=args.transformer_nhead,
            transformer_dim_feedforward=args.transformer_dim_feedforward,
            dilation_depth=args.wavenet_dilation_depth,
            num_repeat=args.wavenet_num_repeat,
            kernel_size=args.wavenet_kernel_size
        ).to(device)
    elif args.model_type == "hybridplus":
        model = ReverbLongModel(
            d_model=args.transformer_d_model,
            nhead=args.transformer_nhead,
            num_layers=args.transformer_layers,
            dim_feedforward=args.transformer_dim_feedforward
        ).to(device)
    elif args.model_type == "hybrid_wavenet_unet":
        model = HybridWaveNetUNet(
            d_model=args.transformer_d_model,
            nhead=args.transformer_nhead,
            num_layers=args.transformer_layers,
            dim_feedforward=args.transformer_dim_feedforward,
            wavenet_channels=args.wavenet_channels,
            wavenet_dilation_depth=args.wavenet_dilation_depth,
            wavenet_num_repeat=args.wavenet_num_repeat,
            use_ir=args.use_ir,
            ir_length=args.ir_length,
            ir_wet=args.ir_wet
        ).to(device)  # 修复：添加 .to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    def count_parameters(model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    print(f"Model created with {total_params:,} trainable parameters")
    
    # 可选：输出更详细的参数信息
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")

    writer = SummaryWriter(log_dir=os.path.join(model_dir, "runs"))
    
    # 优化器 (带梯度裁剪)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # 学习率调度器 (带预热)
    warmup_steps = 100
    total_steps = len(train_loader) * args.max_epochs
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 恢复训练
    start_epoch = 0
    if args.resume and os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

        # 如果需要恢复调度器状态
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except:
            print("Could not restore scheduler state, continuing with new scheduler")
            # 手动推进调度器到正确位置
            current_step = start_epoch * len(train_loader)
            for _ in range(current_step):
                scheduler.step()

    # 训练循环
    target_epoch = start_epoch + args.max_epochs
    print(f"Training from epoch {start_epoch} to {target_epoch}")
    
    for epoch in range(start_epoch, target_epoch):
        current_epoch_display = epoch - start_epoch + 1
        
        model.train()
        total_loss = 0
        
        # 使用 tqdm 创建进度条
        progress_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Epoch {current_epoch_display}/{args.max_epochs}",
            leave=False  # 不保留进度条
        )
        
        for batch_idx, (x, y) in progress_bar:
            x, y = x.to(device), y.to(device)
            
            # 验证输入形状 - 使用动态序列长度
            assert x.shape[1] == 1, f"Expected 1 channel, got {x.shape[1]}"
            assert x.shape[2] == seq_len, f"Expected seq_len={seq_len}, got {x.shape[2]}"
            
            # 前向传播
            optimizer.zero_grad()
            y_pred = model(x)
            
            # 验证输出形状
            assert y_pred.shape == y.shape, f"Output shape {y_pred.shape} != target shape {y.shape}"
            
            # 计算损失
            loss = audio_loss(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()

            if isinstance(model, HybridWaveNetUNet):
                k_value = torch.sigmoid(model.gate_param).item()
                writer.add_scalar("residual/k", k_value, epoch * len(train_loader) + batch_idx)
            
            # 更新进度条显示
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {current_epoch_display}/{args.max_epochs} (absolute: {epoch+1}) | Avg Loss: {avg_loss:.6f}")
        
        # 保存检查点
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(checkpoint, args.model)
        print(f"Checkpoint saved to {args.model}")
        writer.close()

def esr_loss(y_pred, y_true):
    """
    Calculate the Error-to-Signal Ratio (ESR) loss.
    """
    # 确保输入是浮点类型
    y_pred = y_pred.to(torch.float32)
    y_true = y_true.to(torch.float32)

    # 计算误差的能量
    error = y_pred - y_true
    error_energy = torch.sum(error ** 2, dim=-1, keepdim=True) # 在时间维度求和 (B, 1, 1)
    
    # 计算原始信号的能量
    signal_energy = torch.sum(y_true ** 2, dim=-1, keepdim=True) # 在时间维度求和 (B, 1, 1)
    
    # 计算 ESR，添加小量防止除零
    esr = error_energy / (signal_energy + 1e-10)
    
    # 返回平均 ESR 标量
    return torch.mean(esr)

def audio_loss(y_pred, y_true):
    """专业音频损失函数 (包含 ESR)"""
    # 时域MSE
    mse_loss = nn.MSELoss()(y_pred, y_true)
    
    # 预加重损失
    def pre_emphasis(x, coeff=0.95):
        return x[..., :, 1:] - coeff * x[..., :, :-1]
    
    y_pred_emph = pre_emphasis(y_pred)
    y_true_emph = pre_emphasis(y_true)
    emph_loss = nn.MSELoss()(y_pred_emph, y_true_emph)
    
    # 频谱损失
    def spectral_loss(pred, target, n_fft=1024, hop_length=256):
        # 移除通道维度以便STFT
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        spec_pred = torch.stft(
            pred, n_fft, hop_length, 
            window=torch.hann_window(n_fft).to(pred.device),
            return_complex=True
        )
        spec_true = torch.stft(
            target, n_fft, hop_length,
            window=torch.hann_window(n_fft).to(target.device),
            return_complex=True
        )
        
        # 计算幅度谱损失
        mag_loss = nn.L1Loss()(spec_pred.abs(), spec_true.abs())
        return mag_loss

    def multi_resolution_stft_loss(y_pred, y_true, ffts=[2048, 1024, 512], hops=[512, 256, 128]):
        total_loss = 0.0
        for n_fft, hop_length in zip(ffts, hops):
            # spectral_loss 内部逻辑不变，只是被调用多次
            total_loss += spectral_loss(y_pred, y_true, n_fft, hop_length)
        return total_loss / len(ffts)

    try:
        spec_loss = multi_resolution_stft_loss(y_pred, y_true)
    except:
        spec_loss = 0.0

    # --- 新增：计算 ESR 损失 ---
    #esr = esr_loss(y_pred, y_true)

    # 组合损失 (你可以调整权重)
    total_loss = 1.0 * mse_loss + 0.5 * emph_loss + 0.3 * spec_loss
    return total_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio Models for Guitar Effects")
    parser.add_argument("--model", type=str, default="models/Delay/HybridWUNet.ckpt.ckpt")
    parser.add_argument("--data", type=str, default=None, help="Path to prepared dataset (.pickle)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--model_type", type=str, default="transformer", 
                       choices=["transformer", "hybrid", "hybridplus", "hybrid_wavenet_unet"], help="Model architecture type")
    
    # 通用训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    
    # 在 parser.add_argument 部分添加：
    # Hybrid 模型专用参数
    parser.add_argument("--wavenet_channels", type=int, default=16)
    parser.add_argument("--transformer_d_model", type=int, default=64)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_nhead", type=int, default=4)
    parser.add_argument("--transformer_dim_feedforward", type=int, default=128)
    parser.add_argument("--wavenet_dilation_depth", type=int, default=8)
    parser.add_argument("--wavenet_num_repeat", type=int, default=2)
    parser.add_argument("--wavenet_kernel_size", type=int, default=3)
    # U-Net参数
    parser.add_argument("--unet_base_channels", type=int, default=32)
     # IR 分支参数
    parser.add_argument("--use_ir", action="store_true", help="Enable learnable IR convolver branch")
    parser.add_argument("--ir_length", type=int, default=44100, help="Length of learnable IR in samples")
    parser.add_argument("--ir_wet", type=float, default=0.25, help="Initial wet ratio for IR branch (0-1)")
    
    args = parser.parse_args()
    main(args)