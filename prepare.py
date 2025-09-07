import argparse
import pickle
from scipy.io import wavfile
import numpy as np
import os

def normalize(data):
    """归一化音频数据到 [-1, 1] 范围"""
    # 使用最大绝对值进行归一化，以保持信号的动态范围
    data_norm = np.max(np.abs(data))
    if data_norm > 0:
        return data / data_norm
    else:
        return data

def resample_to_target_rate(audio_data, original_rate, target_rate):
    """将音频数据重采样到目标采样率"""
    if original_rate == target_rate:
        return audio_data

    # 计算重采样后的长度
    num_samples = len(audio_data)
    new_length = int(num_samples * target_rate / original_rate)
    
    # 使用简单的线性插值进行重采样。
    # 注意：为了获得最高质量，可以考虑使用 `librosa` 或 `scipy.signal.resample`
    old_indices = np.linspace(0, num_samples - 1, num_samples)
    new_indices = np.linspace(0, num_samples - 1, new_length)
    resampled_data = np.interp(new_indices, old_indices, audio_data)
    
    return resampled_data.astype(np.float32)

def prepare(args):
    """准备训练数据"""
    print("Loading audio files...")
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)
    
    print(f"Input file: rate={in_rate} Hz, length={len(in_data)} samples, dtype={in_data.dtype}")
    print(f"Output file: rate={out_rate} Hz, length={len(out_data)} samples, dtype={out_data.dtype}")

    # --- 数据预处理 ---
    
    # 1. 处理立体声 -> 单声道
    if len(in_data.shape) > 1:
        print("[INFO] Input is stereo, converting to mono by averaging channels.")
        in_data = in_data.mean(axis=1)
    if len(out_data.shape) > 1:
        print("[INFO] Output is stereo, converting to mono by averaging channels.")
        out_data = out_data.mean(axis=1)

    # 2. 转换数据类型到 float32 [-1.0, 1.0]
    for name, data in [('Input', in_data), ('Output', out_data)]:
        if data.dtype in [np.int16, np.int32]:
            bits = 16 if data.dtype == np.int16 else 32
            data = data.astype(np.float32) / (2**(bits - 1))
            print(f"[INFO] {name} data converted from PCM{bits} to float32.")
        elif data.dtype == np.float64:
            data = data.astype(np.float32)
            print(f"[INFO] {name} data converted from float64 to float32.")
    
    # 3. 重采样到目标采样率
    target_rate = args.target_sample_rate
    if in_rate != target_rate:
        print(f"Resampling input audio from {in_rate} Hz to {target_rate} Hz...")
        in_data = resample_to_target_rate(in_data, in_rate, target_rate)
    if out_rate != target_rate:
        print(f"Resampling output audio from {out_rate} Hz to {target_rate} Hz...")
        out_data = resample_to_target_rate(out_data, out_rate, target_rate)

    # 4. 对齐音频长度（保留混响尾巴：用零填充较短的一端而不是裁剪）
    if len(in_data) != len(out_data):
        len_in, len_out = len(in_data), len(out_data)
        print(f"[WARNING] Audio lengths differ (in={len_in}, out={len_out}). Padding shorter to preserve tails.")
        if len_in < len_out:
            pad = len_out - len_in
            in_data = np.pad(in_data, (0, pad), mode='constant')
        else:
            pad = len_in - len_out
            out_data = np.pad(out_data, (0, pad), mode='constant')

    # 5. 归一化 (可选)
    if args.normalize:
        print("Normalizing audio data to [-1, 1]...")
        in_data = normalize(in_data)
        out_data = normalize(out_data)
        print(f"Data normalized. Max input val: {np.max(np.abs(in_data)):.4f}, Max output val: {np.max(np.abs(out_data)):.4f}")

    # --- 关键修改：使用重叠滑动窗口进行数据分段 ---
    sample_size = args.target_sample_size
    hop_size = args.hop_size if args.hop_size is not None else sample_size // 2
    
    print("-" * 30)
    print("Creating overlapping segments for training data...")
    print(f"  Window size: {sample_size} samples ({sample_size/target_rate:.3f} seconds)")
    print(f"  Hop size (step): {hop_size} samples ({hop_size/target_rate:.3f} seconds)")
    print(f"  Overlap: {sample_size - hop_size} samples ({(sample_size - hop_size)/sample_size:.1%})")

    if len(in_data) < sample_size:
        raise ValueError(f"Audio file is too short. It must be at least {sample_size} samples long.")

    x_list, y_list = [], []
    for i in range(0, len(in_data) - sample_size + 1, hop_size):
        x_list.append(in_data[i : i + sample_size])
        y_list.append(out_data[i : i + sample_size])

    x = np.array(x_list)[:, np.newaxis, :].astype(np.float32)
    y = np.array(y_list)[:, np.newaxis, :].astype(np.float32)

    print(f"Data shapes after segmentation: x={x.shape}, y={y.shape}")
    print("-" * 30)

    # --- 数据集分割与保存 ---
    
    # 分割为训练、验证、测试集 (例如 80% 训练, 10% 验证, 10% 测试)
    num_samples = len(x)
    train_end = int(num_samples * 0.95)
    valid_end = int(num_samples * 0.95)
    
    indices = np.arange(num_samples)
    #np.random.shuffle(indices) # 打乱数据以确保随机性
    
    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    d = {}
    d["x_train"], d["x_valid"], d["x_test"] = x[train_indices], x[valid_indices], x[test_indices]
    d["y_train"], d["y_valid"], d["y_test"] = y[train_indices], y[valid_indices], y[test_indices]
    # 保存用于重建的元数据
    d["sample_size"], d["hop_size"], d["target_sample_rate"] = sample_size, hop_size, target_rate
    
    # 计算标准化参数 (仅基于训练集)
    d["mean"], d["std"] = d["x_train"].mean(), d["x_train"].std()
    print(f"Normalization stats from training set: mean={d['mean']:.6f}, std={d['std']:.6f}")

    # 标准化输入数据 x (注意：不要标准化y)
    for key in ["x_train", "x_valid", "x_test"]:
        d[key] = (d[key] - d["mean"]) / d["std"]

    # 确定输出目录与文件名
    out_dir = args.out_dir if args.out_dir else os.path.dirname(args.model)
    if not out_dir:
        out_dir = "."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")

    if args.out_name:
        data_filename = args.out_name
    else:
        if args.model:
            data_filename = os.path.splitext(os.path.basename(args.model))[0] + "_dataDelayOnly.pickle"
        else:
            data_filename = "DelayOnly.pickle"

    data_file_path = os.path.join(out_dir, data_filename)
    with open(data_file_path, "wb") as f:
        pickle.dump(d, f)
    print(f"Data successfully saved to: {data_file_path}")
    
    # 打印最终数据集信息
    print("\n--- Dataset Summary ---")
    print(f"  Train samples: {len(d['x_train'])}")
    print(f"  Valid samples: {len(d['x_valid'])}") 
    print(f"  Test samples:  {len(d['x_test'])}")
    print("-" * 25)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare audio data for training with overlapping windows.")
    parser.add_argument("in_file", help="Path to the input (dry) audio file.")
    parser.add_argument("out_file", help="Path to the output (wet) audio file.")
    parser.add_argument("--model", type=str, default="models/pedalHybrid/ReverbModel.ckpt",
                        help="Path to the model checkpoint (used to determine save directory and data filename).")
    parser.add_argument("--out_dir", type=str, default="preparedData",
                        help="Directory to save prepared dataset (.pickle). Defaults to preparedData.")
    parser.add_argument("--out_name", type=str, default="",
                        help="Custom output dataset filename (e.g., MyData.pickle). If empty, derived from --model.")
    
    # --- 关键参数 ---
    parser.add_argument("--target_sample_rate", type=int, default=44100,
                        help="Target sample rate to resample all audio to.")
    parser.add_argument("--target_sample_size", type=int, default=44100,
                        help="The length of each audio segment in samples. Default is 2 seconds at 44.1kHz.")
    parser.add_argument("--hop_size", type=int, default=None,
                        help="Step size for the sliding window. If None, defaults to 50%% overlap (sample_size / 2).")
    
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Normalize audio to [-1, 1] range before segmentation.")
    parser.add_argument("--no-normalize", dest='normalize', action='store_false',
                        help="Disable normalization.")
                        
    args = parser.parse_args()
    
    prepare(args)