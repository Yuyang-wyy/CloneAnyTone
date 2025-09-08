import pickle
import torch
from scipy.io import wavfile
import argparse
import numpy as np
import os
from tqdm import tqdm

# Import all model classes
from modelTransformer import HybridAudioModel, CausalTransformer, ReverbLongModel, HybridWaveNetUNet

def save(name, data):
    """Save audio file"""
    # Ensure data in [-1, 1], then convert to int16
    data = np.clip(data, -1.0, 1.0)
    wavfile.write(name, 44100, (data * 32767).astype(np.int16))

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def test(args):
    # 1. Initialize model by type
    print(f"Initializing {args.model_type} Model...")
    
    # Initialize different models based on type
    if args.model_type == "hybrid":
        model_params = {
            'wavenet_channels': args.wavenet_channels,
            'transformer_d_model': args.transformer_d_model,
            'num_layers': args.transformer_layers,
            'transformer_nhead': args.transformer_nhead,
            'transformer_dim_feedforward': args.transformer_dim_feedforward,
            'dilation_depth': args.wavenet_dilation_depth,
            'num_repeat': args.wavenet_num_repeat,
            'kernel_size': args.wavenet_kernel_size,
        }
        model = HybridAudioModel(**model_params)
        data_filename = "Overdrive+Reverb+Delay.pickle"
    elif args.model_type == "transformer":
        model_params = {
            'd_model': args.transformer_d_model,
            'nhead': args.transformer_nhead,
            'num_layers': args.transformer_layers,
            'dim_feedforward': args.transformer_dim_feedforward,
            'max_seq_len': 4410,  # Default; handled in forward
        }
        model = CausalTransformer(**model_params)
        data_filename = "Overdrive+Reverb+Delay.pickle"
    elif args.model_type == "hybridplus":
        model_params = {
            'd_model': args.transformer_d_model,
            'nhead': args.transformer_nhead,
            'num_layers': args.transformer_layers,
            'dim_feedforward': args.transformer_dim_feedforward,
        }
        model = ReverbLongModel(**model_params)
        data_filename = "Overdrive+Reverb+Delay.pickle"
    elif args.model_type == "hybrid_wavenet_unet":
        model_params = {
        'wavenet_channels': args.wavenet_channels,
        'wavenet_dilation_depth': args.wavenet_dilation_depth,
        'wavenet_num_repeat': args.wavenet_num_repeat,
        'd_model': args.transformer_d_model,
        'nhead': args.transformer_nhead,
        'num_layers': args.transformer_layers,
        'dim_feedforward': args.transformer_dim_feedforward,
        'use_ir': getattr(args, 'use_ir', False),
        'ir_length': getattr(args, 'ir_length', 32768),
        'ir_wet': getattr(args, 'ir_wet', 0.25),
    }
        model = HybridWaveNetUNet(**model_params)
        data_filename = "Overdrive+Reverb+Delay.pickle"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Model parameters: {model_params}")
    print(f"Model created with {count_parameters(model)} parameters")
    
    # Allow overriding dataset filename via CLI
    if getattr(args, 'data_filename', None):
        data_filename = args.data_filename
    
    # 2. Load checkpoint
    print(f"\nLoading checkpoint from: {args.model}")
    try:
        checkpoint = torch.load(args.model, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model (checkpoint format)")
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.to(device)
    print(f"Model moved to {device}")

    # 3. Load test data
    data_path = os.path.join("preparedData", data_filename)
    print(f"\nLoading data from: {data_path}")
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    x_test = data["x_test"]
    y_test = data["y_test"]
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Get original segment length
    original_sample_length = x_test.shape[-1]
    print(f"Original sample length: {original_sample_length}")
    print(f"Single sample duration: {original_sample_length/44100:.3f} seconds")

    # 4. Data preprocessing
    # Ensure correct data dimensions (num_samples, 1, seq_len)
    if x_test.ndim == 2:
        x_test = x_test[:, np.newaxis, :]  # Add channel dimension
    if y_test.ndim == 2:
        y_test = y_test[:, np.newaxis, :]
        
    print(f"After dimension check - x_test shape: {x_test.shape}")
    print(f"After dimension check - y_test shape: {y_test.shape}")

    # 5. Generate predictions
    y_pred_list = []
    batch_size = args.batch_size
    num_samples = x_test.shape[0]
    
    print(f"\nStarting inference on {num_samples} samples...")
    
    # Progress bar
    progress_bar = tqdm(
        range(0, num_samples, batch_size),
        desc="Generating predictions",
        total=(num_samples + batch_size - 1) // batch_size
    )
    
    for i in progress_bar:
        # Current batch
        end_idx = min(i + batch_size, num_samples)
        batch_x = x_test[i:end_idx]
        
        # Convert to tensor
        batch_x_tensor = torch.from_numpy(batch_x).float().to(device)
        
        # Ensure correct dimensions
        if batch_x_tensor.dim() == 2:
            batch_x_tensor = batch_x_tensor.unsqueeze(1)
        elif batch_x_tensor.dim() == 1:
            batch_x_tensor = batch_x_tensor.unsqueeze(0).unsqueeze(0)
            
        # Generate predictions
        try:
            with torch.no_grad():
                generated = model(batch_x_tensor)
                y_pred_list.append(generated.cpu().numpy())
                
            progress_bar.set_postfix({
                'Batch': f'{i//batch_size + 1}/{(num_samples + batch_size - 1) // batch_size}',
                'Shape': f'{generated.shape}'
            })
        except Exception as e:
            print(f"Error during inference for batch {i//batch_size + 1}: {e}")
            continue
        finally:
            # Cleanup memory
            if 'batch_x_tensor' in locals():
                del batch_x_tensor
            if 'generated' in locals():
                del generated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 6. Merge results
    if not y_pred_list:
        print("\nNo predictions were generated!")
        return

    try:
        y_pred = np.concatenate(y_pred_list, axis=0)
        print(f"\nFinal concatenated y_pred shape: {y_pred.shape}")
    except ValueError as e:
        print(f"\nError concatenating predictions: {e}")
        return

    # 7. Validate results
    if y_pred.shape[0] != x_test.shape[0]:
        print(f"WARNING: Prediction batch size {y_pred.shape[0]} != input batch size {x_test.shape[0]}")
        # Adjust to match size
        if y_pred.shape[0] > x_test.shape[0]:
            y_pred = y_pred[:x_test.shape[0]]
        else:
            # Zero-pad
            padding = np.zeros((x_test.shape[0] - y_pred.shape[0], *y_pred.shape[1:]))
            y_pred = np.concatenate([y_pred, padding], axis=0)

    # 7.1 Overlap-add reconstruction based on metadata (preserve reverb tail)
    # Get hop and window size from dataset metadata
    hop_size = data.get("hop_size")
    sample_size = data.get("sample_size")
    if hop_size is None or sample_size is None:
        # Fallback to 50% overlap
        sample_size = y_pred.shape[-1]
        hop_size = sample_size // 2
        print(f"[INFO] hop/sample_size metadata missing. Using defaults: sample_size={sample_size}, hop_size={hop_size}")

    # Reconstruct using Hann window (overlap sequentially across batches)
    win = np.hanning(sample_size).astype(np.float32)
    # Accumulate weights to prevent energy imbalance
    total_length = hop_size * (y_pred.shape[0] - 1) + sample_size
    recon = np.zeros(total_length, dtype=np.float32)
    weight = np.zeros(total_length, dtype=np.float32)
    for idx in range(y_pred.shape[0]):
        start = idx * hop_size
        end = start + sample_size
        frame = y_pred[idx, 0]  # (sample_size,)
        recon[start:end] += frame * win
        weight[start:end] += win
    # Avoid divide-by-zero
    nz = weight > 1e-8
    recon[nz] /= weight[nz]

    # Reconstruct aligned input and ground truth in the same way for comparison
    recon_in = np.zeros(total_length, dtype=np.float32)
    weight_in = np.zeros(total_length, dtype=np.float32)
    for idx in range(x_test.shape[0]):
        start = idx * hop_size
        end = start + sample_size
        frame = x_test[idx, 0]
        recon_in[start:end] += frame * win
        weight_in[start:end] += win
    nz_in = weight_in > 1e-8
    recon_in[nz_in] /= weight_in[nz_in]

    recon_gt = np.zeros(total_length, dtype=np.float32)
    weight_gt = np.zeros(total_length, dtype=np.float32)
    for idx in range(y_test.shape[0]):
        start = idx * hop_size
        end = start + sample_size
        frame = y_test[idx, 0]
        recon_gt[start:end] += frame * win
        weight_gt[start:end] += win
    nz_gt = weight_gt > 1e-8
    recon_gt[nz_gt] /= weight_gt[nz_gt]

    # Compute metrics using reconstructed 1D signals
    y_pred_for_metrics = recon.reshape(1, 1, -1)
    y_test_for_metrics = recon_gt.reshape(1, 1, -1)

    # 8. Compute evaluation metrics
    print("\nComputing evaluation metrics...")
    # Ensure arrays have the same length for comparison
    min_len = min(len(y_pred), len(y_test))
    mse = np.mean((y_pred[:min_len] - y_test[:min_len])**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred[:min_len] - y_test[:min_len]))
    
    # Compute ESR (Error-to-Signal Ratio)
    def calculate_esr(pred, true):
        error_energy = np.sum((pred - true) ** 2, axis=-1)
        signal_energy = np.sum(true ** 2, axis=-1)
        esr = error_energy / (signal_energy + 1e-10)
        return np.mean(esr)
    
    esr = calculate_esr(y_pred[:min_len].squeeze(), y_test[:min_len].squeeze())
    
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"ESR: {esr:.6f}")

    # 9. Reconstruct full audio via OLA (avoid losing reverb tail)
    print("\nReconstructing full audio via overlap-add...")
    full_prediction = recon
    full_input = recon_in
    full_ground_truth = recon_gt
    print(f"Full prediction duration: {len(full_prediction)/44100:.2f} seconds")

    # 10. Save results
    model_dir = os.path.dirname(args.model)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\nSaving results to: {model_dir}")

    try:
        # Save full-length predicted audio
        save(os.path.join(model_dir, f"y_pred_full_{args.model_type}.wav"), full_prediction)
        print(f"  Saved y_pred_full_{args.model_type}.wav (full-length predicted audio)")
        
        # Save full-length input audio
        save(os.path.join(model_dir, f"x_test_full_{args.model_type}.wav"), full_input)
        print(f"  Saved x_test_full_{args.model_type}.wav (full-length input audio)")
        
        # Save full-length ground-truth audio
        save(os.path.join(model_dir, f"y_test_full_{args.model_type}.wav"), full_ground_truth)
        print(f"  Saved y_test_full_{args.model_type}.wav (full-length ground-truth audio)")
        
        print(f"\nTest completed successfully. Results saved in {model_dir}/")
        print("Files generated:")
        print(f"  - y_pred_full_{args.model_type}.wav: full-length predicted audio (concatenated all test samples)")
        print(f"  - x_test_full_{args.model_type}.wav: full-length input audio (concatenated all test samples)")
        print(f"  - y_test_full_{args.model_type}.wav: full-length ground-truth audio (concatenated all test samples)")

         # 11. If the model has a learnable IR, plot and save
        if hasattr(model, 'ir') and args.use_ir and model.ir is not None:
            print("\n[IR Mode] Plotting learned Impulse Response...")
            try:
                import matplotlib.pyplot as plt

                # Extract IR from model
                ir_data = model.ir.detach().cpu().numpy()
                if ir_data.ndim == 3:
                    ir_data = ir_data[0, 0, :]  # (1, 1, L) → (L,)
                elif ir_data.ndim == 2:
                    ir_data = ir_data[0, :]     # (1, L) → (L,)
                elif ir_data.ndim == 1:
                    pass  # already 1D
                else:
                    print(f"[WARNING] Unexpected IR shape: {ir_data.shape}, skipping plot.")
                    ir_data = None

                if ir_data is not None:
                    plt.figure(figsize=(12, 4))
                    plt.plot(ir_data, linewidth=0.8)
                    plt.title(f'Learned Impulse Response (Length: {len(ir_data)})\nModel: {args.model_type}')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Amplitude')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    ir_plot_path = os.path.join(model_dir, f"learned_ir_{args.model_type}.png")
                    plt.savefig(ir_plot_path, dpi=150)
                    plt.close()

                    print(f" Saved IR plot to: {ir_plot_path}")

                    # Optional: save raw IR data
                    ir_npy_path = os.path.join(model_dir, f"learned_ir_{args.model_type}.npy")
                    np.save(ir_npy_path, ir_data)
                    print(f" Saved IR data to: {ir_npy_path}")

            except Exception as e:
                print(f"[ERROR] Failed to plot IR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n[INFO] IR plotting skipped — model does not expose 'ir' or --use_ir not enabled.")

    except Exception as e:
        print(f"\nError saving files: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Audio Models")
    parser.add_argument("--model", default="models/Overdrive+Reverb/Overdrive+Reverb.ckpt", 
                       help="Path to the model checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to prepared dataset (.pickle)")
    parser.add_argument("--data_filename", type=str, default=None, help="Dataset filename under preparedData (e.g., DelayOnly.pickle)")
    parser.add_argument("--model_type", type=str, default="hybridplus", 
                       choices=["transformer", "hybrid", "hybridplus", "hybrid_wavenet_unet"],
                       help="Model architecture type")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    
    # Common Transformer parameters
    parser.add_argument("--transformer_d_model", type=int, default=64)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_nhead", type=int, default=4)
    parser.add_argument("--transformer_dim_feedforward", type=int, default=128)

    # U-Net parameters
    parser.add_argument("--unet_base_channels", type=int, default=32)
    
    # Hybrid model specific parameters
    parser.add_argument("--wavenet_channels", type=int, default=16)
    parser.add_argument("--wavenet_dilation_depth", type=int, default=8)
    parser.add_argument("--wavenet_num_repeat", type=int, default=2)
    parser.add_argument("--wavenet_kernel_size", type=int, default=3)

    parser.add_argument("--use_ir", action="store_true", help="Enable learnable IR convolver branch")
    parser.add_argument("--ir_length", type=int, default=44100, help="Length of learnable IR in samples (default: 32768)")
    parser.add_argument("--ir_wet", type=float, default=0.25, help="Initial wet ratio for IR branch (0-1, default: 0.25)")
    
    # Testing parameters
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    
    print("Starting Audio Model Test Script...")
    print(f"Using model: {args.model}")
    print(f"Model type: {args.model_type}")
    
    if args.model_type == "hybrid":
        print(f"Hybrid Model parameters:")
        print(f"  WaveNet: channels={args.wavenet_channels}, depth={args.wavenet_dilation_depth}, repeat={args.wavenet_num_repeat}")
        print(f"  Transformer: d_model={args.transformer_d_model}, layers={args.transformer_layers}, nhead={args.transformer_nhead}")
    elif args.model_type == "transformer":
        print(f"Causal Transformer parameters:")
        print(f"  d_model={args.transformer_d_model}, layers={args.transformer_layers}, nhead={args.transformer_nhead}")
    elif args.model_type == "hybridplus":
        print(f"Reverb Long Model parameters:")
        print(f"  d_model={args.transformer_d_model}, layers={args.transformer_layers}, nhead={args.transformer_nhead}")
    elif args.model_type == "hybrid_wavenet_unet":
        print(f"Hybrid WaveNet U-Net parameters:")
        print(f"  WaveNet: channels={args.wavenet_channels}, depth={args.wavenet_dilation_depth}")
        print(f"  U-Net: base_channels={args.unet_base_channels}")
        print(f"  Transformer: d_model={args.transformer_d_model}, layers={args.transformer_layers}, nhead={args.transformer_nhead}")
    
    print(f"Batch size: {args.batch_size}")
    
    test(args)
    print("Script finished.")