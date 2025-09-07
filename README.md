### PedalDiffusion — Working Commands (Windows PowerShell)

#### 1) Prepare datasets
- DelayOnly
```powershell
python prepare.py data/dataset_clean_New.wav data/dataset_delay_New.wav --out_dir preparedData --out_name DelayOnly.pickle
```

- Overdrive+Reverb+Delay
```powershell
python prepare.py data/dataset_clean_New.wav data/dataset_overdrive_reverb_delay_New.wav --out_dir preparedData --out_name Overdrive+Reverb+Delay.pickle
```

#### 2) Train (Hybrid WaveNet U-Net with IR branch)
```powershell
python trainHybrid.py --model models/Delay/HybridWUNet.ckpt --model_type hybrid_wavenet_unet --data preparedData/Overdrive+Reverb+Delay.pickle --use_ir --ir_length 44100 --ir_wet 0.25
```

Optional: force CPU
```powershell
--cpu
```

#### 3) Test (with IR branch enabled)
```powershell
python testHybrid.py --model models/Overdrive+Reverb+Delay/Overdrive+Reverb+Delay.ckpt --model_type hybrid_wavenet_unet --data_filename Overdrive+Reverb+Delay.pickle --use_ir
```

Notes:
- You can also pass a full dataset path via `--data` instead of `--data_filename`.

#### 4) Plot results
```powershell
python plot.py --model models/Overdrive+Reverb+Delay/Overdrive+Reverb+Delay.ckpt --model_type hybrid_wavenet_unet
```


