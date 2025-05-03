# src/run_experiment.py

import os
import numpy as np
import soundfile as sf
import librosa
import torch

from stft_utils import stft, istft
from separation.ica import separate_ica
from separation.nmf import separate_nmf
from separation.spectral import separate_spectral
from separation.masknet import MaskNet, separate_masknet

# 1. 参数 & 目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
out_dir = os.path.join(project_root, 'results', 'separated')
os.makedirs(out_dir, exist_ok=True)

# 2. 加载两段干净语音
s1, sr1 = librosa.load(librosa.example('libri1'), duration=5.0)
s2, sr2 = librosa.load(librosa.example('libri2'), duration=5.0)
assert sr1 == sr2, "采样率不一致"
# 对齐长度
L = min(len(s1), len(s2))
s1, s2 = s1[:L], s2[:L]
# 打印形状确认
print("Loaded sources shape:", np.vstack([s1, s2]).shape)  # (2, L)

# 3. 合成双声道混合——左右声道分别用不同权重叠加
A = np.array([[0.6, 0.4],
              [0.4, 0.6]])        # 线性混合矩阵
mix1 = A[0,0]*s1 + A[0,1]*s2
mix2 = A[1,0]*s1 + A[1,1]*s2
mix = np.stack([mix1, mix2], axis=1)  # (L, 2)
sf.write(os.path.join(out_dir, 'stereo_mix.wav'), mix, sr1)
print("Stereo mix shape:", mix.shape)

# 4. 初始化 MaskNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MaskNet().to(device)
model.eval()

# 5. 调用四种算法分离
# --- ICA (时域 FastICA) ---
ica_est = separate_ica(mix, sr1)    # 返回 [s1_est, s2_est]
for i, sig in enumerate(ica_est, 1):
    sf.write(os.path.join(out_dir, f"ICA_src{i}.wav"), sig, sr1)

# --- NMF (频域 NMF) ---
nmf_est = separate_nmf(mix1 + mix2)    # 传入单通道混合幅度谱的和
for i, sig in enumerate(nmf_est, 1):
    sf.write(os.path.join(out_dir, f"NMF_src{i}.wav"), sig, sr1)

# --- Spectral Clustering (频域) ---
spec_est = separate_spectral(mix1 + mix2)
for i, sig in enumerate(spec_est, 1):
    sf.write(os.path.join(out_dir, f"SPEC_src{i}.wav"), sig, sr1)

# --- MaskNet (深度掩蔽) ---
ml_est = separate_masknet(mix1 + mix2, model)
for i, sig in enumerate(ml_est, 1):
    sf.write(os.path.join(out_dir, f"ML_src{i}.wav"), sig, sr1)

print("All separations done. Outputs in", out_dir)
