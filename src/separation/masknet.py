# src/separation/ica_kmeans.py

import os
import sys
# 确保能 import stft_utils
_script_dir = os.path.dirname(__file__)
_src_dir = os.path.abspath(os.path.join(_script_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from sklearn.cluster import KMeans
from mir_eval.separation import bss_eval_sources
from scipy.signal import correlate           # 新增
from stft_utils import stft                  # 原有

def main():
    # 1) 读取两段干净语音
    s1, sr1 = librosa.load(librosa.example('libri1'), duration=5.0)
    s2, sr2 = librosa.load(librosa.example('libri2'), duration=5.0)
    assert sr1 == sr2, "采样率必须一致"
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]
    sources = np.vstack([s1, s2])  # shape = (2, L)

    # 2) 合成混合信号并生成单通道混合
    A = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
    X = A @ sources               # (2, L)
    mix = X.mean(axis=0)          # (L,)

    # 3) STFT / 功率谱
    n_fft = 1024
    hop_length = 512
    D = stft(mix, n_fft=n_fft, hop_length=hop_length)  # (F, T)
    power = np.abs(D) ** 2
    phase = np.exp(1j * np.angle(D))
    n_freq, n_frames = power.shape

    # 4) KMeans 聚类
    feats = np.log1p(power)
    feats = (feats - feats.mean()) / feats.std()
    X_feats = feats.T  # (T, F)
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(X_feats)  # (T,)

    # 5) 构造 Wiener 掩码并重构
    Vs = []
    for c in [0,1]:
        mask_frame = (labels == c).astype(float)  # (T,)
        Vc = power * mask_frame[np.newaxis, :]    # (F, T)
        Vs.append(Vc)
    V_sum = np.sum(Vs, axis=0) + 1e-8

    S_est = []
    for Vc in Vs:
        mask = Vc / V_sum
        S_tf = mask * phase * np.sqrt(power)      # 保留相位，恢复幅度
        x_est = librosa.istft(S_tf, hop_length=hop_length, length=L)
        S_est.append(x_est)
    S_est = np.vstack(S_est)  # (2, L)

    # 6) 评估
    sdr, sir, sar, _ = bss_eval_sources(sources, S_est)
    print("KMeans 分离评估指标：")
    for idx, (d, i_, a) in enumerate(zip(sdr, sir, sar), 1):
        print(f"  源 {idx}: SDR={d:.2f} dB, SIR={i_:.2f} dB, SAR={a:.2f} dB")

    # 7) 可视化

    # 7.1 时域波形 （原始源 / 混合 / 分离）
    plt.figure(figsize=(12, 6))
    names = ['Source1','Source2','Mixture','Est1','Est2']
    signals = [s1, s2, mix, S_est[0], S_est[1]]
    for i, (sig, name) in enumerate(zip(signals, names), 1):
        plt.subplot(3, 2, i)
        plt.plot(sig)
        plt.title(name)
        plt.tight_layout()
    plt.show()

    # 7.2 频谱图 （混合 + 分离后的两个源）
    plt.figure(figsize=(10, 8))
    items = [(mix, 'Mixture'),
             (S_est[0], 'Est Source1'),
             (S_est[1], 'Est Source2')]
    for i, (sig, title) in enumerate(items, 1):
        plt.subplot(3, 1, i)
        D_sig = stft(sig, n_fft=n_fft, hop_length=hop_length)
        plt.imshow(20*np.log10(np.abs(D_sig)+1e-6), origin='lower', aspect='auto')
        plt.title(title)
        plt.ylabel('Freq bin'); plt.xlabel('Frame')
    plt.tight_layout()
    plt.show()

    # 7.3 波形叠加图：源1 & 源2 各自对比
    t = np.arange(L) / sr1
    for idx, (orig, est) in enumerate(zip([s1, s2], S_est), 1):
        plt.figure(figsize=(6,3))
        plt.plot(t, orig,     label=f'Orig Src{idx}', alpha=0.7)
        plt.plot(t, est, '--', label=f'Est Src{idx}')
        plt.title(f'Overlay Source {idx}')
        plt.xlabel('Time [s]'); plt.legend(); plt.tight_layout()
        plt.show()

    # 7.4 互相关图：源1 & 源2 各自对比
    for idx, (orig, est) in enumerate(zip([s1, s2], S_est), 1):
        corr = correlate(est, orig, mode='full')
        lags = np.arange(-L+1, L)
        plt.figure(figsize=(6,3))
        plt.plot(lags, corr)
        plt.title(f'Cross-correlation Source {idx}')
        plt.xlabel('Lag [samples]'); plt.tight_layout()
        plt.show()

    # 8) 保存结果
    out_dir = os.path.abspath(os.path.join(_src_dir, 'results', 'ml_separated'))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, 'KMeans_src1.wav'), S_est[0], sr1)
    sf.write(os.path.join(out_dir, 'KMeans_src2.wav'), S_est[1], sr1)
    print(f"分离结果已保存到 {out_dir}")

if __name__ == '__main__':
    main()
