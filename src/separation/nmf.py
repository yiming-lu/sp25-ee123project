# src/separation/nmf_spectral.py

import os
import sys
# —— 把项目的 src/ 根目录加入 Python 模块搜索路径 ——
_script_dir = os.path.dirname(__file__)
_src_dir = os.path.abspath(os.path.join(_script_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from sklearn.decomposition import NMF
from mir_eval.separation import bss_eval_sources
from scipy.signal import correlate
from stft_utils import stft

def main():
    # 1) 读取两段干净语音
    s1, sr1 = librosa.load(librosa.example('libri1'), duration=5.0)
    s2, sr2 = librosa.load(librosa.example('libri2'), duration=5.0)
    assert sr1 == sr2, "采样率必须一致"
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]
    sources = np.vstack([s1, s2])  # shape = (2, L)

    # 2) 合成单通道混合
    A = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
    X = A @ sources                # (2, L)
    mix = X.mean(axis=0)           # (L,)

    # 3) STFT & 功率谱
    n_fft = 1024
    hop_length = 512
    D = stft(mix, n_fft=n_fft, hop_length=hop_length)  # (F, T)
    power = np.abs(D)**2
    phase = np.exp(1j * np.angle(D))

    # 4) 改进的 NMF 分解
    n_components = 2
    nmf = NMF(
        n_components=n_components,
        init='nndsvda',
        solver='mu',
        beta_loss='itakura-saito',
        max_iter=500,
        random_state=0,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.5
    )
    W = nmf.fit_transform(power)   # (F, 2)
    H = nmf.components_            # (2, T)

    # 5) Wiener 掩码重构
    Vs = []
    V = np.zeros_like(power)
    for i in range(n_components):
        V_i = np.outer(W[:, i], H[i, :])
        Vs.append(V_i)
        V += V_i
    V += 1e-8

    S_est = []
    for V_i in Vs:
        mask = V_i / V
        S_tf = mask * D
        x_i = librosa.istft(S_tf, hop_length=hop_length, length=L)
        S_est.append(x_i)
    S_est = np.vstack(S_est)  # (2, L)

    # 6) 评估
    sdr, sir, sar, _ = bss_eval_sources(sources, S_est)
    print("改进后 NMF 分离评估指标：")
    for idx, (d, i_, a) in enumerate(zip(sdr, sir, sar), 1):
        print(f"  源 {idx}: SDR={d:.2f} dB, SIR={i_:.2f} dB, SAR={a:.2f} dB")

    # 7.1) 时域波形图
    plt.figure(figsize=(12, 6))
    labels = ['Source1','Source2','Mixture','Est1','Est2']
    sigs    = [s1, s2, mix, S_est[0], S_est[1]]
    for i, (sig, lbl) in enumerate(zip(sigs, labels), 1):
        plt.subplot(3, 2, i)
        plt.plot(sig)
        plt.title(lbl)
        plt.tight_layout()
    plt.show()

    # 7.2) 频谱图
    plt.figure(figsize=(10, 8))
    items = [(mix,    'Mixture'),
             (S_est[0],'Est Src1'),
             (S_est[1],'Est Src2')]
    for i, (sig, title) in enumerate(items, 1):
        plt.subplot(3, 1, i)
        D_sig = stft(sig, n_fft=n_fft, hop_length=hop_length)
        plt.imshow(20*np.log10(np.abs(D_sig)+1e-6),
                   origin='lower', aspect='auto')
        plt.title(title)
        plt.xlabel('Frame'); plt.ylabel('Freq bin')
    plt.tight_layout()
    plt.show()

    # 7.3) 波形叠加图（两个源）
    t = np.arange(L) / sr1
    for idx, (orig, est) in enumerate(zip([s1, s2], S_est), 1):
        plt.figure(figsize=(6, 3))
        plt.plot(t, orig, label=f'Orig Src{idx}', alpha=0.7)
        plt.plot(t, est, '--',   label=f'Est Src{idx}')
        plt.title(f'Overlay Source {idx}')
        plt.xlabel('Time [s]'); plt.legend(); plt.tight_layout()
        plt.show()

    # 7.4) 互相关图（两个源）
    for idx, (orig, est) in enumerate(zip([s1, s2], S_est), 1):
        corr = correlate(est, orig, mode='full')
        lags = np.arange(-L+1, L)
        plt.figure(figsize=(6, 3))
        plt.plot(lags, corr)
        plt.title(f'Cross-correlation Source {idx}')
        plt.xlabel('Lag [samples]'); plt.tight_layout()
        plt.show()

    # 8) 保存结果
    out_dir = os.path.abspath(os.path.join(_src_dir, 'results', 'nmf_separated'))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, 'NMF_src1.wav'), S_est[0], sr1)
    sf.write(os.path.join(out_dir, 'NMF_src2.wav'), S_est[1], sr1)
    print(f"分离结果已保存到 {out_dir}")

if __name__ == '__main__':
    main()
