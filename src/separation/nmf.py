# # src/separation/nmf_spectral.py

# import os
# import sys
# # —— 把项目的 src/ 根目录加入 Python 模块搜索路径 ——
# _script_dir = os.path.dirname(__file__)
# _src_dir = os.path.abspath(os.path.join(_script_dir, '..'))
# if _src_dir not in sys.path:
#     sys.path.insert(0, _src_dir)

# import numpy as np
# import matplotlib.pyplot as plt
# import soundfile as sf
# import librosa
# from sklearn.decomposition import NMF
# from mir_eval.separation import bss_eval_sources
# from scipy.signal import correlate
# from stft_utils import stft


# def main():
#     # 1) 读取两段干净语音
#     s1, sr1 = librosa.load(librosa.example('libri1'), duration=5.0)
#     s2, sr2 = librosa.load(librosa.example('libri2'), duration=5.0)
#     assert sr1 == sr2, "采样率必须一致"
#     L = min(len(s1), len(s2))
#     s1, s2 = s1[:L], s2[:L]
#     sources = np.vstack([s1, s2])  # shape = (2, L)

#     # 2) 合成单通道混合
#     A = np.array([[1.0, 0.5],
#                   [0.5, 1.0]])
#     X = A @ sources                # (2, L)
#     mix = X.mean(axis=0)           # (L,)

#     # 3) STFT & 功率谱
#     n_fft = 1024
#     hop_length = 512
#     D = stft(mix, n_fft=n_fft, hop_length=hop_length)  # (F, T)
#     power = np.abs(D) ** 2

#     # 4) NMF 分解（直接在功率谱上，用 Itakura-Saito 散度）
#     nmf = NMF(
#         n_components=2,
#         init='nndsvda',
#         solver='mu',
#         beta_loss='itakura-saito',
#         max_iter=500,
#         tol=1e-4,
#         random_state=0,
#         alpha_W=0.0,
#         alpha_H=0.0
#     )
#     W = nmf.fit_transform(power)    # (F, 2)
#     H = nmf.components_             # (2, T)

#     # 5) 可选：平滑 H
#     from scipy.ndimage import gaussian_filter1d
#     H = gaussian_filter1d(H, sigma=0.5, axis=1)

#     # 6) 构造掩码（幅度谱）
#     Vs = []
#     V = np.zeros_like(power)
#     for i in range(2):
#         V_i = np.outer(W[:, i], H[i, :])
#         Vs.append(V_i)
#         V += V_i
#     V += 1e-8
#     masks = [V_i / V for V_i in Vs]

#     # 7) 重构时域信号，并转换成 ndarray
#     S_est_list = []
#     for mask in masks:
#         S_tf = mask * np.abs(D) * np.exp(1j * np.angle(D))
#         x_est = librosa.istft(S_tf, hop_length=hop_length, length=L)
#         S_est_list.append(x_est)
#     S_est = np.vstack(S_est_list)    # shape = (2, L)

#     # 8) 评估
#     sdr, sir, sar, _ = bss_eval_sources(sources, S_est)
#     print("改进后 NMF 分离评估指标：")
#     for idx, (d, i_, a) in enumerate(zip(sdr, sir, sar), 1):
#         print(f"  源 {idx}: SDR={d:.2f} dB, SIR={i_:.2f} dB, SAR={a:.2f} dB")

#     # 9) 可视化
#     # 9.1) 时域波形
#     plt.figure(figsize=(12, 6))
#     labels = ['Source1', 'Source2', 'Mixture', 'Est1', 'Est2']
#     sigs = [s1, s2, mix, S_est[0], S_est[1]]
#     for i, (sig, lbl) in enumerate(zip(sigs, labels), 1):
#         plt.subplot(3, 2, i)
#         plt.plot(sig)
#         plt.title(lbl)
#         plt.tight_layout()
#     plt.show()

#     # 9.2) 频谱图
#     plt.figure(figsize=(10, 8))
#     items = [(mix, 'Mixture'), (S_est[0], 'Est Src1'), (S_est[1], 'Est Src2')]
#     for i, (sig, title) in enumerate(items, 1):
#         plt.subplot(3, 1, i)
#         D_sig = stft(sig, n_fft=n_fft, hop_length=hop_length)
#         plt.imshow(20 * np.log10(np.abs(D_sig) + 1e-6), origin='lower', aspect='auto')
#         plt.title(title)
#         plt.xlabel('Frame'); plt.ylabel('Freq bin')
#     plt.tight_layout()
#     plt.show()

#     # 9.3) 波形叠加
#     t = np.arange(L) / sr1
#     for idx, (orig, est) in enumerate(zip([s1, s2], S_est), 1):
#         plt.figure(figsize=(6, 3))
#         plt.plot(t, orig, label=f'Orig Src{idx}', alpha=0.7)
#         plt.plot(t, est, '--', label=f'Est Src{idx}')
#         plt.title(f'Overlay Source {idx}')
#         plt.xlabel('Time [s]'); plt.legend(); plt.tight_layout()
#         plt.show()

#     # 9.4) 互相关
#     for idx, (orig, est) in enumerate(zip([s1, s2], S_est), 1):
#         corr = correlate(est, orig, mode='full')
#         lags = np.arange(-L+1, L)
#         plt.figure(figsize=(6, 3))
#         plt.plot(lags, corr)
#         plt.title(f'Cross-correlation Source {idx}')
#         plt.xlabel('Lag [samples]'); plt.tight_layout()
#         plt.show()

#     # 10) 保存结果
#     out_dir = os.path.abspath(os.path.join(_src_dir, 'results', 'nmf_separated'))
#     os.makedirs(out_dir, exist_ok=True)
#     sf.write(os.path.join(out_dir, 'NMF_src1.wav'), S_est[0], sr1)
#     sf.write(os.path.join(out_dir, 'NMF_src2.wav'), S_est[1], sr1)
#     print(f"分离结果已保存到 {out_dir}")


# if __name__ == '__main__':
#     main()
# src/separation/nmf_spectral_improved.py
"""
改进版：使用监督式 NMF 并集成评估与可视化
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from sklearn.decomposition import NMF
from scipy.signal import correlate
from mir_eval.separation import bss_eval_sources
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
_script_dir = os.path.dirname(__file__)
_src_dir = os.path.abspath(os.path.join(_script_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from stft_utils import stft, istft
# import numpy as np
# import librosa

# def stft(x, n_fft=1024, hop_length=None):
#     """
#     Compute the STFT of a 1D signal.
#     """
#     hop_length = hop_length or n_fft // 2
#     return librosa.stft(x, n_fft=n_fft, hop_length=hop_length)


# def istft(X, hop_length=None):
#     """
#     Inverse STFT back to time-domain signal.
#     """
#     hop_length = hop_length or (X.shape[0] * 2)
#     return librosa.istft(X, hop_length=hop_length)
# 确保模块路径
_script_dir = os.path.dirname(__file__)
_src_dir = os.path.abspath(os.path.join(_script_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 全局参数
n_fft = 1024
hop_length = 512
mix_duration = 5.0
max_iter_update = 500


def compute_spectra(x):
    D = stft(x, n_fft=n_fft, hop_length=hop_length)
    return D, np.abs(D) ** 2


def supervised_nmf(s1, s2, mix, L):
    # 1) 训练字典 W1, W2
    D1, p1 = compute_spectra(s1)
    D2, p2 = compute_spectra(s2)
    K = 20
    nmf1 = NMF(n_components=K, init='nndsvda', solver='mu',
               beta_loss='kullback-leibler', alpha_H=0.1, max_iter=500)
    W1 = nmf1.fit_transform(p1)
    nmf2 = NMF(n_components=K, init='nndsvda', solver='mu',
               beta_loss='kullback-leibler', alpha_H=0.1, max_iter=500)
    W2 = nmf2.fit_transform(p2)
    W = np.concatenate([W1, W2], axis=1)

    # 2) 手动迭代更新 H
    Dmix, pmix = compute_spectra(mix)
    R, T = W.shape[1], pmix.shape[1]
    H = np.random.rand(R, T)
    for _ in range(max_iter_update):
        H *= (W.T @ (pmix / (W @ H + 1e-8))) / (W.T.sum(axis=1)[:, None] + 0.1)
    H = gaussian_filter1d(H, sigma=1.0, axis=1)

    # 3) 构造掩码并重建
    Vs = [np.outer(W[:, i], H[i]) for i in range(R)]
    Vtot = sum(Vs) + 1e-8
    masks = [V / Vtot for V in Vs]
    phase = np.exp(1j * np.angle(Dmix))

    est1 = sum(
        istft(masks[i] * np.abs(Dmix) * phase,
              hop_length=hop_length, length=L)
        for i in range(K)
    )
    est2 = sum(
        istft(masks[i+K] * np.abs(Dmix) * phase,
              hop_length=hop_length, length=L)
        for i in range(K)
    )
    return np.vstack([est1, est2])


def main():
    # 1) 读取干净语音并对齐
    s1, sr1 = librosa.load(librosa.example('libri1'), duration=mix_duration)
    s2, sr2 = librosa.load(librosa.example('libri2'), duration=mix_duration)
    assert sr1 == sr2, "采样率必须一致"
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]

    # 2) 合成混合
    sources = np.vstack([s1, s2])
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    mix = (A @ sources).mean(axis=0)

    # 3) 分离
    S_est = supervised_nmf(s1, s2, mix, L)

    # 4) 评估
    sdr, sir, sar, _ = bss_eval_sources(np.vstack([s1, s2]), S_est)
    print("改进后监督式 NMF 分离评估：")
    for idx, (d, i_, a) in enumerate(zip(sdr, sir, sar), 1):
        print(f"  源 {idx}: SDR={d:.2f} dB, SIR={i_:.2f} dB, SAR={a:.2f} dB")

    # 5) 可视化
    t = np.arange(L) / sr1
    plt.figure(figsize=(12, 6))
    labels = ['Src1 Orig', 'Src2 Orig', 'Mixture', 'Est Src1', 'Est Src2']
    signals = [s1, s2, mix, S_est[0], S_est[1]]
    for i, (sig, lbl) in enumerate(zip(signals, labels), 1):
        plt.subplot(3, 2, i)
        plt.plot(sig)
        plt.title(lbl)
        plt.tight_layout()
    plt.show()

    # 5.2 频谱图
    plt.figure(figsize=(10, 8))
    items = [(mix, 'Mixture'), (S_est[0], 'Est Src1'), (S_est[1], 'Est Src2')]
    for i, (sig, title) in enumerate(items, 1):
        plt.subplot(3, 1, i)
        D_sig, _ = compute_spectra(sig)
        plt.imshow(20 * np.log10(np.abs(D_sig) + 1e-6), origin='lower', aspect='auto')
        plt.title(title)
        plt.xlabel('Frame'); plt.ylabel('Freq bin')
    plt.tight_layout()
    plt.show()

    # 5.3 波形叠加
    for idx, (orig, est) in enumerate(zip([s1, s2], S_est), 1):
        plt.figure(figsize=(6, 3))
        plt.plot(t, orig, label='Orig', alpha=0.7)
        plt.plot(t, est, '--', label='Est')
        plt.title(f'Overlay Source {idx}')
        plt.xlabel('Time [s]'); plt.legend(); plt.tight_layout(); plt.show()

    # 5.4 互相关
    for idx, (orig, est) in enumerate(zip([s1, s2], S_est), 1):
        corr = correlate(est, orig, mode='full')
        lags = np.arange(-L+1, L)
        plt.figure(figsize=(6, 3))
        plt.plot(lags, corr)
        plt.title(f'Cross-corr Src {idx}')
        plt.xlabel('Lag [samples]'); plt.tight_layout(); plt.show()

    # 6) 保存结果
    out_dir = os.path.join(_src_dir, 'results', 'nmf_separated_sup')
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, 'src1.wav'), S_est[0], sr1)
    sf.write(os.path.join(out_dir, 'src2.wav'), S_est[1], sr1)
    print(f"结果已保存到 {out_dir}")

if __name__ == '__main__':
    main()
