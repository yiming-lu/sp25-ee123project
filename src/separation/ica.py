# src/separation/ica.py
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from sklearn.decomposition import FastICA
from mir_eval.separation import bss_eval_sources
import os
from scipy.signal import correlate
import sys

# —— 把项目的 src/ 根目录加入 Python 模块搜索路径 —— 
# 这样才能 import stft_utils
_script_dir = os.path.dirname(__file__)
_src_dir = os.path.abspath(os.path.join(_script_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from stft_utils import stft, istft

def main():
    # 1) 读取两段干净语音
    s1, sr1 = librosa.load(librosa.example('libri1'), duration=5.0)
    s2, sr2 = librosa.load(librosa.example('libri2'), duration=5.0)
    assert sr1 == sr2, "采样率必须一致"
    # 对齐长度
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]
    sources = np.vstack([s1, s2])   # shape = (2, L)

    # 2) 合成混合信号 X = A @ sources
    A = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
    X = A @ sources                # shape = (2, L)

    # 3) FastICA 分离
    ica = FastICA(n_components=2, random_state=0)
    S_est = ica.fit_transform(X.T).T  # shape = (2, L)

    # 4) 评估
    #  mir_eval 要求 shape = (nsrc, nsampl)
    sdr, sir, sar, perm = bss_eval_sources(sources, S_est)
    print("ICA 分离评估指标：")
    for i,(sdr_i, sir_i, sar_i) in enumerate(zip(sdr, sir, sar), 1):
        print(f"  源 {i}: SDR={sdr_i:.2f} dB, SIR={sir_i:.2f} dB, SAR={sar_i:.2f} dB")

    # 5.1) 时域波形对比
    plt.figure(figsize=(10, 8))
    titles = ['Source 1','Source 2','Mix Ch1','Mix Ch2','Est Source1','Est Source2']
    signals = [s1, s2, X[0], X[1], S_est[0], S_est[1]]
    for idx, sig in enumerate(signals, 1):
        plt.subplot(3, 2, idx)
        plt.plot(sig)
        plt.title(titles[idx-1])
        plt.tight_layout()
    plt.show()

    # 5.2) 频谱图 (Spectrograms)
    plt.figure(figsize=(12, 8))
    items = [(X[0], 'Mix Ch1'), (X[1], 'Mix Ch2'),
             (S_est[0], 'Est Src1'), (S_est[1], 'Est Src2')]
    for idx, (sig, title) in enumerate(items, 1):
        plt.subplot(4, 1, idx)
        D = stft(sig, n_fft=1024, hop_length=512)
        plt.imshow(20 * np.log10(np.abs(D) + 1e-6),
                   origin='lower', aspect='auto')
        plt.title(title)
        plt.ylabel('Freq bin')
        plt.xlabel('Frame')
    plt.tight_layout()
    plt.show()

    # 5.3) 波形叠加图 (Original vs Estimated for Src1)
    plt.figure(figsize=(6, 4))
    t = np.arange(L) / sr1
    plt.plot(t, s1, label='Original Src1', alpha=0.7)
    plt.plot(t, S_est[0], '--', label='Estimated Src1')
    plt.title('Waveform Overlay: Source 1')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    t = np.arange(L) / sr2
    plt.plot(t, s2, label='Original Src2', alpha=0.7)
    plt.plot(t, S_est[1], '--', label='Estimated Src2')
    plt.title('Waveform Overlay: Source 2')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5.4) 互相关 (Cross-correlation for Src1)
    corr = correlate(S_est[0], s1, mode='full')
    lags = np.arange(-L+1, L)
    plt.figure(figsize=(6, 4))
    plt.plot(lags, corr)
    plt.title('Cross-correlation: Estimated vs Original (Src1)')
    plt.xlabel('Lag [samples]')
    plt.tight_layout()
    plt.show()

    corr = correlate(S_est[1], s2, mode='full')
    lags = np.arange(-L+1, L)
    plt.figure(figsize=(6, 4))
    plt.plot(lags, corr)
    plt.title('Cross-correlation: Estimated vs Original (Src2)')
    plt.xlabel('Lag [samples]')
    plt.tight_layout()
    plt.show()


    # 6) 保存分离结果
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '..', 'results', 'separated'))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, 'ICA_src1.wav'), S_est[0], sr1)
    sf.write(os.path.join(out_dir, 'ICA_src2.wav'), S_est[1], sr1)
    print(f"分离结果已保存到 {out_dir}")

if __name__ == '__main__':
    main()
