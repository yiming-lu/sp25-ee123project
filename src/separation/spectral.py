import numpy as np
import librosa
from scipy.signal import stft, istft
from sklearn.cluster import KMeans
import soundfile as sf
from mir_eval.separation import bss_eval_sources

# 参数配置
n_fft = 2048
hop_length = 512
win_length = n_fft

def duet_separation(mix, sr):
    """DUET语音分离主函数"""
    assert mix.ndim == 2 and mix.shape[0] == 2, "需要立体声输入"

    # 左右声道STFT
    _, _, Z_l = stft(mix[0], fs=sr, nperseg=win_length, noverlap=win_length-hop_length)
    _, _, Z_r = stft(mix[1], fs=sr, nperseg=win_length, noverlap=win_length-hop_length)

    # 计算幅度和相位
    mag_l, mag_r = np.abs(Z_l), np.abs(Z_r)
    phase_l, phase_r = np.angle(Z_l), np.angle(Z_r)

    # 计算特征
    a = mag_r / (mag_l + 1e-8)
    delta = phase_r - phase_l
    features = np.stack([a.flatten(),
                        np.cos(delta).flatten(),
                        np.sin(delta).flatten()], axis=1)

    # K-means聚类
    kmeans = KMeans(n_clusters=2, n_init=10).fit(features)
    labels = kmeans.labels_.reshape(a.shape)

    # 生成掩码并分离
    masks = [(labels == i) for i in range(2)]
    estimates = [istft(Z_l * mask, fs=sr, nperseg=win_length, 
                      noverlap=win_length-hop_length)[1] for mask in masks]

    # 统一长度并保存
    min_len = min(len(e) for e in estimates)
    return [e[:min_len] for e in estimates]

def evaluate_sources(references, estimates, sr):
    """评估分离质量"""
    max_len = max(max(len(r) for r in references), max(len(e) for e in estimates))
    
    # 对齐长度（修复关键点）
    references = [librosa.util.fix_length(r, size=max_len) for r in references]  # 正确使用size=
    estimates = [librosa.util.fix_length(e, size=max_len) for e in estimates]    # 正确使用size=
    
    # 计算指标
    return bss_eval_sources(np.array(references), np.array(estimates))

if __name__ == "__main__":
    # 1) 生成混合信号
    # 加载示例音频（使用Librosa内置音频）
    s1, sr = librosa.load(librosa.example('libri1'), duration=5.0)
    s2, sr = librosa.load(librosa.example('libri2'), duration=5.0)
    
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]
    sources = np.vstack([s1, s2])  # shape = (2, L)

    # 创建立体声混合矩阵
    A = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
    X = A @ sources  # 立体声混合信号 shape (2, L)

    # 2) 进行分离
    estimates = duet_separation(X, sr)
    
    # 保存结果
    sf.write("estimated1.wav", estimates[0], sr)
    sf.write("estimated2.wav", estimates[1], sr)

    # 3) 评估结果
    try:
        sdr, sir, sar, perm = evaluate_sources([s1, s2], estimates, sr)
        print(f"SDR: {sdr[perm[0]]:.2f} dB, {sdr[perm[1]]:.2f} dB")
        print(f"SIR: {sir[perm[0]]:.2f} dB, {sir[perm[1]]:.2f} dB")
        print(f"SAR: {sar[perm[0]]:.2f} dB, {sar[perm[1]]:.2f} dB")
    except Exception as e:
        print(f"评估失败: {str(e)}")