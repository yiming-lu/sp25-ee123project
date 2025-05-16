import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import stft, istft, correlate
from sklearn.cluster import KMeans
from mir_eval.separation import bss_eval_sources
from skimage.metrics import structural_similarity as ssim  # SSIM
import warnings


n_fft = 2048
hop_length = 512
win_length = n_fft


def duet_separation(mix, sr):
    assert mix.ndim == 2 and mix.shape[0] == 2

    # STFT
    _, _, Z_l = stft(mix[0], fs=sr, nperseg=win_length, noverlap=win_length - hop_length)
    _, _, Z_r = stft(mix[1], fs=sr, nperseg=win_length, noverlap=win_length - hop_length)

 
    mag_l, mag_r = np.abs(Z_l), np.abs(Z_r)
    phase_l, phase_r = np.angle(Z_l), np.angle(Z_r)
    a = mag_r / (mag_l + 1e-8)
    delta = phase_r - phase_l
    features = np.stack([a.flatten(), np.cos(delta).flatten(), np.sin(delta).flatten()], axis=1)

    
    kmeans = KMeans(n_clusters=2, n_init=10).fit(features)
    labels = kmeans.labels_.reshape(a.shape)

    
    masks = [(labels == i) for i in range(2)]
    estimates = [istft(Z_l * mask.astype(float), fs=sr, nperseg=win_length, noverlap=win_length - hop_length)[1] for mask in masks]
    min_len = min(len(e) for e in estimates)
    return [e[:min_len] for e in estimates]


def compute_spectra(x, sr):
    _, _, D = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    return D, np.abs(D) ** 2


def align_estimates(estimates, references):
    min_len = min(min(len(e) for e in estimates), min(len(r) for r in references))
    estimates = [e[:min_len] for e in estimates]
    references = [r[:min_len] for r in references]
    sdr, sir, sar, perm = bss_eval_sources(np.array(references), np.array(estimates))
    aligned = [estimates[i] for i in perm]
    return aligned, [r[:min_len] for r in references], (sdr, sir, sar), perm


def compute_snr(reference, estimate):
    noise = reference - estimate
    return 10 * np.log10(np.sum(reference**2) / (np.sum(noise**2) + 1e-10))


if __name__ == "__main__":
  
    s1, sr = librosa.load(librosa.example('libri1'), duration=5.0)
    s2, _  = librosa.load(librosa.example('libri2'), duration=5.0)
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]
    sources = np.vstack([s1, s2])

    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    X = A @ sources  # shape = (2, L)


    estimates = duet_separation(X, sr)


    estimates_aligned, references_aligned, (sdr, sir, sar), perm = align_estimates(estimates, [s1, s2])
    print("sequence:", perm)
    for i in range(2):
        print(f"source{i+1}: SDR={sdr[i]:.2f} dB, SIR={sir[i]:.2f} dB, SAR={sar[i]:.2f} dB")

   
    for idx, (orig, est) in enumerate(zip(references_aligned, estimates_aligned), 1):
        min_len = min(len(orig), len(est))
        orig, est = orig[:min_len], est[:min_len]
        snr_val = compute_snr(orig, est)
        mse_val = np.mean((orig - est) ** 2)
        ssim_val = ssim(orig, est, data_range=est.max() - est.min())

        print(f"source{idx}: SNR={snr_val:.2f} dB, MSE={mse_val:.6f}, SSIM={ssim_val:.4f}")

        
        t = np.arange(min_len) / sr
        plt.figure(figsize=(8, 3))
        plt.plot(t, orig, label='Original', alpha=0.7)
        plt.plot(t, est, '--', label='Estimated')
        plt.title(f'Overlay Source {idx} - SNR: {snr_val:.2f}dB, MSE: {mse_val:.4e}, SSIM: {ssim_val:.4f}')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.tight_layout()
        plt.show()

     
        corr = correlate(est, orig, mode='full')
        lags = np.arange(-min_len + 1, min_len)
        plt.figure(figsize=(6, 3))
        plt.plot(lags, corr)
        max_lag = lags[np.argmax(np.abs(corr))]
        plt.axvline(max_lag, color='r', linestyle='--', label=f'Max Corr @ {max_lag}')
        plt.title(f'Cross-correlation: Estimated vs Original (Src{idx})')
        plt.xlabel('Lag [samples]')
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    items = [(X[0], 'Mix Ch1'), (X[1], 'Mix Ch2'), (estimates_aligned[0], 'Est Src1'), (estimates_aligned[1], 'Est Src2')]
    plt.figure(figsize=(12, 8))
    for i, (sig, title) in enumerate(items, 1):
        D, _ = compute_spectra(sig, sr)
        plt.subplot(4, 1, i)
        plt.imshow(20 * np.log10(np.abs(D) + 1e-6), origin='lower', aspect='auto')
        plt.title(title)
        plt.xlabel('Frame')
        plt.ylabel('Freq bin')
    plt.tight_layout()
    plt.show()

    
    sf.write("estimated1.wav", estimates_aligned[0], sr)
    sf.write("estimated2.wav", estimates_aligned[1], sr)
