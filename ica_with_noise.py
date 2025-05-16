
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from sklearn.decomposition import FastICA
from mir_eval.separation import bss_eval_sources
import os
from scipy.signal import medfilt2d


def stft(x, n_fft=1024, hop_length=None):

    hop_length = hop_length or n_fft // 2
    return librosa.stft(x, n_fft=n_fft, hop_length=hop_length)


def istft(X, hop_length=None):

    hop_length = hop_length or (X.shape[0] * 2)
    return librosa.istft(X, hop_length=hop_length)


def denoise_signal(sig, n_fft=1024, hop_length=None, kernel_size=(1,7)):
 
    hop_length = hop_length or n_fft // 2
    D = stft(sig, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(D), np.angle(D)

    noise_est = medfilt2d(mag, kernel_size=kernel_size)

    mag_den = np.maximum(mag - noise_est, 0)
    D_den = mag_den * np.exp(1j * phase)
    return librosa.istft(D_den, hop_length=hop_length, length=len(sig))


def main():
    s1, sr1 = librosa.load(librosa.example('libri1'), duration=5.0)
    s2, sr2 = librosa.load(librosa.example('libri2'), duration=5.0)
    assert sr1 == sr2, "fs must be the same for both sources"

 
    example_file = librosa.ex('trumpet') if hasattr(librosa, 'ex') else librosa.example('trumpet')
    bg, sr_bg = librosa.load(example_file, sr=sr1, duration=5.0)


    L = min(len(s1), len(s2), len(bg))
    s1, s2, bg = s1[:L], s2[:L], bg[:L]
    sources = np.vstack([s1, s2, bg])  # shape = (3, L)

    A = np.array([[1.0, 0.5, 0.3],
                  [0.5, 1.0, 0.2],
                  [0.3, 0.2, 1.0]])
    X = A @ sources  # shape = (3, L)


    ica = FastICA(n_components=3, random_state=0)
    S_est = ica.fit_transform(X.T).T  # shape = (3, L)

    S_den = np.vstack([denoise_signal(s, n_fft=1024, hop_length=512) for s in S_est])


    sdr, sir, sar, _ = bss_eval_sources(sources, S_est)
    for i, (sdr_i, sir_i, sar_i) in enumerate(zip(sdr, sir, sar), 1):
        print(f"source {i} : SDR={sdr_i:.2f} dB, SIR={sir_i:.2f} dB, SAR={sar_i:.2f} dB")


    sdr_d, sir_d, sar_d, _ = bss_eval_sources(sources, S_den)
    for i, (sdr_i, sir_i, sar_i) in enumerate(zip(sdr_d, sir_d, sar_d), 1):
        print(f"source {i} : SDR={sdr_i:.2f} dB, SIR={sir_i:.2f} dB, SAR={sar_i:.2f} dB")


    out_dir = os.path.abspath(os.path.join('.', 'results', 'separated'))
    os.makedirs(out_dir, exist_ok=True)

    for ch in range(3):
        sf.write(os.path.join(out_dir, f'MixCh{ch+1}.wav'), X[ch], sr1)

    for ch in range(3):
        sf.write(os.path.join(out_dir, f'ICA_src{ch+1}.wav'), S_est[ch], sr1)
    for ch in range(3):
        sf.write(os.path.join(out_dir, f'ICA_src{ch+1}_denoised.wav'), S_den[ch], sr1)
    print(f"save to  {out_dir}")

if __name__ == '__main__':
    main()
