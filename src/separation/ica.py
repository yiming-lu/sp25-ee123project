
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from sklearn.decomposition import FastICA
from mir_eval.separation import bss_eval_sources
import os
from scipy.signal import correlate
import sys

from skimage.metrics import structural_similarity as ssim


_script_dir = os.path.dirname(__file__)
_src_dir = os.path.abspath(os.path.join(_script_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from stft_utils import stft, istft


def main():

    s1, sr1 = librosa.load(librosa.example('libri1'), duration=5.0)
    s2, sr2 = librosa.load(librosa.example('libri2'), duration=5.0)
    assert sr1 == sr2, "fs must be the same for both sources"

    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]
    sources = np.vstack([s1, s2])   # shape = (2, L)


    A = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
    X = A @ sources                # shape = (2, L)

    ica = FastICA(n_components=2, random_state=0)
    S_est = ica.fit_transform(X.T).T  # shape = (2, L)

    sdr, sir, sar, perm = bss_eval_sources(sources, S_est)
    print("ICA result：")
    for i, (sdr_i, sir_i, sar_i) in enumerate(zip(sdr, sir, sar), 1):
        print(f"  source {i}: SDR={sdr_i:.2f} dB, SIR={sir_i:.2f} dB, SAR={sar_i:.2f} dB")


    mix = X.mean(axis=0)
    print("\nadded result：")


    D_true1 = stft(s1, n_fft=1024, hop_length=512)
    D_true2 = stft(s2, n_fft=1024, hop_length=512)
    log_true1 = 20 * np.log10(np.abs(D_true1) + 1e-6)
    log_true2 = 20 * np.log10(np.abs(D_true2) + 1e-6)

    for i, (est, orig, log_true) in enumerate(zip(S_est, sources, [log_true1, log_true2]), 1):
 
        noise_in = mix - orig
        noise_out = est - orig
        power_signal = np.mean(orig ** 2)
        power_noise_in = np.mean(noise_in ** 2)
        power_noise_out = np.mean(noise_out ** 2)
        snr_in = 10 * np.log10(power_signal / (power_noise_in + 1e-8))
        snr_out = 10 * np.log10(power_signal / (power_noise_out + 1e-8))
        delta_snr = snr_out - snr_in
        mse = np.mean((est - orig) ** 2)

        D_est = stft(est, n_fft=1024, hop_length=512)
        log_est = 20 * np.log10(np.abs(D_est) + 1e-6)
        data_range = np.max(log_true) - np.min(log_true)
        ssim_val = ssim(log_true, log_est, data_range=data_range)

        print(f"  source {i}: ΔSNR={delta_snr:.2f} dB, MSE={mse:.6f}, SSIM={ssim_val:.4f}")


    out_dir = os.path.abspath(os.path.join(_script_dir, '..', 'results', 'separated'))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, 'ICA_src1.wav'), S_est[0], sr1)
    sf.write(os.path.join(out_dir, 'ICA_src2.wav'), S_est[1], sr1)
    print(f"save to  {out_dir}")


if __name__ == '__main__':
    main()
