import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from sklearn.decomposition import NMF
from mir_eval.separation import bss_eval_sources
from scipy.ndimage import gaussian_filter1d
from skimage.metrics import structural_similarity as ssim

# Simple STFT / ISTFT wrappers using librosa
def stft(x, n_fft=1024, hop_length=None):
    hop_length = hop_length or n_fft // 2
    return librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

def istft(X, hop_length=None, length=None):
    hop_length = hop_length or X.shape[0] // 2
    return librosa.istft(X, hop_length=hop_length, length=length)

# Global parameters
n_fft         = 1024
hop_length    = 512
mix_duration  = 5.0
K_baseline    = 2
K_supervised  = 20
max_iter      = 500

def compute_metrics(sources, estimates, mix):

    sdr, sir, sar, _ = bss_eval_sources(sources, estimates)
    delta_snr, mse_vals, ssim_vals = [], [], []


    D_true = [stft(s, n_fft=n_fft, hop_length=hop_length) for s in sources]
    log_true = [20*np.log10(np.abs(D)+1e-6) for D in D_true]

    for i, orig in enumerate(sources):
        est = estimates[i]

        Lc = min(len(orig), len(est), len(mix))
        o = orig[:Lc]; e = est[:Lc]; m = mix[:Lc]

        # ΔSNR
        noise_in  = m - o
        noise_out = e - o
        psig = np.mean(o**2)
        snr_in  = 10*np.log10(psig/(np.mean(noise_in**2)+1e-8))
        snr_out = 10*np.log10(psig/(np.mean(noise_out**2)+1e-8))
        delta_snr.append(snr_out - snr_in)

        # MSE
        mse_vals.append(np.mean((e - o)**2))

        # SSIM on log-spectra
        D_est = stft(e, n_fft=n_fft, hop_length=hop_length)
        log_est = 20*np.log10(np.abs(D_est)+1e-6)
        drange = log_true[i].max() - log_true[i].min()
        ssim_vals.append(ssim(log_true[i], log_est, data_range=drange))

    return np.array(sdr), np.array(sir), np.array(delta_snr), np.array(mse_vals), np.array(ssim_vals)

def baseline_nmf(mix, L):

    D = stft(mix, n_fft=n_fft, hop_length=hop_length)
    power = np.abs(D)**2
    nmf = NMF(n_components=K_baseline,
              beta_loss='itakura-saito',
              solver='mu',
              init='nndsvda',
              max_iter=max_iter,
              random_state=0)
    W = nmf.fit_transform(power)
    H = nmf.components_
    Vs = [np.outer(W[:,i], H[i]) for i in range(K_baseline)]
    Vtot = sum(Vs) + 1e-8
    masks = [V/Vtot for V in Vs]
    phase = np.exp(1j * np.angle(D))
    ests = []
    for mask in masks:
        S_tf = mask * np.abs(D) * phase
        ests.append(istft(S_tf, hop_length=hop_length, length=L))
    return np.vstack(ests)


def annotate_bars(ax, fmt, offset=3):
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(fmt.format(h),
                    xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, offset), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

def main():

    s1, sr1 = librosa.load(librosa.example('libri1'), duration=mix_duration)
    s2, _   = librosa.load(librosa.example('libri2'), duration=mix_duration)
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]
    sources = np.vstack([s1, s2])
    mix = (np.array([[1.0,0.5],[0.5,1.0]]) @ sources).mean(axis=0)


    est_base = baseline_nmf(mix, L)



    # permute so estimate aligns with original
    sdr_b, sir_b, sar_b, perm_b = bss_eval_sources(sources, est_base)
    est_base = est_base[perm_b]
    sdr_s, sir_s, sar_s, perm_s = bss_eval_sources(sources, est_sup)
    est_sup  = est_sup[perm_s]

    print("Baseline NMF:")
    for i, (d, i_, a) in enumerate(zip(sdr_b, sir_b, sar_b), start=1):
        print(f"  source{i}: SDR={d:.2f}, SIR={i_:.2f}, SAR={a:.2f}")
    print("\nSupervised NMF:")
    for i, (d, i_, a) in enumerate(zip(sdr_s, sir_s, sar_s), start=1):
        print(f"  source{i}: SDR={d:.2f}, SIR={i_:.2f}, SAR={a:.2f}")

    # 4) 统一量化指标 ΔSNR, MSE, SSIM
    sdr_b, sir_b, dsnr_b, mse_b, ssim_b = compute_metrics(sources, est_base, mix)
    sdr_s, sir_s, dsnr_s, mse_s, ssim_s = compute_metrics(sources, est_sup,  mix)

    # 5) 美化绘图
    labels = ['Src1 Base','Src2 Base','Src1 Sup','Src2 Sup']
    vals = {
        'SDR (dB)':      (np.hstack([sdr_b, sdr_s]), "{:.2f}"),
        'SIR (dB)':      (np.hstack([sir_b, sir_s]), "{:.2f}"),
        'ΔSNR (dB)':     (np.hstack([dsnr_b, dsnr_s]), "{:.2f}"),
        'MSE':           (np.hstack([mse_b,  mse_s]), "{:.2e}", True),  # log-scale
        'SSIM':          (np.hstack([ssim_b, ssim_s]), "{:.3f}")
    }
    colors = ['#1f77b4','#1f77b4','#ff7f0e','#ff7f0e']  # base / sup
    for title, (data, fmt, *opt) in vals.items():
        logscale = opt[0] if opt else False
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(labels, data, color=colors)
        if logscale: ax.set_yscale('log')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        annotate_bars(ax, fmt)
        plt.xticks(rotation=30, ha='right')
        fig.tight_layout()
        fn = title.lower().replace(' ','_').replace('(', '').replace(')', '') + '_nice.png'
        fig.savefig(fn)
        plt.show()


    out_dir = os.path.join(os.getcwd(), 'results', 'compare_nmf')
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, 'base_src1.wav'), est_base[0], sr1)
    sf.write(os.path.join(out_dir, 'base_src2.wav'), est_base[1], sr1)
    sf.write(os.path.join(out_dir, 'sup_src1.wav'),  est_sup[0],  sr1)
    sf.write(os.path.join(out_dir, 'sup_src2.wav'),  est_sup[1],  sr1)
    print(f"save to {out_dir}")

if __name__ == '__main__':
    main()