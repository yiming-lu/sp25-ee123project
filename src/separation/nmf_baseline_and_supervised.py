import os
import sys

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
from scipy.signal import medfilt
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from mir_eval.separation import bss_eval_sources
from scipy.ndimage import gaussian_filter1d
from skimage.metrics import structural_similarity as ssim
import librosa

_script_dir = os.path.dirname(__file__)
_src_dir = os.path.abspath(os.path.join(_script_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from stft_utils import stft, istft

n_fft = 1024
hop_length = 512
mix_duration = 5.0
max_iter_update = 10000  



K_baseline = 2
K_supervised = 20
max_iter = 500

def compute_spectra(x):
    D = stft(x, n_fft=n_fft, hop_length=hop_length)
    return D, np.abs(D)**2


def supervised_nmf(s1, s2, mix, L):
    D1, p1 = compute_spectra(s1)
    D2, p2 = compute_spectra(s2)
    K = 20
    nmf1 = NMF(n_components=K, beta_loss='kullback-leibler',
               solver='mu', alpha_H=0.1, init='nndsvda', max_iter=500)
    W1 = nmf1.fit_transform(p1)
    nmf2 = NMF(n_components=K, beta_loss='kullback-leibler',
               solver='mu', alpha_H=0.1, init='nndsvda', max_iter=500)
    W2 = nmf2.fit_transform(p2)
    W = np.concatenate([W1, W2], axis=1)

    Dmix, pmix = compute_spectra(mix)
    R, T = W.shape[1], pmix.shape[1]
    H = np.random.rand(R, T)
    for _ in range(500):
        # KL 
        H *= (W.T @ (pmix / (W @ H + 1e-8))) / (W.T.sum(axis=1)[:,None] + 0.1)
    from scipy.ndimage import gaussian_filter1d
    H = gaussian_filter1d(H, sigma=1.0, axis=1)

    Vs = [np.outer(W[:, i], H[i]) for i in range(R)]
    Vtot = sum(Vs) + 1e-8
    masks = [V / Vtot for V in Vs]
    phase = Dmix / (np.abs(Dmix) + 1e-8)

    est1 = sum(istft(masks[i] * np.abs(Dmix) * phase,
                    hop_length=hop_length, length=L) for i in range(K))
    est2 = sum(istft(masks[i+K] * np.abs(Dmix) * phase,
                    hop_length=hop_length, length=L) for i in range(K))
    return np.vstack([est1, est2])
def baseline_nmf(mix, L):
    D = stft(mix, n_fft=n_fft, hop_length=hop_length)
    power = np.abs(D)**2
    nmf = NMF(n_components=K_baseline, init='nndsvda', solver='mu', beta_loss='itakura-saito',
              max_iter=max_iter, random_state=0)
    W = nmf.fit_transform(power)
    H = nmf.components_
    Vs = [np.outer(W[:,i], H[i]) for i in range(K_baseline)]
    Vtot = sum(Vs) + 1e-8
    masks = [V/Vtot for V in Vs]
    est = []
    for mask in masks:
        S_tf = mask * np.abs(D) * np.exp(1j * np.angle(D))
        est.append(istft(S_tf, hop_length=hop_length, length=L))
    return np.vstack(est)


if __name__ == '__main__':

    s1, sr1 = librosa.load(librosa.example('libri1'), duration=mix_duration)
    s2, sr2 = librosa.load(librosa.example('libri2'), duration=mix_duration)
    assert sr1 == sr2, "fs must be the same for both sources"
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]
    sources = np.vstack([s1, s2])


    A = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
    X = A @ sources
    mix = X.mean(axis=0)

    S_est = supervised_nmf(s1, s2, mix, L)


    sdr, sir, sar, _ = bss_eval_sources(np.vstack([s1, s2]), S_est)
    print("SDR:", sdr, "SIR:", sir, "SAR:", sar)
    out_dir = os.path.abspath(os.path.join(os.getcwd(), 'results', 'improved_nmf'))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, 'src1.wav'), S_est[0], sr1)
    sf.write(os.path.join(out_dir, 'src2.wav'), S_est[1], sr1)