

import os
import sys
import numpy as np
import soundfile as sf
import librosa
from sklearn.decomposition import NMF
from mir_eval.separation import bss_eval_sources


def stft(x, n_fft=1024, hop_length=None):
    """
    Compute the STFT of a 1D signal.
    """
    hop_length = hop_length or n_fft // 2
    return librosa.stft(x, n_fft=n_fft, hop_length=hop_length)


n_fft = 1024
hop_length = 512
mix_duration = 5.0


def istft(X, hop_length=None, length=None):
    """
    Inverse STFT back to time-domain signal, with fixed output length.
    """
    hop = hop_length or (X.shape[0] * 2)
    return librosa.istft(X, hop_length=hop, length=length)


def compute_spectra(x):
    D = stft(x, n_fft=n_fft, hop_length=hop_length)
    return D, np.abs(D)**2


def supervised_nmf(s1, s2, mix, L):

    D1, p1 = compute_spectra(s1)
    D2, p2 = compute_spectra(s2)
    K = 20 

    nmf1 = NMF(n_components=K, beta_loss='kullback-leibler', solver='mu', alpha_H=0.1,
               init='nndsvda', max_iter=500)
    W1 = nmf1.fit_transform(p1)
    nmf2 = NMF(n_components=K, beta_loss='kullback-leibler', solver='mu', alpha_H=0.1,
               init='nndsvda', max_iter=500)
    W2 = nmf2.fit_transform(p2)
    W = np.concatenate([W1, W2], axis=1)


    Dmix, pmix = compute_spectra(mix)
    R, T = W.shape[1], pmix.shape[1]
    H = np.random.rand(R, T)

    for _ in range(500):
        H *= (W.T @ (pmix / (W @ H + 1e-8))) / (W.T.sum(axis=1)[:, None] + 0.1)


    Vs = [np.outer(W[:, i], H[i]) for i in range(R)]
    Vtot = sum(Vs) + 1e-8
    masks = [V / Vtot for V in Vs]
    phase = Dmix / (np.abs(Dmix) + 1e-8)

    est1 = sum(istft(masks[i] * np.abs(Dmix) * phase,
                    hop_length=hop_length, length=L) for i in range(K))
    est2 = sum(istft(masks[i+K] * np.abs(Dmix) * phase,
                    hop_length=hop_length, length=L) for i in range(K))
    return np.vstack([est1, est2])


if __name__ == '__main__':

    s1, sr1 = librosa.load(librosa.example('libri1'), duration=mix_duration)
    s2, sr2 = librosa.load(librosa.example('libri2'), duration=mix_duration)
    assert sr1 == sr2, "fs must be the same for both sources"
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]

  
    example = librosa.ex('trumpet') if hasattr(librosa, 'ex') else librosa.example('trumpet')
    bg, sr_bg = librosa.load(example, sr=sr1, duration=mix_duration)
    bg = bg[:L]

    alpha, beta, gamma = 1.0, 0.5, 0.3
    mix = alpha * s1 + beta * s2 + gamma * bg


    S_est = supervised_nmf(s1, s2, mix, L)


    sdr, sir, sar, _ = bss_eval_sources(np.vstack([s1, s2]), S_est)
    print(f"SDR: {sdr}, SIR: {sir}, SAR: {sar}")

    out_dir = os.path.abspath(os.path.join(os.getcwd(), 'results', 'improved_nmf'))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, 'mix_with_bg.wav'), mix, sr1)
    sf.write(os.path.join(out_dir, 'nmf_src1.wav'), S_est[0], sr1)
    sf.write(os.path.join(out_dir, 'nmf_src2.wav'), S_est[1], sr1)
    print(f"save to  {out_dir}")
