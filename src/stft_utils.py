# ee123project/src/stft_utils.py
import numpy as np
import librosa

def stft(x, n_fft=1024, hop_length=None):
    """
    Compute the STFT of a 1D signal.

    Args:
        x (np.ndarray): Input time-domain signal.
        n_fft (int): FFT size.
        hop_length (int, optional): Hop length between frames. Defaults to n_fft // 2.

    Returns:
        np.ndarray: Complex STFT matrix of shape (freq_bins, frames).
    """
    hop = hop_length if hop_length is not None else n_fft // 2
    return librosa.stft(x, n_fft=n_fft, hop_length=hop)

def istft(X, n_fft=1024, hop_length=None):
    """
    Inverse STFT back to time-domain signal.

    Args:
        X (np.ndarray): Complex STFT matrix of shape (freq_bins, frames).
        n_fft (int): FFT size (used to compute hop_length if not provided).
        hop_length (int, optional): Hop length between frames. Defaults to n_fft // 2.

    Returns:
        np.ndarray: Reconstructed time-domain signal.
    """
    hop = hop_length if hop_length is not None else n_fft // 2
    return librosa.istft(X, hop_length=hop)