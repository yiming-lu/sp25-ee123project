import numpy as np
import librosa

def stft(x, n_fft=1024, hop_length=None):
    """
    Compute the STFT of a 1D signal.
    """
    hop_length = hop_length or n_fft // 2
    return librosa.stft(x, n_fft=n_fft, hop_length=hop_length)


def istft(X, hop_length=None):
    """
    Inverse STFT back to time-domain signal.
    """
    hop_length = hop_length or (X.shape[0] * 2)
    return librosa.istft(X, hop_length=hop_length)