# ee123project/src/stft_utils.py
import numpy as np
import librosa

def stft(x, n_fft=1024, hop_length=None):
  
    hop = hop_length if hop_length is not None else n_fft // 2
    return librosa.stft(x, n_fft=n_fft, hop_length=hop)


def istft(X, n_fft=1024, hop_length=None, length=None):
   
    hop = hop_length if hop_length is not None else n_fft // 2
    return librosa.istft(X, hop_length=hop, length=length)