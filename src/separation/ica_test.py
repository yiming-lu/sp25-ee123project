# ee123project/src/ica_test.py
"""
Apply time-domain FastICA on the first WAV in data/two-people/ directory.
Usage:
    python src/ica_test.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import FastICA

# Determine project root (one level up from src)
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Path to two-people data folder
data_dir = os.path.join(project_root, 'data', 'two-people')
if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"Directory not found: {data_dir}")

# Find the first WAV file
ewav_files = [f for f in sorted(os.listdir(data_dir)) if f.lower().endswith('.wav')]
if not wav_files:
    raise FileNotFoundError(f"No WAV files found in {data_dir}")
wav_file = wav_files[0]
mix_path = os.path.join(data_dir, wav_file)

# Read mixture (expect stereo)
mix, sr = sf.read(mix_path)
if mix.ndim != 2 or mix.shape[1] != 2:
    raise ValueError(f"Expected stereo WAV with 2 channels, got shape {mix.shape}")

# Transpose to shape (2, N)
sources = mix.T  # two rows: source1, source2
# Prepare input for ICA: shape (N, 2)
X = sources.T

# Apply FastICA
ica = FastICA(n_components=2, random_state=0)
S_est = ica.fit_transform(X).T  # shape (2, N)

# Plot results
plt.figure(figsize=(10, 8))
titles = ['Mixture Channel 1', 'Mixture Channel 2', 'Estimated Source 1', 'Estimated Source 2']
signals = [sources[0], sources[1], S_est[0], S_est[1]]
for i, sig in enumerate(signals):
    plt.subplot(2, 2, i+1)
    plt.plot(sig)
    plt.title(titles[i])
    plt.tight_layout()
plt.show()

# Save estimated sources
out_dir = os.path.join(project_root, 'results', 'separated')
os.makedirs(out_dir, exist_ok=True)
base = os.path.splitext(wav_file)[0]
for i, sig in enumerate(S_est, 1):
    out_path = os.path.join(out_dir, f"ICA_test_{base}_src{i}.wav")
    sf.write(out_path, sig.astype(np.float32), sr)

print(f"Saved separated sources to {out_dir}")
