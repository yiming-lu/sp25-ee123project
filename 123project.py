import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Generate synthetic signals
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# Create 3 source signals
s1 = np.sin(2 * time)  # Sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Square signal
s3 = np.random.normal(size=n_samples)  # Gaussian noise

# Stack signals and mix them
S = np.c_[s1, s2, s3]
S /= S.std(axis=0)  # Standardize data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Mixed signals

# Apply ICA
ica = FastICA(n_components=3, random_state=0)
S_ = ica.fit_transform(X)  # Reconstructed signals
A_ = ica.mixing_  # Estimated mixing matrix

# Plot results
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.title("Original Signals")
for i, sig in enumerate(S.T):
        plt.plot(sig, label=f"Signal {i+1}")
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
for i, sig in enumerate(X.T):
        plt.plot(sig, label=f"Mixed {i+1}")
plt.legend()

plt.subplot(3, 1, 3)
plt.title("Reconstructed Signals (ICA)")
for i, sig in enumerate(S_.T):
        plt.plot(sig, label=f"Reconstructed {i+1}")
plt.legend()

plt.tight_layout()
plt.show()