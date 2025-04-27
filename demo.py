import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Generate synthetic mixed signals
def generate_signals():
    np.random.seed(0)
    time = np.linspace(0, 10, 1000)

    # Source signals
    s1 = np.sin(2 * time)  # Sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Square wave signal
    s3 = np.random.normal(size=time.shape)  # Random noise

    S = np.c_[s1, s2, s3]
    S /= S.std(axis=0)  # Standardize data

    # Mixing matrix
    A = np.array([[1, 1, 0.5], [0.5, 2, 1], [1.5, 1, 2]])
    X = np.dot(S, A.T)  # Mixed signals
    return X, S

# Perform ICA to separate signals
def separate_signals(X):
    ica = FastICA(n_components=3, random_state=0)
    S_ = ica.fit_transform(X)  # Reconstructed signals
    A_ = ica.mixing_  # Estimated mixing matrix
    return S_, A_

# Plot results
def plot_signals(X, S, S_):
    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.title("Mixed Signals")
    plt.plot(X)
    
    plt.subplot(3, 1, 2)
    plt.title("Original Signals")
    plt.plot(S)
    
    plt.subplot(3, 1, 3)
    plt.title("Separated Signals")
    plt.plot(S_)

    plt.tight_layout()
    plt.show()

def demo():
    """
    Demonstrates the process of generating, mixing, and separating signals.
    """
    print("Generating mixed signals...")
    X, S = generate_signals()
    print("Performing signal separation using ICA...")
    S_, A_ = separate_signals(X)
    print("Plotting results...")
    plot_signals(X, S, S_)

if __name__ == "__main__":
    demo()
