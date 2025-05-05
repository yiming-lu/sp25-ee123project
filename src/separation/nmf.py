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
        # KL 更新
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