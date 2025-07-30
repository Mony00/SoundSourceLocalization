import numpy as np

def esprit_doa(X, d, wavelength, num_sources=1):
    """
    Estimate DOA using ESPRIT from array signal X.

    Parameters:
        X: ndarray, shape (num_mics, num_samples)
        d: spacing between microphones
        wavelength: wavelength of the signal
        num_sources: number of sources to estimate
    Returns:
        DOA estimate in radians
    """
    from scipy.linalg import svd, eig
    R = X @ X.conj().T  # Spatial covariance
    U, S, Vh = svd(R)
    Us = U[:, :num_sources]

    # Partition the steering space
    Us1 = Us[:-1]
    Us2 = Us[1:]
    Phi = np.linalg.pinv(Us1) @ Us2
    eigvals = np.linalg.eigvals(Phi)

    angles = -np.angle(eigvals)
    theta = np.arcsin(angles * wavelength / (2 * np.pi * d))
    return np.real(theta)
