def esprit_doa(X, d, wavelength, num_sources=1):
    """
    Estimate DOA using ESPRIT algorithm for ULA (Uniform Linear Array).

    Parameters:
        X (ndarray): Shape (num_mics, num_snapshots), frequency-domain data at a single frequency bin.
        d (float): Distance between microphones (in meters).
        wavelength (float): Wavelength of the signal (in meters).
        num_sources (int): Number of sources to estimate.

    Returns:
        doa_angles_deg: DOA estimates in degrees.
    """
    import numpy as np
    from scipy.linalg import svd

    R = np.dot(X, X.conj().T) / X.shape[1]  # Covariance matrix
    U, S, Vh = svd(R)
    Es = U[:, :num_sources]  # Signal subspace

    # ESPRIT requires two subarrays: take Es with 1-row shifted versions
    Es1 = Es[:-1, :]
    Es2 = Es[1:, :]

    # Solve the rotational invariance equation
    Psi = np.linalg.pinv(Es1) @ Es2
    eigvals, _ = np.linalg.eig(Psi)

    # Convert phase to angle (theta = arcsin(...))
    angles_rad = np.arcsin(np.angle(eigvals) * wavelength / (2 * np.pi * d))

    # Filter out imaginary values or nan if arcsin fails
    doa_angles_deg = np.degrees(np.real(angles_rad[np.isreal(angles_rad)]))
    return doa_angles_deg
