import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss

# ==== Functions ====
def array_response_vector_2d(array, theta):
    k = 2 * np.pi  # wave number (λ = 1)
    direction = np.array([np.cos(theta), np.sin(theta)])  # unit vector
    phase_shifts = array @ direction  # (N,)
    return np.exp(1j * k * phase_shifts) / np.sqrt(array.shape[0])

def music(CovMat, L, N, array, Angles):
    # Use eigenvalue decomposition (sorted)
    eigvals, eigvecs = LA.eig(CovMat)
    idx = eigvals.argsort()[::-1]  # Sort in descending order
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Noise subspace (should have N-L eigenvectors)
    Qn = eigvecs[:, L:N]
    
    # Compute MUSIC spectrum
    pspectrum = np.zeros(Angles.size)
    for i, angle in enumerate(Angles):
        a = array_response_vector_2d(array, angle)
        pspectrum[i] = 1 / LA.norm(Qn.conj().T @ a)
    
    # Normalize and convert to dB
    psindB = 10 * np.log10(pspectrum / pspectrum.max())
    
    # Improved peak finding
    peaks, _ = ss.find_peaks(psindB, height=-3, distance=10)  # Adjusted parameters
    peak_vals = psindB[peaks]
    topL_idx = peaks[np.argsort(peak_vals)[-L:]]  # Get top L peaks
    
    return np.sort(topL_idx), psindB

def esprit(CovMat, L, N):
    _, U = LA.eig(CovMat)
    S = U[:, 0:L]
    Phi = LA.pinv(S[:-1]) @ S[1:]
    eigs, _ = LA.eig(Phi)
    DoAsESPRIT = np.arcsin(np.angle(eigs) / np.pi)
    return DoAsESPRIT

# ==== Simulation ====
np.random.seed(6)
lamda = 1
L = 2  # number of sources
snr = 10

# 3x3 square array
side = 3
d = 0.5
x, y = np.meshgrid(np.arange(side), np.arange(side))
array = np.column_stack((x.flatten(), y.flatten())) * d  # shape (9,2)
N = array.shape[0]

# Plot array layout
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.plot(array[:, 0], array[:, 1], '^')
plt.title('2D Square Array')
plt.legend(['Antenna'])

# Random DoAs and source amplitudes
Thetas = np.pi * (np.random.rand(L) - 0.5)  # range [-π/2, π/2]
Alphas = (np.random.randn(L) + 1j * np.random.randn(L)) * np.sqrt(1/2)

# Simulated signal h
h = np.zeros(N, dtype=complex)
for i in range(L):
    h += Alphas[i] * array_response_vector_2d(array, Thetas[i])

Angles = np.linspace(-np.pi/2, np.pi/2, 360)
hv = np.zeros(Angles.size)
for j, angle in enumerate(Angles):
    a = array_response_vector_2d(array, angle)
    hv[j] = np.abs(np.inner(h, a.conj()))

powers = np.zeros(L)
for j in range(L):
    a = array_response_vector_2d(array, Thetas[j])
    powers[j] = np.abs(np.inner(h, a.conj()))

# Correlation plot
plt.subplot(232)
plt.plot(Angles, hv)
plt.plot(Thetas, powers, '*')
plt.title('Correlation')
plt.legend(['Correlation power', 'Actual DoAs'])

# Create multiple realizations (snapshots)
numrealization = 100
H = np.zeros((N, numrealization), dtype=complex)
for it in range(numrealization):
    htmp = np.zeros(N, dtype=complex)
    for i in range(L):
        pha = np.exp(1j * 2 * np.pi * np.random.rand(1))
        htmp += pha * Alphas[i] * array_response_vector_2d(array, Thetas[i])
    noise = np.sqrt(0.5 / snr) * (np.random.randn(N) + 1j * np.random.randn(N))
    H[:, it] = htmp + noise

CovMat = H @ H.conj().T

# MUSIC
DoAsMUSIC_idx, psindB = music(CovMat, L, N, array, Angles)
DoAsMUSIC = Angles[DoAsMUSIC_idx]
plt.subplot(233)
plt.plot(Angles, psindB)
plt.plot(DoAsMUSIC, psindB[DoAsMUSIC_idx], 'x')
plt.title('MUSIC')
plt.legend(['pseudo spectrum', 'Estimated DoAs'])

# ESPRIT
DoAsESPRIT = esprit(CovMat, L, N)
plt.subplot(234)
plt.plot(Thetas, np.zeros_like(Thetas), '*')
plt.plot(DoAsESPRIT, np.zeros_like(DoAsESPRIT), 'x')
plt.title('ESPRIT')
plt.legend(['Actual DoAs', 'Estimated DoAs'])

# Polar plot (MUSIC vs Actual)
plt.subplot(235, polar=True)
plt.plot(Thetas, np.ones_like(Thetas), '*', label='Actual DoAs')
plt.plot(DoAsMUSIC, np.ones_like(DoAsMUSIC), 'x', label='MUSIC Estimated DoAs')
plt.title('Polar Plot - MUSIC')
plt.legend(loc='lower left')

# Polar plot (ESPRIT vs Actual)
plt.subplot(236, polar=True)
plt.plot(Thetas, np.ones_like(Thetas), '*', label='Actual DoAs')
plt.plot(DoAsESPRIT, np.ones_like(DoAsESPRIT), 'x', label='ESPRIT Estimated DoAs')
plt.title('Polar Plot - ESPRIT')
plt.legend(loc='lower left')

print('\nActual DoAs:', np.round(np.sort(Thetas), 3))
print('MUSIC DoAs:', np.round(np.sort(DoAsMUSIC), 3))
print('ESPRIT DoAs:', np.round(np.sort(DoAsESPRIT), 3))

plt.tight_layout()
plt.show()
