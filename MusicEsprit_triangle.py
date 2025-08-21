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
    # Use SVD for better numerical stability with small arrays
    _, S, Vh = LA.svd(CovMat)
    Qn = Vh[L:].conj().T  # Noise subspace
    
    # Compute MUSIC spectrum
    pspectrum = np.zeros(Angles.size)
    for i, angle in enumerate(Angles):
        a = array_response_vector_2d(array, angle)
        pspectrum[i] = 1 / LA.norm(Qn.conj().T @ a)
    
    # Normalize and convert to dB
    psindB = 10 * np.log10(pspectrum / pspectrum.max())
    
    # More lenient peak finding for small arrays
    peaks, props = ss.find_peaks(psindB, prominence=1, width=2)
    
    if len(peaks) >= L:
        topL_idx = peaks[np.argsort(props['prominences'])[-L:]]
    else:
        topL_idx = peaks
    
    return np.sort(topL_idx), psindB

def esprit_triangular(CovMat, L, N):
    """Modified ESPRIT for triangular arrays"""
    # Use SVD
    _, _, Vh = LA.svd(CovMat)
    S = Vh[:L].T  # Signal subspace
    
    # Create two subarrays by excluding one microphone at a time
    # This creates an implicit shift invariance
    S1 = S[:-1]  # First subarray (mics 0 and 1)
    S2 = S[1:]   # Second subarray (mics 1 and 2)
    
    # Solve for rotation matrix
    Phi = LA.pinv(S1) @ S2
    eigvals = LA.eigvals(Phi)
    
    # Estimate DoAs
    DoAs = np.arcsin(np.angle(eigvals) / np.pi)
    return DoAs[:L]  # Return only L estimates

# ==== Simulation ====
np.random.seed(6)
lamda = 1
L = 2  # number of sources
snr = 20  # Increased SNR for better performance with few mics

# Equilateral triangle array
radius = 0.5  # Radius of circumscribed circle
angles = np.linspace(0, 2*np.pi, 4)[:-1]  # 3 points at 0, 120, 240 degrees
array = radius * np.column_stack((np.cos(angles), np.sin(angles)))  # shape (3,2)
N = array.shape[0]

# Plot array layout
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.plot(array[:, 0], array[:, 1], '^')
plt.title('Triangular Microphone Array')
for i in range(N):
    plt.text(array[i, 0], array[i, 1], f'Mic {i}')
plt.legend(['Microphones'])

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
numrealization = 200  # More snapshots to compensate for few mics
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
plt.legend(['pseudo spectrum', 'Estimated DoAs'], loc = "lower left")

# ESPRIT
DoAsESPRIT = esprit_triangular(CovMat, L, N)
plt.subplot(234)
plt.plot(Thetas, np.zeros_like(Thetas), '*')
plt.plot(DoAsESPRIT, np.zeros_like(DoAsESPRIT), 'x')
plt.title('ESPRIT (Modified for Triangle)')
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