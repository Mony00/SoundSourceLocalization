import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.signal import stft

# === Simulation Parameters ===
fs = 16000          # Sampling frequency (Hz)
c = 343             # Speed of sound (m/s)
duration = 0.5      # Duration of the signal (seconds)
freq = 1000         # Frequency of the sine wave signal (Hz)

# === Create time signal ===
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
signal = np.sin(2 * np.pi * freq * t)

# === Microphone Array (4 mics in linear configuration) ===
n_mics = 4          # Number of microphones
d = 0.05            # Spacing between microphones (meters)
mic_positions = np.array([
    [0, 0],         # Mic 1 (x,y)
    [d, 0],         # Mic 2
    [2*d, 0],       # Mic 3
    [3*d, 0]        # Mic 4
]).T  # Transpose to (2, n_mics)

# === Create Room ===
room_dim = [5, 4]   # 2D room dimensions (x,y in meters)
room = pra.ShoeBox(room_dim, fs=fs, max_order=0)

# Add microphone array
mics = pra.MicrophoneArray(mic_positions, fs)
room.add_microphone_array(mics)

# === Source location ===
azimuth = 45        # Angle in degrees (0 is front of array)
distance = 2.0      # Distance from array center (meters)

# Convert to Cartesian coordinates
azimuth_rad = np.deg2rad(azimuth)
src_x = distance * np.cos(azimuth_rad)
src_y = distance * np.sin(azimuth_rad)
source_position = [src_x, src_y]

# Add source to room
room.add_source(source_position, signal=signal)

# === Run simulation ===
room.simulate()

# === MUSIC Algorithm Implementation ===

# 1. Get microphone signals
mic_signals = room.mic_array.signals

# 2. STFT parameters
nfft = 256          # FFT size (power of 2)
win = np.hamming(nfft)
noverlap = nfft // 2

# 3. Compute STFT for each microphone
f_stft, t_stft, Zxx = stft(
    mic_signals,
    fs=fs,
    window=win,
    nperseg=nfft,
    noverlap=noverlap,
    return_onesided=True
)

# 4. Find frequency bin closest to our signal frequency
target_bin = np.argmin(np.abs(f_stft - freq))

# 5. Prepare data for MUSIC
# Average across time frames
X_avg = np.mean(Zxx[:, target_bin, :], axis=1)

# Reshape for locate_sources
X_stft = Zxx[:, target_bin, :][:, np.newaxis, :]  # Shape (n_mics, 1, n_frames)
X_stft = np.moveaxis(X_stft, 0, 1)  # Shape (1, n_mics, n_frames)

# 6. Create MUSIC object
doa = pra.doa.MUSIC(
    mic_positions,   # Microphone positions (2D)
    fs=fs,
    c=c,
    nfft=nfft,
    num_src=1,
    mode='far',
    azimuth=np.linspace(0, 360, 361).tolist()
)

# Set the covariance matrix
doa.locate_sources(X_stft, freq_bins=[target_bin])

# === Visualization ===
plt.figure(figsize=(10, 5))

# Polar plot
plt.subplot(121, polar=True)
doa.polar_plt_dirac()
plt.plot([azimuth_rad], [1], 'ro', markersize=10, label='True DOA')
plt.legend()

# Linear plot
plt.subplot(122)
doa.azimuth_recon[doa.azimuth_recon < 0] += 360  # Wrap angles
plt.plot(doa.grid.azimuth, doa.spectrum)
plt.axvline(x=azimuth, color='r', linestyle='--', label='True DOA')
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Spectrum')
plt.legend()

plt.suptitle(f'MUSIC DOA Estimation (True: {azimuth}°)')
plt.tight_layout()
plt.show()

# Print results
print(f"True DOA: {azimuth}°")
print(f"Estimated DOA: {doa.azimuth_recon[0]}°")