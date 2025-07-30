import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

# === Setup Parameters ===
fs = 16000
c = 343
duration = 0.2
freq = 1000
samples = int(fs * duration)

# === Create time signal ===
t = np.linspace(0, duration, samples)
signal = np.sin(2 * np.pi * freq * t)

# === Microphone Array ===
n_mics = 6
d = 0.05  # spacing
mic_xy = np.c_[
    np.linspace(0, (n_mics-1)*d, n_mics),
    np.zeros(n_mics)
].T  # Shape: (2, n_mics)

# === Create Room ===
room_dim = [6, 6]
room = pra.ShoeBox(room_dim, fs=fs, max_order=0)
room.add_microphone_array(mic_xy)

# === Source angle and location ===
angle_deg = 60
angle_rad = np.deg2rad(angle_deg)
src_distance = 2.0
src_xy = [3 + src_distance * np.cos(angle_rad),
          3 + src_distance * np.sin(angle_rad)]
room.add_source(src_xy, signal=signal)

# === Run simulation ===
room.simulate()

# === MUSIC DOA Estimation ===
X = room.mic_array.signals.T
doa = pra.doa.MUSIC(mic_array_geometry =  mic_xy, fs = fs, c = c, num_src=1, nfft=256)
doa.locate_sources(X)

# === Plot DOA + True Angle ===
plt.figure(figsize=(6, 6))
doa.polar_plt_dirac()

# Add true location with star
plt.polar([angle_rad], [1.0], 'k*', markersize=14, label='true location')
plt.legend()
plt.title("MUSIC DOA Spectrum with True Location")
plt.show()

# Print estimated
print("Estimated DOA (degrees):", doa.azimuth_recon * 180 / np.pi)