import numpy as np
import matplotlib.pylab as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample
import pyroomacoustics as pra
import sounddevice as sd

from Esprit import esprit_doa

# a few parameters
c = 343.    # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_range = [300, 3500]
rng = np.random.RandomState(23)
duration_samples = int(fs)

# --- Sound Source Loading and Preparation ---
fs_wav, signal = wavfile.read("Samples/guitar_16k.wav")
# Ensure the signal is float for processing
signal = signal.astype(np.float32)

# Ensure the signal is at the correct sampling frequency
if fs_wav != fs:
    num_samples_resampled = int(len(signal) * fs / fs_wav)
    signal = resample(signal, num_samples_resampled)
    print(f"Resampled guitar signal from {fs_wav} Hz to {fs} Hz.")

# Trim or pad signal to desired duration
if len(signal) > duration_samples:
    signal = signal[:duration_samples]
else:
    signal = np.pad(signal, (0, duration_samples - len(signal)), 'constant')

# --- Room Setup ---
corners = np.array([[0,0], [0,3], [4,3], [4,0]]).T

# Create a shoebox 2D room with NO REVERBERATION (max_order=0)
room = pra.Room.from_corners(corners, fs=fs, max_order=0) 

# Define the source location
source_location = np.array([1., 1.])
room.add_source(source_location, signal=signal)

# Add microphone array
R = pra.square_2D_array(center=[2., 1.5], M=2, N=2, phi=0, d=0.04) 
room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

room.simulate()

# DOA Processing 
X = pra.transform.stft.analysis(room.mic_array.signals.T, nfft, nfft // 2)
X = X.transpose([2, 1, 0])# Shape: (frequency bins, microphones, time frames)

#-----------Music estimation-----------
doa = pra.doa.algorithms['NormMUSIC'](R, fs, nfft, c=c, num_src=1) 
doa.locate_sources(X, freq_range=freq_range) 
spatial_response = doa.grid.values
# Normalize for plotting
min_val = spatial_response.min()
max_val = spatial_response.max()
spatial_response = (spatial_response - min_val) / (max_val - min_val)

# Calculate the actual azimuth angle of the source relative to the mic array center
mic_array_center = R.mean(axis=1)
vector_to_source = source_location - mic_array_center
actual_azimuth = np.arctan2(vector_to_source[1], vector_to_source[0])

# ----Estimate angle (MUSIC)---
estimated_azimuths = doa.azimuth_recon

#-------------ESPRIT DOA estimation--------------
freq_bin = min(2, X.shape[0] - 1) #it has to be within frequency range
X_esprit = X[freq_bin, :, :] # shape: (num_mics, num_frames)

f_esprit = freq_bin * fs / nfft # frequency in Hz
wavelength = c / f_esprit

theta_esprit = esprit_doa(X_esprit, d= 0.04, wavelength= wavelength, num_sources=1)

# --- Print the angles ---
print(f"Actual Source Azimuth (radians): {actual_azimuth:.4f}")
print(f"Actual Source Azimuth (degrees): {np.degrees(actual_azimuth):.2f}°")

if estimated_azimuths.size > 0:
    estimated_angle_rad = estimated_azimuths[0] # Assuming num_src=1
    print(f"Estimated MUSIC DOA (radians): {estimated_angle_rad:.4f}")
    print(f"Estimated MUSIC DOA (degrees): {np.degrees(estimated_angle_rad):.2f}°")
else:
    print("No estimated MUSIC DOA azimuths found.")

if theta_esprit is not None and not np.isnan(theta_esprit[0]):
    print(f"Estimated ESPRIT DOA (radians): {theta_esprit[0]: .4f}")
    print(f"Estimated ESPRIT DOA (degrees): {np.degrees(theta_esprit[0]):.2f}°")
else:
    print("ESPRIT failed to estimate DOA.")

# --- Corrected Simultaneous Plotting ---

# Create a figure (THIS IS THE ONLY plt.figure() YOU SHOULD HAVE)
fig = plt.figure(figsize=(16, 8))

# Create the first subplot (room simulation) as a standard Cartesian plot
ax_room = fig.add_subplot(1, 2, 1) # 1 row, 2 columns, first plot

# Create the second subplot (MUSIC spectrum) explicitly as a polar plot
ax_polar = fig.add_subplot(1, 2, 2, projection='polar') # 1 row, 2 columns, second plot, with polar projection

# Plot 1: Room Simulation
room.plot(ax=ax_room) # IMPORTANT: Ensure you're passing 'ax=ax_room'
ax_room.set_xlim([-1, 5])
ax_room.set_ylim([-1, 4])
ax_room.plot(source_location[0], source_location[1], 'ro', markersize=8, label='Actual Source')
ax_room.legend()
ax_room.set_title("Room Simulation with Actual Source")

# Plot 2: MUSIC Spatial Spectrum (Polar Plot)
phi = doa.grid.azimuth # Azimuth angles for the MUSIC spectrum
ax_polar.plot(phi, spatial_response, 
                linewidth=2,
                color='blue',
                alpha=0.7,
                label='MUSIC Spectrum')

# Format polar plot
ax_polar.set_title("DOA estimation (MUSIC + ESPRIT)", pad=20)
ax_polar.set_theta_zero_location('N')  # 0° at top (North)
ax_polar.set_theta_direction(-1)       # Clockwise
ax_polar.grid(True, alpha=0.5)

# Plot the actual source angle on the polar plot
if actual_azimuth is not None:
    ax_polar.plot([actual_azimuth, actual_azimuth], [0, 1], 
                    color='red', linestyle='--', linewidth=2, label='Actual Source Angle')
    ax_polar.plot(actual_azimuth, 1, 'ro', markersize=8)

# Plot the MUSIC estiamtes
if estimated_azimuths.size > 0:
    for i, est_angle in enumerate(estimated_azimuths):
        ax_polar.plot([est_angle, est_angle], [0, 1], 
                        color='green', linestyle=':', linewidth=2, 
                        label='MUSIC DOA' if i == 0 else "") 
        ax_polar.plot(est_angle, 1, 'go', markersize=8)

# Plot the ESPRIT estimates
if theta_esprit is not None and not np.isnan(theta_esprit[0]):
    for i, est_angle in enumerate(np.atleast_1d(theta_esprit)):
        ax_polar.plot([est_angle, est_angle], [0,1], 
                      color = 'magenta', linestyle = '-.', linewidth = 2,
                      label = 'ESPRIT DOA' if i == 0 else "")

ax_polar.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1)) 
plt.tight_layout() 
plt.show() 