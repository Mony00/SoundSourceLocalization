#!/usr/bin/env python3
"""
DOA Simulation & MUSIC Demo (Pyroomacoustics)
---------------------------------------------
This script builds a reproducible pipeline that matches a typical thesis implementation:
1) Simulate a 2D room in Pyroomacoustics (anechoic or reverberant)
2) Place a *triangular* 3-mic array (equilateral) and a sound source
3) Generate a synthetic signal (wideband chirp) and propagate through the room
4) Add noise to achieve a target SNR
5) Compute and plot:
   - Delay-and-sum beam pattern (for intuition)
   - MUSIC pseudospectrum and the estimated DOA
6) (Optional) Compare against a ULA + ESPRIT baseline (ESPRIT only for ULA)

Usage
-----
python doa_pyroomacoustics.py --help

Example
-------
python doa_pyroomacoustics.py \
  --fs 16000 --duration 2.0 \
  --room 6 5 --rt60 0.3 \
  --array triangular --side 0.05 \
  --src 2.0 1.5 \
  --snr 15 \
  --plot

Notes
-----
- MUSIC works on arbitrary geometries (here: triangular array).
- ESPRIT in this demo is implemented ONLY for ULA as a simple educational baseline.
"""

import argparse
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# Try to import pyroomacoustics, fail gracefully with a clear message
try:
    import pyroomacoustics as pra
except Exception as e:
    raise SystemExit(
        "Pyroomacoustics is required. Install with:\n"
        "  pip install pyroomacoustics numpy scipy matplotlib\n\n"
        f"Original import error: {e}"
    )

C = 343.0  # speed of sound (m/s)

# -------------------------
# Utility helpers
# -------------------------

def db(x):
    return 10*np.log10(np.maximum(x, 1e-12))

def add_awgn(x, snr_db):
    """Add white Gaussian noise to achieve target SNR (per-channel)."""
    rng = np.random.default_rng(0)
    y = np.empty_like(x)
    for ch in range(x.shape[0]):
        s = x[ch]
        ps = np.mean(s**2)
        snr = 10**(snr_db/10)
        pn = ps / snr
        n = rng.normal(0, np.sqrt(pn), size=s.shape)
        y[ch] = s + n
    return y

def equilateral_triangle(side):
    """Return 2D coordinates (x,y) of an equilateral triangle centered at (0,0)."""
    # Vertices at angles 90, 210, 330 degrees around center for symmetry
    R = side / np.sqrt(3)  # circumradius of equilateral triangle
    angles = np.deg2rad([90, 210, 330])
    xy = np.stack([R*np.cos(angles), R*np.sin(angles)], axis=0)  # 2x3
    return xy

def center_array(xy, center):
    """Shift 2xM mic coordinates so centroid is at 'center' (2,)"""
    centroid = np.mean(xy, axis=1, keepdims=True)
    return xy - centroid + np.array(center).reshape(2,1)

def make_ula(M, d):
    """2xM coordinates for a horizontal ULA centered at origin."""
    xs = (np.arange(M) - (M-1)/2.0) * d
    ys = np.zeros_like(xs)
    return np.vstack([xs, ys])

def wideband_chirp(fs, duration, f0=300, f1=4000):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    s = sig.chirp(t, f0=f0, t1=duration, f1=f1, method='logarithmic')
    # apply Hann window fade-in/out to avoid clicks
    win = sig.windows.tukey(len(s), alpha=0.1)
    return s * win

def plot_beam_pattern(mic_xy, fs, nfft=512, steering_az=None, title="Delay-and-sum beam pattern"):
    """
    Very simple far-field delay-and-sum beampattern at one frequency bin (broadband approximation by max over freqs).
    This is for *intuition*, not a rigorous design (MVDR etc.).
    """
    if steering_az is None:
        steering_az = np.linspace(0, 2*np.pi, 360, endpoint=False)
    # Evaluate array response magnitude across azimuths for a small set of frequencies
    freqs = np.linspace(300, 4000, 8)  # coarse
    responses = []
    for az in steering_az:
        # unit vector towards az
        k = np.array([np.cos(az), np.sin(az)])  # 2D
        # array response averaged over freqs
        val = 0.0
        for f in freqs:
            # phase delay per mic
            phase = 2*np.pi*f*(mic_xy.T @ k)/C  # Mx1
            w = np.exp(-1j*phase)  # steering weights (conjugate plane-wave)
            val += np.abs(np.sum(w))  # DAS magnitude
        responses.append(val/len(freqs))
    responses = np.array(responses)
    plt.figure()
    plt.plot(np.rad2deg(steering_az), 20*np.log10(responses/np.max(responses)))
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Normalized response (dB)")
    plt.title(title)
    plt.grid(True)

# -------------------------
# ESPRIT (ULA-only demo)
# -------------------------

def esprit_ula(signal_matrix, d, fs, num_src=1, f0=1000.0):
    """
    Extremely simplified ESPRIT demo for a *narrowband* snapshot near frequency f0.
    Not production-grade. ULA only.
    signal_matrix: M x T (channels x samples), real-valued.
    Returns estimated azimuth(s) in radians (assuming 2D far-field).
    """
    M, T = signal_matrix.shape
    # Create analytic signal via Hilbert and mix to baseband near f0 to approximate narrowband snapshot
    analytic = sig.hilbert(signal_matrix, axis=1)
    t = np.arange(T)/fs
    bb = analytic * np.exp(-1j*2*np.pi*f0*t)  # mix down
    # spatial covariance
    R = (bb @ bb.conj().T) / T  # MxM
    # shift invariance using two subarrays
    X1 = R[:-1, :-1]
    X2 = R[1:, 1:]
    # Solve generalized eigen problem X2 v = lambda X1 v
    evals, evecs = np.linalg.eig(np.linalg.pinv(X1) @ X2)
    # pick num_src largest eigenvalues in magnitude
    idx = np.argsort(-np.abs(evals))[:num_src]
    lambdas = evals[idx]
    # phase progression relates to spatial frequency
    # lambda ~ exp(j*2*pi*f0*(d*cos(theta)/C))
    phase = np.angle(lambdas)
    cos_theta = (phase * C) / (2*np.pi*f0*d)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    thetas = np.arccos(cos_theta)  # radians
    # map to azimuth in [0, pi]; extend symmetry if needed by prior info
    return thetas

# -------------------------
# MUSIC with Pyroomacoustics
# -------------------------

def music_doa(signals, fs, mic_xy, num_src=1, nfft=1024):
    """
    Compute MUSIC DOA using pyroomacoustics.doa.MUSIC for 2D azimuth estimation.
    signals: M x T (channels x samples)
    mic_xy: 2 x M
    """
    R = mic_xy  # pra expects 2xM for 2D arrays
    stft = pra.transform.stft.STFT(nfft, hop=nfft//2, analysis_window=np.hanning(nfft))
    # build multi-channel STFT (F x T_frames x M) as expected by doa
    Xs = []
    for m in range(signals.shape[0]):
        stft.analysis(signals[m])
        Xs.append(stft.STFT)
        stft.refresh()
    X = np.stack(Xs, axis=-1)  # F x T_frames x M

    doa = pra.doa.MUSIC(R, fs, nfft, c=C, num_src=num_src, azimuth=np.linspace(0, 2*np.pi, 360, endpoint=False))
    doa.locate_sources(X)  # fills doa.azimuth_recon
    return doa

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Pyroomacoustics DOA demo with triangular mic array + MUSIC")
    ap.add_argument("--fs", type=int, default=16000, help="Sample rate [Hz]")
    ap.add_argument("--duration", type=float, default=2.0, help="Signal duration [s]")
    ap.add_argument("--room", nargs=2, type=float, default=[6.0, 5.0], help="Room size [Lx Ly] in meters (2D shoebox)")
    ap.add_argument("--rt60", type=float, default=0.0, help="Reverberation time [s]; 0 means anechoic-like")
    ap.add_argument("--array", type=str, choices=["triangular","ula"], default="triangular", help="Array geometry")
    ap.add_argument("--side", type=float, default=0.05, help="Triangular side length [m]")
    ap.add_argument("--mics", type=int, default=3, help="# mics for ULA (ignored for triangular)")
    ap.add_argument("--d", type=float, default=0.05, help="Mic spacing for ULA [m]")
    ap.add_argument("--center", nargs=2, type=float, default=[3.0, 2.5], help="Array center position [x y] in meters")
    ap.add_argument("--src", nargs=2, type=float, default=[2.0, 1.5], help="Source position [x y] in meters")
    ap.add_argument("--snr", type=float, default=15.0, help="Target SNR after propagation [dB]")
    ap.add_argument("--num_src", type=int, default=1, help="# of sources for DOA")
    ap.add_argument("--nfft", type=int, default=1024, help="FFT size for STFT/DOA")
    ap.add_argument("--plot", action="store_true", help="Show plots")
    args = ap.parse_args()

    fs = args.fs
    Lx, Ly = args.room

    # Absorption/RT60 handling
    if args.rt60 > 0:
        # Compute absorption from Sabine's formula (approximate in 2D via 3D helper using height=3m)
        height = 3.0
        e_absorption, _ = pra.inverse_sabine(args.rt60, [Lx, Ly, height])
        absorption = e_absorption
        max_order = pra.parameters.parameters.MAX_ORDER_DEFAULT
    else:
        absorption = 0.0
        max_order = 0  # anechoic-like via image method order=0

    # Build room
    room = pra.ShoeBox([Lx, Ly], fs=fs, absorption=absorption, max_order=max_order)

    # Microphone geometry
    if args.array == "triangular":
        xy = equilateral_triangle(args.side)          # 2 x 3
    else:
        xy = make_ula(args.mics, args.d)              # 2 x M

    xy = center_array(xy, args.center)                # place in room center (or user-specified)

    # Pyroomacoustics wants 3xM mic positions for 3D; for 2D we set z=1.2m (ear height)
    M = xy.shape[1]
    mic_z = np.ones((1, M)) * 1.2
    mic_xyz = np.vstack([xy, mic_z])                  # 3 x M
    mic_array = pra.MicrophoneArray(xy, fs=fs)
    room.add_microphone_array(mic_array)

    # Source
    src_xy = np.array([args.src[0], args.src[1], 1.2])  # same height
    signal = wideband_chirp(fs, args.duration, f0=300, f1=min(0.45*fs, 6000))
    room.add_source(src_xy, signal=signal)

    # Simulate
    room.simulate()

    # Acquire mic signals (M x T)
    sigs = room.mic_array.signals

    # Add noise to target SNR
    sigs_noisy = add_awgn(sigs, args.snr)

    # (Optional) Simple beam pattern for intuition (computed from geometry only)
    if args.plot:
        plot_beam_pattern(xy, fs, nfft=args.nfft, title=f"Delay-and-sum beampattern ({args.array})")

    # MUSIC
    doa = music_doa(sigs_noisy, fs, xy, num_src=args.num_src, nfft=args.nfft)
    est_az = np.rad2deg(doa.azimuth_recon)
    print(f"[MUSIC] Estimated azimuth(s) [deg]: {np.round(est_az, 1)}")

    # Plot MUSIC pseudospectrum
    if args.plot:
        plt.figure()
        az = np.rad2deg(doa.grid.azimuth)
        ps = doa.Pmusic / np.max(doa.Pmusic)
        plt.plot(az, ps)
        plt.xlabel("Azimuth (deg)")
        plt.ylabel("Normalized MUSIC pseudospectrum")
        plt.title("MUSIC Pseudospectrum")
        plt.grid(True)

    # (Optional) ESPRIT baseline for ULA
    if args.array == "ula":
        try:
            thetas = esprit_ula(sigs_noisy, d=args.d, fs=fs, num_src=args.num_src, f0=1000.0)
            print(f"[ESPRIT-ULA] Estimated azimuth(s) [deg]: {np.round(np.rad2deg(thetas), 1)}")
        except Exception as e:
            print("[ESPRIT-ULA] Failed:", e)

    if args.plot:
        plt.show()

if __name__ == "__main__":
    main()
