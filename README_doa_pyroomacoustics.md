# DOA Simulation & MUSIC Demo (Pyroomacoustics)

This package contains a self-contained script to simulate a room, a **triangular microphone array**, and run **MUSIC** for DOA estimation. It also includes an optional **ESPRIT** baseline for **ULA** only (educational, simplified).

## Files
- `doa_pyroomacoustics.py` — main script
- `requirements.txt` — dependencies

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (examples)
Anechoic-like, triangular array (3 mics, side=5 cm), 2 s chirp:
```bash
python doa_pyroomacoustics.py --fs 16000 --duration 2.0 --room 6 5 --rt60 0.0   --array triangular --side 0.05 --center 3.0 2.5 --src 2.0 1.5 --snr 15 --plot
```

Reverberant (RT60=0.3 s), triangular array:
```bash
python doa_pyroomacoustics.py --rt60 0.3 --array triangular --side 0.05 --plot
```

ULA + ESPRIT baseline (for comparison):
```bash
python doa_pyroomacoustics.py --array ula --mics 6 --d 0.04 --plot
```

## What the script produces
- **Console output**: Estimated azimuth(s) from MUSIC (and ESPRIT if ULA).
- **Plots** (with `--plot`):
  - Delay-and-sum **beam pattern** (intuition).
  - **MUSIC pseudospectrum** with peaks at estimated DOA.

## Notes for your thesis
- You can cite this pipeline in **Implementation** as your reusable framework.
- Vary `--rt60`, `--snr`, and room size to build your **Experimental Results** tables (angular error vs. conditions).
- The ESPRIT function here is simplified and meant as a conceptual baseline only; MUSIC is your main method for all geometries, especially **triangular**.

