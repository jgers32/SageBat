"""
Plot the 1-second detected clips from SageBat's detector.

Two views:
  * grid    - thumbnail spectrogram of every clip, for scanning what the
              detector actually caught
  * detail  - single clip: waveform envelope + spectrogram, for looking at
              chirp structure

Resolution defaults are set for millisecond-scale echolocation calls, not
for soundscape averaging: 512-sample window at 384 kHz is ~1.3 ms, hop 128
is ~0.33 ms, so a 2 ms chirp spans roughly 6 frames.
"""

import glob
import os

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# =========================================================
# Configuration
# =========================================================
CLIP_DIR = "/home/jgersey/SageBat/detected_clips"

N_FFT = 512          # ~1.33 ms at 384 kHz
HOP = 128            # ~0.33 ms
F_LO, F_HI = 10_000, 120_000     # display range, Hz
DB_RANGE = 60        # dynamic range below peak to show

WINDOW = np.hanning(N_FFT)


# =========================================================
# Spectrogram
# =========================================================
def clip_spectrogram(path, n_fft=N_FFT, hop=HOP, f_lo=F_LO, f_hi=F_HI):
    """Return (spec_db, freqs_hz, times_s, sr) for one clip."""
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])

    if len(x) < n_fft:
        raise ValueError(f"{os.path.basename(path)}: {len(x)} samples < n_fft {n_fft}")

    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
    keep = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]

    n = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, n_fft), strides=(x.strides[0] * hop, x.strides[0])
    )
    power = np.abs(np.fft.rfft(frames * WINDOW, axis=1)) ** 2
    power = power[:, keep]

    db = 10 * np.log10(np.maximum(power, power.max() * 1e-12))
    db -= db.max()                       # peak-referenced

    times = np.arange(n) * hop / sr
    return db, freqs[keep], times, sr


# =========================================================
# Detail view: one clip
# =========================================================
def plot_clip(path, f_lo=F_LO, f_hi=F_HI):
    db, freqs, times, sr = clip_spectrogram(path, f_lo=f_lo, f_hi=f_hi)
    x, _ = sf.read(path, dtype="float64", always_2d=True)
    x = x[:, 0]
    t_wave = np.arange(len(x)) / sr

    fig, ax = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )

    ax[0].plot(t_wave, x, lw=0.4, color="steelblue")
    ax[0].set_ylabel("amplitude")
    ax[0].set_title(os.path.basename(path))
    ax[0].grid(alpha=0.3)

    im = ax[1].pcolormesh(
        times, freqs / 1000, db.T,
        shading="auto", cmap="inferno", vmin=-DB_RANGE, vmax=0,
    )
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Freq (kHz)")
    plt.colorbar(im, ax=ax[1], label="dB below peak")

    plt.tight_layout()
    plt.show()

    print(f"sr {sr} Hz   {len(x)} samples ({len(x)/sr:.3f} s)")
    print(f"frame {N_FFT/sr*1000:.2f} ms   hop {HOP/sr*1000:.2f} ms   "
          f"{db.shape[0]} frames x {db.shape[1]} bins")
    print(f"peak amplitude {np.abs(x).max():.4f}")

    # Frequency of peak energy per frame - rough call trace
    loud = db.max(axis=1) > -20
    if loud.any():
        peak_f = freqs[db[loud].argmax(axis=1)] / 1000
        print(f"loud frames: {loud.sum()}   "
              f"peak freq range {peak_f.min():.1f}-{peak_f.max():.1f} kHz")


# =========================================================
# Grid view: many clips
# =========================================================
def plot_grid(paths, ncols=4, f_lo=F_LO, f_hi=F_HI, max_clips=16):
    paths = paths[:max_clips]
    nrows = int(np.ceil(len(paths) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False
    )

    for ax, path in zip(axes.ravel(), paths):
        try:
            db, freqs, times, sr = clip_spectrogram(path, f_lo=f_lo, f_hi=f_hi)
        except ValueError as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=7)
            ax.axis("off")
            continue
        ax.pcolormesh(times, freqs / 1000, db.T, shading="auto",
                      cmap="inferno", vmin=-DB_RANGE, vmax=0)
        ax.set_title(os.path.basename(path), fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes.ravel()[len(paths):]:
        ax.axis("off")

    fig.supxlabel("Time (s)")
    fig.supylabel("Freq (kHz)")
    plt.tight_layout()
    plt.show()


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    clips = sorted(glob.glob(os.path.join(CLIP_DIR, "*.wav")))
    print(f"{len(clips)} clips in {CLIP_DIR}")

    if not clips:
        raise SystemExit("no clips found - check CLIP_DIR")

    for c in clips[:5]:
        info = sf.info(c)
        print(f"  {os.path.basename(c)}: {info.frames} samples, "
              f"{info.samplerate} Hz, {info.duration:.3f} s")

    plot_grid(clips)          # scan everything
    plot_clip(clips[0])       # look at one closely