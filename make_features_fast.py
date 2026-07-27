#!/usr/bin/env python3
import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# Parameters
N_TIME_FRAMES = 15000   # matrix rows: time frames
N_FREQ_BINS = 484      # matrix columns: freq bins
WINDOW_S = 0.02         # analysis window length per row (seconds) - reduced for speed
FMIN = 1000.0          # Hz, low cut
DB_RANGE = 80.0        # dB range for scaling
TARGET_CLIP_SEC = 1.0  # duration of each clip (seconds)

def wav_to_fixed_spectrogram_matrix(wav, sr,
                                    duration=TARGET_CLIP_SEC,
                                    n_time_frames=N_TIME_FRAMES,
                                    n_freq_bins=N_FREQ_BINS,
                                    window_s=WINDOW_S,
                                    fmin=FMIN,
                                    db_range=DB_RANGE):
    """
    Convert a waveform (already loaded) to a fixed-size mel-spectrogram matrix.
    Returns spectrogram (n_time_frames, n_freq_bins) and mel frequencies.
    """
    # Ensure we have exactly duration * seconds of audio (pad or trim)
    target_len = int(duration * sr)
    if len(wav) < target_len:
        # pad with zeros at the end
        wav = np.pad(wav, (0, target_len - len(wav)), 'constant')
    elif len(wav) > target_len:
        # trim to center
        start = (len(wav) - target_len) // 2
        wav = wav[start:start + target_len]
    # Now compute spectrogram
    fmax = sr / 2.0  # Nyquist
    hop_length = int((sr * duration) // n_time_frames)
    n_fft = int(round(window_s * sr))
    # mel spectrogram
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_freq_bins, fmin=fmin,
                                       fmax=fmax, window='hann')
    # Force shape
    if S.shape[1] > n_time_frames:
        S = S[:, :n_time_frames]
    elif S.shape[1] < n_time_frames:
        S = np.pad(S, ((0, 0), (0, n_time_frames - S.shape[1])), 'constant')
    # Convert to dB and normalize to [0,1]
    S_db = librosa.power_to_db(S, ref=np.max)
    spec = ((S_db + db_range) / db_range).astype(np.float32)
    mel_freqs = librosa.mel_frequencies(n_mels=n_freq_bins, fmin=fmin, fmax=fmax)
    return spec.T, mel_freqs  # (n_time_frames, n_freq_bins)

def main():
    clip_dir = Path('detected_clips')
    out_dir = Path('features')
    out_dir.mkdir(exist_ok=True)
    wav_files = sorted(clip_dir.glob('*_det*.wav'))
    print(f"Found {len(wav_files)} clips")
    for wav_path in wav_files:
        print(f"Processing {wav_path.name}")
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)  # mono
        # Compute spectrogram matrix
        spec, freqs = wav_to_fixed_spectrogram_matrix(wav, sr)
        print(f"  -> spec shape: {spec.shape}")
        # Save npz
        out_npz = out_dir / f"{wav_path.stem}.npz"
        np.savez_compressed(out_npz, matrix=spec, freqs=freqs)
        print(f"  -> saved {out_npz.name}")
    print("Done.")

if __name__ == '__main__':
    main()