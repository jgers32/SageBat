import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')  # No GUI backend
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters
N_TIME_FRAMES = 15000
N_FREQ_BINS = 484
WINDOW_S = 0.5
FMIN = 1000.0
DB_RANGE = 80.0
TARGET_CLIP_SEC = 1.0

def wav_to_fixed_spectrogram_matrix(wav, sr,
                                    duration=TARGET_CLIP_SEC,
                                    n_time_frames=N_TIME_FRAMES,
                                    n_freq_bins=N_FREQ_BINS,
                                    window_s=WINDOW_S,
                                    fmin=FMIN,
                                    db_range=DB_RANGE):
    target_len = int(duration * sr)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)), 'constant')
    elif len(wav) > target_len:
        start = (len(wav) - target_len) // 2
        wav = wav[start:start + target_len]
    fmax = sr / 2.0
    hop_length = int((sr * duration) // n_time_frames)
    n_fft = int(round(window_s * sr))
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_freq_bins, fmin=fmin,
                                       fmax=fmax, window='hann')
    if S.shape[1] > n_time_frames:
        S = S[:, :n_time_frames]
    elif S.shape[1] < n_time_frames:
        S = np.pad(S, ((0, 0), (0, n_time_frames - S.shape[1])), 'constant')
    S_db = librosa.power_to_db(S, ref=np.max)
    spec = ((S_db + db_range) / db_range).astype(np.float32)
    mel_freqs = librosa.mel_frequencies(n_mels=n_freq_bins, fmin=fmin, fmax=fmax)
    return spec.T, mel_freqs  # (n_time_frames, n_freq_bins)

def save_spectrogram_image(spec, freqs, times, out_png):
    plt.figure(figsize=(10, 4))
    extent = (float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1]))
    plt.imshow(spec.T, origin='lower', aspect='auto', cmap='inferno', extent=extent)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

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
            wav = np.mean(wav, axis=1)
        spec, freqs = wav_to_fixed_spectrogram_matrix(wav, sr)
        duration = TARGET_CLIP_SEC
        times = np.linspace(0, duration, num=N_TIME_FRAMES, endpoint=False)
        out_npz = out_dir / f"{wav_path.stem}.npz"
        np.savez_compressed(out_npz, matrix=spec, freqs=freqs)
        out_png = out_dir / f"{wav_path.stem}.png"
        save_spectrogram_image(spec, freqs, times, out_png)
        print(f"  -> saved {out_npz.name} and {out_png.name}")
    print("Done.")

if __name__ == '__main__':
    main()