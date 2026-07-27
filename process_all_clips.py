import os
import numpy as np
import librosa
from tqdm import tqdm

# Parameters from preprocessing.ipynb
N_TIME_FRAMES = 15000   # matrix rows: time frames / n_samples
N_FREQ_BINS = 484       # matrix columns: freq bins / n_features
DURATION_S = 1.0        # expected clip length in seconds (1 second clips)
WINDOW_S = 0.5          # analysis window length per row, seconds
FMIN = 1000             # Hz, cut very-low-frequency noise/wind
DB_RANGE = 80.0

def wav_to_fixed_spectrogram_matrix(
    audio_path,
    duration=DURATION_S,
    n_time_frames=N_TIME_FRAMES,
    n_freq_bins=N_FREQ_BINS,
    window_s=WINDOW_S,
    fmin=FMIN,
    db_range=DB_RANGE,
):
    """
    Native-rate (no resampling) mel-spectrogram, shaped to exactly
    (n_time_frames, n_freq_bins) — i.e. (n_samples, n_features), the
    convention jeongmin's sparse-coding algorithm expects — so a
    384kHz file maps to one fixed-size matrix without throwing away the
    ultrasonic band. Each row is spaced `duration / n_time_frames` seconds
    apart, built from a `window_s`-second analysis window; its columns are
    that window's mel/frequency feature vector.

    Returns
    -------
    spec : np.ndarray, shape (n_time_frames, n_freq_bins), float32 in [0, 1]
    mel_freqs : np.ndarray, center freq (Hz) of each column
    """
    sr = librosa.get_samplerate(audio_path)
    fmax = sr / 2  # keep full native Nyquist range (no ultrasonic content thrown away)

    # hop_length chosen so the clip maps to exactly n_time_frames rows;
    # n_fft set directly from window_s (analysis window per row)
    hop_length = int((sr * duration) // n_time_frames)
    n_fft = int(round(window_s * sr))

    data, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    n_samples = int(duration * sr)
    if len(data) < n_samples:
        data = np.pad(data, (0, n_samples - len(data)))

    S = librosa.feature.melspectrogram(
        y=data, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_freq_bins, fmin=fmin, fmax=fmax,
        window="hann",
    )  # (n_freq_bins, time_frames)

    # librosa's centering can produce a frame off by one; force exact shape
    if S.shape[1] > n_time_frames:
        S = S[:, :n_time_frames]
    elif S.shape[1] < n_time_frames:
        S = np.pad(S, ((0, 0), (0, n_time_frames - S.shape[1])))

    S_db = librosa.power_to_db(S, ref=np.max, top_db=db_range)
    spec = ((S_db + db_range) / db_range).astype(np.float32)

    mel_freqs = librosa.mel_frequencies(n_mels=n_freq_bins, fmin=fmin, fmax=fmax)
    return spec.T, mel_freqs  # (n_samples, n_features) = (n_time_frames, n_freq_bins)

def main():
    base_dir = os.getcwd()
    clips_dir = os.path.join(base_dir, "detected_clips")
    matrices_dir = os.path.join(base_dir, "matrices")
    os.makedirs(matrices_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(clips_dir) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} wav files in {clips_dir}")

    for wav_file in tqdm(wav_files, desc="Processing clips"):
        wav_path = os.path.join(clips_dir, wav_file)
        try:
            matrix, freqs = wav_to_fixed_spectrogram_matrix(wav_path)
            # Save matrix and freqs
            clip_name = os.path.splitext(wav_file)[0]
            matrix_path = os.path.join(matrices_dir, f"{clip_name}.npz")
            np.savez_compressed(matrix_path, matrix=matrix, freqs=freqs)
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()