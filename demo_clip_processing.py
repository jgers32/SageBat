import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load one of the detected clips
clip_path = "detected_clips/20260414_100939_det0000.wav"
y, sr = sf.read(clip_path)
print(f"Clip shape: {y.shape}, sample rate: {sr} Hz")
print(f"Duration: {len(y)/sr:.3f} s")

# Compute mel-spectrogram with parameters similar to the notebook but using the clip's duration
# In the notebook they used: n_time_frames=15000, n_freq_bins=484, window_s=0.5, fmin=1000, duration=600s
# For a short clip we can compute a spectrogram with a reasonable hop length to get a time-frequency image.
# Let's use a hop length that gives about 100 time frames (adjustable).
n_fft = 2048
hop_length = 512  # samples
n_mels = 484
fmin = 1000
fmax = sr / 2

S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                   n_mels=n_mels, fmin=fmin, fmax=fmax,
                                   window='hann')
S_db = librosa.power_to_db(S, ref=np.max)

print(f"Mel spectrogram shape: {S_db.shape} (freq bins, time frames)")

# Plot
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel',
                         fmin=fmin, fmax=fmax)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram of detected clip')
plt.tight_layout()
plt.show()

# Also compute the fixed-size matrix as in the notebook (if we wanted to match exactly)
def wav_to_fixed_spectrogram_matrix(audio_path, duration=None, n_time_frames=15000, n_freq_bins=484,
                                   window_s=0.5, fmin=1000, db_range=80.0):
    import librosa
    sr = librosa.get_samplerate(audio_path)
    fmax = sr / 2
    # If duration not provided, use the actual length of the audio
    if duration is None:
        # we need to load the audio to know its length; but we already have y and sr
        duration = len(y) / sr
    hop_length = int((sr * duration) // n_time_frames)
    n_fft = int(round(window_s * sr))
    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    # Pad or truncate to exactly duration*sr samples
    target_len = int(duration * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_freq_bins, fmin=fmin, fmax=fmax,
                                       window='hann')
    if S.shape[1] > n_time_frames:
        S = S[:, :n_time_frames]
    elif S.shape[1] < n_time_frames:
        S = np.pad(S, ((0, 0), (0, n_time_frames - S.shape[1])))
    S_db = librosa.power_to_db(S, ref=np.max, top_db=db_range)
    spec = ((S_db + db_range) / db_rate).astype(np.float32)
    mel_freqs = librosa.mel_frequencies(n_mels=n_freq_bins, fmin=fmin, fmax=fmax)
    return spec.T, mel_freqs  # (n_time_frames, n_freq_bins)

# Note: The above function has a typo (db_rate vs db_range). Let's fix it quickly:
def wav_to_fixed_spectrogram_matrix_correct(audio_path, duration=None, n_time_frames=15000, n_freq_bins=484,
                                           window_s=0.5, fmin=1000, db_range=80.0):
    import librosa
    sr = librosa.get_samplerate(audio_path)
    fmax = sr / 2
    if duration is None:
        # we need to load to get duration; we'll just load full
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        duration = len(y) / sr
    hop_length = int((sr * duration) // n_time_frames)
    n_fft = int(round(window_s * sr))
    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    target_len = int(duration * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_freq_bins, fmin=fmin, fmax=fmax,
                                       window='hann')
    if S.shape[1] > n_time_frames:
        S = S[:, :n_time_frames]
    elif S.shape[1] < n_time_frames:
        S = np.pad(S, ((0, 0), (0, n_time_frames - S.shape[1])))
    S_db = librosa.power_to_db(S, ref=np.max, top_db=db_range)
    spec = ((S_db + db_range) / db_range).astype(np.float32)  # scale to [0,1]
    mel_freqs = librosa.mel_frequencies(n_mels=n_freq_bins, fmin=fmin, fmax=fmax)
    return spec.T, mel_freqs

# Try on the clip
spec, freqs = wav_to_fixed_spectrogram_matrix_correct(clip_path)
print(f"Fixed matrix shape: {spec.shape}")
print(f"Frequency range: {freqs[0]:.0f} - {freqs[-1]:.0f} Hz")