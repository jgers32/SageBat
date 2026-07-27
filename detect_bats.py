import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
import json
import os

# -------------------------- USER PARAMETERS --------------------------
LOWCUT   = 20_000      # Hz
HIGHCUT  = 100_000     # Hz
FORDER   = 4           # Butterworth order
ENV_LP_CUT = 500.0     # Hz, low‑pass for envelope
MED_WIN_SEC = 0.5      # seconds for running median of envelope
THRESH_FACTOR = 4.0    # multiplier over median
MIN_DUR_SEC = 0.001    # minimum length of a detection (s)
MAX_GAP_SEC = 0.005    # merge gap if shorter than this (s)
PAD_PRE_SEC = 0.010    # seconds of padding before onset
PAD_POST_SEC = 0.010   # seconds of padding after offset
TARGET_CLIP_SEC = 1.0  # <-- Desired length of each saved wav clip (seconds)
OUTPUT_DIR = Path("detected_clips")
SAVE_AS_INT16 = True   # write PCM_16 wav (smaller, standard)
# -------------------------------------------------------------------

def design_bandpass(fs, low, high, order=4):
    nyq = 0.5 * fs
    low_n  = low / nyq
    high_n = high / nyq
    b, a = signal.butter(order, [low_n, high_n], btype='band')
    return b, a

def envelope(sig, fs, lowpass_hz):
    analytic = signal.hilbert(sig)
    amp = np.abs(analytic)
    nyq = 0.5 * fs
    wn = lowpass_hz / nyq
    b, a = signal.butter(2, wn, btype='low')
    env = signal.filtfilt(b, a, amp)
    return env

def detect_segments(env, fs, thresh, min_dur, max_gap):
    above = env > thresh
    # Convert bool to int for diff
    diff = np.diff(above.astype(int))
    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1
    if above[0]:
        onsets = np.insert(onsets, 0, 0)
    if above[-1]:
        offsets = np.append(offsets, len(above))
    starts = onsets
    ends   = offsets
    min_samples = int(np.ceil(min_dur * fs))
    valid = (ends - starts) >= min_samples
    starts = starts[valid]
    ends   = ends[valid]
    max_gap_samples = int(np.ceil(max_gap * fs))
    merged_st = []
    merged_en = []
    s = starts[0]
    e = ends[0]
    for ns, ne in zip(starts[1:], ends[1:]):
        if ns - e <= max_gap_samples:
            e = ne
        else:
            merged_st.append(s)
            merged_en.append(e)
            s, e = ns, ne
    merged_st.append(s)
    merged_en.append(e)
    return list(zip(merged_st, merged_en))

def write_clip_centered(wav_data, sr, start, end, out_path, meta=None):
    """
    Write a clip that is exactly TARGET_CLIP_SEC long,
    centred on the detection [start, end).  If the detection
    is shorter we pad with zeros; if longer we take the
    central TARGET_CLIP_SEC portion.
    """
    det_len = end - start                     # raw detection length in samples
    target_len = int(TARGET_CLIP_SEC * sr)    # desired number of samples

    # Compute the ideal start index so that the detection is centred
    ideal_start = start - (target_len - det_len) // 2
    # Clip to file bounds
    ideal_start = max(0, ideal_start)
    ideal_end   = ideal_start + target_len
    if ideal_end > len(wav_data):
        ideal_end = len(wav_data)
        ideal_start = max(0, ideal_end - target_len)

    segment = wav_data[ideal_start:ideal_end]

    # If we still fell short (e.g. detection at the very beginning/end)
    # pad with zeros to reach target_len
    if len(segment) < target_len:
        pad = target_len - len(segment)
        if ideal_start == 0:
            segment = np.pad(segment, (pad, 0), 'constant')
        else:
            segment = np.pad(segment, (0, pad), 'constant')

    # Write out
    if SAVE_AS_INT16:
        seg_int = np.int16(np.clip(segment, -1.0, 1.0) * 32767)
        sf.write(out_path, seg_int, sr, subtype='PCM_16')
    else:
        sf.write(out_path, segment, sr, subtype='FLOAT')
    if meta:
        json_path = out_path.with_suffix('.json')
        with json_path.open('w') as f:
            json.dump(meta, f, indent=2)

def main(wav_path):
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    wav, sr = sf.read(wav_path, always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    print(f'Loaded {Path(wav_path).name}: {len(wav)/sr:.2f} s @ {sr} Hz')
    
    b, a = design_bandpass(sr, LOWCUT, HIGHCUT, FORDER)
    filtered = signal.filtfilt(b, a, wav)
    
    env = envelope(filtered, sr, ENV_LP_CUT)
    
    win_len = int(MED_WIN_SEC * sr)
    if win_len % 2 == 0:
        win_len += 1
    med = signal.medfilt(env, kernel_size=win_len)
    thresh = med * THRESH_FACTOR
    
    segs = detect_segments(env, sr, thresh, MIN_DUR_SEC, MAX_GAP_SEC)
    print(f'Found {len(segs)} raw segments after thresholding.')
    
    saved = 0
    for i, (s, e) in enumerate(segs):
        t_start = s / sr
        t_end   = e / sr
        peak_amp = np.max(np.abs(wav[s:e]))
        meta = {
            "source_file": str(Path(wav_path).name),
            "start_sec": round(t_start, 6),
            "end_sec":   round(t_end, 6),
            "duration_s": round(t_end - t_start, 6),
            "peak_amplitude": float(peak_amp),
            "detection_index": i
        }
        out_name = f"{Path(wav_path).stem}_det{i:04d}.wav"
        out_path = out_dir / out_name
        write_clip_centered(wav, sr, s, e, out_path, meta=meta)
        saved += 1
    print(f"Wrote {saved} clips to {out_dir}/")

if __name__ == "__main__":
    import sys
    wav_file = sys.argv[1] if len(sys.argv) > 1 else "20260414_100939.wav"
    main(wav_file)