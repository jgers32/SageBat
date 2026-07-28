# SageBat

## Project Overview

This project has three parts:
- 1. **`ultramic`**: a pip-installable Python package for recording from the 384kHz ultrasound mic, making it simple to add ultrasound sensing to any SAGE (or non-SAGE) edge node

![Ultramic](ultramic.jpg)

- 2. **Edge detection + classification pipeline**: band-filtering to isolate bat chirps at the edge, then off-the-shelf ML (NABat ML) for species-level classification

![Identification](identification.png)

- 3. **Iintrinsic dimensionality**: investigation bat sound spectrograms using a framework called Pairwise Distortion Distribution (PDD). The estimated intrinsic dimension is then used to analyze the latent representation learned by an autoencoder and evaluate reconstruction quality.

![Pipeline](pipeline.png)

---

## 1. `ultramic`: the recording package
**Motivation**: existing deployments needed a simple, dependency-light add-on USB mic library that would "just work" across edge nodes without platform-specific audio backends.

- Dependency-light continuous recording library for the Dodotronic Ultramic384k_evo USB microphone
- Built on `sounddevice` (PortAudio) — runs identically on Linux, macOS, and Windows with no platform-specific audio backend
- Published on PyPI: https://pypi.org/project/ultramic/
- Install: `pip install ultramic`

**Features**:
- Automatic device detection by name (no hardcoded device indices)
- Waits for USB enumeration with a configurable timeout
- Verifies the device supports the target sample rate before recording
- Continuous recording with automatic file rotation (new `.wav` every N seconds)
- Simple CLI entry point, plus a Python API for use in custom scripts
- Zero required dependencies beyond `sounddevice`, `soundfile`, and `numpy`
- Companion `ultramic-viz` CLI for quickly visualizing a recorded `.wav` (waveform, spectrogram, log-scale spectrogram, mel spectrogram)

**Usage** — install and record within seconds:
```bash
ultramic --audio-dir ./audio
# UltraMic found (device index 0)
# UltraMic OK at 384000 Hz - starting recording
```

Or from Python:
```python
from pathlib import Path
from ultramic import wait_for_device, verify_sample_rate, record_continuous

device = wait_for_device(timeout=30)
verify_sample_rate(device, sample_rate=384000)
record_continuous(device, output_dir=Path("./audio"), file_duration_sec=5)
```

Quickly inspect what was recorded:
```bash
ultramic-viz record.wav
```

---

## 2. Enabling SageBat at the Edge
Three-step pipeline for turning continuous 384kHz audio into classified bat detections on-node:

**Step 1 — Chirp detection via band filtering**: energy in the relevant ultrasonic band is tracked over time and thresholded against a running median (median + 8 dB) to flag candidate bat chirps against background cave/environmental noise.

![chirp-detection](chirp-detection.png)


**Step 2 — Clip & write metadata**: each detection is clipped out and logged with metadata, e.g.:
```json
{
  "source_file": "20260414_100939.wav",
  "start_sec": 289.718672,
  "end_sec": 289.720651,
  "duration_s": 0.001979,
  "peak_amplitude": 0.0438232421875,
  "detection_index": 2
}
```


**Step 3 — ML inference**: clipped detections are passed to off-the-shelf classification models. Evaluated/considered: NABat ML, BatDetect2, and BattyBirdNET-Analyzer (a fork of birdnet-team/BirdNET-Analyzer adapted for bat calls).

## Data Used
- **Raw Audio**: 384kHz ultrasound recordings (.wav files) from [NSF Center for Pandemic Insights](https://www.pandemicinsights.org/) dataset

## Species Classification (NABat ML)
Used the pretrained **NABat ML** model (Khalighifar et al., 2022 — deep learning for crowdsourced, automated bat population monitoring) off-the-shelf for species-level classification of detected calls.

- Example output: a detected event spanning 289.6–290.5s containing 14 calls classified as ANPA/COTO
- Cross-referenced against the known bat species list at Pinnacles National Park (14 species, e.g. Western Pipistrelle, Western Red Bat, Hoary Bat, Townsend's Big-eared Bat, Pallid Bat, Big Brown Bat, and several *Myotis* species)
- Top predictions for the example event: **Pallid Bat** (*Antrozous pallidus*, ANPA) and **Townsend's Big-eared Bat** (*Corynorhinus townsendii*, COTO)

![Identification](identification.png)


**Challenges**:
- Very few bat calls present in the deployment dataset
- Low SNR — substantial background cave noise
- Fixed mic location within a large talus cave limiting spatial diversity of recordings

---

## 3. Motivation

High-dimensional spectrograms often lie on a much lower-dimensional manifold. Estimating the intrinsic dimension helps us

- understand the complexity of bat vocalizations,
- determine appropriate latent dimensions,
- evaluate representation efficiency, and
- guide compression and downstream learning.

## Dataset & Model

- Input: A matrix of bat sound spectrograms
- Representation: Flattened spectrogram vectors
- Preprocessing:
  - Normalization
  - Vectorization
  - Train/test split

- Model: Fully connected auto encoder 
  - We adopt a standard fully connected autoencoder as a baseline. The encoder and decoder architectures are application-dependent and can be replaced with more suitable models, such as a Vision Transformer (ViT) for image data.

## Reconstruction Analysis

Compare

- Original spectrogram
- Reconstructed spectrogram
- Reconstruction error (MSE)

Analyze how reconstruction quality changes with latent dimension and compare it with the estimated intrinsic dimension.

Results

![Pipeline](results.png)

The reconstruction performance improves as the latent dimension increases until 30 as the algorithm indicates, with the reconstruction MSE decreasing rapidly at lower dimensions. Beyond a latent dimension of approximately 30, the performance plateaus, indicating that additional latent dimensions provide negligible improvement. This suggests that the intrinsic dimension of the data is around 30, beyond which the autoencoder primarily learns redundant representations.

## Conclusion

The results indicate that a 30-dimensional latent representation is sufficient to capture the essential structure of the data, as reconstruction performance shows negligible improvement beyond this point. This enables compact storage and transmission of data from edge devices, allowing only the 30-dimensional latent vectors to be sent to and stored on the server, thereby reducing communication and storage costs.
Future work should investigate matrix generation methods that better preserve the underlying data structure, as the quality of the generated matrix directly impacts the effectiveness of intrinsic dimension estimation and downstream reconstruction.

## Reference
[1] C.Nuzman, J. Chae, K. Mestav, & U. Mitra, (2026, May). Pairwise Distortion Distribution for Compression and Quantization. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 51-55). IEEE.

[2] Camastra, Francesco, and Antonino Staiano. "Intrinsic dimension estimation: Advances and open problems." Information Sciences 328 (2016): 26-41.



