# Faraday Wave Data — Liquid Metal Interfacial Tension Measurement

## Contents

### Image Processing Scripts
- `v5_improved.py` — Main pipeline: zero-padded FFT + physical scoring peak detection
- `v5_prominence.py` — Cross-validation: prominence-based peak detection
- `container_detector.py` — Automatic container center/radius detection
- `exposure_check.py` — Exposure quality check
- `compare_exposures.py` — Cross-experiment exposure comparison

### Processed Data (`实验结果汇总/`)
Radial power spectra, time-averaged images, and background-subtracted images for each tested frequency (30–80 Hz, 5 Hz steps).

### Raw Images
Raw camera images (~11 GB, 1000 fps × 1000 frames per condition) are available from the corresponding author upon reasonable request.

## Citation
Wang, J.; Liu, D.; He, S.; et al. Measuring Liquid Metal Interfacial Tension by Faraday Wave Dispersion and Zero-Padded FFT. *Langmuir*, submitted, 2026.
