# 🥈 Silver Medal Solution — Axell AI Super-Resolution Contest (SIGNATE 2024)

**Competition:** [AI Super-Resolution Model Challenge](https://signate.jp) by Axell AI (株式会社アクセル)  
**Metric:** PSNR (Peak Signal-to-Noise Ratio)  
**Constraint:** Inference ≤ 0.035 s/image on Tesla T4 (enables real-time 4× upscaling of 1024×1024 → 4096×4096 at 60 fps)  
**Participants:** 232 teams, 1 774 total submissions

## Task

Upscale images 4× (1024×1024 → 4096×4096) under a strict real-time latency budget using a model ≤ 1 GB, evaluated on average PSNR over a held-out test set.

## Model architecture

The model is an [ESPCN](https://github.com/leftthomas/ESPCN)-style network (`ESPCN4x` in `train.py`) with several modifications that improved both speed and quality:

- **Sub-pixel convolution (PixelShuffle ×4)** — keeps all computation at low resolution and reshuffles channels into spatial pixels, which is significantly faster than transposed convolutions or bilinear upsampling.
- **Residual / delta learning** — the network predicts a *correction* on top of a bicubic baseline (`base + tanh(nn(x))`). This reduces the learning burden and stabilizes early training.
- **3-channel RGB processing** — operating directly in RGB (rather than a luminance-only YCbCr approach) avoided colour fringing artefacts and simplified the pipeline.
- **PReLU activations** — learnable negative slopes gave a consistent PSNR improvement over ReLU throughout the network.
- **Mixed-precision training (AMP)** — enabled larger batch sizes and faster iteration without quality loss.

## Data

The organiser provided 851 high-resolution training images. Adding external data gave large accuracy boost:

| Source | Size |
|---|---|
| Competition images | ~850 |
| ImageNet validation (filtered ≥ 512 px) | ~25 k |
| ImageNet train (filtered large) | ~100 k |
| Japan 160k dataset | ~160 k |
| Unsplash / Flickr high-res | variable |

Random 512×512 crops with horizontal/vertical flips were applied at training time. A model soup (weight averaging of top checkpoints) was used for the final submission.
