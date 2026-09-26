# SPATIALLY-ADAPTIVE-CONVOLUTIONS-VIA-ATTENTION-ROUTED-EXPERT-MIXTURES

Official implementation of **SPATIALLY-ADAPTIVE-CONVOLUTIONS-VIA-ATTENTION-ROUTED-EXPERT-MIXTURES**.

This repository contains the code for our study of spatially adaptive
convolution using a fixed bank of structurally heterogeneous 3D convolutional
experts and spatially varying attention-based routing.

Rather than dynamically modifying convolutional kernel parameters, our
formulation keeps the expert operators fixed and learns spatially varying
weights over their outputs:

\[
y(p) = \sum_{k=1}^{K} w_k(p) F_k(x)(p),
\]

where \(F_k\) denotes the transformation performed by expert \(k\), and
\(w_k(p)\) is its spatially varying routing weight.

We use 3D MRI reconstruction as a testbed and focus on characterizing the
resulting expert responses, spatial routing behavior, structural associations,
and frequency-dependent responses.

---

## Overview

Conventional convolution applies the same learned transformation across the
spatial extent of a feature map. Our approach instead provides a fixed bank of
heterogeneous convolutional transformations and learns how their contributions
should vary spatially.

The expert bank contains operators with different local structures:

| Expert | Operator | Intended structural bias |
|--------|----------|---------------------------|
| \(E_0\) | \(1\times3\times3\) average pooling + \(1\times1\times1\) convolution | In-plane smoothing |
| \(E_1\) | \(1\times1\times1\) convolution | Pointwise transformation |
| \(E_2\) | \(1\times3\times3\) convolution | In-plane spatial mixing |
| \(E_3\) | \(3\times1\times1\) convolution | Through-plane/depth mixing |
| \(E_4\) | \(1\times1\times1\) convolution | Independent pointwise transformation |

The encoder uses localized spatial windows to construct routing tokens.
Self-attention processes these tokens and produces expert logits, which are
interpolated to obtain a dense spatial routing field.

The resulting expert contributions are combined using a soft spatial mixture.

### Key properties

- Fixed convolutional expert parameters
- Structurally heterogeneous 3D operators
- Spatially varying expert composition
- Attention-based routing in the encoder
- Soft rather than hard expert assignment
- Analysis of expert responses with respect to image structure and frequency
- Expert-composition and expert-removal ablations
- 3D MRI reconstruction evaluation

---
## Repository Structure

```text
├── configs/
│   ├── ae.py
│   ├── data.py
│
├── data/
│   ├── __init__.py
│   └── dataset.py
│
├── models/
│   ├── __init__.py
│   ├── ae.py
│   ├── ShapedEncoder3D.py
│   └── Decoder.py
│
├── train/
│   ├── __init__.py
│   ├── train.py
│   └── checkpoint.py
│
├── visualization/
│   ├── __init__.py
│   ├── routing.py
│   ├── wta.py
│   ├── expert_analysis.py
│   └── freq_response.py
│
├── scripts/
│   ├── train_ae.py
│   └── analyze_ae.py
│
├── notebooks/
├── figures/
├── README.md
├── CITATION.cff
└── LICENSE
```
