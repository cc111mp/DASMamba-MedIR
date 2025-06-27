# Directional Adaptive Shuffle-Based Visual State-Space Models for Medical Image Restoration (MICCAI 2025)
PyTorch implementation of Directional Adaptive Shuffle Mamba (MICCAI 2025) for MRI SR, CT denoising &amp; PET synthesis (coming soon)


> **Abstract:**
Medical image restoration (MedIR) demands precise modeling of anisotropic spatial dependencies, where directional anatomical patterns are frequently degraded by conventional methods. We propose Directional Adaptive Shuffle Mamba (DASMamba), a state-space model architecture that addresses this challenge through two novel components: (1) the Directional Adaptive Shuffle Module (DASM), which captures long-range dependencies via directional adaptive random shuffle and selective scanning, and (2) the Dual-path Feedforward Network (DPFN), enhancing feature representation through multi-scale learning and dynamic channel fusion. By integrating these modules into a hierarchical U-shaped architecture, DASMamba achieves state-of-the-art performance on MRI super-resolution, CT denoising, and PET synthesis tasks while maintaining linear computational complexity. Our framework’s ability to preserve diagnostically critical structural details underscores its clinical value.

## Acknowledgment  
This repository builds upon the code and dataset provided by the authors of [Restore-RWKV](https://github.com/Yaziwel/Restore-RWKV).  
We thank them for making their work publicly available. Thanks to the authors for their excellent work.
