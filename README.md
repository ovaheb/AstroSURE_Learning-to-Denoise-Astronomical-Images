# AstroSURE: Learning to Denoise Astronomical Images

AstroSURE is a denoising framework designed to train deep learning models to remove noise from astronomical images without requiring access to ground truth data. The models included in this repository were trained on simulated images generated by Galsim, as well as raw data from the Hubble Space Telescope and the Canada-France-Hawaii Telescope. The approach combines Noise2Noise training with SURE (Stein's Unbiased Risk Estimator) training loss.

## Features

- **Noise2Noise Training**: Trains models using pairs of noisy images without needing clean, ground truth images.
- **SURE Training Loss**: Utilizes Stein's Unbiased Risk Estimator for effective denoising without ground truth.
- **Multiple Datasets**: Includes models trained on simulated Galsim images and raw data from Hubble and Canada-France-Hawaii telescopes.

## Installation

To get started with AstroSURE, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/ovaheb/AstroSURE_Learning-to-Denoise-Astronomical-Images.git
cd AstroSURE_Learning-to-Denoise-Astronomical-Images
pip install -r requirements.txt
```

## Usage

```bash
python train.py
```

## Datasets

The following datasets are supported:

- **Galsim**: Simulated astronomical images.
- **Hubble Space Telescope**: Raw observational data.
- **Canada-France-Hawaii Telescope**: Raw observational data.

## Acknowledgements

We would like to thank the developers of Galsim, the Hubble Space Telescope, and the Canada-France-Hawaii Telescope for providing the data used in this project.
