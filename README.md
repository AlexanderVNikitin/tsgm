<div style="text-align:center">
<img src="https://github.com/AlexanderVNikitin/tsgm/raw/main/docs/_static/logo.png">
</div>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l2VB6eUwvrxyu8iB30faGiQM5AKthc82?usp=sharing)
[![Pypi version](https://img.shields.io/pypi/v/tsgm)](https://pypi.org/project/tsgm/)
[![unit-tests](https://github.com/AlexanderVNikitin/tsgm/actions/workflows/test.yml/badge.svg?event=push)](https://github.com/AlexanderVNikitin/tsgm/actions?query=workflow%3ATests+branch%3Amain)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/AlexanderVNikitin/tsgm/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/AlexanderVNikitin/tsgm)](https://github.com/AlexanderVNikitin/tsgm/commits/main)

[![arXiv](https://img.shields.io/badge/arXiv-2305.11567-b31b1b.svg)](https://arxiv.org/abs/2305.11567)
[![codecov](https://codecov.io/gh/AlexanderVNikitin/tsgm/branch/main/graph/badge.svg?token=UD38ANZ0M1)](https://codecov.io/gh/AlexanderVNikitin/tsgm)

# Time Series Generative Modeling (TSGM)

[Documentation](https://tsgm.readthedocs.io/en/latest/) |
[Tutorials](https://github.com/AlexanderVNikitin/tsgm/tree/main/tutorials)

## About TSGM

TSGM is an open-source framework for synthetic time series generation and augmentation. 

The framework can be used for:
- creating synthetic data, using historical data, black-box models, or a combined approach,
- augmenting time series data,
- evaluating synthetic data with respect to consistency, privacy, downstream performance, and more.


## Install TSGM
```bash
pip install tsgm
```

### TensorFlow 2.16 or higher
TensorFlow 2.16 uses Keras 3.0. In order to use TensorFlow Keras backend, you need to additionally install `tf-keras`:
```bash
pip install tf-keras
```

### M1 and M2 chips:
To install `tsgm` on Apple M1 and M2 chips:
```bash
# Install tensorflow
conda install -c conda-forge tensorflow=2.9.1

# Install tsgm without dependencies
pip install tsgm --no-deps

# Install rest of the dependencies (separately here for clarity)
conda install tensorflow-probability scipy antropy statsmodels dtaidistance networkx optuna prettytable seaborn scikit-learn yfinance tqdm
```


## Train your generative model

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l2VB6eUwvrxyu8iB30faGiQM5AKthc82?usp=sharing) Introductory Tutorial "[Getting started with TSGM](https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/GANs/cGAN.ipynb)"
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Vw9t4TlI1Nek_t6bMPyKcPPPqCiXfOK3?usp=sharing) Tutorial on using [Time Series Augmentations](https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/augmentations.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hubtddSX94KyLzuCTwmU6pAFBgBeiEB-?usp=sharing) Tutorial on [Evaluation of Synthetic Time Series Data](https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/evaluation.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wpf9WeNVj5TkUcPF6EavVx-hUCOfyvUd?usp=sharing) Tutorial on using [Multiple GPUs or TPU with TSGM](https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/Using%20Multiple%20GPUs%20or%20TPU.ipynb)

For more examples, see [our tutorials](./tutorials).

```python
import tsgm

# ... Define hyperparameters ...
# dataset is a tensor of shape n_samples x seq_len x feature_dim

# Zoo contains several prebuilt architectures: we choose a conditional GAN architecture
architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
    seq_len=seq_len, feat_dim=feature_dim,
    latent_dim=latent_dim, output_dim=0)
discriminator, generator = architecture.discriminator, architecture.generator

# Initialize GAN object with selected discriminator and generator
gan = tsgm.models.cgan.GAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)
gan.fit(dataset, epochs=N_EPOCHS)

# Generate 100 synthetic samples
result = gan.generate(100)
```


## Getting started

We provide:
* [Documentation](https://tsgm.readthedocs.io/en/latest/) with a complete overview of the implemented methods,
* [Tutorials](https://github.com/AlexanderVNikitin/tsgm/tree/main/tutorials) that describe practical use-cases of the framework.


#### For contributors
```bash
git clone github.com/AlexanderVNikitin/tsgm
cd tsgm
pip install -e .
```

Run tests:
```bash
python -m pytest
```

To check static typing:
```bash
mypy
```

## CLI
We provide two CLIs for convenient synthetic data generation:
- `tsgm-gd` generates data by a stored sample,
- `tsgm-eval` evaluates the generated time series.

Use `tsgm-gd --help` or `tsgm-eval --help` for documentation.


## Datasets
TSGM provides API for convenient use of many time-series datasets (currently more than 15 datasets). The comprehensive list of the datasets in the [documentation](https://tsgm.readthedocs.io/en/latest/guides/datasets.html)

## Augmentations
TSGM provides a number of time series augmentations.

| Augmentation  | Class in TSGM | Reference     |
| ------------- | ------------- | ------------- |
| Gaussian Noise / Jittering  | `tsgm.augmentations.GaussianNoise` | -  |        
| Slice-And-Shuffle  | `tsgm.augmentations.SliceAndShuffle` | - |
| Shuffle features  | `tsgm.augmentations.Shuffle` | - |
| Magnitude warping  | `tsgm.augmentations.MagnitudeWarping` | [Data Augmentation of Wearable Sensor Data for Parkinson’s Disease Monitoring using Convolutional Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3136755.3136817) |
| Window warping  | `tsgm.augmentations.WindowWarping` | [Data Augmentation for Time Series Classification using Convolutional Neural Networks](https://shs.hal.science/halshs-01357973/document) |


## Contributing
We appreciate all contributions. To learn more, please check [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing
If you find this repo useful, please consider citing our paper:
```
@article{
  nikitin2023tsgm,
  title={TSGM: A Flexible Framework for Generative Modeling of Synthetic Time Series},
  author={Nikitin, Alexander and Iannucci, Letizia and Kaski, Samuel},
  journal={arXiv preprint arXiv:2305.11567},
  year={2023}
}
```

## License
[Apache License 2.0](LICENSE)
