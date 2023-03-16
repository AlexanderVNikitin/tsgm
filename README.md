<div style="text-align:center">
<img src="./docs/_static/logo.png">
</div>

[![unit-tests](https://github.com/AlexanderVNikitin/tsgm/actions/workflows/test.yml/badge.svg?event=push)](https://github.com/AlexanderVNikitin/tsgm/actions?query=workflow%3ATests+branch%3Amain)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/AlexanderVNikitin/tsgm/blob/main/LICENSE)


# Time Series Generative Modeling Framework (TSGM)

[Documentation](https://tsgm.readthedocs.io/en/latest/) |
[Tutorials](https://github.com/AlexanderVNikitin/tsgm/tree/main/tutorials)

## About TSGM

TSGM is an open-source framework for synthetic time series generation.

The framework can be used for:
- creating synthetic data, using historical data, black-box models, or a combined approach,
- researching new approaches for synthetic data generation,
- comparison of different generative methods for time series.


## Getting started

We provide:
* [Documentation](https://tsgm.readthedocs.io/en/latest/) where we give complete overview of the implemented methods,
* [Tutorials](https://github.com/AlexanderVNikitin/tsgm/tree/main/tutorials) with tutorials that describe practical use-cases of the framework.


## Install TSGM

To install TSGM run
```
pip install tsgm
```

#### For contributors
```bash
git clone github.com/AlexanderVNikitin/tsgm
cd tsgm
make install
```

Run tests:
```bash
make test
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

## Train your generative model
Here, we provide the gist of the framework, for the completed examples see [our tutorials](./tutorials).
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
    nikitin2023gen,
    author = {Alexander Nikitin and Samuel Kaski},
    title = {TSGM --- A Flexible Framework for Synthetic Time Series Generative Modeling},
    year = {2023},
}
```

## License
[Apache License 2.0](LICENSE)
