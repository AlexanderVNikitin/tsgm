<div style="text-align:center">
<img src="./docs/_static/logo.png">
</div>

![unit-tests](https://github.com/AlexanderVNikitin/tsgm/actions/workflows/test.yml/badge.svg?event=push)


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

## Datasets
TSGM provides API for convenient use of many time-series datasets. The comprehensive list of the datasets in the [documentation](https://tsgm.readthedocs.io/en/latest/guides/datasets.html)

## Contributing
We appreciate all contributions. To learn more, please check [CONTRIBUTING.md](CONTRIBUTING.md).

## License
[Apache License 2.0](LICENSE)