# Table of Contents

<!-- toc -->

- [Contributing to TSGM](#contributing-to-tsgm)
- [Development](#development)
- [Documenting](#documenting)

<!-- tocstop -->

## Contributing to TSGM
Thanks for your interest and willingness to help!

Please, (1) open an issue for a new feature or comment on the existing issue in [the issue tracker](https://github.com/AlexanderVNikitin/tsgm/issues), (2) open a pull request with the issue (see [the list of pull request](https://github.com/AlexanderVNikitin/tsgm/pulls)).

The easiest way to make a pull request is to fork the repo, see [GitHub documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

## Development
To install TSGM in development mode, first install prerequisites:
```
pip install -r requirements/requirements.txt
pip install -r requirements/tests_requirements.txt
pip install -r requirements/docs_requirements.txt
```

and then, install TSGM in development mode:
```
python setup.py develop
```

To run tests, use pytest, for example:
```
pytest tests/test_cgan.py::test_temporal_cgan
```

To run linters, use:
```
flake8 tsgm/
```

## Documenting
We aim to produce high-quality documentation to help our users to use the library. In your contribution, please edit corresponding documentation pages in [./docs](https://github.com/AlexanderVNikitin/tsgm/tree/main/docs).

To build the documentation, use:
```
cd docs
make html
```
