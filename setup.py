import os
from setuptools import setup
from setuptools import find_packages


# Function to read version from __version__.py
def get_version():
    with open(os.path.join(os.path.dirname(__file__), 'tsgm/version.py')) as f:
        exec(f.read())
    return locals()['__version__']


name = "tsgm"
version = get_version()

keywords = [
    "machine learning",
    "deep learning",
    "signal processing",
    "temporal signal",
    "time series",
    "generative modeling",
    "neural networks",
    "GAN",
]

author = "Alexander Nikitin"
url = "https://github.com/AlexanderVNikitin/tsgm"

license = "Apache-2.0"

classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]


def read_file(filename: str) -> str:
    with open(filename, encoding="utf-8") as f:
        return f.read().strip()


readme_text = read_file("README.md")


setup(name='tsgm',
      version=version,
      description='Time Series Generative Modelling Framework',
      author=author,
      author_email='',
      maintainer=author,
      maintainer_email='',
      url=url,
      download_url='',
      keywords=keywords,
      long_description=readme_text,
      long_description_content_type='text/markdown',
      license=license,
      entry_points={
        "console_scripts": ["tsgm-gd=cli.gd:main", "tsgm-eval=cli.eval:main"],
      },
      install_requires=[
          "scipy>=1.9.0",
          "numpy>=2.0",
          "networkx>=3.1,<3.3",
          "seaborn>=0.13.2",
          "scikit-learn>=1.6.1",
          "prettytable==3.16.0",
          "antropy==0.1.6",
          "yfinance==0.2.61",
          "tqdm>=4.67.1",
          "dtaidistance==2.3.13",
          "keras>=3.10.0",
          "statsmodels==0.14.5"
      ],
      extras_require={
          "tensorflow": ["tensorflow>=2.19.0", "tensorflow-probability>=0.25.0", "tf-keras>=2.19.0"],
          "torch": ["torch>=2.6.0", "torchvision>=0.21.0"],
          "jax": ["jax>=0.4.30", "jaxlib>=0.4.30"],
          "all": ["tensorflow>=2.19.0", "tensorflow-probability>=0.25.0", "tf-keras>=2.19.1", "torch>=2.6.0", "torchvision>=0.21.0", "jax>=0.4.30", "jaxlib>=0.4.30"]
      },
      package_data={'tsgm': ['README.md']},
      packages=find_packages())
