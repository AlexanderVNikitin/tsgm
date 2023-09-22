from setuptools import setup
from setuptools import find_packages


name = "tsgm"

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
      version='0.0.3',
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
          "scipy",
          "numpy",          
          "networkx",
          "seaborn",
          "scikit-learn",
          "prettytable",
          "yfinance",
          "tqdm",
          "dtaidistance >= 2.3.10",
          "tensorflow",
          "tensorflow-probability",
      ],
      package_data={'tsgm': ['README.md']},
      packages=find_packages())
