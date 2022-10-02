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


setup(name='tsgm',
      version='0.0.0',
      description='Time Series Generative Modelling Framework',
      author=author,
      author_email='',
      maintainer=author,
      maintainer_email='',
      url=url,
      download_url='',
      keywords=keywords,
      long_description="",
      license=license,
      install_requires=[
          "tensorflow==2.9.1",
          "scipy>=1.7.3"
          "numpy>=1.21.6",
      ],
      package_data={'tsgm': ['README.md']},
      packages=find_packages())
