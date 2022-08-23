from setuptools import setup
from setuptools import find_packages

setup(name='tsgm',
      version='0.0',
      description='Time Series Generative Modelling Framework',
      author='',
      author_email='',
      url='',
      download_url='',
      license='Apache-2.0',
      install_requires=[
          "tensorflow>=2.9",
          "scipy>=1.7.3"
          "numpy>=1.23.2",
      ],
      package_data={'tsgm': ['README.md']},
      packages=find_packages())
