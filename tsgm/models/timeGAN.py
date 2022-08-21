import tensorflow as tf
from tensorflow import keras

import logging

import tsgm


logger = logging.getLogger('models')
logger.setLevel(logging.DEBUG)


class TimeGAN(keras.Model):
    """
    Time-series Generative Adversarial Networks (TimeGAN)

    Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
    "Time-series Generative Adversarial Networks,"
    Neural Information Processing Systems (NeurIPS), 2019.

    Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
    """
    def __init__(self):
        pass