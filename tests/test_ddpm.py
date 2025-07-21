import pytest
import tsgm

import tensorflow as tf
import numpy as np
from tensorflow import keras


def test_ddpm():
    seq_len = 12
    feat_dim = 1

    model_type = tsgm.models.architectures.zoo["ddpm_denoiser"]
    architecture = model_type(seq_len=seq_len, feat_dim=feat_dim)

    denoiser_model = architecture.model

    X = tsgm.utils.gen_sine_dataset(50, seq_len, feat_dim, max_value=20)

    scaler = tsgm.utils.TSFeatureWiseScaler((0, 1))
    X = scaler.fit_transform(X).astype(np.float64)

    ddpm_model = tsgm.models.ddpm.DDPM(denoiser_model, model_type(seq_len=seq_len, feat_dim=feat_dim).model, 1000)
    ddpm_model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(0.0003))
    
    with pytest.raises(ValueError):
        ddpm_model.generate(7)

    ddpm_model.fit(X, epochs=1, batch_size=128)

    x_samples = ddpm_model.generate(7)
    assert x_samples.shape == (7, seq_len, feat_dim)

    x_decoded = ddpm_model.generate(3)
    assert x_decoded.shape == (3, seq_len, feat_dim)
