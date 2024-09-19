import pytest
import copy
import itertools
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tsgm
import keras


def _get_gunpoint_dataset():
    data_manager = tsgm.utils.UCRDataManager(ds="GunPoint")
    X_train, y_train, X_test, y_test  = data_manager.get()
    X_train, X_test = X_train[:, :, None], X_test[:, :, None]
    y_train = keras.utils.to_categorical(y_train - 1)
    y_test = keras.utils.to_categorical(y_test - 1)
    return X_train, y_train, X_test, y_test


def test_classification_conv():
    X_train, y_train, X_test, y_test = _get_gunpoint_dataset()

    seq_len, feat_dim, output_dim = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = tsgm.models.zoo["clf_cn"](seq_len=seq_len, feat_dim=feat_dim, output_dim=output_dim).model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=False)

    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=0)

    y_prediction = model.predict(X_train)
    assert y_prediction.shape == y_train.shape


def test_classification_conv_3_layers():
    X_train, y_train, X_test, y_test = _get_gunpoint_dataset()

    seq_len, feat_dim, output_dim = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = tsgm.models.zoo["clf_cn"](seq_len=seq_len, feat_dim=feat_dim, output_dim=output_dim, n_conv_blocks=3).model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=False)

    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=0)

    y_prediction = model.predict(X_train)
    assert y_prediction.shape == y_train.shape


def test_classification_lstm_conv_3_layers():
    X_train, y_train, X_test, y_test = _get_gunpoint_dataset()

    seq_len, feat_dim, output_dim = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = tsgm.models.zoo["clf_cl_n"](
        seq_len=seq_len, feat_dim=feat_dim,
        output_dim=output_dim, n_conv_lstm_blocks=3
    ).model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=False)

    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=0)

    y_prediction = model.predict(X_train)
    assert y_prediction.shape == y_train.shape


def test_classification_blocks_3_layers():
    X_train, y_train, X_test, y_test = _get_gunpoint_dataset()

    seq_len, feat_dim, output_dim = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    block = [keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
             keras.layers.Dropout(0.2),
             keras.layers.LSTM(32, activation="relu", return_sequences=True),
             keras.layers.Dropout(0.2)]
    blocks = list(itertools.chain(*[copy.deepcopy(block) for _ in range(3)]))
    for i, b in enumerate(blocks):
        b._name = b._name + str(i)

    model = tsgm.models.zoo["clf_block"](
        seq_len=seq_len, feat_dim=feat_dim,
        output_dim=output_dim, blocks=blocks
    ).model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=False)

    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=0)

    y_prediction = model.predict(X_train)
    assert y_prediction.shape == y_train.shape
