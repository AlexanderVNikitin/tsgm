import argparse
import pickle
import copy
import functools
import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras

from io import BytesIO

from keras import layers

import tsgm


N_EPOCHS_DEFAULT = 10
BATCH_SIZE_DEFAULT = 256
LATENT_DIM_DEFAULT = 16


def _gen_dataset(X, batch_size=BATCH_SIZE_DEFAULT, scale=(-1, 1)):
    scaler = tsgm.utils.TSFeatureWiseScaler(scale)
    X_train = scaler.fit_transform(X).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(X_train)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset, X_train


def parse_arguments():
    parser = argparse.ArgumentParser(description='Experiments with data generations')

    parser.add_argument('--source-data', type=str, help='Path to the pickled data you want to model',
                        default="source_data")
    parser.add_argument('--source-data-labels', type=str, help="Path to the pickled source data labels", 
                        default="")

    parser.add_argument('--dest-data', type=str, help='Destination path for the pickled data',
                        default="generated_data")
    parser.add_argument('--n-epochs', type=int, help='Destination path for the pickled data',
                        default=N_EPOCHS_DEFAULT)
    parser.add_argument('--batch-size', type=int, help='Batch size for training',
                        default=BATCH_SIZE_DEFAULT)
    parser.add_argument('--latent-dim', type=int, help='Latent dimensionality',
                        default=LATENT_DIM_DEFAULT)
    parser.add_argument("--dump-model-path", type=str, help="Path where the serialized model is saved",
                        default="./saved_model.pkl")
    parser.add_argument("--architecture", type=str, help="[GAN|TimeGAN|VAE]", default="GAN")

    return parser.parse_args()


if __name__ == "__main__":
    tsgm.utils.fix_seeds()
    args = parse_arguments()

    print(f"Training {args.architecture} with latent_dim={args.latent_dim} and n_epochs={args.n_epochs}")

    X_source = pickle.load(open(args.source_data, "rb"))
    if args.source_data_labels != "":
        y_sourse = pickle.load(open(args.source_data, "rb"))
    assert len(X_source.shape) == 3
    n, seq_len, feature_dim = X_source.shape
    print(f"n={n}, seq_len={seq_len} feature_dim={feature_dim}")
    if args.architecture.lower() == "gan":
        dataset, X_preprocessed = _gen_dataset(X_source, batch_size=args.batch_size)
        architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
            seq_len=seq_len, feat_dim=feature_dim,
            latent_dim=LATENT_DIM_DEFAULT, output_dim=0)
        discriminator, generator = architecture.discriminator, architecture.generator

        model = tsgm.models.cgan.GAN(
            discriminator=discriminator, generator=generator, latent_dim=args.latent_dim
        )
        model.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss_fn=keras.losses.BinaryCrossentropy(),
        )
        model.fit(dataset, epochs=args.n_epochs)
    elif args.architecture.lower() == "timegan":
        dataset, X_preprocessed = _gen_dataset(X_source, batch_size=args.batch_size, scale=(0, 1))
        model = tsgm.models.timeGAN.TimeGAN(seq_len=X_preprocessed.shape[1],
            n_features=X_preprocessed.shape[2], module='lstm', epochs=args.n_epochs)
        model.compile()
        model.fit(X_preprocessed, epochs=args.n_epochs)
    elif args.architecture.lower() == "vae":
        dataset, X_preprocessed = _gen_dataset(X_source, batch_size=args.batch_size, scale=(0, 1))
        architecture = tsgm.models.architectures.zoo["vae_conv5"](
            seq_len=seq_len, feat_dim=feature_dim,
            latent_dim=args.latent_dim)
        encoder, decoder = architecture.encoder, architecture.decoder
        model = tsgm.models.cvae.BetaVAE(encoder, decoder)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003))
        model.fit(X_preprocessed, epochs=args.n_epochs)
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    X_syn = model.generate(n)
    pickle.dump(X_syn, open(args.dest_data, "wb"))
    print(f"The generated dataset is saved to {args.dest_data}")
