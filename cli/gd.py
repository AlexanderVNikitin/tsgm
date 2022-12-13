import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

import tsgm


N_EPOCHS_DEFAULT = 10
BATCH_SIZE_DEFAULT = 256
LATENT_DIM_DEFAULT = 16
DEFAULT_GAN_ARCHITECTURE = "cgan_base_c4_l1"
DEFAULT_VAE_ARCHITECTURE = "vae_conv5"


def _gen_dataset(X, y, batch_size=BATCH_SIZE_DEFAULT, scale=(-1, 1)):
    scaler = tsgm.utils.TSFeatureWiseScaler(scale)
    X_train = scaler.fit_transform(X).astype(np.float32)
    if y is None:
        y_train = None
        dataset = tf.data.Dataset.from_tensor_slices(X_train)
    else:
        y_train = keras.utils.to_categorical(y, 2).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset, X_train, y_train


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
    parser.add_argument("--architecture-type", type=str, help="[GAN|TimeGAN|VAE]", default="GAN")
    parser.add_argument("--architecture", type=str, help="Specific name of the architecture in Zoo", default=None)

    return parser.parse_args()


def _get_dims(X_source, y_source):
    n, seq_len, feature_dim = X_source.shape
    output_dim = y_source.shape[1] if y_source is not None else 0

    return n, seq_len, feature_dim, output_dim


def _get_gan_model(X_source, y_source, args, architecture_name=None):
    architecture_name = architecture_name or DEFAULT_GAN_ARCHITECTURE
    n, seq_len, feature_dim, output_dim = _get_dims(X_source, y_source)
    architecture = tsgm.models.architectures.zoo[architecture_name](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=args.latent_dim, output_dim=output_dim)
    discriminator, generator = architecture.discriminator, architecture.generator

    # TODO: make a unified class for GANs and conditional GANs
    if output_dim == 0:
        model = tsgm.models.cgan.GAN(
            discriminator=discriminator, generator=generator, latent_dim=args.latent_dim
        )
    else:
        model = tsgm.models.cgan.ConditionalGAN(
            discriminator=discriminator, generator=generator, latent_dim=args.latent_dim,
        )
    return model


# TODO: allow temporal labels
def _get_vae_model(X_source, y_source, args, architecture_name=None):
    architecture_name = architecture_name or DEFAULT_VAE_ARCHITECTURE
    n, seq_len, feature_dim, output_dim = _get_dims(X_source, y_source)
    architecture = tsgm.models.architectures.zoo[architecture_name](
        seq_len=seq_len, feat_dim=feature_dim, latent_dim=args.latent_dim, output_dim=output_dim)
    encoder, decoder = architecture.encoder, architecture.decoder
    if output_dim == 0:
        model = tsgm.models.cvae.BetaVAE(encoder, decoder)
    else:
        model = tsgm.models.cvae.cBetaVAE(encoder, decoder, latent_dim=args.latent_dim, temporal=False)
    return model


def main():
    tsgm.utils.fix_seeds()
    args = parse_arguments()

    print(f"Training {args.architecture_type} with latent_dim={args.latent_dim} and n_epochs={args.n_epochs}")

    X_source = pickle.load(open(args.source_data, "rb"))
    y_source = None
    if args.source_data_labels:
        y_source = pickle.load(open(args.source_data_labels, "rb"))
    assert len(X_source.shape) == 3 and X_source.shape[0] == y_source.shape[0]
    n, seq_len, feature_dim = X_source.shape
    print(f"n={n}, seq_len={seq_len} feature_dim={feature_dim}")

    normalised_arch_type = args.architecture_type.strip().lower()
    if normalised_arch_type == "gan":
        dataset, X_preprocessed, y_preprocessed = _gen_dataset(X_source, y_source, batch_size=args.batch_size)
        model = _get_gan_model(X_preprocessed, y_preprocessed, args, architecture_name=args.architecture)
        model.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss_fn=keras.losses.BinaryCrossentropy(),
        )
        model.fit(dataset, epochs=args.n_epochs)
    elif normalised_arch_type == "timegan":
        dataset, X_preprocessed, y_preprocessed = _gen_dataset(X_source, batch_size=args.batch_size, scale=(0, 1))
        model = tsgm.models.timeGAN.TimeGAN(
            seq_len=X_preprocessed.shape[1], n_features=X_preprocessed.shape[2],
            module='lstm', epochs=args.n_epochs)
        model.compile()
        model.fit(X_preprocessed, y_preprocessed, epochs=args.n_epochs)
    elif normalised_arch_type == "vae":
        dataset, X_preprocessed, y_preprocessed = _gen_dataset(X_source, y_source, batch_size=args.batch_size, scale=(0, 1))
        model = _get_vae_model(X_preprocessed, y_preprocessed, args, architecture_name=args.architecture)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003))
        model.fit(X_preprocessed, y_preprocessed, epochs=args.n_epochs)
    else:
        raise ValueError(f"Unknown architecture: {normalised_arch_type}")
    if y_preprocessed is None:
        X_syn = model.generate(n)
    else:
        rand_idx = np.random.randint(y_preprocessed.shape[0], size=n)
        X_syn = model.generate(y_preprocessed[rand_idx, :])
    pickle.dump(X_syn, open(args.dest_data, "wb"))
    print(f"The generated dataset is saved to {args.dest_data}")


if __name__ == "__main__":
    main()
