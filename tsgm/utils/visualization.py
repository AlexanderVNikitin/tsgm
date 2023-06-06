import sklearn
import sklearn.manifold
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

import tsgm


DEFAULT_PALETTE_TSNE = {"hist": "red", "gen": "blue"}


def visualize_dataset(
    dataset: tsgm.dataset.Dataset,
    obj_id: int = 0,
    palette: dict = DEFAULT_PALETTE_TSNE,
    path: str = "/tmp/generated_data.pdf",
) -> None:
    """
    :param dataset: A time series dataset.
    :type dataset: tsgm.dataset.DatasetOrTensor.

    The function visualizes time series dataset with target values.
    It can be handy for regression problems.
    """
    plt.figure(
        num=None, figsize=(8, 4), dpi=80, palette=palette, facecolor="w", edgecolor="k"
    )

    T = dataset.X.shape[-1]

    sns.lineplot(np.arange(0, T, 1), dataset.X[obj_id, -3], label="Feature #1")
    sns.lineplot(np.arange(0, T, 1), dataset.X[obj_id, -2], label="Feature #2")
    sns.lineplot(np.arange(0, T, 1), dataset.X[obj_id, -1], label="Feature #3")

    plt.xlabel("Time")
    plt.ylabel("Absolute value (measurements)")

    print([int(el) for el in dataset.y[obj_id]])
    plt.ylabel("Target value(y)")
    plt.title("Generated data")

    plt.savefig(path)


def visualize_tsne_unlabeled(
    X: tsgm.types.Tensor,
    X_gen: tsgm.types.Tensor,
    palette: dict = DEFAULT_PALETTE_TSNE,
    alpha: float = 0.25,
    path: str = "/tmp/tsne_embeddings.pdf",
    fontsize: int = 20,
    markerscale: int = 3,
    markersize: int = 1,
    feature_averaging: bool = False,
):
    """
    Visualizes t-SNE embeddings of unlabeled data.

    :param X: The original data tensor of shape (num_samples, num_features).
    :type X: tsgm.types.Tensor
    :param X_gen: The generated data tensor of shape (num_samples, num_features).
    :type X_gen: tsgm.types.Tensor
    :param palette: A dictionary mapping class labels to colors. Defaults to DEFAULT_PALETTE_TSNE.
    :type palette: dict, optional
    :param alpha: The transparency level of the plotted points. Defaults to 0.25.
    :type alpha: float, optional
    :param path: The path to save the visualization as a PDF file. Defaults to "/tmp/tsne_embeddings.pdf".
    :type path: str, optional
    :param fontsize: The font size of the class labels in the legend. Defaults to 20.
    :type fontsize: int, optional
    :param markerscale: The scaling factor for the size of the markers in the legend. Defaults to 3.
    :type markerscale: int, optional
    :param markersize: The size of the markers in the scatter plot. Defaults to 1.
    :type markersize: int, optional
    :param feature_averaging: Whether to compute the average features for each class. Defaults to False.
    :type feature_averaging: bool, optional
    """
    tsne = sklearn.manifold.TSNE(n_components=2, learning_rate="auto", init="random")

    point_styles = ["hist"] * X.shape[0] + ["gen"] * X_gen.shape[0]

    if feature_averaging:
        X_all = np.concatenate((np.mean(X, axis=2), np.mean(X_gen, axis=2)))

        X_emb = tsne.fit_transform(np.resize(X_all, (X_all.shape[0], X_all.shape[1])))
    else:
        X_all = np.concatenate((X, X_gen))

        X_emb = tsne.fit_transform(
            np.resize(X_all, (X_all.shape[0], X_all.shape[1] * X_all.shape[2]))
        )

    plt.figure(figsize=(8, 6), dpi=80)
    sns.scatterplot(
        x=X_emb[:, 0],
        y=X_emb[:, 1],
        hue=point_styles,
        style=point_styles,
        markers={"hist": "<", "gen": "H"},
        palette=palette,
        alpha=alpha,
        s=markersize,
    )
    plt.box(False)
    plt.axis("off")
    plt.tight_layout()
    plt.legend(
        bbox_to_anchor=(1, 1),
        loc=1,
        borderaxespad=0,
        fontsize=fontsize,
        markerscale=markerscale,
    )
    plt.savefig(path)


def visualize_tsne(
    X: tsgm.types.Tensor,
    y: tsgm.types.Tensor,
    X_gen: tsgm.types.Tensor,
    y_gen: tsgm.types.Tensor,
    path: str = "/tmp/tsne_embeddings.pdf",
    feature_averaging: bool = False,
):
    """
    Visualizes t-SNE embeddings of real and synthetic data.

    This function generates a scatter plot of t-SNE embeddings for real and synthetic data.
    Each data point is represented by a marker on the plot, and the colors of the markers
    correspond to the corresponding class labels of the data points.

    :param X: The original real data tensor of shape (num_samples, num_features).
    :type X: tsgm.types.Tensor
    :param y: The labels of the original real data tensor of shape (num_samples,).
    :type y: tsgm.types.Tensor
    :param X_gen: The generated synthetic data tensor of shape (num_samples, num_features).
    :type X_gen: tsgm.types.Tensor
    :param y_gen: The labels of the generated synthetic data tensor of shape (num_samples,).
    :type y_gen: tsgm.types.Tensor
    :param path: The path to save the visualization as a PDF file. Defaults to "/tmp/tsne_embeddings.pdf".
    :type path: str, optional
    :param feature_averaging: Whether to compute the average features for each class. Defaults to False.
    :type feature_averaging: bool, optional
    """
    tsne = sklearn.manifold.TSNE(n_components=2, learning_rate="auto", init="random")

    if feature_averaging:
        X_all = np.concatenate((np.mean(X, axis=2), np.mean(X_gen, axis=2)))

        X_emb = tsne.fit_transform(np.resize(X_all, (X_all.shape[0], X_all.shape[1])))
    else:
        X_all = np.concatenate((X, X_gen))

        X_emb = tsne.fit_transform(
            np.resize(X_all, (X_all.shape[0], X_all.shape[1] * X_all.shape[2]))
        )

    y_all = np.concatenate((y, y_gen))

    c = np.argmax(y_all, axis=1)
    colors = {0: "class 0", 1: "class 1"}
    c = [colors[el] for el in c]
    point_styles = ["hist"] * X.shape[0] + ["gen"] * X_gen.shape[0]

    plt.figure(figsize=(8, 6), dpi=80)
    sns.scatterplot(
        x=X_emb[:, 0],
        y=X_emb[:, 1],
        hue=c,
        style=point_styles,
        markers={"hist": "<", "gen": "H"},
        alpha=0.7,
    )
    plt.legend()
    plt.box(False)
    plt.axis("off")
    plt.savefig(path)


def visualize_ts(ts: tsgm.types.Tensor, num: int = 5):
    """
    Visualizes time series tensor.

    This function generates a plot to visualize time series data. It displays a specified number of time series
    from the input tensor.

    :param ts: The time series data tensor of shape (num_samples, num_timesteps, num_features).
    :type ts: tsgm.types.Tensor
    :param num: The number of time series to display. Defaults to 5.
    :type num: int, optional

    Raises:
        AssertionError: If the input tensor does not have three dimensions.

    Example:
        >>> visualize_ts(time_series_tensor, num=10)
    """
    assert len(ts.shape) == 3

    fig, axs = plt.subplots(num, 1, figsize=(14, 10))

    ids = np.random.choice(ts.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        axs[i].imshow(ts[sample_id].T, aspect="auto")


def visualize_ts_lineplot(
    ts: tsgm.types.Tensor,
    ys: tsgm.types.OptTensor = None,
    num: int = 5,
    unite_features: bool = True,
):
    assert len(ts.shape) == 3

    fig, axs = plt.subplots(num, 1, figsize=(14, 10))
    if num == 1:
        axs = [axs]

    ids = np.random.choice(ts.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        if not unite_features:
            feature_id = np.random.randint(ts.shape[2])
            sns.lineplot(
                x=range(ts.shape[1]),
                y=ts[sample_id, :, feature_id],
                ax=axs[i],
                label=rf"feature \#{feature_id}",
            )
        else:
            for feat_id in range(ts.shape[2]):
                sns.lineplot(
                    x=range(ts.shape[1]), y=ts[sample_id, :, feat_id], ax=axs[i]
                )
        if ys is not None:
            if len(ys.shape) == 1:
                axs[i].set_title(ys[sample_id])
            elif len(ys.shape) == 2:
                sns.lineplot(
                    x=range(ts.shape[1]),
                    y=ys[sample_id],
                    ax=axs[i].twinx(),
                    color="g",
                    label="Target variable",
                )
            else:
                raise ValueError("ys contains too many dimensions")


def visualize_original_and_reconst_ts(
    original: tsgm.types.Tensor,
    reconst: tsgm.types.Tensor,
    num: int = 5,
    vmin: int = 0,
    vmax: int = 1,
):
    assert original.shape == reconst.shape

    fig, axs = plt.subplots(num, 2, figsize=(14, 10))

    ids = np.random.choice(original.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        axs[i, 0].imshow(original[sample_id].T, aspect="auto", vmin=vmin, vmax=vmax)
        axs[i, 1].imshow(reconst[sample_id].T, aspect="auto", vmin=vmin, vmax=vmax)


def visualize_training_loss(
    loss_vector: tsgm.types.Tensor,
    labels: tuple = (),
    path: str = "/tmp/training_loss.pdf",
):
    """
    Plot training losses as a function of the epochs
    :param loss_vector: np.array, having shape num of metrics times number of epochs
    :param labels: list of strings
    :param path: str, where to save the plot
    """
    num_of_metrics = loss_vector.shape[0]
    num_of_epochs = loss_vector[0].shape[0]
    _colors = [
        {"color": "orange", "linewidth": 1, "alpha": 0.8},
        {"color": "darkorchid"},
        {"color": "pink"},
        {"color": "blue"},
        {"color": "red"},
        {"color": "green"},
        {"color": "black", "linewidth": 2},
    ]
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for i in range(num_of_metrics):
        _label = labels[i] if i < len(labels) else None
        _loss = loss_vector[i]

        # scale loss to be in range [0, 0.xxx]
        _max_magnitude = math.floor(math.log10(np.max(_loss)))
        if _max_magnitude >= 0:
            _exp = _max_magnitude + 1
            _loss /= 10 ** _exp
            _label += f" ($10^{_exp}$)"

        if i < len(_colors):
            # use custom styles until a style is defined
            ax.plot(range(num_of_epochs), _loss, label=_label, **_colors[i])
        else:
            ax.plot(
                range(num_of_epochs),
                _loss,
                label=_label,
            )

    plt.legend()
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    plt.savefig(path, dpi=80)
