import sklearn
import sklearn.manifold
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import tsgm


def visualize_dataset(dataset: tsgm.dataset.Dataset, obj_id: int = 0, path: str = "/tmp/generated_data.pdf") -> None:
    """
    The function visualizes time series dataset with target values.
    It can be handy for regression problems.
    :param dataset: A time series dataset.
    :type dataset: tsgm.dataset.DatasetOrTensor.
    """
    plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')

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
        X: tsgm.types.Tensor, X_gen: tsgm.types.Tensor, palette="deep",
        alpha=0.25,
        path: str = "/tmp/tsne_embeddings.pdf"):
    """
    Visualizes TSNE of real and synthetic data.
    """
    tsne = sklearn.manifold.TSNE(n_components=2, learning_rate='auto', init='random')

    X_all = np.concatenate((X, X_gen))

    colors = {"hist": "red", "gen": "blue"}
    point_styles = ["hist"] * X.shape[0] + ["gen"] * X_gen.shape[0]
    X_emb = tsne.fit_transform(np.resize(X_all, (X_all.shape[0], X_all.shape[1] * X_all.shape[2])))

    plt.figure(figsize=(8, 6), dpi=80)
    sns.scatterplot(x=X_emb[:, 0], y=X_emb[:, 1], hue=point_styles,
                    style=point_styles, markers={"hist": "<", "gen": "H"}, palette=colors, alpha=0.25)
    plt.legend(fontsize=14)
    plt.box(False)
    plt.axis('off')
    plt.savefig(path)


def visualize_tsne(X: tsgm.types.Tensor, y: tsgm.types.Tensor, X_gen: tsgm.types.Tensor, y_gen: tsgm.types.Tensor,
                   path: str = "/tmp/tsne_embeddings.pdf"):
    """
    Visualizes TSNE of real and synthetic data.
    """
    tsne = sklearn.manifold.TSNE(n_components=2, learning_rate='auto', init='random')

    X_all = np.concatenate((X, X_gen))
    y_all = np.concatenate((y, y_gen))

    c = np.argmax(y_all, axis=1)
    colors = {0: "class 0", 1: "class 1"}
    c = [colors[el] for el in c]
    point_styles = ["hist"] * X.shape[0] + ["gen"] * X_gen.shape[0]
    X_emb = tsne.fit_transform(np.resize(X_all, (X_all.shape[0], X_all.shape[1] * X_all.shape[2])))

    plt.figure(figsize=(8, 6), dpi=80)
    sns.scatterplot(x=X_emb[:, 0], y=X_emb[:, 1], hue=c, style=point_styles, markers={"hist": "<", "gen": "H"}, alpha=0.7)
    plt.legend()
    plt.box(False)
    plt.axis('off')
    plt.savefig(path)


def visualize_ts(ts: tsgm.types.Tensor, num: int = 5):
    """
    Visualizes time series tensor.
    """
    assert len(ts.shape) == 3

    fig, axs = plt.subplots(num, 1, figsize=(14, 10))

    ids = np.random.choice(ts.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        axs[i].imshow(ts[sample_id].T, aspect='auto')


def visualize_ts_lineplot(ts: tsgm.types.Tensor, ys: tsgm.types.OptTensor = None, num: int = 5):
    assert len(ts.shape) == 3

    fig, axs = plt.subplots(num, 1, figsize=(14, 10))
    if num == 1:
        axs = [axs]

    ids = np.random.choice(ts.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        feature_id = np.random.randint(ts.shape[2])
        sns.lineplot(x=range(ts.shape[1]), y=ts[sample_id, :, feature_id], ax=axs[i], label=f"feature #{feature_id}")
        if ys is not None:
            if len(ys.shape) == 1:
                axs[i].set_title(ys[sample_id])
            elif len(ys.shape) == 2:
                sns.lineplot(x=range(ts.shape[1]), y=ys[sample_id], ax=axs[i].twinx(), color="g", label="Target variable")
            else:
                raise ValueError("ys contains too many dimensions")


def visualize_original_and_reconst_ts(original, reconst, num=5, vmin=0, vmax=1):
    assert original.shape == reconst.shape

    fig, axs = plt.subplots(num, 2, figsize=(14, 10))

    ids = np.random.choice(original.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        axs[i, 0].imshow(original[sample_id].T, aspect='auto', vmin=vmin, vmax=vmax)
        axs[i, 1].imshow(reconst[sample_id].T, aspect='auto', vmin=vmin, vmax=vmax)
