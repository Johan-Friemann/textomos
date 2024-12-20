import numpy as np
import matplotlib.pyplot as plt
import tifffile
from matplotlib_scalebar.scalebar import ScaleBar
import hdf5_utils

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


"""
This file contains small utility scripts for inspecting a dataset.
It includes plotting individual slices and dataset histograms.
"""


def compute_dataset_histogram(path, bins, lims, normalize=True):
    counts = np.zeros(bins, dtype=np.float32)
    try:
        num_samples = hdf5_utils.get_database_shape(path)[0]
        for idx in range(num_samples):
            sample = hdf5_utils.get_reconstruction_from_database(path, idx)
            counts += np.histogram(sample, bins=bins, range=lims)[0]
    except:
        sample = tifffile.imread(path)
        counts += np.histogram(sample, bins=bins, range=lims)[0]

    if normalize:
        counts /= np.sum(counts)

    return counts


def plot_histogram(
    counts, bins, lims, title, xlabel, ylabel, scale, savepath, legends=None
):
    edges = np.arange(lims[0], lims[1], (lims[1] - lims[0]) / (bins + 1))
    plt.figure(figsize=(5.00 * scale, 3.76 * scale), layout="constrained")
    linestyles = ("-", "--", "-.", ":")
    for i, count in enumerate(counts):
        plt.stairs(count, edges=edges, fill=i == 0, linestyle=linestyles[i])
    plt.xticks(fontsize=11)
    plt.xlabel(xlabel)
    plt.yticks(fontsize=11)
    plt.ylabel(ylabel)
    plt.ticklabel_format()
    plt.title(title, fontsize=11)
    plt.grid(visible=True, alpha=0.5)
    if not legends is None:
        plt.legend(legends)
    plt.savefig(savepath)


def plot_slice(
    slice,
    lims,
    dx,
    scale,
    txt,
    txtclr,
    savepath,
):
    plt.figure(figsize=(5.00 * scale, 5.00 * scale), layout="constrained")
    scalebar = ScaleBar(
        dx=dx,
        units="um",
        box_alpha=0.0,
        color="r",
        length_fraction=0.5,
    )
    plt.imshow(
        slice,
        cmap="gray",
        vmin=lims[0],
        vmax=lims[1],
        origin="lower",
    )
    plt.text(20, 25, txt, color=txtclr, backgroundcolor="w")
    plt.gca().add_artist(scalebar)
    plt.axis("off")
    plt.savefig(savepath)


if __name__ == "__main__":
    database_path = "./tex-ray/training_set"
    lims = (0, 3)
    slice = hdf5_utils.get_reconstruction_from_database(database_path, 100)[
        :, :, 236
    ]
    plot_slice(
        slice,
        lims,
        46.19,
        0.48,
        "$yz$",
        "g",
        "./tex-ray/sim_yz_slice.pdf",
    )
