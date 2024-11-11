import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import torch
from torch.utils.data import DataLoader
from torch_segmentation import TexRayDataset, TIFFDataset

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


"""
This file contains small utility scripts for inspecting a dataset.
It includes plotting individual slices and dataset histograms.
"""

def compute_dataset_histogram(dataloader, bins, lims):
    counts = torch.zeros(bins, dtype=torch.float32)
    iterator = iter(dataloader)
    for slice in iterator:
        if type(slice) is list:  # Deal with TIFFDataset vs TexRayDataset
            slice = slice[0]
        counts += torch.histc(slice, bins=bins, min=lims[0], max=lims[1])
    return counts


def plot_histogram(counts, bins, lims, title, xlabel, ylabel, scale, savepath):
    edges = torch.arange(lims[0], lims[1], (lims[1] - lims[0]) / (bins + 1))
    plt.figure(figsize=(5.00*scale, 3.76*scale), layout="constrained")
    plt.stairs(counts, edges=edges, fill=True)
    plt.xticks(fontsize=11)
    plt.xlabel(xlabel)
    plt.yticks(fontsize=11)
    plt.ylabel(ylabel)
    plt.ticklabel_format()
    plt.title(title, fontsize=11)
    plt.grid(visible=True, alpha=0.5)
    plt.savefig(savepath)


def plot_slice(
    dataloader,
    slice_idx,
    dx,
    scale,
    txt,
    txtclr,
    savepath,
):
    iterator = iter(dataloader)
    for i in range(slice_idx + 1):
        slice = next(iterator)
    if type(slice) is list:  # Deal with TIFFDataset vs TexRayDataset
        slice = slice[0]
    plt.figure(figsize=(5.00*scale, 5.00*scale), layout="constrained")
    scalebar = ScaleBar(
        dx=dx,
        units="um",
        box_alpha=0.0,
        color="r",
        length_fraction=0.5,
    )
    plt.text(20,25,txt,color=txtclr,backgroundcolor="w")
    plt.gca().add_artist(scalebar)
    plt.axis("off")
    plt.savefig(savepath)


if __name__ == "__main__":
    input_path = "./tex-ray/reconstructions"
    dataset = TIFFDataset(input_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    bins = 10000
    lims = (1e-6, 0.6)
    counts = compute_dataset_histogram(dataloader, bins, lims)
    plot_histogram(
        counts,
        bins,
        "Histogram of voxel values",
        "Attenuation (1/cm)",
        "Counts (-)",
        0.75,
        "./tex-ray/histogram.pdf",
    )

    plot_slice(
        dataloader,
        320,
        46.96,
        0.48,
        "$yz$",
        "g",
        "./tex-ray/yz_slice.pdf",
    )
