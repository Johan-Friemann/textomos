import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import torch
from torch.utils.data import DataLoader
from torch_segmentation import TexRayDataset, TIFFDataset

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def compute_dataset_histogram(dataloader, bins, lims):
    counts = torch.zeros(bins, dtype=torch.float32)
    iterator = iter(dataloader)
    for slice in iterator:
        if type(slice) is list:  # Deal with TIFFDataset vs TexRayDataset
            slice = slice[0]
        counts += torch.histc(slice, bins=bins, min=lims[0], max=lims[1])
    return counts


def plot_histogram(counts, bins, lims, title, xlabel, ylabel):
    edges = torch.arange(lims[0], lims[1], (lims[1] - lims[0]) / (bins + 1))
    plt.figure(figsize=(4.77, 3.58), layout="constrained")
    plt.stairs(counts, edges=edges)
    plt.xticks(fontsize=11)
    plt.xlabel(xlabel)
    plt.yticks(fontsize=11)
    plt.ylabel(ylabel)
    plt.ticklabel_format()
    plt.title(title, fontsize=11)
    # plt.savefig("./tex-ray/line_spread.pdf")
    plt.show()


def plot_slice(dataloader, slice_idx, dx, lims, title, xlabel, ylabel, cbarlabel):
    iterator = iter(dataloader)
    for i in range(slice_idx + 1):
        slice = next(iterator)
    if type(slice) is list:  # Deal with TIFFDataset vs TexRayDataset
        slice = slice[0]
    plt.figure(figsize=(4.77, 3.58), layout="constrained")
    plt.imshow(slice[0, 0, ...], cmap="gray", vmin=lims[0], vmax=lims[1])
    scalebar = ScaleBar(
        dx=dx,
        units="um",
        box_alpha=0.0,
        color="r",
        length_fraction=0.5,
    )
    plt.colorbar(label=cbarlabel)
    plt.gca().add_artist(scalebar)
    plt.title(title, fontsize=11)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":

    inferenece_input_path = "./tex-ray/reconstructions/BAM2x2x3UC_LFOV_AIR_40kV_10W_5s_BIN4_so80mm_od150mm.tiff"
    test_set = TIFFDataset(inferenece_input_path)
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=1
    )
    """
    dataset = TexRayDataset("./tex-ray/training_set")
    dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), 50))
    test_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1
    )
    """
    bins = 10000
    lims = (0.0001, 0.6)
    counts = compute_dataset_histogram(test_loader, bins, lims)
    plot_histogram(
        counts,
        bins,
        lims,
        "Histogram of experiment reconstruction voxel values",
        "Attenuation (1/cm)",
        "Counts (-)",
    )

    plot_slice(
        test_loader,
        255,
        46.96,
        lims,
        "Slice 255 in the $yz$-plane",
        "$y$ (pixels)",
        "$z$ (pixels)",
        "Attenuation (1/cm)",
    )
