import sys
import os
import random
import itertools
import collections
import tifffile
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
    fcn_resnet50,
    FCN_ResNet50_Weights,
    fcn_resnet101,
    FCN_ResNet101_Weights,
)
from torch_models import *
from hdf5_utils import (
    get_reconstruction_chunk_handle,
    get_segmentation_chunk_handle,
    get_metadata,
    get_database_shape,
)


class TextomosDatasetBase(Dataset):
    """A base class containing utility functions for dataset loading and
    metrics.
    """

    def __init__(self, database_path, z_score, separate_centers=False):
        self.database_path = database_path
        self.mean = z_score[0]
        self.std = z_score[1]
        # data_0 is guaranteed to exist
        detector_rows = get_metadata(database_path, 0, "detector_rows")
        binning = get_metadata(database_path, 0, "binning")
        self.slices_per_sample = detector_rows // binning
        self.num_samples, self.num_chunks, self.chunk_size = get_database_shape(
            self.database_path
        )
        self.separate_centers = separate_centers
        self.num_yarn_types = (  # -1 for matrix
            len(get_metadata(database_path, 0, "phase_densities")) - 1
        )

    def compute_min_max(self, device):
        """Compute the dataset min max normalization statistics. This is very
        slow for large datasets. It is therefore recommended to do it in advance
        and save it to avoid waiting during model prototyping.

        Args:
            device (torch device): Device to perform computations on.

        Keyword args:
            -

        Returns:
            min (float): The dataset minimum value.
            max (float): The dataset maximum value.
        """
        min_candidate = torch.tensor([0.0]).to(device)
        max_candidate = torch.tensor([0.0]).to(device)
        min_val = torch.tensor([100.0]).to(device)
        max_val = torch.tensor([-100.0]).to(device)
        for global_idx in range(self.num_samples):
            chunk_idx = global_idx // self.chunk_size
            local_idx = global_idx % self.chunk_size
            recon_sample = torch.tensor(
                get_reconstruction_chunk_handle(
                    self.database_path, chunk_idx
                ).get("reconstruction_" + str(local_idx))[:, :, :]
            ).to(device)
            min_candidate = torch.min(recon_sample)
            max_candidate = torch.max(recon_sample)
            min_val = torch.min(min_val, min_candidate)
            max_val = torch.max(max_val, max_candidate)

        return min_val.item(), max_val.item()

    def compute_z_score(self, device):
        """Compute the dataset z-score normalization statistics. This is VERY
        slow for large datasets. It is therefore recommended to do it in advance
        and save it to avoid waiting during model prototyping.

        Args:
            device (torch device): Device to perform computations on.

        Keyword args:
            -

        Returns:
            mean (float): The dataset mean value.
            std (float): The dataset standard deviation.

        """
        mean = torch.tensor([0.0]).to(device)
        var = torch.tensor([0.0]).to(device)
        for global_idx in range(self.num_samples):
            chunk_idx = global_idx // self.chunk_size
            local_idx = global_idx % self.chunk_size
            recon_sample = torch.tensor(
                get_reconstruction_chunk_handle(
                    self.database_path, chunk_idx
                ).get("reconstruction_" + str(local_idx))[:, :, :]
            ).to(device)
            mean += torch.mean(recon_sample)
        mean /= self.num_samples

        for global_idx in range(self.num_samples):
            chunk_idx = global_idx // self.chunk_size
            local_idx = global_idx % self.chunk_size
            recon_sample = torch.tensor(
                get_reconstruction_chunk_handle(
                    self.database_path, chunk_idx
                ).get("reconstruction_" + str(local_idx))[:, :, :]
            ).to(device)
            var += torch.sum((recon_sample - mean) ** 2)
        var /= self.num_samples * self.slices_per_sample**3 - 1

        return mean.item(), numpy.sqrt(var.item())

    def compute_loss_weights(self, device):
        """Compute the dataset median frequency balancing weights. This is very
        slow for large datasets. It is therefore recommended to do it in advance
        and save it to avoid waiting during model prototyping.

        Args:
            device (torch device): Device to perform computations on.

        Keyword args:
            -

        Returns:
            weights (torch tensor): A tensor of length corresponding to the
                                    number of phases that contains the
                                    frequency balancing weights. They are:
                                    given as [air, phase_1, ..., phase_n].
        """
        class_freq = torch.zeros(
            2 + self.num_yarn_types * (1 + self.separate_centers)
        ).to(device)
        for global_idx in range(self.num_samples):
            chunk_idx = global_idx // self.chunk_size
            local_idx = global_idx % self.chunk_size
            seg_sample = torch.tensor(
                get_segmentation_chunk_handle(
                    self.database_path, chunk_idx
                ).get("segmentation_" + str(local_idx))[:, :, :]
            ).to(device)
            for i in range(
                2 + self.num_yarn_types * (1 + self.separate_centers)
            ):
                class_freq[i] += torch.sum(seg_sample == i)
        median = torch.median(class_freq)
        return median / class_freq


class TextomosDataset2D(TextomosDatasetBase):
    """See pytorch dataset class documentation for specifics of __method__
    type methods that are required by the dataset interface.

    Params:
        database_path (str): The absolute path to the database folder.

    Keyword params:
        z_score (tuple(float, float)): Mean and standard deviation to use for
                                       z-score normalization. If either is None
                                       no normalization is performed.
    """

    def __init__(self, database_path, z_score=(None, None)):
        super().__init__(database_path, z_score)

    def __len__(self):
        return self.num_samples * self.slices_per_sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        global_idx = idx // self.slices_per_sample
        chunk_idx = global_idx // self.chunk_size
        local_idx = global_idx % self.chunk_size
        slice_idx = idx % self.slices_per_sample

        recon = get_reconstruction_chunk_handle(
            self.database_path, chunk_idx
        ).get("reconstruction_" + str(local_idx))

        # Unsqueeze to create a channel axis for consistency.
        recon_slice = torch.tensor(recon[slice_idx]).unsqueeze(0)
        if (not self.mean is None) and (not self.std is None):
            recon_slice = (recon_slice - self.mean) / self.std

        seg = get_segmentation_chunk_handle(self.database_path, chunk_idx).get(
            "segmentation_" + str(local_idx)
        )
        seg_slice = torch.tensor(seg[slice_idx])
        seg_mask = torch.zeros(
            (
                self.num_yarn_types + 2,
                self.slices_per_sample,
                self.slices_per_sample,
            )
        )
        # air
        seg_mask[0] = seg_slice == 0
        # yarns
        for i in range(self.num_yarn_types):
            seg_mask[i]
            seg_mask[i + 1] = seg_slice == i + 1
            # For 2D mode we dont track center lines
            seg_mask[i + 1] += seg_slice == i + 1 + self.num_yarn_types
        # matrix
        seg_mask[1 + self.num_yarn_types] = seg_slice == 1 + self.num_yarn_types

        return recon_slice, seg_mask


class TextomosDataset3D(TextomosDatasetBase):
    """See pytorch dataset class documentation for specifics of __method__
    type methods that are required by the dataset interface.

    Params:
        database_path (str): The absolute path to the database folder.

        patch_size (int): The side length of the cubic patch of the volume
                          used as input.

        stride (int): The stride between input patches. If equal to patch_size
                      there will be no overlap or skipped voxels.


    Keyword params:
        z_score (tuple(float, float)): Mean and standard deviation to use for
                                       z-score normalization. If either is None
                                       no normalization is performed.

    """

    def __init__(
        self,
        database_path,
        patch_size,
        stride,
        z_score=(None, None),
        separate_centers=False,
    ):
        super().__init__(
            database_path, z_score, separate_centers=separate_centers
        )
        self.patch_size = patch_size
        self.stride = stride

        patch_indices = []
        for sample_idx in range(self.num_samples):
            for i in range(
                0, self.slices_per_sample - self.patch_size + 1, stride
            ):
                for j in range(
                    0, self.slices_per_sample - self.patch_size + 1, stride
                ):
                    for k in range(
                        0, self.slices_per_sample - self.patch_size + 1, stride
                    ):
                        patch_indices.append((sample_idx, i, j, k))
        self.patch_indices = numpy.array(patch_indices)

    def __len__(self):
        return (
            self.num_samples
            * ((self.slices_per_sample - self.patch_size) // self.stride + 1)
            ** 3
        )

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        global_idx = self.patch_indices[idx][0]
        chunk_idx = global_idx // self.chunk_size
        local_idx = global_idx % self.chunk_size

        # Unsqueeze to create a channel axis for consistency.
        recon = get_reconstruction_chunk_handle(
            self.database_path, chunk_idx
        ).get("reconstruction_" + str(local_idx))
        recon_patch = torch.tensor(
            recon[
                self.patch_indices[idx][1] : self.patch_indices[idx][1]
                + self.patch_size,
                self.patch_indices[idx][2] : self.patch_indices[idx][2]
                + self.patch_size,
                self.patch_indices[idx][3] : self.patch_indices[idx][3]
                + self.patch_size,
            ]
        ).unsqueeze(0)
        if (not self.mean is None) and (not self.std is None):
            recon_patch = (recon_patch - self.mean) / self.std
        seg = get_segmentation_chunk_handle(self.database_path, chunk_idx).get(
            "segmentation_" + str(local_idx)
        )
        seg_patch = torch.tensor(
            seg[
                self.patch_indices[idx][1] : self.patch_indices[idx][1]
                + self.patch_size,
                self.patch_indices[idx][2] : self.patch_indices[idx][2]
                + self.patch_size,
                self.patch_indices[idx][3] : self.patch_indices[idx][3]
                + self.patch_size,
            ]
        )
        seg_mask = torch.zeros(
            (
                2 + self.num_yarn_types * (1 + self.separate_centers),
                self.patch_size,
                self.patch_size,
                self.patch_size,
            )
        )

        # air
        seg_mask[0] = seg_patch == 0
        # yarns
        for i in range(self.num_yarn_types):
            seg_mask[i + 1] = seg_patch == i + 1
            if self.separate_centers:
                seg_mask[i + 2 + self.num_yarn_types] = seg_patch == (
                    i + 2 + self.num_yarn_types
                )
            else:
                seg_mask[i + 1] += seg_patch == (i + 2 + self.num_yarn_types)
        # matrix
        seg_mask[1 + self.num_yarn_types] = seg_patch == (
            1 + self.num_yarn_types
        )

        return recon_patch, seg_mask


class TiffDatasetBase(Dataset):
    """A base class containing dataset loader."""

    def __init__(
        self, tiff_path, z_score, cutoff, slice_axis, ground_truth_path
    ):
        self.tiff_path = tiff_path

        self.mean = z_score[0]
        self.std = z_score[1]
        self.min_val = cutoff[0]
        self.max_val = cutoff[1]

        self.reconstruction = torch.tensor(tifffile.imread(tiff_path))
        self.slices_per_sample = self.reconstruction.shape[0]
        if slice_axis == "x":
            self.reconstruction = torch.transpose(self.reconstruction, 0, 2)
            self.reconstruction = torch.transpose(self.reconstruction, 1, 2)
        elif slice_axis == "y":
            self.reconstruction = torch.transpose(self.reconstruction, 0, 1)
        elif slice_axis != "z":
            raise ValueError("slice_axis can only be 'x', 'y', or 'z'.")

        self.segmentation = None
        self.num_phases = None
        if not ground_truth_path is None:
            self.segmentation = torch.tensor(tifffile.imread(ground_truth_path))
            self.num_phases = torch.max(self.segmentation).item() + 1
            if slice_axis == "x":
                self.segmentation = torch.transpose(self.segmentation, 0, 2)
                self.segmentation = torch.transpose(self.segmentation, 1, 2)
            elif slice_axis == "y":
                self.segmentation = torch.transpose(self.segmentation, 0, 1)
            elif slice_axis != "z":
                raise ValueError("slice_axis can only be 'x', 'y', or 'z'.")


class TIFFDataset2D(TiffDatasetBase):
    """See pytorch dataset class documentation for specifics of __method__
    type methods that are required by the dataset interface.

    Params:
        tiff_path (str): The absolute path to the tiff-file.

    Keyword params:
        ground_truth_path (str): The path to ground truth data if available.

        slice_axis (str): The axis along which to take slices.

        z_score (tuple(float, float)): Mean and standard deviation to use for
                                       z-score normalization. If either is None
                                       no normalization is performed.

        cutoff (tuple(float, float)): Lower and upper cutoff values. Sets
                                      values in reconstruction slices to
                                      cutoff[0] if lower than cutofff[0] and
                                      sets values to cutoff[1] if larger
                                      than cutoff[1]. If an entry is 'None',
                                      that bound is skipped.
    """

    def __init__(
        self,
        tiff_path,
        ground_truth_path=None,
        slice_axis="x",
        z_score=(None, None),
        cutoff=(-1.0, 1.0),
    ):
        super().__init__(
            tiff_path, z_score, cutoff, slice_axis, ground_truth_path
        )

    def __len__(self):
        return self.reconstruction.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Unsqueeze to create a channel axis for consistency.
        recon_slice = self.reconstruction[idx].unsqueeze(0)

        if not self.min_val is None:
            recon_slice[recon_slice < self.min_val] = self.min_val

        if not self.max_val is None:
            recon_slice[recon_slice > self.max_val] = self.max_val

        if (not self.mean is None) and (not self.std is None):
            recon_slice = (recon_slice - self.mean) / self.std

        seg_mask = []
        if not self.segmentation is None:
            seg_slice = self.segmentation[idx].unsqueeze(0)
            seg_mask = torch.zeros((self.num_phases, 512, 512))
            for i in range(self.num_phases):
                seg_mask[i] = seg_slice == i

        return recon_slice, seg_mask


class TIFFDataset3D(TiffDatasetBase):
    """See pytorch dataset class documentation for specifics of __method__
    type methods that are required by the dataset interface.

    Params:
        tiff_path (str): The absolute path to the tiff-file.

        patch_size (int): The side length of the cubic patch of the volume
                          used as input.

        stride (int): The stride between input patches. If equal to patch_size
                      there will be no overlap or skipped voxels.

    Keyword params:
        ground_truth_path (str): The path to ground truth data if available.

        z_score (tuple(float, float)): Mean and standard deviation to use for
                                       z-score normalization. If either is None
                                       no normalization is performed.

        cutoff (tuple(float, float)): Lower and upper cutoff values. Sets
                                      values in reconstruction slices to
                                      cutoff[0] if lower than cutofff[0] and
                                      sets values to cutoff[1] if larger
                                      than cutoff[1]. If an entry is 'None',
                                      that bound is skipped.
    """

    def __init__(
        self,
        tiff_path,
        patch_size,
        stride,
        slice_axis="x",
        ground_truth_path=None,
        z_score=(None, None),
        cutoff=(-1.0, 1.0),
    ):
        super().__init__(
            tiff_path, z_score, cutoff, slice_axis, ground_truth_path
        )
        self.patch_size = patch_size
        self.stride = stride

        patch_indices = []
        for i in range(0, self.slices_per_sample - self.patch_size + 1, stride):
            for j in range(
                0, self.slices_per_sample - self.patch_size + 1, stride
            ):
                for k in range(
                    0, self.slices_per_sample - self.patch_size + 1, stride
                ):
                    patch_indices.append((i, j, k))
        self.patch_indices = numpy.array(patch_indices)

    def __len__(self):
        return (
            (self.slices_per_sample - self.patch_size) // self.stride + 1
        ) ** 3

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        recon_patch = self.reconstruction[
            self.patch_indices[idx][0] : self.patch_indices[idx][0]
            + self.patch_size,
            self.patch_indices[idx][1] : self.patch_indices[idx][1]
            + self.patch_size,
            self.patch_indices[idx][2] : self.patch_indices[idx][2]
            + self.patch_size,
        ].unsqueeze(0)

        if not self.min_val is None:
            recon_patch[recon_patch < self.min_val] = self.min_val

        if not self.max_val is None:
            recon_patch[recon_patch > self.max_val] = self.max_val

        if (not self.mean is None) and (not self.std is None):
            recon_patch = (recon_patch - self.mean) / self.std

        seg_mask = []
        if not self.segmentation is None:
            seg_patch = self.segmentation[
                self.patch_indices[idx][0] : self.patch_indices[idx][0]
                + self.patch_size,
                self.patch_indices[idx][1] : self.patch_indices[idx][1]
                + self.patch_size,
                self.patch_indices[idx][2] : self.patch_indices[idx][2]
                + self.patch_size,
            ]
            seg_mask = torch.zeros(
                (
                    self.num_phases,
                    self.patch_size,
                    self.patch_size,
                    self.patch_size,
                )
            )
            for i in range(self.num_phases):
                seg_mask[i] = seg_patch == i

        return recon_patch, seg_mask


class JaccardLoss(nn.Module):
    """See pytorch module documentation for specifics of forward type method
       that is required by the nn.Module interface.

    Compute the soft Jaccard Loss (intersection over union).

    Params:
        -

    Keyword params:
        -
    """

    def __init__(self, eps=1e-6):
        super(JaccardLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, 1)
        intersection = torch.sum(inputs * targets, (2, 3, 4))
        union = torch.sum(inputs + targets, (2, 3, 4)) - intersection
        jaccard_loss = 1 - (intersection + self.eps) / (union + self.eps)
        return torch.mean(jaccard_loss)


class PixelLoss(nn.Module):
    """See pytorch module documentation for specifics of forward type method
       that is required by the nn.Module interface.

    Compute the soft pixel wise Loss.

    Params:
        -

    Keyword params:
        -
    """

    def __init__(self):
        super(PixelLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, 1)
        pixel_loss = 1 - torch.sum(inputs * targets, (1, 2, 3)) / (
            inputs.shape[2] * inputs.shape[3]
        )
        return torch.mean(pixel_loss)


def seed_all(rng_seed):
    """Seed all random number generators for reproducibility.

    Args:
        rng_seed (int): The seed to use for relevant random number generators.

    Keyword args:
        -

    Returns:
        generator (Pytorch generator): A pytorch random number generator based
                                       with seed set to rng_seed.
    """
    generator = torch.manual_seed(rng_seed)
    numpy.random.seed(rng_seed)
    random.seed(rng_seed)

    return generator


def build_model(
    model="deeplabv3_resnet50",
    input_channels=1,
    output_channels=4,
    pre_trained=False,
):
    """Build the segmentation model. The model is adapted to use for segmenting
       images into n distinct classes.

    Args:
        -

    Keyword args:
        model (str): The model to build. Can be 'deeplabv3_resnet50',
                     'deeplabv3_resnet101', 'fcn_resnet_50', or
                     'fcn_resnet_101'.

        input_channels (int): The number of channels of the input data.

        pre_trained (bool): Will initialize with weights trained on a subset
                            of COCO. See pytorch documentation.


    Returns:
        segmentation_model (pytorch model): The segmentation model to train/use.
    """
    weights = None
    if model == "deeplabv3_resnet50":
        if pre_trained:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
        segmentation_model = deeplabv3_resnet50(weights=weights)
    elif model == "deeplabv3_resnet101":
        if pre_trained:
            weights = DeepLabV3_ResNet101_Weights.DEFAULT
        segmentation_model = deeplabv3_resnet101(weights=weights)
    elif model == "deeplabv3_mobilenet_v3":
        if pre_trained:
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        segmentation_model = deeplabv3_mobilenet_v3_large(weights=weights)
    elif model == "fcn_resnet50":
        if pre_trained:
            weights = FCN_ResNet50_Weights.DEFAULT
        segmentation_model = fcn_resnet50(weights=weights)
    elif model == "fcn_resnet101":
        if pre_trained:
            weights = FCN_ResNet101_Weights.DEFAULT
        segmentation_model = fcn_resnet101(weights=weights)
    else:
        raise ValueError("Unsupported model: '" + model + "'.")

    # Replace input to accomodate n channel input data.
    if model == "deeplabv3_mobilenet_v3":
        segmentation_model.backbone["0"] = nn.Conv2d(
            input_channels,
            16,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
    else:
        segmentation_model.backbone.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    # Different architectures have different last layer shape.
    if model in ["fcn_resnet50", "fcn_resnet101"]:
        last_layer_input_size = 512
    if model in [
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
        "deeplabv3_mobilenet_v3",
    ]:
        last_layer_input_size = 256

    # Replace output such that there are n classes to classify.
    segmentation_model.classifier[4] = nn.Conv2d(
        last_layer_input_size,
        output_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
    )

    return segmentation_model


def seed_worker(worker_id):
    """Seed a pytorch worker for reproducibility. This needs to be defined
       in global scope.

    Args:
        worker_id (int): Pytorch worker id.

    Keyword args:
        -

    Returns:
        -
    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders_from_single_set(
    dataset,
    batch_size,
    num_workers,
    generator,
    split=[0.8, 0.2],
    shuffle=True,
):
    """Build the model dataloader dictionary from a single dataset.

    Args:
        dataset (pytorch dataset): The data set to use for constructing the
                                   loaders.

        batch_size (int): The dataloader batch size.

        num_workers (int): The number of cpu workers to use while loading.

        generator (torch generator): The random number generator to use for data
                                     sampling.

    Keyword args:
        split (list[float]): A list of either length 2 or 3 whose entries sum to
                             1. If length 2: train/validation split, if length 3
                             train/validation/test split.

        shuffle (bool): Will shuffle the datasets between epochs if True.

    Returns:
        dataloaders (dict[pytorch dataloader]): The dataloader dictionary.
    """
    if sum(split) != 1.0:
        raise ValueError("Dataloader split must sum to 1.")

    if len(split) not in (2, 3):
        raise ValueError("Dataloader split can only be of length 2 or 3.")

    datasets = torch.utils.data.random_split(
        dataset, split, generator=generator
    )
    labels = ("training", "validation", "testing")
    dataloaders = {}

    for idx in range(len(split)):
        dataloaders[labels[idx]] = DataLoader(
            datasets[idx],
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=generator,
            num_workers=num_workers,
        )

    return dataloaders


def build_dataloaders_from_multiple_sets(
    datasets,
    batch_size,
    num_workers,
    generator,
    shuffle=True,
):
    """Build the model dataloader dictionary from multiple datasets.

    Args:
        datasets (list[pytorch dataset]): The data sets to use for constructing
                                          the loaders.

        batch_size (int): The dataloader batch size.

        num_workers (int): The number of cpu workers to use while loading.

        generator (torch generator): The random number generator to use for data
                                     sampling.

    Keyword args:
        shuffle (bool): Will shuffle the datasets between epochs if True.

    Returns:
        dataloaders (dict[pytorch dataloader]): The dataloader dictionary.
    """
    if len(datasets) not in (2, 3):
        raise ValueError(
            "Dataloaders can only be constructed from 2 or 3 datasets."
        )
    labels = ("training", "validation", "testing")
    dataloaders = {}

    for idx in range(len(datasets)):
        dataloaders[labels[idx]] = DataLoader(
            datasets[idx],
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=generator,
            num_workers=num_workers,
        )

    return dataloaders


def draw_image_with_masks(
    input,
    prediction,
    alpha=0.2,
    save=False,
    show=True,
    save_path="./textomos/segmented_image.jpg",
):
    """Draw the input image in grayscale and overlay the predicted segmentation
    as a colored mask. Red refers to air, blue refers to matrix, green refers to
    weft, and yellow refers to warp.


    Args:
        input (torch tensor): A tensor of size 1 batch x 1 channel x height
                              x width. It is a normalized X-ray slice.

        prediction (torch tensor): A tensor of size 1 batch x 1 channel
                                   x height x width.

    Keyword args:
        alpha (float): A number between 0.0 and 1.0 that determines the opacity
                       of the colored mask.

        save (bool): Will save the image to save_path if True.

        show(bool): Will display the image if true.

        save_path (str): The absolute path (including file name) to where to
                         save the image if save is True.

    Returns:
        -
    """
    # Need to separate the prediction into boolean mask arrays
    air_mask = (prediction == 0)[0]
    weft_mask = (prediction == 1)[0]
    warp_mask = (prediction == 2)[0]
    matrix_mask = (prediction == 3)[0]
    masks = torch.stack([air_mask, matrix_mask, weft_mask, warp_mask])

    # draw_segmentation_masks requires RGB.
    # [batch, channel, height, width] --> [red, green, blue, height, width]
    rgb_input = input[0, 0, ...].repeat(3, 1, 1)

    input_with_mask = draw_segmentation_masks(
        rgb_input,
        masks=masks,
        alpha=alpha,
        colors=["red", "green", "yellow", "blue"],
    )

    pil_image = v2.ToPILImage()(input_with_mask)
    if save:
        pil_image.save(
            save_path,
        )
    if show:
        pil_image.show()

    return None


def one_epoch(
    model,
    criterion,
    optimizer,
    dataloaders,
    device,
    epoch=0,
    writer=None,
    scheduler=None,
    scaler=None,
):
    """Process one (1) epoch.


    Args:
        model (torch model): The model to train/validate.

        criterion (torch loss function): The loss function to use for training/
                                         validation.

        optimizer (torch optimizer): Optimizer to use for training.

        dataloaders (dict{torch dataloader}): Dataloaders for loading data for
                                              training and validation.

        device (torch device): The device (cpu/gpu) that will be used to load/
                               train the model.


    Keyword args:
        scheduler (torch learning rate scheduler): The learning rate scheduler.
                                                   to utilze during training.
                                                   Is called once per iteration
                                                   if not ReduceLROnPlateau.

        epoch (int): The current epoch number. Used to compute the iteration
                     number for the summary writer (if it exists).

        writer (torch summary writer): A summary writer object that logs the
                                       iteration loss values.

        scaler (torch amp grad scaler): Gradient scaler object that prevents
                                        overflow when working with float16.
                                        float16 casting is disabled if None.

    Returns:
        epoch_loss (float): The epoch loss. This is the sum of all batch losses
                            divided by the number of batches in the epoch.
    """
    num_batches = len(dataloaders["training"])
    num_val = len(dataloaders["validation"])

    repeats = num_batches // num_val + (num_batches % num_val > 0)

    iter_loader = {
        "training": iter(dataloaders["training"]),
        "validation": itertools.chain.from_iterable(
            itertools.repeat(dataloaders["validation"], repeats)
        ),
    }

    training_epoch_loss = 0.0
    validation_epoch_loss = 0.0

    for batch_idx in range(num_batches):
        loss_str = ""
        for mode in ["training", "validation"]:
            inputs, labels = next(iter_loader[mode])
            inputs = inputs.to(device)
            labels = labels.to(device)

            if mode == "training":
                model.train()
            else:
                model.eval()

            optimizer.zero_grad()
            with torch.set_grad_enabled(mode == "training"):
                with torch.amp.autocast(
                    device_type=str(device),
                    dtype=torch.float16,
                    enabled=not scaler is None,
                ):
                    outputs = model(inputs)
                    # Handle pytorch model.
                    if type(outputs) is collections.OrderedDict:
                        outputs = outputs["out"]
                    loss = criterion(outputs, labels)
                if mode == "training":
                    if scaler is None:
                        loss.backward()
                        optimizer.step()
                    else:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    if not scheduler is None and not isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step()
                    training_loss = loss.item()
                    loss_str += "{:.6f}, ".format(training_loss)
                    training_epoch_loss += training_loss
                else:
                    validation_loss = loss.item()
                    loss_str += "{:.6f}".format(validation_loss)
                    validation_epoch_loss += validation_loss
        if not writer is None:
            writer.add_scalars(
                "Loss",
                {"Training": training_loss, "Validation": validation_loss},
                epoch * num_batches + batch_idx,
            )
        progress_str = "{:5.2%}".format(batch_idx / num_batches)
        num_space = 24 - len(loss_str) - len(progress_str)
        print(
            "|| Progress: ",
            progress_str + " | Training, Validation: " + loss_str,
            " " * (num_space) + "||",
            end="\r",
        )

    print(" " * 67, end="\r")  # Clear such that following text prints cleanly.
    return (
        training_epoch_loss / num_batches,
        validation_epoch_loss / num_batches,
    )


def train_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    device,
    num_epochs,
    writer=None,
    start_epoch=0,
    end_epoch=None,
    best_loss=None,
    scheduler=None,
    state_dict_path=None,
    float16=False,
):
    """Train a torch model for one or more epochs.

    Args:
        model (torch model): The model to train.

        criterion (torch loss function): The loss function to use for training.

        optimizer (torch optimizer): Optimizer to use for training.

        dataloaders (dict(torch dataloader)): A dictionary containing two items.
                                              The first key "training" refers to
                                              the dataloader to use for training
                                              while the second key "validation"
                                              refers to the dataloader to use
                                              for validation.

        device (torch device): The device (cpu/gpu) that will be used to load/
                               train the model.

        num_epochs (int): The number of epochs to train the model.

    Keyword args:
        writer (torch summary writer): A summary writer object that logs the
                                       iteration loss values.

        start_epoch (int): The epoch number to start from. This is useful when
                           continuing training from a state dict.

        end_epoch (int): The epoch number to end at. This is useful when
                         training in one epoch at a time from a batch script.
                         The default value is None, which will result in
                         end_epoch being equal to num_epochs.

        best_loss (float): The best loss value so far. Only used to mark prints
                           as new current best. The default value none results
                           in an arbitrary large number. Useful to pass when
                           training one epoch at a time from a batch script.

        state_dict_path (str): The absolute path to where to save the state
                               dictionary. The function continuously saves the
                               model weights and optimizer state.

        scheduler (torch learning rate scheduler): The learning rate scheduler.
                                                   to utilze during training.
                                                   Is called once per epoch for
                                                   ReduceLROnPlateau and once
                                                   per iteration else.

        float16 (Bool): Will cast to float16 if True.
    Returns:
        -
    """
    if end_epoch is None:
        end_epoch = num_epochs
    scaler = torch.amp.GradScaler(str(device)) if float16 else None
    num_space = 32 - len(model.__class__.__name__) - len(str(num_epochs))
    print(
        "-" * 66 + "\n||" + " " * (num_space // 2 - 1),
        f"Training model: '{model.__class__.__name__}' for {num_epochs} epochs",
        " " * (num_space // 2 + (num_space % 2) - 1) + "||",
        "\n" + "-" * 66,
    )
    current_lr_str = "{:.6f}".format(scheduler.get_last_lr()[0])
    num_space = 36 - len(current_lr_str)
    print(
        "||" + " " * (num_space // 2 - 1),
        "Current learning rate is: " + current_lr_str,
        " " * (num_space // 2 + (num_space % 2) - 1) + "||",
    )
    print("-" * 66)
    if best_loss is None:
        best_loss = (
            10e6  # Just put a big value to ensure non-convergence at step 1.
        )

    for epoch in range(start_epoch, end_epoch):
        num_space = 55 - len(str(epoch + 1)) - len(str(num_epochs))
        print(
            "||" + " " * (num_space // 2 - 1),
            f"Epoch {epoch + 1}/{num_epochs}",
            " " * (num_space // 2 + (num_space % 2) - 1) + "||",
        )

        epoch_training_loss, epoch_validation_loss = one_epoch(
            model,
            criterion,
            optimizer,
            dataloaders,
            device,
            epoch=epoch,
            writer=writer,
            scheduler=scheduler,
            scaler=scaler,
        )
        training_loss_str = "{:.6f}".format(epoch_training_loss)
        validation_loss_str = "{:.6f}".format(epoch_validation_loss)
        num_space = 41 - len(training_loss_str)
        print(
            "||" + " " * (num_space // 2 - 1),
            "Training epoch loss: " + training_loss_str,
            " " * (num_space // 2 + (num_space % 2) - 1) + "||",
        )

        # Reduce LR on plateau should be called once per epoch.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_training_loss)
        if epoch_validation_loss < best_loss:
            best_loss = epoch_validation_loss
            validation_loss_str += " *"
        if state_dict_path is not None:
            torch.save(
                {
                    "epoch": epoch
                    + 1,  # +1 since we save after finishing epoch
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                state_dict_path,
            )
        num_space = 39 - len(validation_loss_str)
        print(
            "||" + " " * (num_space // 2 - 1),
            "Validation epoch loss: " + validation_loss_str,
            " " * (num_space // 2 + (num_space % 2) - 1) + "||",
        )

        current_lr_str = "{:.6f}".format(scheduler.get_last_lr()[0])
        num_space = 36 - len(current_lr_str)
        print(
            "||" + " " * (num_space // 2 - 1),
            "Current learning rate is: " + current_lr_str,
            " " * (num_space // 2 + (num_space % 2) - 1) + "||",
        )
        print("-" * 66)

    return None


def segment_slice_from_dataloader(model, dataloader, slice_idx):
    """Use model to segment the slice with index slice_idx from dataloader.


    Args:
        model (torch model): The model to utilize.

        dataloader (torch dataloader):  The dataloader from which the slice is
                                        to be segmented is loaded.

        slice_idx (int): The index of the slice to segment.

    Keyword args:
        -

    Returns:
        input (torch tensor):  A tensor of size 1 batch x num channels x height
                               x width, that contains the input that was used
                               for the segmentation model.

        prediction (torch tensor): A tensor of size 1 batch x 1 channel x height
                                   x width, that contains the predicted
                                   segmentation.
    """
    device = next(model.parameters()).device

    iterator = iter(dataloader)
    for _ in range(slice_idx):  # Skip all entries up until idx - 1
        next(iterator)
    input = next(iterator)
    if type(input) is list:  # Handle TIFFDataset vs TexRayDataset.
        input = input[0]
    input = input.to(device)

    prediction = model(input)["out"].argmax(1)

    return input, prediction


def segment_volume_from_slice_dataloader(model, dataloader, slice_axis="x"):
    """Use model to segment volume slice by slice from dataloader.

    Args:
        model (torch model): The model to utilize.

        dataloader (torch dataloader):  The dataloader from which the volume
                                        to be segmented is loaded.


    Keyword args:
        slice_axis (str): The axis along which the segmentation is sliced.

    Returns:

        segmented_volume (torch tensor): A tensor of size len(dataloader) ^ 3
                                         that contains the segmented volume.

    """
    device = next(model.parameters()).device

    dim = len(dataloader)
    segmented_volume = torch.zeros((dim, dim, dim), dtype=torch.uint8)
    segmented_volume = segmented_volume.to(device)
    iterator = iter(dataloader)
    for slice_idx, input in enumerate(iterator):
        input = input[0]  # Extract recon only
        input = input.to(device)
        prediction = model(input)
        if type(prediction) is collections.OrderedDict:  # Handle pytorch model.
            prediction = prediction["out"]
        prediction = prediction.argmax(1)
        segmented_volume[slice_idx] = prediction

    # We need to permute our axes back to match original tiff-file.
    if slice_axis == "x":
        segmented_volume = torch.transpose(segmented_volume, 1, 2)
        segmented_volume = torch.transpose(segmented_volume, 0, 2)
    elif slice_axis == "y":
        segmented_volume = torch.transpose(segmented_volume, 0, 1)
    elif slice_axis != "z":
        raise ValueError("slice_axis can only be 'x', 'y', or 'z'.")

    return segmented_volume


def segment_volume_from_patch_dataloader(
    model,
    dataloader,
    patch_size,
    stride,
    slice_axis="x",
    volume_shape=512,
    float16=False,
):
    """Use model to segment volume patch by patch from dataloader.

    Args:
        model (torch model): The model to utilize.

        dataloader (torch dataloader):  The dataloader from which the volume
                                        to be segmented is loaded.

        patch_size (int): The side length of the cubic patch of the volume
                          used as input.

        stride (int): The stride between input patches. If equal to patch_size
                      there will be no overlap or skipped voxels.


    Keyword args:
        volume_shape (int): The side length of the volume to segment.

        slice_axis (str): The axis that was had the first index in the array
                          during model training.

        float16 (Bool): Will cast to float16 if True.

    Returns:

        segmented_volume (torch tensor): A tensor of size len(dataloader) ^ 3
                                         that contains the segmented volume.

    """
    device = next(model.parameters()).device
    with torch.amp.autocast(
        device_type=str(device), dtype=torch.float16, enabled=float16
    ):
        dummy_in = torch.randn(1, 1, patch_size, patch_size, patch_size).to(
            device
        )
        dummy_seg = model(dummy_in)
        predictions = torch.zeros(
            (dummy_seg.shape[1], volume_shape, volume_shape, volume_shape),
            dtype=torch.float32,
        )
        iterator = iter(dataloader)

        patch_indices = []
        for i in range(0, volume_shape - patch_size + 1, stride):
            for j in range(0, volume_shape - patch_size + 1, stride):
                for k in range(0, volume_shape - patch_size + 1, stride):
                    patch_indices.append((i, j, k))
        patch_indices = numpy.array(patch_indices)

        for idx, input in enumerate(iterator):
            input = input[0]  # Extract recon only
            input = input.to(device)
            seg = model(input)
            if type(seg) is collections.OrderedDict:  # Handle pytorch model.
                seg = seg["out"]
            pred = seg.to("cpu").detach()
            predictions[
                :,
                patch_indices[idx][0] : patch_indices[idx][0] + patch_size,
                patch_indices[idx][1] : patch_indices[idx][1] + patch_size,
                patch_indices[idx][2] : patch_indices[idx][2] + patch_size,
            ] += pred[
                0
            ]  # If we use overlapping patches we sum up logits

    segmented_volume = predictions.argmax(0)
    # We need to permute our axes back to match original tiff-file.
    if slice_axis == "x":
        segmented_volume = torch.transpose(segmented_volume, 1, 2)
        segmented_volume = torch.transpose(segmented_volume, 0, 2)
    elif slice_axis == "y":
        segmented_volume = torch.transpose(segmented_volume, 0, 1)
    elif slice_axis != "z":
        raise ValueError("slice_axis can only be 'x', 'y', or 'z'.")

    return segmented_volume


def compute_dataset_loss(model, dataloader, device, loss_function):
    """Compute the loss of an entire data set.

    Args:
        model (torch model): The model to utilize.

        dataloader (torch dataloader):  The dataloader from which the data set
                                        to be evaluated is loaded.

        device (torch device): The device (cpu/gpu) that will be used to load
                               the model.

        loss_function (torch loss function):  The loss function with which to
                                              evaluate the loss.

    Keyword args:
        -

    Returns:
        mean_loss (float): The average loss of the data set.

        min_loss (float): The minimum loss in the data set.

        max_loss (float): The maximum loss in the data set.
    """
    min_loss = 1000.0
    max_loss = -1000.0
    mean_loss = 0.0
    i = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            i += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)["out"]
            loss = loss_function(outputs, labels)
            loss_val = loss.item()
            mean_loss += loss_val
            min_loss = loss_val if loss_val < min_loss else min_loss
            max_loss = loss_val if loss_val > max_loss else max_loss
        mean_loss /= len(dataloader)
    return mean_loss, min_loss, max_loss


if __name__ == "__main__":
    training_data_path = "./textomos/training_set"
    validation_data_path = "./textomos/validation_set"
    testing_data_path = "./textomos/testing_set"
    state_dict_path = "./textomos/state_dict_unet3d.pt"
    continue_from_state = True
    inferenece_input_path = (
        "./textomos/reconstructions/real_layer2layer_sample_reconstruction.tiff"
    )
    inferenece_output_path = "./textomos/ml_segmentation.tiff"
    data_mean = 0.21940
    data_std = 0.26584
    inference = True
    train = True
    normalize = True
    shuffle = True
    batch_size = 2
    num_epochs = 10
    learn_rate = 0.001
    weight_decay = 0.00001
    num_workers = 4
    rng_seed = 0
    model = UNet3D(1, 6)

    generator = seed_all(rng_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    end_epoch=num_epochs
    if len(sys.argv) > 1:
        end_epoch = int(sys.argv[1])
    if len(sys.argv) > 2:
        writer = SummaryWriter(sys.argv[2],flush_secs=1)
    else:
        writer = SummaryWriter(flush_secs=1)
    if train:
        training_set = TextomosDataset3D(
            training_data_path,
            128,
            128,
            z_score=(data_mean, data_std),
            separate_centers=True,
        )
        validation_set = TextomosDataset3D(
            validation_data_path,
            128,
            128,
            z_score=(data_mean, data_std),
            separate_centers=True,
        )
        testing_set = TextomosDataset3D(
            testing_data_path,
            128,
            128,
            z_score=(data_mean, data_std),
            separate_centers=True,
        )
        dataloaders = build_dataloaders_from_multiple_sets(
            [training_set, validation_set, testing_set],
            batch_size,
            num_workers,
            generator,
            shuffle=shuffle,
        )

        criterion = JaccardLoss()
        optimizer = optim.RAdam(
            model.parameters(),
            lr=learn_rate,
            weight_decay=weight_decay,
            decoupled_weight_decay=True,
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1.0,
            0.0,
            num_epochs * len(training_set) // batch_size,
        )

        start_epoch = 0
        best_loss = None
        if continue_from_state and os.path.isfile(state_dict_path):
            state_dict = torch.load(state_dict_path)
            start_epoch = state_dict["epoch"]
            best_loss = state_dict["best_loss"]
            model.load_state_dict(state_dict["model_state_dict"])
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        train_model(
            model,
            criterion,
            optimizer,
            dataloaders,
            device,
            num_epochs,
            writer=writer,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            best_loss=best_loss,
            scheduler=scheduler,
            state_dict_path=state_dict_path,
            float16=True,
        )

    if inference:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
        torch.torch.set_grad_enabled(False)
        test_set = TIFFDataset3D(
            inferenece_input_path,
            128,
            128,
            z_score=(data_mean, data_std),
            cutoff=(0.0, 0.65),
        )
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=1
        )
        volume = segment_volume_from_patch_dataloader(
            model, test_loader, 128, 128, float16=True
        )
        volume = volume.cpu().numpy().astype(numpy.int32)
        tifffile.imwrite(inferenece_output_path, volume)
