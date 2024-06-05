import torch
from torch.utils.data import Dataset
from hdf5_utils import (
    get_reconstruction_chunk_handle,
    get_segmentation_chunk_handle,
    get_metadata,
    get_database_shape,
)


"""
This file contains a custom pytorch dataset class that is tailormade for the
Tex-Ray hdf5 database format.
"""


class TexRayDataset(Dataset):
    """See pytorch dataset class documentation for specifics of __method__
    type methods that are required by the dataset interface.

    Params:
        database_path (str): The absolute path to the database folder.

    Keyword params:
        transform (pytorch transform): The transform to apply to the input data.
    """

    def __init__(self, database_path, transform=None):
        self.database_path = database_path

        self.transform = transform

        # data_0 is guaranteed to exist
        detector_rows = get_metadata(database_path, 0, "detector_rows")
        binning = get_metadata(database_path, 0, "binning")
        self.slices_per_sample = detector_rows // binning
        self.num_samples, self.num_chunks, self.chunk_size = get_database_shape(
            self.database_path
        )

        self.recon_data = []
        self.seg_data = []
        for i in range(self.num_chunks):
            recon_file_handle = get_reconstruction_chunk_handle(
                self.database_path, i
            )
            self.recon_data.append([])
            seg_file_handle = get_segmentation_chunk_handle(
                self.database_path, i
            )
            self.seg_data.append([])
            for j in range(self.chunk_size):
                recon_data_handle = recon_file_handle.get(
                    "reconstruction_" + str(j)
                )
                if recon_data_handle is not None:
                    self.recon_data[i].append(recon_data_handle)
                seg_data_handle = seg_file_handle.get("segmentation_" + str(j))
                if seg_data_handle is not None:
                    self.seg_data[i].append(seg_data_handle)

    def __len__(self):
        return self.num_samples * self.slices_per_sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        global_idx = idx // self.slices_per_sample
        chunk_idx = global_idx // self.chunk_size
        local_idx = global_idx % self.chunk_size
        slice_idx = idx % self.slices_per_sample

        recon_slice = torch.tensor(
            self.recon_data[chunk_idx][local_idx][slice_idx]
        )
        if self.transform:
            recon_slice = self.transform(recon_slice)

        seg_slice = torch.tensor(self.seg_data[chunk_idx][local_idx][slice_idx])
        seg_mask = torch.zeros(
            (4, self.slices_per_sample, self.slices_per_sample)
        )
        for i in range(4):
            seg_mask[i] = seg_slice == i

        return recon_slice, seg_mask

    def get_loss_weights(self):
        """Get dataset median frequency balancing weights.

        Args:
            -

        Keyword args:
            -

        Returns:
            weights (torch tensor): A tensor of length 4 that contains the
                                    frequency balancing weights. They are:
                                    given as [air, matrix, weft, warp].
        """
        class_freq = torch.tensor([0, 0, 0, 0])
        for idx in range(self.num_samples * self.slices_per_sample):
            global_idx = idx // self.slices_per_sample
            chunk_idx = global_idx // self.chunk_size
            local_idx = global_idx % self.chunk_size
            slice_idx = idx % self.slices_per_sample

            seg_slice = torch.tensor(
                self.seg_data[chunk_idx][local_idx][slice_idx]
            )
            for i in range(4):
                class_freq[i] += torch.sum(seg_slice == i)

        median = torch.median(class_freq)
        return median / class_freq
