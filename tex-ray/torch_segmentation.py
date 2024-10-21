import random
import tifffile
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
    fcn_resnet50,
    FCN_ResNet50_Weights,
    fcn_resnet101,
    FCN_ResNet101_Weights,
)
from hdf5_utils import (
    get_reconstruction_chunk_handle,
    get_segmentation_chunk_handle,
    get_metadata,
    get_database_shape,
)


class TexRayDataset(Dataset):
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

        # Unsqueeze to create a channel axis for consistency.
        recon_slice = torch.tensor(
            self.recon_data[chunk_idx][local_idx][slice_idx]
        ).unsqueeze(0)
        if (not self.mean is None) and (not self.std is None):
            recon_slice = (recon_slice - self.mean) / self.std

        seg_slice = torch.tensor(self.seg_data[chunk_idx][local_idx][slice_idx])
        seg_mask = torch.zeros(
            (4, self.slices_per_sample, self.slices_per_sample)
        )
        for i in range(4):
            seg_mask[i] = seg_slice == i

        return recon_slice, seg_mask

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
                self.recon_data[chunk_idx][local_idx][:]
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
                self.recon_data[chunk_idx][local_idx][:]
            ).to(device)
            mean += torch.mean(recon_sample)
        mean /= self.num_samples

        for global_idx in range(self.num_samples):
            chunk_idx = global_idx // self.chunk_size
            local_idx = global_idx % self.chunk_size
            recon_sample = torch.tensor(
                self.recon_data[chunk_idx][local_idx][:]
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
            weights (torch tensor): A tensor of length 4 that contains the
                                    frequency balancing weights. They are:
                                    given as [air, matrix, weft, warp].
        """
        class_freq = torch.tensor([0, 0, 0, 0]).to(device)
        for global_idx in range(self.num_samples):
            chunk_idx = global_idx // self.chunk_size
            local_idx = global_idx % self.chunk_size
            seg_sample = torch.tensor(
                self.seg_data[chunk_idx][local_idx][:]
            ).to(device)
            for i in range(4):
                class_freq[i] += torch.sum(seg_sample == i)
        median = torch.median(class_freq)
        return median / class_freq


class TIFFDataset(Dataset):
    """See pytorch dataset class documentation for specifics of __method__
    type methods that are required by the dataset interface.

    Params:
        tiff_path (str): The absolute path to the tiff-file.

    Keyword params:
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
        slice_axis="x",
        z_score=(None, None),
        cutoff=(-1.0, 1.0),
    ):
        self.tiff_path = tiff_path

        self.mean = z_score[0]
        self.std = z_score[1]
        self.min_val = cutoff[0]
        self.max_val = cutoff[1]

        self.reconstruction = torch.tensor(tifffile.imread(tiff_path))
        if slice_axis == "x":
            self.reconstruction = torch.transpose(self.reconstruction, 0, 2)
            self.reconstruction = torch.transpose(self.reconstruction, 1, 2)
        elif slice_axis == "y":
            self.reconstruction = torch.transpose(self.reconstruction, 0, 1)
        elif slice_axis != "z":
            raise ValueError("slice_axis can only be 'x', 'y', or 'z'.")

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

        return recon_slice


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
    model="deeplabv3_resnet50", input_channels=1, pre_trained=False
):
    """Build the segmentation model. The model is adapted to use for segmenting
       images into 4 distinct classes.

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
    if model in ["deeplabv3_resnet50", "deeplabv3_resnet101"]:
        last_layer_input_size = 256

    # Replace output such that there are 4 classes to classify.
    segmentation_model.classifier[4] = nn.Conv2d(
        last_layer_input_size, 4, kernel_size=(1, 1), stride=(1, 1)
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
    save_path="./tex-ray/segmented_image.jpg",
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


def one_epoch(model, criterion, optimizer, dataloader, device, mode):
    """Process one (1) epoch. This can be either training or validation.


    Args:
        model (torch model): The model to train/validate.

        criterion (torch loss function): The loss function to use for training/
                                         validation.

        optimizer (torch optimizer): Optimizer to use for training.

        dataloader (torch dataloader): Dataloader for loading data for training/
                                       validation.

        device (torch device): The device (cpu/gpu) that will be used to load/
                               train the model.

        mode (str): If set to "training" the model will be trained, else the
                    model is validated.

    Keyword args:
        -

    Returns:
        epoch_loss (float): The epoch loss. This is the sum of all batch losses
                            divided by the number of batches in the epoch.
    """
    if mode == "training":
        header = "Training"
        model.train()
    else:
        header = "Evaluation"
        model.eval()

    num_batches = len(dataloader)
    epoch_loss = 0.0
    current_batch = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        current_batch += 1

        optimizer.zero_grad()
        with torch.set_grad_enabled(mode == "training"):
            outputs = model(inputs)["out"]
            loss = criterion(outputs, labels)
            if mode == "training":
                loss.backward()
                optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        loss_str = "{:.6f}".format(batch_loss)
        progress_str = "{:7.2%}".format(current_batch / num_batches)

        num_space = 28 - len(header) - len(loss_str) - len(progress_str)
        print(
            "||" + " " * (num_space // 2 - 1) + header + " progress: ",
            progress_str + " | Current batch loss: " + loss_str,
            " " * (num_space // 2 + (num_space % 2) - 1) + "||",
            end="\r",
        )

    print(" " * 67, end="\r")  # Clear such that following text prints cleanly.
    return epoch_loss / num_batches


def train_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    device,
    num_epochs,
    scheduler=None,
    state_dict_path=None,
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
        state_dict_path (str): The absolute path to where to save the state
                               dictionary. The function continuously saves the
                               set of weights that results in the current lowest
                               validation loss

        scheduler (torch learning rate scheduler): The learning rate scheduler.
                                                   to utilze during training.
                                                   Is called once per epoch.
    Returns:
        -
    """
    num_space = 32 - len(model.__class__.__name__) - len(str(num_epochs))

    print(
        "-" * 66 + "\n||" + " " * (num_space // 2 - 1),
        f"Training model: '{model.__class__.__name__}' for {num_epochs} epochs",
        " " * (num_space // 2 + (num_space % 2) - 1) + "||",
        "\n" + "-" * 66,
    )

    best_loss = (
        10e6  # Just put a big value to ensure non-convergence at step 1.
    )
    for epoch in range(num_epochs):
        num_space = 55 - len(str(epoch + 1)) - len(str(num_epochs))
        print(
            "||" + " " * (num_space // 2 - 1),
            f"Epoch {epoch + 1}/{num_epochs}",
            " " * (num_space // 2 + (num_space % 2) - 1) + "||",
        )

        for mode in ["training", "validation"]:
            epoch_loss = one_epoch(
                model, criterion, optimizer, dataloaders[mode], device, mode
            )
            loss_str = "{:.6f}".format(epoch_loss)
            if mode == "training":
                num_space = 41 - len(loss_str)
                print(
                    "||" + " " * (num_space // 2 - 1),
                    "Training epoch loss: " + loss_str,
                    " " * (num_space // 2 + (num_space % 2) - 1) + "||",
                )
            if mode == "validation":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    loss_str += " *"
                    if state_dict_path is not None:
                        torch.save(model.state_dict(), state_dict_path)
                num_space = 39 - len(loss_str)
                print(
                    "||" + " " * (num_space // 2 - 1),
                    "Validation epoch loss: " + loss_str,
                    " " * (num_space // 2 + (num_space % 2) - 1) + "||",
                )

        if scheduler is not None:
            scheduler.step()
            current_lr_str = "{:.6f}".format(scheduler.get_last_lr()[0])
            num_space = 35 - len(current_lr_str)
            print(
                "||" + " " * (num_space // 2 - 1),
                "Updating learning rate to: " + current_lr_str,
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


def segment_volume_from_dataloader(model, dataloader, slice_axis="x"):
    """Use model to segment slice with index slice_idx from dataloader.

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
        input = input.to(device)
        prediction = model(input)["out"].argmax(1)
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


if __name__ == "__main__":
    # max: 4.202881813049316
    # min: 0.0
    # mean: 0.20771875977516174
    # std: 0.25910180825541423
    # balancing weights: [0.1242, 1.0038, 1.0000, 0.6890]
    training_data_path = "./tex-ray/training_set"
    validation_data_path = "./tex-ray/validation_set"
    testing_data_path = "./tex-ray/testing_set"
    state_dict_path = "./tex-ray/state_dict.pt"
    inferenece_input_path = "./tex-ray/reconstructions/reconstruction.tiff"
    inferenece_output_path = "./tex-ray/ml_segmentation.tiff"

    data_mean = 0.20772
    data_std = 0.25910
    data_weight = [0.1242, 1.0038, 1.0000, 0.6890]
    inference = True
    train = True
    normalize = True
    batch_size = 8
    num_epochs = 20
    learn_rate = 0.001
    weight_decay = 0.001
    momentum = 0.9
    num_workers = 4
    rng_seed = 0

    generator = seed_all(rng_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model = model.to(device)

    if train:
        training_set = TexRayDataset(
            training_data_path, z_score=(data_mean, data_std)
        )
        validation_set = TexRayDataset(
            validation_data_path, z_score=(data_mean, data_std)
        )
        testing_set = TexRayDataset(
            testing_data_path, z_score=(data_mean, data_std)
        )
        weight = torch.tensor(data_weight).to(device)
        dataloaders = build_dataloaders_from_multiple_sets(
            [training_set, validation_set, testing_set],
            batch_size,
            num_workers,
            generator,
        )

        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = optim.SGD(
            model.parameters(),
            lr=learn_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=num_epochs, power=0.9
        )

        train_model(
            model,
            criterion,
            optimizer,
            dataloaders,
            device,
            num_epochs,
            scheduler=scheduler,
            state_dict_path=state_dict_path,
        )

    if inference:
        model.load_state_dict(torch.load(state_dict_path))
        model.eval()
        test_set = TIFFDataset(
            inferenece_input_path, z_score=(data_mean, data_std)
        )
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=1
        )
        volume = segment_volume_from_dataloader(model, test_loader)
        volume = volume.cpu().numpy()
        tifffile.imwrite(inferenece_output_path, volume)
