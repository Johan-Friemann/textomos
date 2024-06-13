import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from torch_dataset import TexRayDataset, TIFFDataset
from torch_train import train_model


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


def build_dataloaders(
    dataset,
    batch_size,
    num_workers,
    split=[0.8, 0.2],
    shuffle=True,
):
    """Build the model dataloader dictionary.

    Args:
        dataset (pytorch dataset): The data set to use for constructing the
                                   loaders.

        batch_Size (int): The dataloader batch size.

        num_workers (int): The number of cpu workers to use while loading.

    Keyword args:
        split (list[float]): A list of either length 2 or 3 whose entries sum to
                             1. If length 2: train/validation split, if length 3
                             train/validation/test split.

        shuffle (bool): Will shuffle the datasets if True.

    Returns:
        dataloaders (dict[pytorch dataloader]): The dataloader dictionary.
    """

    if len(split) == 2:
        training, validation = torch.utils.data.random_split(dataset, split)
    elif len(split) == 3:
        training, validation, testing = torch.utils.data.random_split(
            dataset, split
        )
    else:
        raise ValueError("Dataloader split can only be of length 2 or 3.")

    if sum(split) != 1.0:
        raise ValueError("Dataloader split must sum to 1.")

    dataloaders = {
        "training": DataLoader(
            training,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "validation": DataLoader(
            validation,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
    }

    if len(split) == 3:
        dataloaders["testing"] = DataLoader(
            testing,
            batch_size=batch_size,
            shuffle=True,
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

        prediction (torch tensor): A tensor of size 1 batch x 4 channels
                                   x height x width. Each channel refers to the
                                   probabilities of a pixel belonging to the
                                   respective material classes.

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
    air_mask = (prediction.argmax(1) == 0)[0]
    matrix_mask = (prediction.argmax(1) == 1)[0]
    weft_mask = (prediction.argmax(1) == 2)[0]
    warp_mask = (prediction.argmax(1) == 3)[0]
    masks = torch.stack([air_mask, matrix_mask, weft_mask, warp_mask])

    # draw_segmentation_masks requires RGB.
    # [batch, channel, height, width] --> [red, green, blue, height, width]
    rgb_input = input[0, 0, ...].repeat(3, 1, 1)

    input_with_mask = draw_segmentation_masks(
        rgb_input,
        masks=masks,
        alpha=alpha,
        colors=["red", "blue", "green", "yellow"],
    )

    pil_image = v2.ToPILImage()(input_with_mask)
    if save:
        pil_image.save(
            save_path,
        )
    if show:
        pil_image.show()

    return None


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_idx = 250

    batch_size = 8
    num_epochs = 100
    learn_rate = 0.001
    weight_decay = 0.001
    momentum = 0.9
    num_workers = 4
    state_dict_path = "./tex-ray/state_dict.pt"
    normalize = True

    model = build_model()
    model = model.to(device)

    train = True
    if train:
        dataset = TexRayDataset("./tex-ray/dbase", normalize=normalize)
        weight = dataset.get_loss_weights().to(device)
        dataloaders = build_dataloaders(dataset, batch_size, num_workers)

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

    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    test_set = TIFFDataset(
    "./tex-ray/reconstructions/real_binned_recon.tiff", slice_axis="x"
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=1
    )

    iterator = iter(test_loader)
    for i in range(test_idx):
        next(iterator)
    inputs = next(iterator)
    if type(inputs) is list:  # Handle TIFFDataset vs TexRayDataset.
        inputs = inputs[0]
    inputs[inputs > 1.0] = 1.0 # Remove zingers that can mess up normalization.
    inputs = inputs.to(device)

    # We normalize here instead of inside the dataloader such that we can show
    # the original image together with the segmentation.
    transformed_input = (inputs - torch.min(inputs)) / (
        torch.max(inputs) - torch.min(inputs)
    )
    transformed_input = (
        transformed_input - torch.mean(transformed_input)
    ) / torch.std(transformed_input)

    prediction = model(transformed_input)["out"]
    draw_image_with_masks(inputs, prediction, alpha=0.2)
