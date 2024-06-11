import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks


def build_model():
    """DOCSTRING"""
    return None


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

    rgb_input = input.repeat(3, 1, 1)  # PIL requires RGB, becomes greyscale.

    input_with_mask = draw_segmentation_masks(
        rgb_input,
        masks=masks,
        alpha=alpha,
        colors=["red", "blue", "green", "yellow"],
    )

    pil_image = to_pil_image(input_with_mask)
    if save:
        pil_image.save(
            save_path,
        )
    if show:
        pil_image.show()

    return None

def test_model():
    """DOCSTRING"""
    return None


if __name__ == "__main__":
    print("main")
