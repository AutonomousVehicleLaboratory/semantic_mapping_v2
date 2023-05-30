import numpy as np
import torch

from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def make_grid_np(image, num_rows=1):
    """
    A numpy implementation of make_grid

    Modified from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    Args:
        image: Input image with shape (batch, height, width, intensity)
        num_rows: Number of rows in the output image

    Returns:
        Reshaped output image

    """
    assert len(image.shape) == 4
    num_batch, height, width, intensity = image.shape
    num_cols = num_batch // num_rows
    assert num_batch == num_rows * num_cols

    # Want result.shape == (height*num_rows, width*num_cols, intensity)
    image = image.reshape(num_rows, num_cols, height, width, intensity)
    image = np.swapaxes(image, 1, 2).reshape(height * num_rows, width * num_cols, intensity)
    return image


def visualize_training_data(data_batch):
    """
    Visualize the training data fed into the network
    Args:
        data_batch: A dictionary with two keys "image" and "label". Both are tensor.
            - image is normalized using the same formula here https://pytorch.org/docs/stable/torchvision/models.html
            - label: assume index 255 means ignored label
    """
    image = make_grid(data_batch["image"]).cpu().numpy()
    image = np.transpose(image, axes=(1, 2, 0))
    label = make_grid(data_batch["label"].unsqueeze(dim=1)).cpu().numpy()
    label = np.transpose(label, axes=(1, 2, 0))

    # Denormalize the data
    image *= np.array((0.229, 0.224, 0.225))
    image += np.array((0.485, 0.456, 0.406))
    image = np.clip(image, 0, None)

    # Remove additional identical channels inflated by make_grid()
    label = label[:, :, 0]

    # Reduce the value of 255 so other colors can stand out
    label[label == 255] = -1

    ax = plt.subplot(1, 2, 1)
    ax.imshow(image)
    ax.set_title("Image")
    ax = plt.subplot(1, 2, 2)
    ax.imshow(label)
    ax.set_title("Label")
    plt.show()


def visualize_network_output(data_batch, preds, color_fn, *color_fn_args, tb_writer=None, tag="", step=0):
    """
    Visualize the network output

    Args:
        data_batch (tensor): Data feeds into the network
        preds (tensor): Raw network output
        color_fn: A function that converts label to color
        color_fn_args: color_fn arguments
        tb_writer: TensorBoard writer. If None, then we will plot images with matplotlib
        tag (str): For TensorBoard, the root tag of the images
        step (int): For TensorBoard, the step
    """
    image = make_grid(data_batch["image"]).cpu().numpy()
    image = np.transpose(image, axes=(1, 2, 0))

    # Denormalize the image
    image *= np.array((0.229, 0.224, 0.225))
    image += np.array((0.485, 0.456, 0.406))
    image = np.clip(image, 0, None)

    label = data_batch["label"].cpu().numpy()
    color_label = color_fn(label, *color_fn_args)
    color_label = make_grid_np(color_label)

    preds_np = torch.argmax(preds, dim=1).cpu().numpy()
    color_preds = color_fn(preds_np, *color_fn_args)
    color_preds = make_grid_np(color_preds)

    if tb_writer is not None:
        tb_writer.add_image("{}/input".format(tag), image, dataformats='HWC', global_step=step)
        tb_writer.add_image("{}/label".format(tag), color_label, dataformats='HWC', global_step=step)
        tb_writer.add_image("{}/pred".format(tag), color_preds, dataformats='HWC', global_step=step)
    else:
        plt.figure()
        ax = plt.subplot(3, 1, 1)
        ax.imshow(image)
        ax = plt.subplot(3, 1, 2)
        ax.imshow(color_label.astype(int))
        ax = plt.subplot(3, 1, 3)
        ax.imshow(color_preds.astype(int))
        plt.show()
