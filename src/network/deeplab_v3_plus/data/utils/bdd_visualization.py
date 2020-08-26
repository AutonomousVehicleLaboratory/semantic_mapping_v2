import numpy as np

from matplotlib import pyplot as plt

import deeplab_v3_plus.data.dataset.bdd as bdd

BDD_SEM_SEG_IGNORE_INDEX = 255
BDD_SEM_SEG_VOID_INDEX = 20
BDD_SEM_SEG_COLORMAP = {}
# BDD semantic segmentation colormap; Maps label to color
for l in bdd.labels:
    if l.trainId == BDD_SEM_SEG_IGNORE_INDEX:
        # Maps all the ignore labels to (0, 0, 0)
        BDD_SEM_SEG_COLORMAP[BDD_SEM_SEG_VOID_INDEX] = (0, 0, 0)
    else:
        BDD_SEM_SEG_COLORMAP[l.trainId] = l.color


def convert_label_to_color(label_array, colormap=None):
    """
    Convert segmentation class label into RGB color
    Args:
        label_array: array with shape (b x h x w)
        colormap: Maps labels to RGB colors

    Returns:
        color image: (b x h x w x 3)

    """
    assert len(label_array.shape) == 3
    num_batch, height, width = label_array.shape

    if colormap is None:
        colormap = BDD_SEM_SEG_COLORMAP

    color_label = np.zeros((num_batch, height, width, 3), dtype=np.uint8)
    for index, color in colormap.items():
        color_label[label_array == index] = color
    return color_label


def display_labels(blocking=False, alpha=1):
    """
    Display the semantic segmentation labels

    Modified from: https://matplotlib.org/2.0.2/examples/color/named_colors.html

    Args:
        blocking (bool): True if the plot blocks the process
        alpha (value): The transparent_alpha value of RGBA color.
    """
    # Maps the trainId to label names
    id_to_name = {}
    for l in bdd.labels:
        if l.trainId == BDD_SEM_SEG_IGNORE_INDEX:
            id_to_name[BDD_SEM_SEG_VOID_INDEX] = "void"
        else:
            id_to_name[l.trainId] = l.name

    num_id = len(id_to_name)
    num_row = num_id
    num_col = 1
    fig, ax = plt.subplots(figsize=(4, 5))

    # Get figure's height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    height = Y / (num_row + 1)
    width = X / num_col
    for i, id in enumerate(id_to_name):
        name = id_to_name[id]
        if name == "void":
            # For void class, it is pure black.
            color = np.array(list(BDD_SEM_SEG_COLORMAP[id]))
        else:
            color = np.array(list(BDD_SEM_SEG_COLORMAP[id]) + [alpha * 255])
        # Normalize color to 0-1 range because ax.hlines() requires it
        color = color / 255

        col = i % num_col
        row = i // num_col
        y = Y - (row * height) - height

        xi_line = width * (col + 0.05)
        xf_line = width * (col + 0.25)
        xi_text = width * (col + 0.3)
        ax.text(xi_text, y, name, fontsize=(height * 0.8), horizontalalignment='left', verticalalignment='center')

        ax.hlines(y + height * 0.1, xi_line, xf_line, color=color, linewidth=(height * 0.6))

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    if blocking:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.001)
