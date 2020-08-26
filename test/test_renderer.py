import numpy as np

from matplotlib import pyplot as plt

from src.renderer import fill_black, fill_black_for_loop, render_bev_map_with_thresholds, label_colors, apply_filter

try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp


def test_filter():
    import cv2
    from matplotlib import pyplot as plt
    img = cv2.imread('/home/henry/Pictures/global_map.png')
    # img = img[900:1300, 400:800]#  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[0:6000, 0:5000]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.canvas.manager.full_screen_toggle()

    img_filled = fill_black(img)
    print('done firts')
    img_filled = fill_black_for_loop(img_filled)

    cv2.imwrite('/home/henry/Pictures/global_map_rendered.png', img_filled)

    ax1.imshow(img)
    ax2.imshow(img_filled)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()


def convert_map_from_log_to_probability(map):
    """
    Convert the map from log value to probability
    Args:
        map: (H, W, C). The value is recorded in log. We will normalize them with respect to the channels.

    Returns: A new map with probability normalized in the channel axis.
    """
    # Normalize the map and convert it into probability
    channel_log_sum = logsumexp(map, axis=2)
    map_normalized = map - np.expand_dims(channel_log_sum, axis=2)
    map_normalized = np.exp(map_normalized)
    return map_normalized


def visualize_map_layer(filepath):
    """ Visualize each layer of the map """
    # Assume the map is from counting method.
    map = np.load(filepath)[300:1000, 400:800]

    unknown_region = (np.sum(map, axis=2) == 0)

    # Mask the unknown region of the map to zero.
    # map[unknown_region] = -1

    # If True,
    binarize = False

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    im1 = ax1.imshow((map[:, :, 0] > 0).astype(np.uint8) if binarize else map[:, :, 0])
    ax1.set_title('road layer')

    im2 = ax2.imshow((map[:, :, 1] > 0).astype(np.uint8) if binarize else map[:, :, 1])
    ax2.set_title('crosswalk layer')

    im3 = ax3.imshow((map[:, :, 2] > 0).astype(np.uint8) if binarize else map[:, :, 2])
    ax3.set_title('lane layer')

    im4 = ax4.imshow((map[:, :, 4] > 0).astype(np.uint8) if binarize else map[:, :, 4])
    ax4.set_title('sidewalk layer')

    if not binarize:
        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        fig.colorbar(im3, ax=ax3)
        fig.colorbar(im4, ax=ax4)

    # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()


def test_separate_map(filepath):
    map_local = np.load(filepath)

    # visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    im1 = ax1.imshow(map_local[:, :, 2])
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('lane layer')

    im2 = ax2.imshow((map_local[:, :, 2] > 0).astype(np.uint8))
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('larger than 5')

    im3 = ax3.imshow((map_local[:, :, 2] > 2).astype(np.uint8))
    fig.colorbar(im3, ax=ax3)
    ax3.set_title('larger than 10')

    im4 = ax4.imshow((map_local[:, :, 2] > 5).astype(np.uint8))
    fig.colorbar(im4, ax=ax4)
    ax4.set_title('larger than 20')

    plt.figure()
    plt.imshow((map_local[:, :, 2] > 0).astype(np.uint8))

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()


def test_render_portion(filepath):
    priority = [3, 4, 0, 2, 1]
    # map_local = np.load('/home/henry/Pictures/map_local_small.npy')[550:-200, 150:450]

    # np.random.seed(2077)
    # map_local = np.random.rand(300, 400, 5)
    map_local = np.load(filepath)

    # visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    im1 = ax1.imshow(
        render_bev_map_with_thresholds(map_local, label_colors, priority, thresholds=[0.01, 0.01, 0.01, 0.01, 0.01]))
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('side walk layer')

    im2 = ax2.imshow(
        render_bev_map_with_thresholds(map_local, label_colors, priority, thresholds=[0.1, 0.1, 0.5, 0.15, 0.05]))
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('larger than 0.05')

    map_local = apply_filter(map_local)

    im3 = ax3.imshow(render_bev_map_with_thresholds(map_local, label_colors,
                                                    priority, thresholds=[0.1, 0.1, 0.5, 0.1, 0.05]))
    fig.colorbar(im3, ax=ax3)
    ax3.set_title('larger than 0.15')

    im4 = ax4.imshow(render_bev_map_with_thresholds(map_local, label_colors,
                                                    priority, thresholds=[0.1, 0.1, 0.5, 0.12, 0.05]))
    fig.colorbar(im4, ax=ax4)
    ax4.set_title('larger than 0.5')

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()


if __name__ == '__main__':
    filepath = "/home/qinru/avl/playground/vision_semantic_segmentation/outputs/tune_the_threshold/version_2/map.npy"
    # visualize_map_layer(filepath)
    # test_separate_map(filepath)
    test_render_portion(filepath)