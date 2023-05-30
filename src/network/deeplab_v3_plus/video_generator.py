import argparse
import cv2
import numpy as np
import os
import os.path as osp
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, osp.dirname(__file__) + '/..')

import deeplab_v3_plus.data.utils.bdd_visualization as bdd_visl
import deeplab_v3_plus.data.utils.mapillary_visualization as mapillary_visl

from core.utils.benchmark import model_timer
from core.utils.checkpoint import Checkpoint
from deeplab_v3_plus.models.build import build_model
from tqdm import tqdm


def visualize_label(cfg):
    """Visualize labels"""
    if cfg.TRAIN_DATASET == "BDD":
        bdd_visl.display_labels()
    elif cfg.TRAIN_DATASET == "Mapillary":
        mapillary_visl.display_labels(cfg.TRAIN_DATASET_DIR)
    else:
        raise NotImplementedError


def generate_semantic_image(input, output, color_fn, *color_fn_args, transparent_alpha=0.5):
    """
    Generate semantic segmentation image.

    Args:
        input (tensor): Input to the network. Size (1 , c, h, w) - c is the color channels
        output (tensor): Output to the network. Size (1, n, h, w) - n is number of classes
        transparent_alpha (float): Transparent parameter. The larger it is, the lighter the labels are. In range [0,1]

    Returns:
        A dictionary that contains
        - Denormalize input image
        - argmax network prediction
        - colored network prediction
        - Blended Semantic segmentation image (in BGR format)
    """
    image = input.squeeze().cpu().numpy()
    image = np.transpose(image, axes=(1, 2, 0))

    # Denormalize the input image
    image *= np.array((0.229, 0.224, 0.225))
    image += np.array((0.485, 0.456, 0.406))
    image = np.clip(image, 0, None)
    image = image.astype(np.float)  # Make sure image has the same type as color_output

    # Resize the image to match the dimension of network output
    height, width = output.shape[2:]
    image = cv2.resize(image, (width, height))

    output = torch.argmax(output, dim=1).cpu().numpy()
    colored_output = color_fn(output, *color_fn_args)
    colored_output = np.squeeze(colored_output)
    colored_output = colored_output.astype(np.float)

    # normalize color output to the scale of [0,1] because opencv will rescale a float type back to 0-255 range
    colored_output /= 255

    blended_image = cv2.addWeighted(image, transparent_alpha, colored_output, 1 - transparent_alpha, 0)

    # Because opencv imshow use BGR color ordering, we need to transpose our RGB image into BGR
    blended_image = blended_image[:, :, [2, 1, 0]]
    colored_output = colored_output[:, :, [2, 1, 0]]

    out_dict = {
        "image": image,
        "pred": output,
        "colored_pred": colored_output,
        "blended_image": blended_image,
    }

    return out_dict


def generate_video(cfg, output_dir):
    # Build model
    model = build_model(cfg)[0]
    model = nn.DataParallel(model).cuda()
    model.eval()

    # Setup checkpoint
    checkpoint = Checkpoint(model, save_dir=output_dir)
    checkpoint.load(cfg.MODEL.WEIGHT, resume=False, resume_states=False)

    # Determine the color function to plot labels
    if cfg.TRAIN_DATASET == "BDD":
        color_fn = bdd_visl.convert_label_to_color
        color_fn_args = None
    elif cfg.TRAIN_DATASET == "Mapillary":
        color_fn = mapillary_visl.apply_color_map
        color_fn_args = mapillary_visl.get_labels(cfg.TRAIN_DATASET_DIR)
    else:
        raise NotImplementedError

    # Load the video
    video_path = osp.abspath(cfg.DATASET.ROOT_DIR)
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    num_frame = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Loaded video from {path}.\nHeight:{height} Width:{width} FPS:{fps}".format(
        path=video_path,
        height=height,
        width=width,
        fps=fps
    ))

    # Output video parameters
    # Note that the output size of the VideoWriter must be the same as the input image size, otherwise VideoWriter will fail silently!
    # Reference:
    # https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    # out_image_height = 1440
    # out_image_width = 1920
    out_image_height = 356
    out_image_width = 476 * 2
    fps = 10  # FPS is measured from model_timer

    # Build transform
    transform = T.Compose([
        T.ToPILImage(),
        # T.Resize((out_image_height, out_image_width)),
        # T.CenterCrop((out_image_height, out_image_width)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True),
    ])

    # Create a Video Writer
    if cfg.OUTPUT_NAME:
        video_name = cfg.OUTPUT_NAME
    else:
        video_name = osp.splitext(osp.basename(video_path))[0] + "_out"
    save_path = osp.join(output_dir, "{}.avi".format(video_name))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # If XVID is not working, consider use MJPG
    vwriter = cv2.VideoWriter(save_path, fourcc, fps, (out_image_width, out_image_height))

    # Visualize label
    visualize_label(cfg)

    progress_bar = tqdm(total=num_frame)
    with torch.no_grad():
        while (vcapture.isOpened()):
            success, image = vcapture.read()
            if success == True:
                assert image.shape[2] == 3  # Assert image has only three channels
                image_tensor = transform(image)
                image_tensor = image_tensor.unsqueeze(dim=0).cuda()
                preds = model(image_tensor, upsample_pred=False)
                # preds = model_timer(model, image_tensor, upsample_pred=False)

                image_dict = generate_semantic_image(image_tensor, preds, color_fn, color_fn_args,
                                                     transparent_alpha=0.5)

                # Stack the image side by side
                blended_image = image_dict["blended_image"]
                colored_pred = image_dict["colored_pred"]
                display_image = np.concatenate((blended_image, colored_pred), axis=1)

                # Resize the image for display
                # blended_image_for_display = cv2.resize(blended_image, (out_image_width, out_image_height,))
                # cv2.imshow("Semantic Segmentation Output", display_image)

                # Save the image
                # Video writer requires the input array to be uint8. Pretty stupid but this is how to handle it.
                vwriter.write((display_image * 255).astype(np.uint8))

                progress_bar.update(1)

                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # When everything done, release the video capture object
        vcapture.release()
        vwriter.release()
        # Closes all the frames
        cv2.destroyAllWindows()

        progress_bar.close()


def parse_args():
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser(description='DeepLab Training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from deeplab_v3_plus.config.demo import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace('experiments', 'outputs')
        output_dir = output_dir.replace('@', config_path)
        os.makedirs(output_dir, exist_ok=True)

    generate_video(cfg, output_dir)


if __name__ == '__main__':
    main()
