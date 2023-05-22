import argparse
import numpy as np
import os
import os.path as osp
import sys

# Assume this program is called at deeplab/deeplab_v3_plus/data/utils
sys.path.insert(0, osp.abspath('../../../'))

import deeplab_v3_plus.data.dataset.bdd as bdd

from collections import defaultdict
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="BDD100K Data Preparation")

    parser.add_argument(
        "-l", "--label_dir",
        required=True,
        default="",
        type=str,
        help="path to the original label",
    )
    parser.add_argument(
        "-s", "--save_dir",
        default="",
        type=str,
        help="path to save the preprocessed data. (default: [label_dir]_preprocessed in the same folder as [label_dir])",
    )
    parser.add_argument(
        "-t", "--task",
        default="sem_seg",
        type=str,
        help="preprocess task. It can be semantic segmentation (sem_seg)."  # [other task to be supported in the future]
    )
    parser.add_argument(
        "-w", "--workers",
        default=os.cpu_count(),
        metavar="INT",
        help="number of workers that run in parallel [default is the number of CPU cores you have]."
    )

    return parser.parse_args()


def sem_seg_convert_color_to_label(input):
    """
    This is the worker function that converts color into semantic segmentation labels

    Note that Pool requires that worker function must be a top-level function.

    Args:
        input: We pack all the inputs into a tuple so that this worker function can be processed by Pool.map()
            load_path (str)
            save_path (str)
            IGNORE_INDEX (int) Index indicates that the label of the pixel is not available.
            VOID_INDEX (int) We will map all the unlabeled pixel to this void index
    """
    load_path, save_path, IGNORE_INDEX, VOID_INDEX = input
    label_image = np.array(Image.open(load_path))
    # If label has 4 channels, remove the last transparent_alpha channel
    if label_image.shape[2] != 3:
        label_image = label_image[:, :, :3]

    # Convert label to class index
    out_image = np.full(label_image.shape[:-1], VOID_INDEX)
    for l in bdd.labels:
        out_image[(label_image == l.color).all(axis=2)] = l.trainId if l.trainId != IGNORE_INDEX else VOID_INDEX

    out_image = Image.fromarray(out_image.astype(np.uint8()))
    out_image.save(save_path)


def generate_semantic_segmentation(label_dir, save_dir, num_workers):
    """
    Convert RGB value into semantic segmentation labels
    We add a void class into the label.

    Args:
        label_dir: absolute path to the color_labels
        save_dir: absolute path to save the output images

    """
    # Map Ignore index to void class
    IGNORE_INDEX = 255
    VOID_INDEX = 19  # Remember indexing starts from 0

    # Get the file list in the label directory
    # label directory contains multiple directories: train and val
    # filenames is a dictionary with the name of subdirectory as key and the list of files in that subdirectory as value
    filenames = defaultdict(list)
    for d in os.listdir(label_dir):
        dir_path = osp.join(label_dir, d)
        assert osp.isdir(dir_path)

        # Get the file list in the current directory
        for f in os.listdir(dir_path):
            assert osp.isfile(osp.join(dir_path, f))
            filenames[dir_path].append(f)

    # Convert them into semantic segmentation label
    for dir, file_list in filenames.items():
        save_subdir = osp.join(save_dir, osp.basename(dir))
        os.makedirs(save_subdir, exist_ok=True)
        print("Processing files in {}".format(dir))

        # Build the absolute load path and save path for each file
        worker_input = [(osp.join(dir, file), osp.join(save_subdir, file), IGNORE_INDEX, VOID_INDEX)
                        for file in file_list]
        num_tasks = len(worker_input)

        with Pool(num_workers) as p:
            # chunksize must be 1 so that tqdm has a correct speed estimation
            # Wrap the tqdm with a list so that we can trigger the iterator
            list(tqdm(p.imap_unordered(sem_seg_convert_color_to_label, worker_input, chunksize=1), total=num_tasks))

    print("Complete the process of semantic segmentation data.\nSaved in:{}".format(save_dir))


def main():
    args = parse_args()

    label_dir = osp.abspath(args.label_dir)
    save_dir = args.save_dir
    # If save_dir is not provided, by default we save it in the same folder as label directory
    if not save_dir:
        root_dir = osp.dirname(label_dir)
        base_name = osp.basename(label_dir)
        save_dir = osp.join(root_dir, "{}_preprocessed".format(base_name))

    # Create save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    num_workers = args.workers
    assert num_workers is not None and num_workers > 0, "Invalid number of workers {}".format(num_workers)

    if args.task == "sem_seg":
        generate_semantic_segmentation(label_dir, save_dir, num_workers)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    """
    Usage:
    python bdd_preparation.py -l [path to]/bdd100k/seg/color_labels
    """
    main()
