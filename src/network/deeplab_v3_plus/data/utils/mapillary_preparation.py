import argparse
import json
import numpy as np
import os
import os.path as osp
import sys

from multiprocessing import Manager, Pool, Process
from PIL import Image
from shutil import copyfile
from tqdm import tqdm

# Assume this program is called at deeplab/deeplab_v3_plus/data/utils
sys.path.insert(0, osp.abspath('../../../'))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Determine the label generation scheme
# white_list - only include the labels included in the WHITE_LIST
# black_list - exclude the labels in IGNORE_LABEL
SCHEME = "white_list"

if SCHEME == "white_list":
    WHITE_LIST = [2, 8, 13, 15, 17, 19, 20, 21, 24, 27, 30, 41, 45, 50, 52, 54, 55, 57, 61]

    # Merge values to key
    # The number is the label index in original label system
    #
    # Merge terrian into vegetation
    # Merge all human type into human-person
    # Merge traffic-sign--back into traffic-sign--front
    # Merge motorcycle into bicycle
    MERGE_LABEL = {
        8: [23],    # crosswalk
        17: [16],   # bridge to building
        19: [22],   # human
        30: [29],   # vegetation
        50: [49],   # traffic sign
    }
    # Build a value to key map
    MERGE_LABEL_REVERSE_MAP = {i: k for k, v in MERGE_LABEL.items() for i in v}

    # Assert the keys of MERGE_LABEL are in the WHITE_LIST
    for k in MERGE_LABEL.keys():
        assert k in WHITE_LIST

    # For id that is not in the WHITE_LIST or not a value of a key in MERGE_LABEL, mark it as IGNORE_LABEL
    IGNORE_LABEL = [i for i in range(66) if i not in WHITE_LIST and i not in MERGE_LABEL_REVERSE_MAP]
elif SCHEME == "black_list":
    # Ignored labels do not produce gradient
    IGNORE_LABEL = [0, 1, 12, 18, 22, 25, 26, 28, 31, 34, 36, 37, 39, 40, 42, 43, 46, 53, 56, 58, 62, 63, 65]
    # Merge values to key
    # The number is the label index in original label system
    #
    # Do not merge the pedestrian area and side walk. Side walk must have road around whereas pedestrain area does not have road around
    # Merge service land into road
    # Merge curb-cut into side walk
    # Merge guard-rail into other-barrier
    # Merge fence into wall
    MERGE_LABEL = {
        13: [14],
        15: [9],
        5: [4],
        6: [3],
    }
    # Build a value to key map
    MERGE_LABEL_REVERSE_MAP = {i: k for k, v in MERGE_LABEL.items() for i in v}
else:
    raise NotImplementedError

# Index used to indicate the ignore label
# Maximum number is 255!
IGNORE_INDEX = 255


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_dir_list(dir):
    """Collect all the directories in the dir"""
    content = os.listdir(dir)
    dir_list = []
    for c in content:
        if osp.isdir(osp.join(dir, c)):
            dir_list.append(c)
    return dir_list


def get_file_list(dir, no_ext=False):
    """Collect all the file in the dir
    If no_ext, remove file extension
    """
    content = os.listdir(dir)
    file_list = []
    for c in content:
        if osp.isfile(osp.join(dir, c)):
            if no_ext:
                c = osp.splitext(c)[0]
            file_list.append(c)
    return file_list


def progress_monitor(total_size, progress_queue):
    """
    Monitor the progress with progress bar
    Args:
        total_size: total size of the task
        progress_queue: progress_queue
    """
    count = 0
    pbar = tqdm(total=total_size)
    while count < total_size:
        c = progress_queue.get()
        count += c
        pbar.update(c)
    pbar.close()


def label_converter(to_new_label_map, input_queue, output_queue, terminate_signal):
    """
    A worker that convert the labels
    Args:
        to_new_label_map: A map that maps label to new label index
        input_queue: Worker receives input from this queue
        output_queue: Worker outputs the return value of the function into this queue
        terminate_signal: If input == terminate_signal, worker will go home and rest.
    """
    # print("Worker {} awakens.".format(os.getpid()))
    while True:
        label_path, save_label_path = input_queue.get()
        if label_path == terminate_signal:
            break

        label_image = Image.open(label_path)

        # Convert labeled data to numpy arrays for better handling
        label_array = np.array(label_image)

        # Change the label to new label system
        # Pre-full the array with IGNORE_INDEX and only do the replacement when the new_label is not IGNORE_INDEX
        # This implementation is faster than the naive implementation like the following:
        #       for old_label, new_label in to_new_label_map.items():
        #           label_array[label_array == old_label] = new_label
        # because replacement takes most of the computation time.
        new_label_array = np.full_like(label_array, fill_value=IGNORE_INDEX)
        for old_label, new_label in to_new_label_map.items():
            if new_label != IGNORE_INDEX:
                new_label_array[label_array == old_label] = new_label
        label_array = new_label_array

        # Save labels to file
        out_image = Image.fromarray(label_array.astype(np.uint8()))
        out_image.save(save_label_path)

        output_queue.put(1)


def generate_semantic_segmentation(data_dir, save_dir, num_workers, verbose=True):
    """
    Generate semantic segmentation label

    Args:
        data_dir: Absolute path to the Mapillary dataset
        save_dir: Absolute path to the output directory
        num_workers: Number of workers to be created
        verbose: If True, it will print out the operation that it is doing
    """
    print_fn = print if verbose else lambda x: 0

    # Read in config file
    config_file = osp.join(data_dir, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    labels = config["labels"]

    # Show user the original label
    print_fn("Original label contains")
    for label_id, label in enumerate(labels):
        print_fn("{:>30} ({:2d}): {:<40} has instances: {}".format(
            label["readable"], label_id, label["name"], label["instances"]))

    # Create new labels
    new_labels = []
    # Maps the index from original label to new label system
    to_new_label_map = {}
    for label_id, label in enumerate(labels):
        new_label_id = IGNORE_INDEX  # Presume that label will be invalid
        if label_id in IGNORE_LABEL:
            print_fn("Removing label {:>30} ({:2d}): {:<40}".format(label["readable"], label_id, label["name"]))
        elif label_id in MERGE_LABEL_REVERSE_MAP:
            dest_label_id = MERGE_LABEL_REVERSE_MAP[label_id]

            # Ignore the merge if dest_label_id id is in IGNORE_LABEL
            if dest_label_id not in IGNORE_LABEL:
                temp_label = labels[dest_label_id]
                print_fn("Merging label  {:>30} ({:2d}): {:<40} to {:>30} ({:2d}): {:<40}".format(
                    label["readable"], label_id, label["name"], temp_label["readable"], dest_label_id,
                    temp_label["name"]))
        else:
            new_labels.append(label)
            new_label_id = len(new_labels) - 1  # label index starts from 0
        to_new_label_map[label_id] = new_label_id

    # Only after we remove merged labels and ignored labels, can we know how to rearrange the labels
    for k, v in MERGE_LABEL.items():
        for vi in v:
            to_new_label_map[vi] = to_new_label_map[k]

    print_fn("\nNew label contains")
    for label_id, label in enumerate(new_labels):
        print_fn("{:>30} ({:2d}): {:<40} has instances: {}".format(
            label["readable"], label_id, label["name"], label["instances"]))

    # Create a new config file
    new_config = config.copy()
    new_config["labels"] = new_labels
    with open(osp.join(save_dir, "config.json"), 'w') as f:
        json.dump(new_config, f, indent=2)

    manager = Manager()
    input_queue = manager.Queue()
    progress_queue = manager.Queue()
    with Pool(num_workers) as pool:
        # Summon workers
        print_fn("Creates {} workers".format(num_workers))
        for i in range(num_workers):
            pool.apply_async(label_converter, args=(to_new_label_map, input_queue, progress_queue, ""))

        # Process the files in the directory
        dir_list = get_dir_list(data_dir)
        for split_dir in dir_list:
            image_dir = osp.join(data_dir, split_dir, "images")
            label_dir = osp.join(data_dir, split_dir, "labels")
            save_image_dir = osp.join(save_dir, split_dir, "images")
            save_label_dir = osp.join(save_dir, split_dir, "labels")

            # Create save folder
            os.makedirs(save_image_dir, exist_ok=True)
            os.makedirs(save_label_dir, exist_ok=True)

            # Push all the labels into the input_queue
            label_list = get_file_list(label_dir)
            if label_list:
                for file in label_list:
                    label_path = osp.join(label_dir, file)
                    save_label_path = osp.join(save_label_dir, file)
                    input_queue.put((label_path, save_label_path), block=False)
                # Create a process to monitor the workers' progress
                monitor = Process(target=progress_monitor, args=(len(label_list), progress_queue))
                monitor.start()
            else:
                monitor = None

            # Copy image into new folder
            print_fn("Copying Image to {}".format(save_image_dir))
            image_list = get_file_list(image_dir)
            for file in image_list:
                # Each operation takes about 5e-3 sec, which is affordable
                copyfile(osp.join(image_dir, file), osp.join(save_image_dir, file))

            # Wait for workers to finish current jobs
            if monitor:
                monitor.join()

        # Terminate workers
        for i in range(num_workers):
            input_queue.put("")


def parse_args():
    parser = argparse.ArgumentParser(description="Mapillary Vistas Data Preparation")

    parser.add_argument(
        "-d", "--data_dir",
        required=True,
        default="",
        type=str,
        metavar="PATH",
        help="path to the root directory of mapillary vistas dataset"
    )
    parser.add_argument(
        "-s", "--save_dir",
        default="",
        type=str,
        help="path to save the processed dataset. If not provided, a folder will be created under the same directory as label_dir"
    )
    parser.add_argument(
        "-w", "--workers",
        default=os.cpu_count(),
        metavar="INT",
        type=int,
        help="number of workers to process the data. By default is the number of CPU cores in the system."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = osp.abspath(args.data_dir)
    save_dir = args.save_dir
    # If save_dir is not provided, by default we save it in the same folder as label directory
    if not save_dir:
        root_dir = osp.dirname(data_dir)
        base_name = osp.basename(data_dir)
        save_dir = osp.join(root_dir, "{}_processed".format(base_name))
    # Create save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    num_workers = args.workers
    assert num_workers is not None and num_workers > 0, "Invalid number of workers {}".format(num_workers)

    generate_semantic_segmentation(data_dir, save_dir, num_workers)


if __name__ == '__main__':
    main()
