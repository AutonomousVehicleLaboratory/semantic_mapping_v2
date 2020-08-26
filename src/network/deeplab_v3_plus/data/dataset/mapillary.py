import json
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset


class MapillaryVistas(Dataset):
    """
    Mapillary Vistas Dataset
    """

    def __init__(self, root_dir, type, transform=None):
        """

        Args:
            root_dir (str): The root directory of the dataset.
            type (str): The type of dataset, e.g. ['train', 'val', 'test']
            transform: Data transformation (augmentation)
        """
        assert type in ("train", "test", "val")
        self.root_dir = osp.abspath(root_dir)
        self.transform = transform
        self.type = type

        # Read in config file
        config_file = osp.join(self.root_dir, "config.json")
        with open(config_file) as f:
            config = json.load(f)
        self.labels = config["labels"]

        # Set up images and labels directories
        subdir = {
            "train": "training",
            "test": "testing",
            "val": "validation"
        }
        self.image_dir = osp.join(self.root_dir, subdir[type], "images")
        self.label_dir = osp.join(self.root_dir, subdir[type], "labels")

        # Collect Image IDs
        self.image_ids = self.get_filenames(self.image_dir)
        self.image_ids.sort()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = osp.join(self.image_dir, "{}.jpg".format(image_id))
        label_path = osp.join(self.label_dir, "{}.png".format(image_id))

        # load images
        base_image = Image.open(image_path)
        label_image = Image.open(label_path)

        out_dict = {
            "image": base_image,
            "label": label_image,
        }
        if self.transform is not None:
            out_dict = self.transform(out_dict)

        return out_dict

    @staticmethod
    def get_filenames(dir):
        """ Get the list of files in a directory """
        content = os.listdir(dir)
        filenames = []
        for c in content:
            if osp.isfile(osp.join(dir, c)):
                # Remove the .xxx extension
                c = osp.splitext(c)[0]
                filenames.append(c)
        return filenames


if __name__ == '__main__':
    def plot_image(image, label, instance):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 15))
        ax[0].imshow(image)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[0].set_title("Base image")
        ax[1].imshow(label)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[1].set_title("Labels")
        ax[2].imshow(instance)
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        ax[2].set_title("Instance IDs")

        plt.show()


    root_dir = "/home/qinru/avl/semantic/deeplab/data/mapillary-vistas-dataset_public_v1.1_processed"
    # root_dir = "/home/qinru/avl/semantic/deeplab/data/mapillary-vistas-dataset_public_v1.1"
    mapillary = MapillaryVistas(root_dir, "val")
    print("Dataset size: {}".format(len(mapillary)))

    index = np.random.randint(0, len(mapillary))
    print(index)
    out_dict = mapillary[index]

    base_image = out_dict["image"]
    label_image = out_dict["label"]
    # instance_image = out_dict["instance"]

    # convert labeled data to numpy arrays for better handling
    label_array = np.array(label_image)
    # instance_array = np.array(instance_image, dtype=np.uint16)

    # now we split the instance_array into labels and instance ids
    # instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
    # instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)

    # plot_image(base_image, label_array, instance_ids_array)

    target_label = 0
    for i in reversed(range(len(mapillary))):
        print("Searching {}".format(i))
        out_dict = mapillary[i]

        base_image = out_dict["image"]
        label_image = out_dict["label"]

        # convert labeled data to numpy arrays for better handling
        label_array = np.array(label_image)

        if np.any(label_array == target_label):
            # Highlight the target label
            highlight = np.zeros_like(label_array)
            highlight[label_array == target_label] = 1

            plot_image(base_image, label_array, highlight)
            # break
