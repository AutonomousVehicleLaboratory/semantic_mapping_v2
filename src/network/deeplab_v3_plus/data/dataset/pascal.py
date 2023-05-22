import numpy as np
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset


class VOCSegmentation(Dataset):
    """
    Pascal VOC 2012 Semantic Segmentation Dataset
    """
    # Map the type to the txt file of image list
    type_map = {
        'train': 'train.txt',
        'val': 'val.txt',
    }

    def __init__(self, root_dir, type, transform=None):
        """

        Assume the file structure is the same as the VOC package downloaded directly from PASCAL
        The root directory of the data should be the path of "VOC2012"
        The split of segmentation set is assumed in the root_dir/ImageSets/Segmentation

        Args:
            root_dir (str): the root directory of data
            type (str): the type of dataset, e.g. ['train', 'test']
            transform: methods to transform images
        """
        self.root_dir = osp.abspath(root_dir)
        self.split_file = osp.join(self.root_dir, "ImageSets/Segmentation", self.type_map[type])
        self.image_dir = osp.join(self.root_dir, "JPEGImages")
        self.seg_label_dir = osp.join(root_dir, "SegmentationClass")
        self.transform = transform

        # Read the split file
        self.image_id_list = self._read_file(self.split_file)

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        image = Image.open(osp.join(self.image_dir, image_id) + ".jpg")
        label = Image.open(osp.join(self.seg_label_dir, image_id) + ".png")

        sample = {
            "image": image,
            "label": label,
        }
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _read_file(self, filename):
        with open(filename) as f:
            content = f.readlines()
            lines = []
            for c in content:
                line = c[:-1] if c[-1] == '\n' else c
                lines.append(line)
        return lines


class Overfit_VOCSegmentation(Dataset):
    """VOC Segmentation Dataset for training overfitting"""
    # Map the type to the txt file of image list
    type_map = {
        'train': 'train.txt',
        'val': 'val.txt',
    }

    def __init__(self, root_dir, type, size, transform=None):
        """
        Args:
            size: size of the dataset
        """
        self.root_dir = osp.abspath(root_dir)
        self.split_file = osp.join(self.root_dir, "ImageSets/Segmentation", self.type_map[type])
        self.image_dir = osp.join(self.root_dir, "JPEGImages")
        self.seg_label_dir = osp.join(root_dir, "SegmentationClass")
        self.transform = transform

        # Read the split file
        self.image_id_list = self._read_file(self.split_file)

        num_id = len(self.image_id_list)
        assert size <= num_id
        self.image_id_list = self.image_id_list[:size]

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        image = Image.open(osp.join(self.image_dir, image_id) + ".jpg")
        label = Image.open(osp.join(self.seg_label_dir, image_id) + ".png")
        assert label.size == image.size

        sample = {
            "image": image,
            "label": label,
        }
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _read_file(self, filename):
        with open(filename) as f:
            content = f.readlines()
            lines = []
            for c in content:
                line = c[:-1] if c[-1] == '\n' else c
                lines.append(line)
        return lines


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = VOCSegmentation("../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012", "train", None)
    num_data = len(dataset)
    print(num_data)

    index = np.random.choice(num_data)
    data = dataset[index]
    image = data["image"]
    label = data["label"]

    ax = plt.subplot(1, 2, 1)
    ax.imshow(image)
    ax = plt.subplot(1, 2, 2)
    ax.imshow(label)
    plt.show()
