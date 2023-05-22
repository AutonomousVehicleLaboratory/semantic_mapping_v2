"""
Use PIL.Image to process image
For numerical operation, use torch.tensor

Modified from https://github.com/NVIDIA/semantic-segmentation/blob/master/transforms/joint_transforms.py
"""
import numbers
import numpy as np
import PIL.Image
import random
import torch
import torchvision.transforms as T
import warnings


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, sample):
        """
        sample is a dictionary that has two keys: "image" and "label"
        """
        image = np.array(sample["image"])
        label = np.array(sample["label"])

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        label = torch.from_numpy(label).float()

        out_dict = {
            "image": image,
            "label": label,
        }

        return out_dict


class Normalize(object):
    """
    Follow the normalization requirement of pytorch pretrained model, as specified here
    https://pytorch.org/docs/stable/torchvision/models.html
    """

    def __init__(self, mean, std, inplace=False):
        self.normalize = T.Normalize(mean, std, inplace)

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

        # Only normalize the image
        # scale the image to the range of [0, 1] first then normalize it
        image = self.normalize(image / 255)

        out_dict = {
            "image": image,
            "label": label,
        }

        return out_dict


class Resize(object):
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        assert image.size == label.size

        image = image.resize(self.size, PIL.Image.BILINEAR)
        label = label.resize(self.size, PIL.Image.NEAREST)

        out_dict = {
            "image": image,
            "label": label,
        }
        return out_dict


class RandomHorizontalFlip(object):
    """Refer to: torchvision.transforms.RandomHorizontalFlip"""

    def __init__(self, p=0.5):
        """
        Args:
            p: probability of the image being flipped. Default value is 0.5
        """
        self.prob = p

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        if random.random() < self.prob:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            label = label.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        out_dict = {
            "image": image,
            "label": label,
        }
        return out_dict


class RandomRotate(object):
    def __init__(self, degrees):
        """

        Args:
            degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

        angle = random.uniform(self.degrees[0], self.degrees[1])

        image = image.rotate(angle, PIL.Image.BILINEAR)
        label = label.rotate(angle, PIL.Image.NEAREST)

        out_dict = {
            "image": image,
            "label": label,
        }
        return out_dict


class RandomCrop(object):
    """
    Take a random crop from the image.

    First the image or crop size may need to be adjusted if the incoming image
    is too small...

    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         the label is padded too, filled with ignore_indx.
         unless 'nopad', in which case the crop size is shrunk to fit the image

    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """

    def __init__(self, size, ignore_index=0, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, sample, centroid=None):
        image = sample["image"]
        label = sample["label"]
        assert image.size == label.size

        w, h = image.size
        th, tw = self.size
        if w == tw and h == th:
            return sample

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                image = PIL.ImageOps.expand(image, border=border, fill=self.pad_color)
                label = PIL.ImageOps.expand(label, border=border, fill=self.ignore_index)
                # Update the width and height of image
                w, h = image.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)

        image = image.crop((x1, y1, x1 + tw, y1 + th))
        label = label.crop((x1, y1, x1 + tw, y1 + th))

        out_dict = {
            "image": image,
            "label": label,
        }
        return out_dict


class RandomSizeAndCrop(object):
    """
    Randomly scale the image then crop it.
    """

    def __init__(self, size, scale=(0.5, 2), ignore_index=0, crop_nopad=False, pre_size=None):
        """
        Args:
            size (int or a pair of int): Output image size
            scale (a pair of int): Random scale the image in the range defined by
            ignore_index (int): Index to pad on labels
            crop_nopad (bool): refer to RandomCrop
            pre_size (int): Resize image shorter edge to this before doing data augmentation
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        if (scale[0] > scale[1]):
            warnings.warn("range should be of kind (min, max)")

        self.crop = RandomCrop(self.size, ignore_index=ignore_index, nopad=crop_nopad)
        self.scale = scale
        self.pre_size = pre_size

    def __call__(self, sample, centroid=None):
        image = sample["image"]
        label = sample["label"]
        assert image.size == label.size

        scale_min, scale_max = self.scale

        # first, resize such that shorter edge is pre_size
        if self.pre_size is None:
            scale_amt = 1.
        else:
            short_edge = min(image.size)
            scale_amt = self.pre_size / short_edge
        scale_amt = scale_amt * random.uniform(scale_min, scale_max)
        w, h = [int(i * scale_amt) for i in image.size]

        if centroid is not None:
            centroid = [int(c * scale_amt) for c in centroid]

        image = image.resize((w, h), PIL.Image.BILINEAR)
        label = label.resize((w, h), PIL.Image.NEAREST)

        sample = {
            "image": image,
            "label": label,
        }
        out_dict = self.crop(sample, centroid)
        return out_dict


class FixScaleCenterCrop(object):
    """
    Crop the image in the center while preserving the image scale

    We want to preserve as much information as possible, so we will scale the image first to make sure that its shortest
    length is equal to the crop size, then we center crop the image. The aspect ratio of the image will be preserved
    during scale so that the image will not be distorted.

    Based on FixScaleCrop: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/dataloaders/custom_transforms.py

    test if crop size is small than image size
    """

    def __init__(self, size):
        """
        Args:
            size (int or tuple of int): crop size (height, width)
        """
        if isinstance(size, tuple):
            self.size = size
        else:
            assert isinstance(size, numbers.Number)
            self.size = (size, size)

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

        width, height = image.size
        c_height, c_width = self.size
        # Find the required scale ratio on each side of the image and then pick the largest one
        w_ratio = c_width / width
        h_ratio = c_height / height

        scale_ratio = max(w_ratio, h_ratio)
        s_width = int(width * scale_ratio)
        s_height = int(height * scale_ratio)

        image = image.resize((s_width, s_height), PIL.Image.BILINEAR)
        label = label.resize((s_width, s_height), PIL.Image.NEAREST)

        # Center crop
        x1 = int((s_width - c_width) // 2)
        y1 = int((s_height - c_height) // 2)
        image = image.crop((x1, y1, x1 + c_width, y1 + c_height))
        label = label.crop((x1, y1, x1 + c_width, y1 + c_height))

        out_dict = {
            "image": image,
            "label": label,
        }
        return out_dict


class CenterCropWithPad(object):
    """
    Center crop the image, add padding if necessary.
    """

    def __init__(self, size, ignore_index=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        assert image.size == label.size

        w, h = image.size
        tw, th = self.size

        if w < tw:
            pad_x = tw - w
        else:
            pad_x = 0
        if h < th:
            pad_y = th - h
        else:
            pad_y = 0

        if pad_x or pad_y:
            # left, top, right, bottom
            image = PIL.ImageOps.expand(image, border=(pad_x, pad_y, pad_x, pad_y), fill=0)
            label = PIL.ImageOps.expand(label, border=(pad_x, pad_y, pad_x, pad_y), fill=self.ignore_index)
            # Update the width and height of image
            w, h = image.size

        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        image = image.crop((x1, y1, x1 + tw, y1 + th))
        label = label.crop((x1, y1, x1 + tw, y1 + th))
        out_dict = {
            "image": image,
            "label": label,
        }
        return out_dict


class MaxSizeCenterCrop(object):
    """Center crop the image if its size is larger than a max size"""

    def __init__(self, size, ignore_index=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.center_crop = CenterCropWithPad(size, ignore_index)

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        assert image.size == label.size

        w, h = image.size
        tw, th = self.size
        if w > tw or h > th:
            return self.center_crop(sample)
        else:
            return sample
