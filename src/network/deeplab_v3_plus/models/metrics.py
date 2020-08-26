from __future__ import absolute_import
import numpy as np
import torch
import torch.distributed as distributed

from core.utils.metric import GenericMetric


class MeanIOU(GenericMetric):
    """
    Mean IoU accuracy

    Note:
         mean IoU is computed across all the images per class, not an image per class. Therefore, it is meaningless
    to use AverageMeter here.
        The runtime of this metric is 0.16 sec for a prediction size (8, 21, 513, 513). It is possible to accelerate
    this process by optimizing the numpy code (with 8 times faster) but we did not implement it here.

    Reference: https://stackoverflow.com/questions/31653576/how-to-calculate-the-mean-iu-score-in-image-segmentation
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((num_class, num_class))

    def reset(self):
        self.confusion_matrix = np.zeros_like(self.confusion_matrix)

    def evaluate(self, preds, labels):
        """

        Args:
            preds (tensor): network prediction b x c x h x w
                This is the direct softmax output from the network. To understand which class it represent, you need to
                interpret it through argmax.
            labels (tensor): ground truth b x h x w

        Returns:
            mIoU averaged over all class

        Note that we strictly assume the input is a tensor because torch.argmax() is 10 times faster than the np.argmax().
        For a prediction size 8x513x513. torch.argmax() takes 0.03 sec while np.argmax() takes 0.2 sec.
        """
        num_class = preds.shape[1]
        preds = torch.argmax(preds, dim=1)
        # Sanity check
        assert num_class == self.num_class
        assert preds.shape == labels.shape

        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        # Ignore all the pixels whose label is beyond the num_class
        # This is a bincounting hack copied from:
        # https://github.com/davidtvs/PyTorch-ENet/blob/d897f3d5e9d44bfc43efbb538f1318cda2cc0b26/metric/confusionmatrix.py#L6
        # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
        mask = (labels >= 0) & (labels < self.num_class)
        x = self.num_class * labels[mask] + preds[mask]
        count = np.bincount(x.astype(np.int), minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        self.confusion_matrix += confusion_matrix

    def synchronize_between_processes(self):
        if not distributed.is_available() or not distributed.is_initialized():
            return
        t = torch.tensor(self.confusion_matrix, dtype=torch.float64, device='cuda')
        distributed.barrier()
        distributed.all_reduce(t)
        self.confusion_matrix = t.cpu().numpy()

    @property
    def global_avg(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=0) + np.sum(self.confusion_matrix, axis=1) - intersection
        # If union == 0, then both label_mask and pred_mask are empty. We should ignore this class when computing
        # the mean IoU. We do it by assigning it to np.nan and use np.nanmean() to compute the mean.
        iou = np.divide(intersection, union, out=np.full(union.shape, np.nan), where=(union != 0))
        miou = np.nanmean(iou)
        return miou
