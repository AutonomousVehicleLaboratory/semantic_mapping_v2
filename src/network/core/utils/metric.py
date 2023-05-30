"""
Meter is the basic class of measurement.
Metric is built on top of that.
"""
from collections import defaultdict, deque

import numbers
import numpy as np
import torch
import torch.distributed as distributed


class GenericMetric():
    """
    Basic class of metric. Metric is used to evaluate the performance of network with human interpretable metrics.
    """

    def __init__(self):
        super(GenericMetric, self).__init__()

    def evaluate(self, preds, labels):
        """
        Evaluate the metric
        Args:
            preds: Network predictions
            labels: ground truth
        """
        raise NotImplementedError()

    def __str__(self):
        return '{:.4f}'.format(self.global_avg)

    def synchronize_between_processes(self):
        """
        Used for distributed training

        Warnings: You should not synchronize the metric during the training process, for the same reason explained in
        AverageMeter().synchronize_between_processes().
        """
        raise NotImplementedError()

    @property
    def global_avg(self):
        """This API is aligned with MeterLogger and AverageMeter API"""
        raise NotImplementedError()

    @property
    def summary_str(self):
        """We need this to support the API of MeterLogger and AverageMeter"""
        return self.__str__()


class AverageMeter(object):
    """
    AverageMeter tracks a series of values and provide access to smoothed values over a window or the global series
    average.
    """

    def __init__(self, window_size=20):
        self.value_queue = deque(maxlen=window_size)
        self.count_queue = deque(maxlen=window_size)
        self.sum = 0
        self.count = 0

    def update(self, value, count=1):
        """
        Args:
            value: It can be a 1D numpy array or a scalar
            count: It can be a 1D numpy array or a scalar
        """
        assert isinstance(value, numbers.Number) or value.ndim == 1
        assert isinstance(count, numbers.Number) or count.ndim == 1

        self.value_queue.append(value)
        self.count_queue.append(count)
        self.sum += np.sum(value)
        self.count += np.sum(count)

    def synchronize_between_processes(self):
        """
        Used for distributed training

        Warnings: Does not synchronize the deque.

        Warnings: You cannot synchronize average meter during the training process, otherwise the average log is wrong.
            Example: Assume we synchronize the average meter during the training process.
            Suppose we have 2 processes and they have sum = 8, count = 2 from previous synchronization.
            Now each process read 1 data with value 1 and 2. Then process one has (sum=9, count=3) and process two has
            (sum=10, count=3). After synchronization, (sum=19, count=6) for all the processes. The average is 3.1667,
            but the actually average should be (8 + 1 + 2) / 4 = 2.75.
        """
        if not distributed.is_available() or not distributed.is_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        distributed.barrier()
        distributed.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = int(t[1])

    @property
    def avg(self):
        count_sum = np.sum(self.count_queue)
        return np.sum(self.value_queue) / count_sum if count_sum != 0 else float('nan')

    @property
    def global_avg(self):
        return self.sum / self.count if self.count != 0 else float('nan')

    def reset(self):
        self.value_queue.clear()
        self.count_queue.clear()
        self.sum = 0
        self.count = 0

    def __str__(self):
        """Print the average and global average of the meter"""
        return '{:.4f} ({:.4f})'.format(self.avg, self.global_avg)

    @property
    def summary_str(self):
        """Print the global average"""
        return '{:.4f}'.format(self.global_avg)


class MeterLogger():
    """
    Keep track of a list of meters and report their average movement
    """

    def __init__(self, delimiter=' '):
        """
        Args:
            delimiter: delimiter determines how the meter logger will separate each meter it during the print out.
                default value is a space.
        """
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update the meters
        Args:
            **kwargs: key-value pair. The key will be the name of the meter and the value can either be a 1D array or
            a scalar.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                value = v.sum().item()
                count = v.numel()
            elif isinstance(v, np.ndarray):
                value = v.sum().item()
                count = v.size
            elif isinstance(v, numbers.Number):
                value = v
                count = 1
            else:
                raise NotImplementedError

            self.meters[k].update(value, count)

    def bind(self, metric):
        """
        Unfortunately metric cannot be added into meter logger by update()
        That is why we create this function to bind the metric into meter logger
        Args:
            metric:
        """
        assert isinstance(metric, GenericMetric)
        self.meters[type(metric).__name__] = metric

    def synchronize_between_processes(self):
        """Used for distributed training"""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append('{}: {}'.format(name, str(meter)))
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append('{}: {}'.format(name, meter.summary_str))
        return self.delimiter.join(metric_str)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()
