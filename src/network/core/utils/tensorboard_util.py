import numbers
import os.path as osp

from core.utils.metric import GenericMetric, AverageMeter


def add_scalars(tb_writer, tag, meters, step):
    """
    A helper function to add scales into Tensorboard
    Args:
        tb_writer: Tensorboard writer
        meters (dict): meters may contains customized metrics
        step (int): global step size
    """
    for name, meter in meters.items():
        if isinstance(meter, (GenericMetric, AverageMeter)):
            v = meter.global_avg
        elif isinstance(meter, numbers.Number):
            v = meter
        else:
            raise TypeError('Unknown meter type {}.'.format(type(meter).__name__))
        tb_writer.add_scalar(osp.join(tag, name), v, global_step=step)
