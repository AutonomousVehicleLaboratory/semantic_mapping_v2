import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(__file__) + '/..')

import argparse
import logging
import time
import torch
import torch.nn as nn

from core.nn.freezer import freeze_bn
from core.optim.build import build_optimizer, build_scheduler
from core.utils.checkpoint import Checkpoint
from core.utils.logger import setup_logger
from core.utils.metric import MeterLogger
from core.utils.tensorboard_util import add_scalars
from core.utils.torch_util import set_random_seed
from deeplab_v3_plus.data.build import build_dataloader
from deeplab_v3_plus.data.utils.bdd_visualization import convert_label_to_color
from deeplab_v3_plus.data.utils.mapillary_visualization import apply_color_map
from deeplab_v3_plus.data.utils.visualization import visualize_network_output
from deeplab_v3_plus.models.build import build_model
from torch.utils.tensorboard import SummaryWriter

# This is the prefix of all the loggers in this script
# We will overwrite it with task specific name
LOGGER_BASENAME = 'deeplab'


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


def train_one_epoch(model,
                    loss_fn,
                    metric,
                    dataloader,
                    optimizer,
                    scheduler,
                    log_period=-1):
    logger = logging.getLogger(LOGGER_BASENAME + '.train')
    # If you want to know something's average movement, put it into the meter logger.
    metric.reset()
    meter_logger = MeterLogger()
    meter_logger.bind(metric)
    # set training mode
    model.train()
    loss_fn.train()

    start_time = time.time()
    for iteration, data_batch in enumerate(dataloader):
        load_data_time = time.time() - start_time

        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        preds = model(data_batch["image"])
        loss = loss_fn(preds, data_batch["label"])

        with torch.no_grad():
            metric.evaluate(preds, data_batch["label"])
        meter_logger.update(loss=loss, data_time=load_data_time)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - start_time
        start_time = time.time()

        if log_period > 0 and iteration % log_period == 0:
            logger.info(
                'iter:{iter:4d} {meters} lr:{lr:.2e} max mem: {memory:.0f} MiB'.format(
                    iter=iteration,
                    meters=str(meter_logger),
                    lr=optimizer.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024 ** 2)
                ))

    return meter_logger


def validate(model,
             loss_fn,
             metric,
             dataloader,
             tb_writer,
             log_period=-1,
             curr_epoch=0):
    """
    Validate the model
    Args:
        model:
        loss_fn:
        metric:
        dataloader:
        tb_writer: TensorBoard writer
        log_period: Print out the logger info per log_period iterations.
        curr_epoch: Current training epoch, use it for TensorBoard
    """
    logger = logging.getLogger(LOGGER_BASENAME + '.validate')
    metric.reset()
    meter_logger = MeterLogger()
    meter_logger.bind(metric)
    # set evaluate mode
    model.eval()
    loss_fn.eval()

    with torch.no_grad():
        start_time = time.time()
        for iteration, data_batch in enumerate(dataloader):
            load_data_time = time.time() - start_time

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

            preds = model(data_batch["image"])
            loss = loss_fn(preds, data_batch["label"])

            metric.evaluate(preds, data_batch["label"])
            meter_logger.update(loss=loss, data_time=load_data_time)

            batch_time = time.time() - start_time
            start_time = time.time()

            # Visualize a specific set of data
            if iteration == 3:
                visualize_network_output(data_batch, preds, apply_color_map, dataloader.dataset.labels,
                                         tb_writer=tb_writer, tag="Validation", step=curr_epoch)

            if log_period > 0 and iteration % log_period == 0:
                logger.info(
                    'iter:{iter:4d} {meters} max mem: {memory:.0f} MiB'.format(
                        iter=iteration,
                        meters=str(meter_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024 ** 2)
                    ))

        return meter_logger


def train(cfg, output_dir):
    """  Training Pipeline  """
    logger = logging.getLogger(LOGGER_BASENAME + '.train')

    # Control the random seed for reproducibility
    set_random_seed(cfg.RNG_SEED)

    # Build model
    model, loss_fn, train_metric, val_metric = build_model(cfg)
    logger.info('Build model:\n{}'.format(str(model)))
    model = nn.DataParallel(model).cuda()

    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    # Build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # Build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader = build_dataloader(cfg, mode="train")
    val_dataloader = build_dataloader(cfg, mode="val")
    train_period = cfg.TRAIN.LOG_PERIOD
    val_period = cfg.VALIDATE.PERIOD
    val_log_period = cfg.VALIDATE.LOG_PERIOD

    # Setup TensorBoard
    tb_writer = SummaryWriter(output_dir)

    # Setup checkpoint
    checkpoint = Checkpoint(model, optimizer, scheduler, output_dir, logger)
    checkpoint_data = checkpoint.load(cfg.MODEL.WEIGHT, cfg.AUTO_RESUME, cfg.RESUME_STATES)

    # Setup the best model parameters
    train_metric_name = type(train_metric).__name__
    best_metric_name = 'best_{}'.format(train_metric_name)
    best_metric = checkpoint_data.get(best_metric_name, None)

    # Check if we need to freeze batch normalization
    if cfg.TRAIN.FREEZE_BATCHNORM:
        freeze_bn(model, True, False, verbose=True, logger=logger)

    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = checkpoint_data.get('epoch', 0)
    logger.info('Start training from epoch {}'.format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        curr_epoch = epoch + 1

        start_time = time.time()
        train_log = train_one_epoch(model, loss_fn, train_metric, train_dataloader, optimizer, scheduler, train_period)
        end_time = time.time() - start_time
        logger.info('Epoch[{}]-Train {} - Total Time:{:.2f}s'.format(curr_epoch, train_log.summary_str, end_time))

        add_scalars(tb_writer, "Train", train_log.meters, curr_epoch)

        # Save the latest model
        checkpoint_data['epoch'] = curr_epoch
        checkpoint_data[best_metric_name] = best_metric
        checkpoint.save('latest_model', **checkpoint_data)

        # Validate
        if val_period > 0 and (curr_epoch % val_period == 0 or curr_epoch == max_epoch):
            start_time = time.time()
            val_log = validate(model, loss_fn, val_metric, val_dataloader, tb_writer, val_log_period, curr_epoch)
            end_time = time.time() - start_time
            logger.info('Epoch[{}]-Validate {} - Total Time:{:.2f}s'.format(curr_epoch, val_log.summary_str, end_time))

            add_scalars(tb_writer, "Validate", val_log.meters, curr_epoch)

            # Save the model with best validation
            if train_metric_name in val_log.meters:
                curr_metric = val_log.meters[train_metric_name].global_avg
                if best_metric is None or curr_metric > best_metric:
                    best_metric = curr_metric
                    checkpoint_data['epoch'] = curr_epoch
                    checkpoint_data[best_metric_name] = best_metric
                    checkpoint.save('model_best', **checkpoint_data)

    # Print train summary
    logger.info('Best val-{} = {}'.format(train_metric_name, best_metric))


def main():
    """
    Parse user's arguments, setup logger, and call the train()
    """
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from deeplab_v3_plus.config.deeplab_v3_plus import cfg
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

    logger = setup_logger(LOGGER_BASENAME, output_dir, prefix='train')
    # display arguments
    logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
    logger.info(args)

    # display configuration
    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    train(cfg, output_dir)


if __name__ == '__main__':
    main()
