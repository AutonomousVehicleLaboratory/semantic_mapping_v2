"""
*** Distributed Network Training ***

Assume the network is trained on a single node (i.e. a single computer) with multiple GPUs. Process with rank 0 is in
charge of logging and model saving.

This script should be called by torch.distributed.launch

"""
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

import argparse
import logging
import time
import torch
import torch.distributed as distributed

from core.optim.build import build_optimizer, build_scheduler
from core.utils.checkpoint import Checkpoint
from core.utils.logger import setup_logger
from core.utils.metric import MeterLogger
from core.utils.tensorboard_util import add_scalars
from core.utils.torch_util import set_random_seed
from deeplab_v3_plus.data.build import build_dataloader
from deeplab_v3_plus.data.utils.mapillary_visualization import apply_color_map
from deeplab_v3_plus.data.utils.visualization import visualize_network_output
from deeplab_v3_plus.models.build import build_model
from torch.utils.tensorboard import SummaryWriter

MASTER_LOCAL_RANK = 0
LOGGER_BASENAME = 'deeplab'


def parse_args():
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser(description='Distributed DeepLab Training')
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        required=True,
        help='local rank of this process (i.e. process id)'
    )
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
                    is_master,
                    log_period=-1):
    """
    Train one epoch

    Args:
        model:
        loss_fn:
        metric:
        dataloader:
        optimizer:
        is_master:
        log_period:
    """
    # Set up logger
    logger = logging.getLogger(LOGGER_BASENAME + '.train') if is_master else None

    # Set up metric
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

        start_time = time.time()

        if is_master and log_period > 0 and iteration % log_period == 0:
            logger.info(
                'iter:{iter:4d} {meters} lr:{lr:.2e} max mem: {memory:.0f} MiB'.format(
                    iter=iteration,
                    meters=str(meter_logger),
                    lr=optimizer.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024 ** 2)
                ))

    # Synchronize the training result across all the processes
    #
    # Warning: You cannot synchronize the meter_logger during the training process. Please
    # refer to the documentation of AverageMeter().synchronize_between_processes()
    meter_logger.synchronize_between_processes()

    return meter_logger


def validate(model,
             loss_fn,
             metric,
             dataloader,
             tb_writer,
             is_master,
             log_period=-1,
             curr_epoch=0):
    """
    Validate the model

    Assume only master process will run this function

    Args:
        model:
        loss_fn:
        metric:
        dataloader:
        tb_writer: TensorBoard writer
        is_master: True if this process is the master
        log_period: Print out the logger info per log_period iterations.
        curr_epoch: Current training epoch, use it for TensorBoard
    """
    logger = logging.getLogger(LOGGER_BASENAME + '.validate') if is_master else None
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

            start_time = time.time()

            # Visualize a specific set of data
            if is_master and iteration == 3:
                visualize_network_output(data_batch, preds, apply_color_map, dataloader.dataset.labels,
                                         tb_writer=tb_writer, tag="Validation", step=curr_epoch)

            if is_master and log_period > 0 and iteration % log_period == 0:
                logger.info(
                    'iter:{iter:4d} {meters} max mem: {memory:.0f} MiB'.format(
                        iter=iteration,
                        meters=str(meter_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024 ** 2)
                    ))

        meter_logger.synchronize_between_processes()

        return meter_logger


def distributed_training(cfg, output_dir, local_rank):
    """

    Args:
        cfg: Config file
        output_dir: Output directory
        local_rank: The local rank of the process

    Returns:

    """
    is_master = local_rank == MASTER_LOCAL_RANK
    # Setup logger
    logger = logging.getLogger(LOGGER_BASENAME + '.train') if is_master else None

    # Build model
    model, loss_fn, train_metric, val_metric = build_model(cfg)
    if logger: logger.info('Build model:\n{}'.format(str(model)))
    # Model must be converted to cuda device first before it is passed into DistributedDataParallel
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    # Build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # Build data loader
    train_dataloader = build_dataloader(cfg, mode="train", distributed=True)
    train_period = cfg.TRAIN.LOG_PERIOD
    # Run validation only on master process
    val_dataloader = build_dataloader(cfg, mode="val", distributed=True)
    val_period = cfg.VALIDATE.PERIOD
    val_log_period = cfg.VALIDATE.LOG_PERIOD

    # Setup TensorBoard
    tb_writer = SummaryWriter(output_dir) if is_master else None

    # Setup checkpoint
    checkpoint = Checkpoint(model, optimizer, scheduler, output_dir, logger)
    checkpoint_data = checkpoint.load(cfg.MODEL.WEIGHT, cfg.AUTO_RESUME, cfg.RESUME_STATES,
                                      map_location=lambda storage, loc: storage.cuda(local_rank))
    # Use distributed.barrier() to ensure all the processes load the checkpoint
    distributed.barrier()

    # Setup the best model parameters
    train_metric_name = type(train_metric).__name__
    best_metric_name = 'best_{}'.format(train_metric_name)
    best_metric = checkpoint_data.get(best_metric_name, None)

    # Start training
    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = checkpoint_data.get('epoch', 0)
    if logger: logger.info('Start training from epoch {}'.format(start_epoch))

    # Set the train sampler to the start_epoch
    # It is important to set the sampler to the right epoch because it uses this as the random seed to shuffle data.
    # Reference: https://github.com/pytorch/vision/blob/master/references/detection/train.py
    train_dataloader.sampler.set_epoch(start_epoch)

    for epoch in range(start_epoch, max_epoch):
        curr_epoch = epoch + 1

        start_time = time.time()
        train_log = train_one_epoch(model, loss_fn, train_metric, train_dataloader, optimizer, is_master, train_period)
        # For distributed training, it doesn't make sense to update scheduler per iteration (because the total
        # iteration should be the sum across all processes.
        scheduler.step()
        end_time = time.time() - start_time

        if is_master:
            logger.info('Epoch[{}]-Train {} - Total Time:{:.2f}s'.format(curr_epoch, train_log.summary_str, end_time))
            add_scalars(tb_writer, "Train", train_log.meters, curr_epoch)

            checkpoint_data['epoch'] = curr_epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpoint.save('latest_model', **checkpoint_data)

        # Validate
        if val_period > 0 and (curr_epoch % val_period == 0 or curr_epoch == max_epoch):
            start_time = time.time()
            val_log = validate(model, loss_fn, val_metric, val_dataloader, tb_writer, is_master,
                               val_log_period, curr_epoch)
            end_time = time.time() - start_time

            if is_master:
                logger.info('Epoch[{}]-Validate {} - Total Time:{:.2f}s'.format(curr_epoch, val_log.summary_str,
                                                                                end_time))
                add_scalars(tb_writer, "Validate", val_log.meters, curr_epoch)

                # Save the model with best validation
                if train_metric_name in val_log.meters:
                    curr_metric = val_log.meters[train_metric_name].global_avg
                    if best_metric is None or curr_metric > best_metric:
                        best_metric = curr_metric
                        checkpoint_data['epoch'] = curr_epoch
                        checkpoint_data[best_metric_name] = best_metric
                        checkpoint.save('model_best', **checkpoint_data)

        # Wait for master to save the checkpoint
        distributed.barrier()

    # Print train summary
    if is_master:
        logger.info('Best val-{} = {}'.format(train_metric_name, best_metric))


def main():
    args = parse_args()

    # Load the configuration
    # import on-the-fly to avoid overwriting cfg
    from deeplab_v3_plus.config.deeplab_v3_plus import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Determine current process's local rank
    local_rank = args.local_rank
    is_master = (local_rank == MASTER_LOCAL_RANK)

    # Process is only visible to one unique GPU
    torch.cuda.set_device(local_rank)

    # Initialize distributed training
    # rank and world_size are not specified here because we assume the torch.distributed.launch has already set them
    # up properly in the environment variables
    # FYI: rank == process_id; world_size == total number of processes
    assert 'WORLD_SIZE' in os.environ and 'RANK' in os.environ
    assert 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ
    distributed.init_process_group('nccl', init_method='env://')

    # Explicitly setting seed to make sure that models created in different processes
    # start from same random weights and biases.
    seed = 2077 if cfg.RNG_SEED < 0 else cfg.RNG_SEED
    set_random_seed(seed)

    # Set up output directory
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        # replace '@' with config path
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace('experiments', 'outputs')
        output_dir = output_dir.replace('@', config_path)
        if is_master:
            os.makedirs(output_dir, exist_ok=True)
    # Make sure all the processes can see the output_dir
    distributed.barrier()

    if is_master:
        logger = setup_logger(LOGGER_BASENAME, output_dir, prefix='train')
        # display arguments
        logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
        logger.info(args)

        # display configuration
        logger.info('Loaded configuration file {}'.format(args.config_file))
        logger.info('Running with config:\n{}'.format(cfg))

    # print(torch.backends.cudnn.benchmark)
    # Train
    distributed_training(cfg, output_dir, local_rank)


if __name__ == '__main__':
    # An example to run
    # python -m torch.distributed.launch --nproc_per_node=2 deeplab_v3_plus/distributed_train.py --cfg experiments/test_dist_train.yaml
    main()
