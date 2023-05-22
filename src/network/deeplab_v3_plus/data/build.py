import deeplab_v3_plus.data.transforms as T

from deeplab_v3_plus.data.dataset.bdd import BDDSegmentation
from deeplab_v3_plus.data.dataset.mapillary import MapillaryVistas
from deeplab_v3_plus.data.dataset.pascal import VOCSegmentation
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_transform(augmentation):
    """
    Build the transform (aka data augmentation)

    Args:
        augmentation: A tuple of transform methods. A method can be a name of a method that takes no argument
        (a string), or a method that takes argument (a tuple starts with a string, and follows by the arguments)

    Returns:

    """
    transform_list = []
    for method in augmentation:
        if isinstance(method, tuple):
            name = method[0]
            args = method[1:]
        else:
            name = method
            args = None

        if not hasattr(T, name):
            raise NotImplementedError

        trans = getattr(T, name)
        if args:
            transform_list.append(trans(*args))
        else:
            transform_list.append(trans())

    transform = T.Compose(transform_list)
    return transform


def build_dataloader(cfg, mode='train', distributed=False):
    """
    Build the dataloader
    Args:
        cfg: configuration dictionary
        mode (str): Must be one of three modes ['train', 'val', 'test']
            mode help us to control which part of the configuration file we should read
        distributed (bool): True if we are using distributed data parallel

    Returns:

    """
    if mode == 'train':
        batch_size = cfg.TRAIN.BATCH_SIZE
        augmentation = cfg.TRAIN.AUGMENTATION
    elif mode == 'val':
        batch_size = cfg.VALIDATE.BATCH_SIZE
        augmentation = cfg.VALIDATE.AUGMENTATION
    elif mode == 'test':
        batch_size = cfg.TEST.BATCH_SIZE
        augmentation = cfg.TEST.AUGMENTATION
    else:
        raise NotImplementedError

    # Load transform
    transform = build_transform(augmentation)

    if cfg.DATASET.NAME == "Pascal":
        dataset = VOCSegmentation(root_dir=cfg.DATASET.ROOT_DIR,
                                  type=mode,
                                  transform=transform)
    elif cfg.DATASET.NAME == "BDD":
        dataset = BDDSegmentation(root_dir=cfg.DATASET.ROOT_DIR,
                                  type=mode,
                                  transform=transform,
                                  ignore_index=255)
    elif cfg.DATASET.NAME == "Mapillary":
        dataset = MapillaryVistas(root_dir=cfg.DATASET.ROOT_DIR,
                                  type=mode,
                                  transform=transform)
    else:
        raise NotImplementedError('Unsupported dataset: {}'.format(cfg.DATASET.NAME))

    is_train = (mode == 'train')
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
        is_train = False
        # Warning: when using DistributedSampler() the batch_size is not the total batch size
        # it is the batch size per process.
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        sampler=sampler,
        drop_last=(is_train and cfg.DATALOADER.DROP_LAST),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
    )
    return dataloader
