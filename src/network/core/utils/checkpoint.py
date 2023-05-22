import logging
import os
import torch


class Checkpoint(object):
    """
    Provide checkpoint functionality for model, optimizer, scheduler and other customized variables
    """

    def __init__(self,
                 model,
                 optimizer=None,
                 scheduler=None,
                 save_dir='.',
                 logger=None):
        """

        Args:
            model: model to be monitored
            optimizer: optimizer to be monitored
            scheduler: scheduler to be monitored
            save_dir: the directory where we save the checkpoint files
            logger:
        """
        self.save_dir = save_dir
        # default logger is Checkpoint itself
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save(self, filename, **kwargs):
        """
        Save checkpoint into a file

        Args:
            filename: checkpoint file name
            **kwargs: additional variables that user wants to save into the checkpoint file
        """
        if not self.save_dir:
            self.logger.warning("Invalid save directory path.")
            return
        if not filename:
            self.logger.warning("Empty filename")
            return

        data = dict()
        data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data['scheduler'] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, '{}.pth'.format(filename))
        self.logger.info('Saving checkpoint to {}'.format(os.path.abspath(save_file)))
        torch.save(data, save_file)

        # Update the last checkpoint file
        self.update_last_checkpoint(save_file)

    def load(self, filename=None, resume=True, resume_states=True, map_location=torch.device('cpu')):
        """
        Load the checkpoint file
        Args:
            filename: Checkpoint filename
            resume: If True, model's weight will be reloaded from the latest checkpoint
            resume_states: If True, optimizer and scheduler will be reset back to their checkpoint states
            map_location: The device where model's parameters will be saved. (Refer to torch.load())

        Returns:

        """
        if resume and self.has_checkpoint():
            # override argument with existing checkpoint
            filename = self.get_checkpoint_file()
        if not filename:
            # no checkpoint could be found
            self.logger.info('No checkpoint found. Initializing model from scratch')
            return {}
        self.logger.info('Loading checkpoint from {}'.format(filename))

        checkpoint = torch.load(filename, map_location=map_location)
        self.model.load_state_dict(checkpoint.pop('model'))
        if resume_states:
            if 'optimizer' in checkpoint and self.optimizer:
                self.logger.info('Loading optimizer from {}'.format(filename))
                self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
            if 'scheduler' in checkpoint and self.scheduler:
                self.logger.info('Loading scheduler from {}'.format(filename))
                self.scheduler.load_state_dict(checkpoint.pop('scheduler'))
        else:
            checkpoint = {}

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        """Determines if there are checkpoints available."""
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        """Find the path of the last checkpoint file"""
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        try:
            with open(save_file, 'r') as f:
                last_saved = f.read()
            # If not absolute path, add save_dir as prefix
            if not os.path.isabs(last_saved):
                last_saved = os.path.join(self.save_dir, last_saved)
        except IOError:
            # If file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ''
        return last_saved

    def update_last_checkpoint(self, last_filename):
        """Update the filename stored in the last_checkpoint file"""
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        # If not absolute path, only save basename
        if not os.path.isabs(last_filename):
            last_filename = os.path.basename(last_filename)
        with open(save_file, 'w') as f:
            f.write(last_filename)


def test_checkpoint():
    conv = torch.nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3)
    checkpoint = Checkpoint(conv)
    checkpoint.save("abc")
    c = checkpoint.load()


if __name__ == '__main__':
    test_checkpoint()
