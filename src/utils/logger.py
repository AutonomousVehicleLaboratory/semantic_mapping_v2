import logging
import os
import sys
import time
import socket
import errno

from .file_io import makedirs


class MyLogger:
    """ My customized logger """

    def __init__(self, name, save_dir='', version=None, use_timestamp=True):
        """
        Set up the logger

        Args:
            name: The name of the logger
            save_dir: If provided, save the logger result to save_dir
            use_timestamp: If True, saved log file name will include the timestamp
        """
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Create console handler and set level to debug
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)

        # Create console handler and set level to debug, add formatter to ch
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            # Determine the version of the logger
            if version is None:
                version = self._get_next_version(save_dir)

            # Update the save_dir and create this folder if empty
            save_dir = os.path.join(save_dir, "version_" + str(version))
            makedirs(save_dir, exist_ok=True)

            filename = 'log'
            if use_timestamp:
                current_time = time.strftime('%m-%d_%H-%M-%S')
                filename += '.' + current_time + '.' + socket.gethostname()
            log_file = os.path.join(save_dir, filename + '.txt')

            # Create file handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        self.logger = logger
        self.save_dir = save_dir

    def log(self, msg, level="info"):
        if level == "info":
            self.logger.info(msg)
        elif level == "debug":
            self.logger.debug(msg)
        elif level == "warning":
            self.logger.warning(msg)
        elif level == "critical":
            self.logger.critical(msg)
        else:
            raise NotImplementedError

    def _get_next_version(self, save_dir):
        """ Get the next version from the save_dir """
        root_dir = os.path.join(save_dir)

        if not os.path.isdir(root_dir):
            try:
                os.makedirs(root_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
