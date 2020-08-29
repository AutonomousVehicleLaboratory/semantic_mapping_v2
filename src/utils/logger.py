import logging
import os
import sys
import time
import socket
import errno

from .file_io import makedirs


class MyLogger:
    """ My customized logger """

    def __init__(self,
                 name,
                 save_dir='',
                 version=None,
                 use_timestamp=True,
                 log_level=logging.DEBUG,
                 log_format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                 ):
        """
        Set up the logger

        Args:
            name (str): The name of the logger
            save_dir (str) : If provided, save the logger result to save_dir. It will create the folder if it is not
                exist.
            version (str): The version of the log. If not provide, it will be 1 plus the maximum version in save_dir.
            use_timestamp (bool): If True, saved log file name will include the timestamp
            log_level: The logging level
            log_format (str): The log output format
        """
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        # Create console handler and set level to debug
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(log_level)

        # Create console handler and set level to debug, add formatter to ch
        formatter = logging.Formatter(log_format)
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

    def debug(self, msg):
        return self.logger.debug(msg)

    def info(self, msg):
        return self.logger.info(msg)

    def warning(self, msg):
        return self.logger.warning(msg)

    def error(self, msg):
        return self.logger.error(msg)

    def critical(self, msg):
        return self.logger.critical(msg)

    def exception(self, msg):
        return self.logger.exception(msg)

    def log(self, msg, level="info"):
        if level == "info":
            self.info(msg)
        elif level == "debug":
            self.debug(msg)
        elif level == "warning":
            self.warning(msg)
        elif level == "error":
            self.error(msg)
        elif level == "critical":
            self.critical(msg)
        elif level == "exception":
            self.exception(msg)
        else:
            raise NotImplementedError
