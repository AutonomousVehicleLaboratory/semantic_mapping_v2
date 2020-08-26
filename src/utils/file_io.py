import errno
import os
import os.path as osp
import shutil


def get_dir_list(dir):
    """
    Collect all the directories in the dir
    :param dir: the directory path
    """
    content = os.listdir(dir)
    dir_list = []
    for c in content:
        if osp.isdir(osp.join(dir, c)):
            dir_list.append(c)
    return dir_list


def get_file_list(dir, no_ext=False):
    """
    Collect all the file in the dir
    :param dir:
    :param no_ext: If True, return file list will not have file extension. e.g. "image.png" -> "image"
    """
    content = os.listdir(dir)
    file_list = []
    for c in content:
        if osp.isfile(osp.join(dir, c)):
            if no_ext:
                c = osp.splitext(c)[0]
            file_list.append(c)
    return file_list


def move(src, des):
    """Move a file/directory from src to des"""
    shutil.move(src, des)


def remove(file_path, recursive=True):
    """
    Remove the file_path. If recursive == False, then only remove file. Exception will raise if file_path
    is a directory. Remove both file and directory type if recursive == True.
    """
    if not osp.exists(file_path):
        return

    if not recursive or osp.isfile(file_path):
        # Note that exception will raise if file_path is a directory
        os.remove(file_path)
    else:
        shutil.rmtree(file_path)


def makedirs(save_dir, exist_ok=False):
    """ Make directory specifically for Python 2.7 """
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            if not exist_ok:
                raise
        else:
            raise
