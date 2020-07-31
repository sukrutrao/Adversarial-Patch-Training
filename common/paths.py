import os
from .log import log, LogLevel

# This file holds a bunch of specific paths used for experiments and
# data. The intention is to have all important paths at a central location, while
# allowing to easily prototype new experiments.

# Base directory for data
BASE_DATA = '/BS/srao/work/AP/data/'

# Base directory for experiments (model and perturbation files)
BASE_EXPERIMENTS = '/BS/srao/work/AP/ril-adversarial-patches-final/experiments/'

# Base directory for logs (e.g. Tensorboard)
BASE_LOGS = '/BS/srao/work/AP/ril-adversarial-patches-final/logs/'

if not os.path.exists(BASE_DATA):
    BASE_DATA = os.path.join(os.path.expanduser('~'), 'data') + '/'
    log('[Warning] changed data directory: %s' % BASE_DATA, LogLevel.WARNING)
    BASE_EXPERIMENTS = os.path.join(
        os.path.expanduser('~'), 'experiments') + '/'
    log('[Warning] changed experiments directory: %s' %
        BASE_EXPERIMENTS, LogLevel.WARNING)

if not os.path.exists(BASE_DATA):
    log('[Error] could not find data directory %s' % BASE_DATA, LogLevel.ERROR)
    raise Exception('Data directory %s not found.' % BASE_DATA)

# Common extension types used.
TXT_EXT = '.txt'
HDF5_EXT = '.h5'
STATE_EXT = '.pth.tar'
LOG_EXT = '.log'
PNG_EXT = '.png'
PICKLE_EXT = '.pkl'
TEX_EXT = '.tex'
MAT_EXT = '.mat'
GZIP_EXT = '.gz'


# Naming conventions.
def data_file(directory, name, ext=HDF5_EXT):
    """
    Generate path to data file.

    :param name: name of file
    :type name: str
    :param ext: extension (including period)
    :type ext: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_DATA, directory, name) + ext


def train_images_file(dataset):
    """
    Train images.

    :param dataset: name of directory containing dataset 
    :type dataset: str
    :return: filepath
    :rtype: str
    """

    return data_file(dataset, 'train_images', HDF5_EXT)


def test_images_file(dataset):
    """
    Test images.

    :param dataset: name of directory containing dataset 
    :type dataset: str
    :return: filepath
    :rtype: str
    """

    return data_file(dataset, 'test_images', HDF5_EXT)


def train_labels_file(dataset):
    """
    Train labels.

    :param dataset: name of directory containing dataset 
    :type dataset: str
    :return: filepath
    :rtype: str
    """

    return data_file(dataset, 'train_labels', HDF5_EXT)


def test_labels_file(dataset):
    """
    Test labels.

    :param dataset: name of directory containing dataset 
    :type dataset: str
    :return: filepath
    :rtype: str
    """

    return data_file(dataset, 'test_labels', HDF5_EXT)


def experiment_dir(directory):
    """
    Generate path to experiment directory.

    :param directory: directory
    :type directory: str
    """

    return os.path.join(BASE_EXPERIMENTS, directory)


def experiment_file(directory, name, ext=''):
    """
    Generate path to experiment file.

    :param directory: directory
    :type directory: str
    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_EXPERIMENTS, directory, name) + ext


def log_dir(directory):
    """
    Generate path to log directory.

    :param directory: directory
    :type directory: str
    """

    return os.path.join(BASE_LOGS, directory)


def log_file(directory, name, ext=''):
    """
    Generate path to log file.

    :param directory: directory
    :type directory: str
    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_LOGS, directory, name) + ext
