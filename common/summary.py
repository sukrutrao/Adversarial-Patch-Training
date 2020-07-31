import numpy
import time
import os
from .log import log
import pickle
import torch
import tensorboard.backend.event_processing.event_accumulator


class SummaryWriter:
    """
    Summary dummy or interface to work like the Tensorboard SummaryWriter.
    """

    def __init__(self, log_dir='', **kwargs):
        """
        Constructor.

        :param log_dir: summary directory
        :type log_dir: str
        """

        pass

    def add_scalar(self, tag, value, global_step=None, walltime=None):
        """
        Add scalar value.

        :param tag: tag for scalar
        :type tag: str
        :param value: value
        :type value: mixed
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        pass

    def add_scalars(self, tag, tag_scalar_dict, global_step=None, walltime=None):
        """
        Add scalar values.

        :param tag: tag for scalar
        :type tag: str
        :param tag_scalar_dict: values
        :type tag_scalar_dict: dict
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        pass

    def add_histogram(self, tag, values, global_step=None, bins='auto', walltime=None, max_bins=None):
        """
        Add histogram data.

        :param tag: tag
        :type tag: str
        :param values: values
        :type values: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param bins: binning method
        :type bins: str
        :param walltime: time
        :type walltime: int
        :param max_bins: maximum number of bins
        :type max_bins: int
        """

        pass

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """
        Add image.

        :param tag: tag
        :type tag: str
        :param img_tensor: image
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        pass

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param img_tensor: images
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        pass

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        """
        Add figure.

        :param tag: tag
        :type tag: str
        :param figure: test
        :type figure: matplotlib.pyplot.figure
        :param global_step: global step
        :type global_step: int
        :param close: whether to automatically close figure
        :type close: bool
        :param walltime: time
        :type walltime: int
        """

        pass

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param text_string: test
        :type text_string: str
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        pass

    def flush(self):
        """
        Flush.
        """

        pass
