import os
import torch
import torch.utils.data  # needs to be imported separately
from . import utils
import numpy
from . import paths
import skimage.transform
from .log import log
from . import numpy as cnumpy


class CleanDataset(torch.utils.data.Dataset):
    """
    General, clean dataset used for training, testing and attacking.
    """

    def __init__(self, images, labels, indices=None, resize=None):
        """
        Constructor.

        :param images: images/inputs
        :type images: str or numpy.ndarray
        :param labels: labels
        :type labels: str or numpy.ndarray
        :param indices: indices
        :type indices: numpy.ndarray
        :param resize: resize in [channels, height, width
        :type resize: resize
        """

        self.images_file = None
        """ (str) File images were loaded from. """

        self.labels_file = None
        """ (str) File labels were loaded from. """

        if isinstance(images, str):
            self.images_file = images
            images = utils.read_hdf5(self.images_file)
            log('read %s' % self.images_file)
        if not images.dtype == numpy.float32:
            images = images.astype(numpy.float32)

        if isinstance(labels, str):
            self.labels_file = labels
            labels = utils.read_hdf5(self.labels_file)
            log('read %s' % self.labels_file)
        labels = numpy.squeeze(labels)
        if not labels.dtype == int:
            labels = labels.astype(int)

        assert isinstance(images, numpy.ndarray)
        assert isinstance(labels, numpy.ndarray)
        assert images.shape[0] == labels.shape[0]

        if indices is None:
            indices = range(images.shape[0])
        assert numpy.min(indices) >= 0
        assert numpy.max(indices) < images.shape[0]

        images = images[indices]
        labels = labels[indices]

        if resize is not None:
            assert isinstance(resize, list)
            assert len(resize) == 3

            size = images.shape
            assert len(size) == 4

            # resize
            if resize[1] != size[1] or resize[2] != size[2]:
                out_images = numpy.zeros(
                    (size[0], resize[1], resize[2], size[3]), dtype=numpy.float32)
                for n in range(out_images.shape[0]):
                    out_images[n] = skimage.transform.resize(
                        images[n], (resize[1], resize[2]))
                images = out_images

            # update!
            size = images.shape

            # color to grayscale
            if resize[0] == 1 and size[3] == 3:
                out_images = numpy.zeros(
                    (size[0], size[1], size[2], 1), dtype=numpy.float32)
                for n in range(out_images.shape[0]):
                    out_images[n, :, :, 0] = 0.2125 * images[n, :, :, 0] + \
                        0.7154 * images[n, :, :, 1] + \
                        0.0721 * images[n, :, :, 2]
                images = out_images

            # grayscale to color
            if resize[0] == 3 and size[3] == 1:
                out_images = numpy.zeros(
                    (size[0], size[1], size[2], 3), dtype=numpy.float32)
                for n in range(out_images.shape[0]):
                    out_images[n, :, :, 0] = images[n, :, :, 0]
                    out_images[n, :, :, 1] = images[n, :, :, 0]
                    out_images[n, :, :, 2] = images[n, :, :, 0]
                images = out_images

        self.images = images
        """ (numpy.ndarray) Inputs. """

        self.labels = labels
        """ (numpy.ndarray) Labels. """

        self.targets = None
        """ (numpy.ndarray) Possible attack targets. """

        self.indices = indices
        """ (numpy.ndarray) Indices. """

    def add_targets(self, targets):
        """
        Add attack targets.
        :param targets: targets
        :type targets: numpy.ndarray
        """

        assert numpy.min(self.indices) >= 0
        assert numpy.max(self.indices) < targets.shape[0]
        self.targets = targets[self.indices]

    def __getitem__(self, index):
        assert index < len(self)
        if self.targets is not None:
            return self.images[index], self.labels[index], self.targets[index]
        else:
            return self.images[index], self.labels[index]

    def __len__(self):
        assert self.images.shape[0] == self.labels.shape[0]
        return self.images.shape[0]

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])
