import random
import numpy as np
from enum import Enum

""" 
Stores the four directions in the form of the change in pixel values along each dimension of
the image
"""
MaskDirection = {
    'left': (0, -1),
    'right': (0, 1),
    'up': (-1, 0),
    'down': (1, 0)
}


class PatchGenerator:
    def __init__(self, img_dims):

        assert len(img_dims) == 2
        self.img_dims = img_dims

    def random_location(self, n=1):
        raise NotImplementedError

    def random_batch(self, batch_size, n_channels):
        raise NotImplementedError

    def get_masks(self, mask_coords, n_channels):
        raise NotImplementedError

    def move_coords(self, mask_coords, direction, stride=1):
        raise NotImplementedError

    def move_coords_single(self, mask_coord, direction, stride=1):
        raise NotImplementedError

    def move_coords_random(self, mask_coords, stride=1):
        raise NotImplementedError


class MaskGenerator(PatchGenerator):
    """
    Generates a random mask location for applying an adversarial patch to an image
    """

    def __init__(self, img_dims, mask_dims, include_list=None, exclude_list=None):
        """
        Constructor. Exactly one of include_list and exclude_list must be specified.

        :param img_dims: image dimensions
        :type img_dims: tuple
        :param mask_dims: mask dimensions
        :type mask_dims: tuple
        :param include_list: list of boxes in image to include among possible mask locations, defaults to None
        :type include_list: numpy.array, optional
        :param exclude_list: list of boxes in image to exclude among possible mask locations, defaults to None
        :type exclude_list: numpy.array, optional
        """
        super(MaskGenerator, self).__init__(img_dims)
        assert len(mask_dims) == 2
        assert include_list is None or exclude_list is None
        assert include_list is not None or exclude_list is not None
        assert mask_dims <= img_dims

        self.mask_dims = mask_dims
        self.include_list = include_list
        self.exclude_list = exclude_list
        if self.include_list is not None:
            self.allowed_pixels = self._parse_include_list()
        else:
            self.allowed_pixels = self._parse_exclude_list()

    def _parse_include_list(self):
        """
        Generates list of allowed pixels to start mask

        :return: list of allowed pixels
        :rtype: set
        """
        assert len(self.include_list.shape) == 2
        allowed_pixels = set()
        for box in self.include_list:
            y, x, h, w = box
            assert x >= 0 and y >= 0 and h > 0 and w > 0
            assert y+h < self.img_dims[0] and x+w < self.img_dims[1]
            y_range = np.arange(y, y+h-self.mask_dims[0]+1)
            x_range = np.arange(x, x+w-self.mask_dims[1]+1)
            pixels = [(y, x) for y in y_range for x in x_range]
            allowed_pixels.update(pixels)
        return allowed_pixels

    def _parse_exclude_list(self):
        """
        Generates list of disallowed pixels to start mask

        :return: list of disallowed pixels
        :rtype: set
        """
        assert len(self.exclude_list.shape) == 2
        all_pixels = [(y, x) for y in range(self.img_dims[0]-self.mask_dims[0])
                      for x in range(self.img_dims[1]-self.mask_dims[1])]
        allowed_pixels = set(all_pixels)
        for box in self.exclude_list:
            y, x, h, w = box
            assert x >= 0 and y >= 0 and h > 0 and w > 0
            assert y+h < self.img_dims[0] and x+w < self.img_dims[1]
            y_range = np.arange(max(0, y-self.mask_dims[0]+1), y+h)
            x_range = np.arange(max(0, x-self.mask_dims[1]+1), x+w)
            pixels = [(y, x) for y in y_range for x in x_range]
            allowed_pixels = allowed_pixels.difference(pixels)
        return allowed_pixels

    def random_location(self, n=1):
        """
        Generates n mask coordinates randomly from allowed locations

        :param n: number of masks to generate, defaults to 1
        :type n: int, optional
        :return: n masks
        :rtype: numpy.array
        """
        assert n >= 1
        assert len(self.allowed_pixels) > 0
        start_pixels = random.choices(tuple(self.allowed_pixels), k=n)
        return np.array([(y, x, self.mask_dims[0], self.mask_dims[1]) for (y, x) in start_pixels])

    def random_batch(self, batch_size, n_channels):
        """
        Generates random masks of specified shape

        :param batch_size: number of masks to generate
        :type batch_size: int
        :param n_channels: number of channels in image, used as second dimension of masks
        :type n_channels: int
        :return: generated masks
        :rtype: numpy.array
        """
        assert batch_size >= 1 and n_channels >= 1
        assert len(self.allowed_pixels) > 0
        mask_coords = random.choices(tuple(self.allowed_pixels), k=batch_size)
        masks = np.zeros(
            (batch_size, n_channels, self.img_dims[0], self.img_dims[1]), dtype=np.float32)
        for b in range(batch_size):
            masks[b, :, mask_coords[b][0]:mask_coords[b][0]+self.mask_dims[0],
                  mask_coords[b][1]:mask_coords[b][1]+self.mask_dims[1]] = 1
        return masks

    def get_masks(self, mask_coords, n_channels):
        """
        Gets mask in image shape given mask coordinates

        :param mask_coords: mask coordinates for the batch of masks
        :type mask_coords: numpy.array
        :param n_channels: number of channels in image, used as second dimension of masks
        :type n_channels: int
        :return: generated masks
        :rtype: numpy.array
        """
        assert n_channels >= 1
        batch_size = len(mask_coords)
        masks = np.zeros(
            (batch_size, n_channels, self.img_dims[0], self.img_dims[1]), dtype=np.float32)
        for b in range(batch_size):
            masks[b, :, mask_coords[b][0]:mask_coords[b][0]+self.mask_dims[0],
                  mask_coords[b][1]:mask_coords[b][1]+self.mask_dims[1]] = 1
        return masks

    def move_coords(self, mask_coords, direction, stride=1):
        """
        Moves mask coordinates in a specified direction using a specified stride. Each coordinate
        is moved only if the new location is an allowed location.

        :param mask_coords: mask coordinates to move
        :type mask_coords: numpy.array
        :param direction: direction to move mask coordinate, key in MaskDirection dict
        :type direction: str
        :param stride: stride when moving coordinate, defaults to 1
        :type stride: int, optional
        :return: moved coordinates
        :rtype: numpy.array
        """
        batch_size = len(mask_coords)
        new_coords = np.copy(mask_coords)
        for b in range(batch_size):
            new_coords[b] = self.move_coords_single(mask_coords[b], direction, stride)
        return new_coords

    def move_coords_single(self, mask_coord, direction, stride=1):
        """
        Moves a single mask coordinate in a specified direction using a specified stride. The
        coordinate is moved only if the new location is an allowed location.

        :param mask_coord: mask coordinate to move
        :type mask_coord: numpy.array
        :param direction: direction to move coordinate, key in MaskDirection dict
        :type direction: str  
        :param stride: stride when moving coordinate, defaults to 1
        :type stride: int, optional
        :return: moved mask coordinate
        :rtype: numpy.array
        """
        new_coord = np.copy(mask_coord)
        new_coord[0] += MaskDirection[direction][0] * stride
        new_coord[1] += MaskDirection[direction][1] * stride
        if self.isallowed(new_coord):
            return new_coord
        return mask_coord

    def move_coords_random(self, mask_coords, stride=1):
        """
        Moves mask coordinates in random directions. If the chosen direction leads to an invalid
        location, the corresponding coordinate is not moved (a new direction is not chosen).

        :param mask_coords: mask coordinates to move
        :type mask_coords: numpy.array
        :param stride: stride when moving coordinates, defaults to 1
        :type stride: int, optional
        :return: moved mask coordinates
        :rtype: numpy.array
        """
        batch_size = len(mask_coords)
        new_coords = np.copy(mask_coords)
        random_directions = random.choices(list(MaskDirection.keys()), k=batch_size)
        mask_perturbations = np.array([np.hstack((MaskDirection[direction], 0, 0))
                                       for direction in random_directions])
        for idx in range(len(mask_coords)):
            new_coord = mask_coords[idx] + stride * mask_perturbations[idx]
            if self.isallowed(new_coord):
                new_coords[idx] = new_coord
        return new_coords, random_directions

    def set_include_list(self, include_list):
        """
        Sets the include_list and recomputes allowed pixels

        :param include_list: list of boxes in image to include among possible mask locations
        :type include_list: numpy.array
        """
        assert include_list is not None
        assert len(include_list.shape) == 2

        self.include_list = include_list
        self.allowed_pixels = self._parse_include_list()

    def set_exclude_list(self, exclude_list):
        """
        Sets the exclude_list and recomputes allowed pixels

        :param exclude_list: list of boxes in image to exclude among possible mask locations
        :type exclude_list: numpy.array
        """
        assert exclude_list is not None
        assert len(exclude_list.shape) == 2

        self.exclude_list = exclude_list
        self.allowed_pixels = self._parse_exclude_list()

    def isallowed(self, mask):
        """
        Checks if a given mask is in an allowed location

        :param mask: mask coordinates (y, x, h, w)
        :type mask: tuple
        :return: if the mask is in an allowed location
        :rtype: bool
        """
        assert len(mask) == 4
        y, x, h, w = mask
        if self.mask_dims != (h, w):
            return False
        if (y, x) in self.allowed_pixels:
            return True
        return False