import os
import time
import torch
import numpy
import random
import common.torch
import common.summary
import common.numpy
from common.log import log
import attacks
import math


class TrainingInterface:
    """
    Training interface.

    Will only take care of the training and test step. This means that loading of model, optimizer,
    scheduler as well as data (including data augmentation in form of a dataset) needs to be
    taken care of separately.
    """

    def __init__(self, writer=common.summary.SummaryWriter()):
        """
        Constructor.

        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        """

        self.writer = writer
        """ (torch.util.tensorboardSummarWriter or equivalent) Summary writer. """

    def progress(self, epoch, batch, batches):
        """
        Report progress.

        :param epoch: epoch
        :type epoch: int
        :param batch: batch
        :type batch: int
        :param batches: batches
        :type batches: int
        """

        if batch == 0:
            log(' %d .' % epoch, end='')
        elif batch == batches - 1:
            log('. done', end="\n", context=False)
        else:
            log('.', end='', context=False)

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        raise NotImplementedError()

    def test(self, epoch):
        """
        Test step.

        :param epoch: epoch
        :type epoch: int
        """

        raise NotImplementedError()

    def step(self, epoch):
        """
        Training + test step.

        :param epoch: epoch
        :type epoch: int
        """

        self.train(epoch)
        self.test(epoch)