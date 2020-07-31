#! /usr/bin/env python3

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import models
from common.state import State
import common.test
from common.eval import CleanEvaluation, AdversarialEvaluation
import common.train
import common.imgaug
import common.paths as paths
from common.mask import MaskGenerator
from common.datasets import CleanDataset
from attacks.norms import LInfNorm
from attacks.adversarial_patch import AdversarialPatch
from attacks.objectives import UntargetedF0Objective


def run(args):

    # Get the data
    train_data = CleanDataset(paths.train_images_file(
        args.dataset), paths.train_labels_file(args.dataset))
    test_data = CleanDataset(paths.test_images_file(
        args.dataset), paths.test_labels_file(args.dataset))

    trainset = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    testset = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Create or load saved model
    if args.saved_model_file:
        state = State.load(paths.experiment_file(
            args.models_dir, args.saved_model_file))
        model = state.model
        if args.cuda:
            model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = common.train.get_exponential_scheduler(
            optimizer, batches_per_epoch=len(trainset), gamma=args.lr_decay)
        optimizer.load_state_dict(state.optimizer)
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.cuda()
        scheduler.load_state_dict(state.scheduler)
        initial_epoch = state.epoch
    else:
        model = models.ResNet(args.n_classes, [3, 32, 32], channels=12, blocks=[
                              3, 3, 3], clamp=True)
        if args.cuda:
            model.cuda()
        optimizer = SGD(model.parameters(), lr=args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = common.train.get_exponential_scheduler(
            optimizer, batches_per_epoch=len(trainset), gamma=args.lr_decay)
        initial_epoch = -1

    # Logging
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
    else:
        from common.summary import SummaryWriter
    writer = SummaryWriter(paths.log_dir(args.log_dir), max_queue=100)

    # Augmentation parameters
    augmentation_crop = True
    augmentation_contrast = True
    augmentation_add = False
    augmentation_saturation = False
    augmentation_value = False
    augmentation_flip = args.use_flip
    augmentation = common.imgaug.get_augmentation(noise=False, crop=augmentation_crop, flip=augmentation_flip, contrast=augmentation_contrast,
                                                  add=augmentation_add, saturation=augmentation_saturation, value=augmentation_value)

    # Create attack objects
    img_dims = (32, 32)
    if args.location == 'random':
        mask_gen = MaskGenerator(img_dims, tuple(args.mask_dims),
                                 exclude_list=np.array([args.exclude_box]))
    else:
        mask_gen = MaskGenerator(img_dims, tuple(args.mask_dims),
                                 include_list=np.array([args.mask_pos + args.mask_dims]))
    attack = AdversarialPatch(mask_gen, args.epsilon, args.iterations,
                              args.optimize_location, args.opt_type, args.stride, args.signed_grad)
    attack.norm = LInfNorm()
    objective = UntargetedF0Objective()

    if args.mode == 'adversarial':
        trainer = common.train.AdversarialTraining(model, trainset, testset, optimizer, scheduler,
                                                   attack, objective, fraction=args.adv_frac, augmentation=augmentation, writer=writer, cuda=args.cuda)
    elif args.mode == 'normal':
        trainer = common.train.NormalTraining(
            model, trainset, testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=args.cuda)

    trainer.summary_gradients = False

    # Train model
    for e in range(initial_epoch + 1, args.epochs):
        trainer.step(e)
        writer.flush()

        # Save model snapshot
        if (e + 1) % args.snapshot_frequency == 0:
            State.checkpoint(paths.experiment_file(
                args.models_dir, args.model_prefix + '_' + str(e + 1)), model, optimizer, scheduler, e)

    # Save final model
    State.checkpoint(paths.experiment_file(
        args.models_dir, args.model_prefix + '_complete_' + str(e + 1)), model, optimizer, scheduler, args.epochs)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Flag to enable running on GPU')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Learning rate for attack')
    parser.add_argument('--mask_pos', type=int, nargs='+',
                        help='Coordinates of top-left corner of mask in (y, x) format')
    parser.add_argument('--mask_dims', type=int, nargs='+',
                        help='Dimensions of mask in (h, w) format')
    parser.add_argument('--mode', type=str,
                        choices={'normal', 'adversarial'}, required=True, help='Training mode to use')
    parser.add_argument('--dataset', type=str,
                        required=True, help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images to train in a batch')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations to train. Currently the full number of iterations is reached even if all images in the batch flip earlier.')
    parser.add_argument('--location', type=str, default='fixed', choices={
                        'fixed', 'random'}, help='Selects whether mask is to be placed in a fixed or random location. Overrides --mask if random is selected.')
    parser.add_argument('--exclude_box', type=int, nargs='+',
                        help='Pixels to exclude for random mask, in (y, x, h, w) format')
    parser.add_argument('--optimize_location', action='store_true', default=False,
                        help='Flag to enable location optimization of the patch')
    parser.add_argument('--opt_type', type=str, default='full',
                        choices={'full', 'random'}, help='Type of location optimization to use, if enabled')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for moving mask when location optimization is enabled')
    parser.add_argument('--lr', type=float, default=0.075,
                        help='Learning rate for adversarial/normal training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for adversarial/normal training')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='Decay value for training learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay value for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs for adversarial/normal training')
    parser.add_argument('--adv_frac', type=float, default=0.5,
                        help='Fraction of adversarial examples per batch in adversarial training')
    parser.add_argument('--log_dir', type=str, default='',
                        help='Directory to log tensorboard summary. Directory base path to be specified in common/paths.py.')
    parser.add_argument('--snapshot_frequency', type=int, default=10,
                        help='Frequency of saving model during training')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory to store the trained model and intermediate snapshots. Directory base path to be specified in common/paths.py.')
    parser.add_argument('--signed_grad', action='store_true', default=False,
                        help='Flag to use sign of gradient when updating patch')
    parser.add_argument('--saved_model_file', type=str, default=None,
                        help='File name of model to continue training from. Directory base path to be specified in common/paths.py, and model directory to be specified in models_dir.')
    parser.add_argument('--model_prefix', type=str, default='model',
                        help='Prefix to use when saving models to file')
    parser.add_argument('--use_tensorboard', action='store_true',
                        default=False, help='Flag to enable Tensorboard logging')
    parser.add_argument('--use_flip', action='store_true', default=False,
                        help='Flag to enable flipping during augmentation')
    parser.add_argument('--n_classes', type=int, required=True,
                        help='Number of classes in data')
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
