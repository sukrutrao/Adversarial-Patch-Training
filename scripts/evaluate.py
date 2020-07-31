#! /usr/bin/env python3

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
from attacks.objectives import UntargetedF0Objective
from attacks.adversarial_patch import AdversarialPatch
from attacks.norms import LInfNorm
from common.datasets import CleanDataset
from common.mask import MaskGenerator
import common.paths as paths
from common.eval import CleanEvaluation, AdversarialEvaluation
import common.test
from common.state import State
import models
import torch
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def run(args):

    # Get the data
    test_data = CleanDataset(paths.test_images_file(
        args.dataset), paths.test_labels_file(args.dataset))
    adversarial_data = CleanDataset(paths.test_images_file(
        args.dataset), paths.test_labels_file(args.dataset), indices=list(range(args.adv_set_size)))

    testset = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    adversarialset = DataLoader(
        adversarial_data, batch_size=args.batch_size, shuffle=False)

    # Logging
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
    else:
        from common.summary import SummaryWriter
    writer = SummaryWriter(paths.log_dir(args.log_dir), max_queue=100)

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

    # Load model
    state = State.load(paths.experiment_file(
        args.models_dir, args.saved_model_file))
    model = state.model
    model.eval()
    if args.cuda:
        model.cuda()

    # Run model
    probabilities = common.test.test(model, testset, cuda=args.cuda)

    if args.mode in {'all', 'clean'}:
        # Perform clean evaluation on trained model
        evaluator = CleanEvaluation(
            probabilities, testset.dataset.labels, validation=0)
        print("Clean Test Error:", evaluator.test_error())
        writer.add_text('results/clean_test_error',
                        str(evaluator.test_error()))

    if args.mode in {'all', 'adversarial'}:
        # Attack trained model
        perturbations, adversarial_probabilities, errors = common.test.attack(
            model, adversarialset, attack, objective, attempts=args.attempts, writer=writer, cuda=args.cuda)

        if args.perturbations_file:
            common.utils.write_hdf5(paths.experiment_file(args.models_dir, args.perturbations_file), [perturbations, adversarial_probabilities, errors], keys=[
                'perturbations', 'adversarial_probabilities', 'errors'])

        # Perform adversarial evaluation on attacked model
        evaluator = AdversarialEvaluation(probabilities[:len(
            adversarialset.dataset)], adversarial_probabilities, adversarialset.dataset.labels, validation=0, errors=errors)
        print("Robust Test Error, Success Rate, Test Error:")
        print(evaluator.robust_test_error(),
              evaluator.success_rate(), evaluator.test_error())
        writer.add_text('results/robust_test_error',
                        str(evaluator.robust_test_error()))
        writer.add_text('results/success_rate', str(evaluator.success_rate()))
        writer.add_text('results/test_error', str(evaluator.test_error()))


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
                        choices={'clean', 'adversarial', 'all'}, default='all', help='Mode for evaluation')
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
    parser.add_argument('--attempts', type=int, default=1,
                        help='Number of attempts to run the attack. Useful for random initialization.')
    parser.add_argument('--optimize_location', action='store_true', default=False,
                        help='Flag to enable location optimization of the patch')
    parser.add_argument('--opt_type', type=str, default='full',
                        choices={'full', 'random'}, help='Type of location optimization to use, if enabled')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for moving mask when location optimization is enabled')
    parser.add_argument('--adv_set_size', type=int,
                        default=1000, help='Size of adversarial set')
    parser.add_argument('--log_dir', type=str, required=False,
                        help='Directory to log tensorboard summary. Directory base path to be specified in common/paths.py.')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory to store the trained model and intermediate snapshots. Directory base path to be specified in common/paths.py.')
    parser.add_argument('--signed_grad', action='store_true', default=False,
                        help='Flag to use sign of gradient when updating patch')
    parser.add_argument('--saved_model_file', type=str, default=None,
                        help='File name of model to continue training from. Directory base path to be specified in common/paths.py, and model directory to be specified in models_dir.')
    parser.add_argument('--use_tensorboard', action='store_true',
                        default=False, help='Flag to enable Tensorboard logging')
    parser.add_argument('--perturbations_file', type=str, default=None,
                        help='Name of file to store perturbations, errors, and probabilities after evaluation. Directory base path to be specified in common/paths.py.')
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
