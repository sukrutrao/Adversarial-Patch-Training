#! /usr/bin/env python3

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
import common.utils as utils
import common.paths as paths
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def create_hdf5(img_h5_path, label_h5_path, img_label_path, sep=',', keys='tensor'):
    assert os.path.isfile(img_label_path)

    df = pd.read_csv(img_label_path, sep=sep, names=['name', 'label'])

    # Store image names and labels in numpy arrays
    img_paths = df['name'].to_numpy().astype(str)
    labels = df['label'].to_numpy().astype(np.int)

    # Check that labels are scalar integers
    assert len(labels.shape) == 1

    # Check that number of images and labels are equal
    assert img_paths.shape[0] == labels.shape[0]

    transform = transforms.ToTensor()

    images = np.empty((len(img_paths),) + tuple(transform(Image.open(
        img_paths[0]).convert('RGB')).permute(1, 2, 0).shape), dtype=np.float32)

    for idx, img_path in enumerate(img_paths):
        img = transform(Image.open(img_path).convert('RGB'))

        # Change from C x H x W format to H x W x C format as expected by test.attack
        assert len(img.shape) == 3
        img = img.permute(1, 2, 0)
        images[idx] = img

    print("Writing image hdf5...")
    utils.write_hdf5(img_h5_path, images, keys)

    print("Writing label hdf5...")
    utils.write_hdf5(label_h5_path, labels, keys)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_csv', type=str, default=None,
                        help='Path to CSV for training data')
    parser.add_argument('--test_csv', type=str, default=None,
                        help='Path to CSV for test data')
    parser.add_argument('--dataset', type=str,
                        required=True, help='Name of dataset')
    parser.add_argument('--sep', type=str, default=',',
                        help='CSV delimiting character')
    args = parser.parse_args()
    if args.train_csv:
        create_hdf5(paths.train_images_file(args.dataset),
                    paths.train_labels_file(args.dataset), args.train_csv, args.sep)
    if args.test_csv:
        create_hdf5(paths.test_images_file(args.dataset),
                    paths.test_labels_file(args.dataset), args.test_csv, args.sep)


if __name__ == "__main__":
    main()
