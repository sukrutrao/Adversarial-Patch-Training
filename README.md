# Adversarial Training against Location-Optimized Adversarial Patches

[arXiv](https://arxiv.org/abs/2005.02313) | [Code](#) 

Code for the paper:

Sukrut Rao, David Stutz, Bernt Schiele. Adversarial Training against Location-Optimized Adversarial Patches. ECCV CVCOPS 2020 (to appear).

## Setup

### Requirements
* Python 3.7 or above
* PyTorch
* scipy
* h5py
* scikit-image
* scikit-learn

### Optional requirements
#### To use script to convert data to HDF5 format
* torchvision
* Pillow
* pandas
#### To use Tensorboard logging
* tensorboard

With the exception of Python and PyTorch, all requirements can be installed directly using pip:
```bash
$ pip install -r requirements.txt
```

### Setting the paths

In [`common/paths.py`](common/paths.py), set the following variables:
* `BASE_DATA`: base path for datasets.
* `BASE_EXPERIMENTS`: base path for trained models and perturbations after attacks.
* `BASE_LOGS`: base path for tensorboard logs (if used).

## Data

Data needs to be provided in the HDF5 format. To use a dataset, use the following steps:
* In [`common/paths.py`](common/paths.py), set `BASE_DATA` to the base path where data will be stored.
* For each dataset, create a directory named `<dataset-name>` in `BASE_DATA`
* Place the following files in this directory:
  * `train_images.h5`: Training images
  * `train_labels.h5`: Training labels
  * `test_images.h5`: Test images
  * `test_labels.h5`: Test labels

A script [create_dataset_h5.py](scripts/create_dataset_h5.py) has been provided to convert data in a comma-separated CSV file consisting of full paths to images and their corresponding labels to a HDF5 file. To use this script, first set `BASE_DATA` in [`common/paths.py`](common/paths.py). If the files containing training and test data paths and labels are `train.csv` and `test.csv` respectively, use:
```bash
$ python scripts/create_dataset_h5.py --train_csv /path/to/train.csv --test_csv /path/to/test.csv --dataset dataset_name
```
where `dataset_name` is the name for the dataset.

## Training and evaluating a model
### Training
To train a model, use:
```bash
$ python scripts/train.py [options]
```

A list of available options and their descriptions can be found by using:
```bash
$ python scripts/train.py -h
```
### Evaluation
To evaluate a trained model, use:
```bash
$ python scripts/evaluate.py [options]
```

A list of available options and their descriptions can be found by using:
```bash
$ python scripts/evaluate.py -h
```



## Using models and attacks from the paper

The following provides the arguments to use with the training and evaluation scripts to train the models and run the attacks described in the paper. The commands below assume that the dataset is named `cifar10` and has 10 classes.

### Models
#### Normal
```bash
$ python scripts/train.py --cuda --dataset cifar10 --n_classes 10 --cuda --mode normal --log_dir logs --snapshot_frequency 5 --models_dir models --use_tensorboard --use_flip
```

#### Occlusion
```bash
$ python scripts/train.py --cuda --dataset cifar10 --n_classes 10 --mask_dims 8 8 --mode adversarial --location random --exclude_box 11 11 10 10 --epsilon 0.1 --signed_grad --max_iterations 1 --log_dir logs --snapshot_frequency 5 --models_dir models --use_tensorboard --use_flip
```

#### AT-Fixed
```bash
$ python scripts/train.py --cuda --dataset cifar10 --n_classes 10 --mask_pos 3 3 --mask_dims 8 8 --mode adversarial --location fixed --exclude_box 11 11 10 10 --epsilon 0.1 --signed_grad --max_iterations 25 --log_dir logs --snapshot_frequency 5 --models_dir models --use_tensorboard --use_flip
```

#### AT-Rand
```bash
$ python scripts/train.py --cuda --dataset cifar10 --n_classes 10 --mask_dims 8 8 --mode adversarial --location random --exclude_box 11 11 10 10 --epsilon 0.1 --signed_grad --max_iterations 25 --log_dir logs --snapshot_frequency 5 --models_dir models --use_tensorboard --use_flip
```

#### AT-RandLO
```bash
$ python scripts/train.py --cuda --dataset cifar10 --n_classes 10 --mask_dims 8 8 --mode adversarial --location random --exclude_box 11 11 10 10 --epsilon 0.1 --signed_grad --max_iterations 25 --optimize_location --opt_type random --stride 2 --log_dir logs --snapshot_frequency 5 --models_dir models --use_tensorboard --use_flip
```

#### AT-FullLO
```bash
$ python scripts/train.py --cuda --dataset cifar10 --n_classes 10 --mask_dims 8 8 --mode adversarial --location random --exclude_box 11 11 10 10 --epsilon 0.1 --signed_grad --max_iterations 25 --optimize_location --opt_type full --stride 2 --log_dir logs --snapshot_frequency 5 --models_dir models --use_tensorboard --use_flip
```

### Attacks

The arguments used here correspond to using 100 iterations and 30 attempts. These can be changed by appropriately setting `--iterations` and `--attempts` respectively.

#### AP-Fixed
```bash
$ python scripts/evaluate.py --cuda --dataset cifar10 --n_classes 10 --mask_pos 3 3 --mask_dims 8 8 --mode adversarial --log_dir logs --models_dir models --saved_model_file model_complete_200 --attempts 30 --location fixed --epsilon 0.05 --iterations 100 --signed_grad --perturbations_file perturbations --use_tensorboard
```

#### AP-Rand
```bash
$ python scripts/evaluate.py --cuda --dataset cifar10 --n_classes 10 --mask_dims 8 8 --mode adversarial --log_dir logs --models_dir models --saved_model_file model_complete_200 --attempts 30 --location random --epsilon 0.05 --iterations 100 --exclude_box 11 11 10 10 --signed_grad --perturbations_file perturbations --use_tensorboard
```

#### AP-RandLO
```bash
$ python scripts/evaluate.py --cuda --dataset cifar10 --n_classes 10 --mask_dims 8 8 --mode adversarial --log_dir logs --models_dir models --saved_model_file model_complete_200 --attempts 30 --location random --epsilon 0.05 --iterations 100 --exclude_box 11 11 10 10 --optimize_location --opt_type random --stride 2 --signed_grad --perturbations_file perturbations --use_tensorboard
```

#### AP-FullLO
```bash
$ python scripts/evaluate.py --cuda --dataset cifar10 --n_classes 10 --mask_dims 8 8 --mode adversarial --log_dir logs --models_dir models --saved_model_file model_complete_200 --attempts 30 --location random --epsilon 0.05 --iterations 100 --exclude_box 11 11 10 10 --optimize_location --opt_type full --stride 2 --signed_grad --perturbations_file perturbations --use_tensorboard
```

## Citation

Please cite the paper as follows:
```
@InProceedings{Rao2020Adversarial,
author = {Sukrut Rao and David Stutz and Bernt Schiele},
title = {Adversarial Training against Location-Optimized Adversarial Patches},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV) Workshops},
year = {2020}
} 
```

## Acknowledgement

This repository uses code from [davidstutz/confidence-calibrated-adversarial-training](https://github.com/davidstutz/confidence-calibrated-adversarial-training).

## License
Copyright (c) 2020 Sukrut Rao, David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying
documentation before you download and/or use this software and associated
documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge
right to copy, modify, merge, publish, distribute, and sublicense the Software
for the sole purpose of performing non-commercial scientific research,
non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited.
This includes, without limitation, incorporation in a commercial product, use in
a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide
either maintenance services, update services, notices of latent defects, or
corrections of defects with regard to the Software. The authors nevertheless
reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. You agree to cite the
corresponding papers (see above) in documents and papers that report on research
using the Software.

