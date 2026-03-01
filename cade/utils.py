"""
utils.py
~~~~~~~~

Helper functions for setting up the environment.

"""

import argparse
import logging
import os
import random
import sys
import traceback
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.random import seed
from scipy.optimize import linear_sum_assignment
from tensorflow import set_random_seed

os.environ['PYTHONHASHSEED'] = '0'


random.seed(1)
seed(1)

set_random_seed(2)


@dataclass(frozen=True)
class SimConfig:
    """Configuration for drift detection and explanation experiments."""

    data: str
    classifier: Literal['mlp', 'rf']
    stage: Literal['detect', 'explanation']
    pure_ae: int
    quiet: int
    cae_hidden: str
    cae_batch_size: int
    cae_lr: float
    cae_epochs: int
    cae_lambda_1: float
    similar_ratio: float
    margin: float
    display_interval: int
    mad_threshold: float
    exp_method: Literal['distance_mm1', 'approximation_loose']
    exp_lambda_1: float
    mlp_retrain: int
    mlp_hidden: str
    mlp_batch_size: int
    mlp_lr: float
    mlp_epochs: int
    mlp_dropout: float
    newfamily_label: int
    tree: int
    rf_retrain: int


def parse_args() -> SimConfig:
    """Parse the command line configuration for a particular run.

    Raises:
        ValueError: if the tree value for RandomForest is negative.

    Returns:
        SimConfig -- a typed dataclass containing the parsed arguments.
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--data', required=True, help='The dataset to use.')

    # classifier
    p.add_argument(
        '-c',
        '--classifier',
        default='mlp',
        choices=['mlp', 'rf'],
        help='The target classifier to use.',
    )

    # experiment stages
    p.add_argument(
        '--stage',
        default='detect',
        choices=['detect', 'explanation'],
        help='Stage to run: "detect" (only detection) or "explanation" (both).',
    )
    p.add_argument(
        '--pure-ae',
        default=0,
        type=int,
        choices=[0, 1],
        help='Use standard autoencoder (1) or contrastive autoencoder (0).',
    )
    p.add_argument(
        '--quiet',
        default=1,
        type=int,
        choices=[0, 1],
        help='Whether to print the debugging logs.',
    )

    # Contrastive Autoencoder / Drift Detection
    p.add_argument(
        '--cae-hidden',
        default='512-128-32',
        help='Hidden layers of CAE (e.g., "512-128-32").',
    )
    p.add_argument('--cae-batch-size', default=64, type=int, help='CAE batch size.')
    p.add_argument(
        '--cae-lr', default=0.001, type=float, help='CAE Adam learning rate.'
    )
    p.add_argument('--cae-epochs', default=250, type=int, help='CAE training epochs.')
    p.add_argument(
        '--cae-lambda-1', default=1e-1, type=float, help='CAE loss function lambda_1.'
    )
    p.add_argument(
        '--similar-ratio',
        default=0.25,
        type=float,
        help='Ratio of similar samples in CAE batch.',
    )
    p.add_argument(
        '--margin', default=10.0, type=float, help='Max margins for dissimilar samples.'
    )
    p.add_argument(
        '--display-interval',
        default=10,
        type=int,
        help='Log frequency during CAE training.',
    )
    p.add_argument(
        '--mad-threshold',
        default=3.5,
        type=float,
        help='Threshold for MAD outlier detection.',
    )

    # Explanation module
    p.add_argument(
        '--exp-method',
        default='distance_mm1',
        choices=['distance_mm1', 'approximation_loose'],
        help='Explanation method: our method or baseline.',
    )
    p.add_argument(
        '--exp-lambda-1',
        default=1e-3,
        type=float,
        help='lambda_1 for explanation loss.',
    )

    # MLP sub-arguments
    p.add_argument(
        '--mlp-retrain',
        default=0,
        type=int,
        choices=[0, 1],
        help='Retrain the MLP classifier.',
    )
    p.add_argument(
        '--mlp-hidden', default='100-30', help='MLP hidden layers (e.g., "100-30").'
    )
    p.add_argument('--mlp-batch-size', default=32, type=int, help='MLP batch size.')
    p.add_argument(
        '--mlp-lr', default=0.001, type=float, help='MLP Adam learning rate.'
    )
    p.add_argument('--mlp-epochs', default=50, type=int, help='MLP epochs.')
    p.add_argument('--mlp-dropout', default=0.2, type=float, help='MLP dropout rate.')

    # Data specific
    p.add_argument(
        '--newfamily-label',
        default=7,
        type=int,
        help='Label used for new family in testing.',
    )

    # RandomForest [Deprecated]
    p.add_argument('--tree', type=int, default=100, help='n_estimators for RF.')
    p.add_argument(
        '--rf-retrain', default=0, type=int, choices=[0, 1], help='Retrain RF.'
    )

    args = p.parse_args()

    if args.tree < 0:
        raise ValueError('Tree value for Random Forest cannot be negative.')

    return SimConfig(**vars(args))


def get_model_dims(
    model_name: str, input_layer_num: int, hidden_layer_num: str, output_layer_num: int
) -> list[int]:
    """
    Convert hidden layer arguments to the architecture of a model (list).

    Args:
        model_name: Name of the model ('MLP' or 'Contrastive AE').
        input_layer_num: The number of the features.
        hidden_layer_num: The '-' connected numbers indicating the number
            of neurons in hidden layers.
        output_layer_num: The number of the classes.

    Returns:
        A list representing the model architecture dimensions.
    """
    try:
        hidden_layers = [int(dim) for dim in hidden_layer_num.split('-')]
        dims = [input_layer_num, *hidden_layers, output_layer_num]

        logging.debug(f'{model_name} dims: {dims}')

    except Exception:
        logging.error(f'get_model_dims {model_name}\n{traceback.format_exc()}')
        sys.exit(-1)

    return dims


def create_folder(name: str) -> None:
    if not os.path.exists(name):
        os.makedirs(name)


def create_parent_folder(file_path: str) -> None:
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def redo_flag(args: argparse.Namespace, path_same: str, path_diff: str) -> bool:
    return not ('newfamily' in args.data and os.path.exists(path_diff)) or (
        'evolve' in args.data
        and os.path.exists(path_same)
        and os.path.exists(path_diff)
    )


def get_cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate clustering accuracy by finding the optimal permutation of labels.

    Args:
        y_true: Ground truth labels, numpy.array with shape `(n_samples,)`.
        y_pred: Predicted labels, numpy.array with shape `(n_samples,)`.

    Returns:
        accuracy: The best matching accuracy in the range [0, 1].
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    assert y_pred.size == y_true.size

    # Create the cost/weight matrix
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Solve the linear assignment problem (Kuhn-Munkres algorithm)
    # We use max() - w because linear_sum_assignment minimizes the cost
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Sum the matches and divide by total samples
    accuracy = sum(w[row_ind, col_ind]) / y_pred.size

    return float(accuracy)


def plot_confusion_matrix(
    cm: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dataset_name: str,
    newfamily: int,
    save_fig_name: str,
) -> None:
    """
    Plot and save a confusion matrix heatmap using seaborn.

    Args:
        cm: The confusion matrix array.
        y_pred: Predicted labels from the classifier.
        y_true: Ground truth labels.
        dataset_name: Name of the dataset for conditional logic.
        newfamily: Default index for a new family if bluehex is used.
        save_fig_name: Full path (including filename) to save the plot.
    """
    logging.getLogger('matplotlib.font_manager').disabled = True
    fig, ax = plt.subplots()
    ax = sns.heatmap(cm, annot=True, fmt='d', annot_kws={'size': 12})

    if 'drebin' in dataset_name:
        new = 7
    elif 'IDS' in dataset_name:
        new = 3
    elif 'bluehex' in dataset_name:
        new = newfamily
    else:
        new = newfamily

    no_of_axes = len(np.unique(y_pred)) + 1
    logging.debug(f'no_of_axes: {no_of_axes}')
    if new in np.unique(y_pred):
        label_of_axes = np.unique(y_pred)
    else:
        label_of_axes = sorted(np.append(np.unique(y_pred), [new]))
    ax.set_xticks(np.arange(no_of_axes) + 0.5)
    ax.set_yticks(np.arange(no_of_axes) + 0.5)
    ax.set_xticklabels(label_of_axes, fontsize=16)
    ax.set_yticklabels(np.arange(len(np.unique(y_true))), fontsize=16)
    ax.set_title('Confusion matrix', fontsize=20)
    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)

    fig.tight_layout()
    create_parent_folder(save_fig_name)
    fig.savefig(save_fig_name, dpi=200)
    plt.clf()
