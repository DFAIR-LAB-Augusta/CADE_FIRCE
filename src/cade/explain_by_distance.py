"""
explain_by_distance.py
~~~~~~~

Functions for explaining why a sample is an drift (distance-based).
Perturb the testing set drift sample to be closer to the centroid of its closest family.
If flipping a feature can make the drift sample be closer to the centroid, then we think it's important.
In this case, we can't rank the importance of a feature, just two cases: important or unimportant.

Two design options:
1. use mask only, use mask_exp_by_distance_mask_only.py
2. use mask * m1, use mask_exp_by_distance_mask_m1.py

"""  # noqa: E501

import logging
import os
import random
import re
import traceback
from typing import Literal

import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.models import Model
from numpy.random import seed
from tqdm import tqdm

import cade.utils as utils
from cade.autoencoder import Autoencoder

from .utils import SimConfig

os.environ['PYTHONHASHSEED'] = '0'


random.seed(1)
seed(1)

tf.random.set_seed(2)


def explain_drift_samples_per_instance(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    _y_test: np.ndarray,
    config: SimConfig,
    one_by_one_check_result_path: str,
    training_info_for_detect_path: str,
    cae_weights_path: str,
    mask_file_path: str,
) -> None:
    """
    Generate feature-level explanations for drift samples by comparing them to family centroids.

    This function identifies samples flagged as drift, calculates statistical boundaries
    (MAD-based distance lower bounds) for their closest historical families, and
    optimizes a feature mask for each instance to explain why it deviated from
    the family's training distribution.

    Args:
        x_train: Training set feature vectors.
        y_train: Training set ground truth labels.
        x_test: Testing set feature vectors.
        _y_test: Testing set labels (unused).
        config: Typed SimConfig object containing experiment hyperparameters.
        one_by_one_check_result_path: Path to the CSV containing drift detection results.
        training_info_for_detect_path: Path to the compressed training distribution statistics.
        cae_weights_path: Path to the pre-trained Contrastive Autoencoder weights.
        mask_file_path: Output path to save the generated explanation masks (.npz).

    Returns:
        None. Saves the 'masks' array to the specified file path.
    """  # noqa: E501
    if os.path.exists(mask_file_path):
        logging.info(
            f'explanation result file {mask_file_path} exists, no need to run explanation module'  # noqa: E501
        )
    else:
        drift_samples_idx_list, _drift_samples_real_labels, drift_samples_closest = (
            get_drift_samples_to_explain(one_by_one_check_result_path)
        )

        mad_threshold = config.mad_threshold

        cae_dims = utils.get_model_dims(
            'Contrastive AE',
            x_train.shape[1],
            config.cae_hidden,
            len(np.unique(y_train)),
        )

        family_centroid_dict = {}
        for family in np.unique(drift_samples_closest):
            _z_train, _z_closest_family, centroid, dis_to_centroid, mad = (
                load_training_info(training_info_for_detect_path, family)
            )
            distance_lowerbound = mad * mad_threshold + np.median(dis_to_centroid)
            dis_to_centroid_inds = np.array(
                dis_to_centroid
            ).argsort()  # distance ascending
            x_train_family = x_train[np.where(y_train == family)[0]]
            closest_to_centroid_sample = x_train_family[dis_to_centroid_inds][0]
            logging.debug(
                f'family-{family} closest distance to centroid: {np.min(dis_to_centroid)}'  # noqa: E501
            )

            family_centroid_dict[family] = [
                centroid,
                distance_lowerbound,
                closest_to_centroid_sample,
            ]

        masks = []
        x_drift_list = []
        for idx, sample_idx in tqdm(
            enumerate(drift_samples_idx_list),
            total=len(drift_samples_idx_list),
            desc='explain drift',
        ):
            try:
                x_target = x_test[sample_idx]
                x_drift_list.append(x_target)
                closest_family = drift_samples_closest[idx]

                [centroid, distance_lowerbound, closest_to_centroid_sample] = (
                    family_centroid_dict[closest_family]
                )

                diff = x_target - closest_to_centroid_sample
                diff_idx = np.where(diff != 0)[0]

                mask = explain_instance(
                    x_target,
                    config.exp_method,
                    diff_idx,
                    centroid,
                    closest_to_centroid_sample,
                    distance_lowerbound,
                    config.exp_lambda_1,
                    cae_dims,
                    cae_weights_path,
                )
                masks.append(mask)
            except Exception:
                masks.append(None)
                logging.error(f'idx: {idx}, sample_idx: {sample_idx}')
                logging.error(traceback.format_exc())

        np.savez_compressed(mask_file_path, masks=masks)


def get_drift_samples_to_explain(
    one_by_one_check_result_path: str,
) -> tuple[list, list, list]:
    """
    Extract a subset of detected drift samples for feature-level explanation.

    This function parses the cumulative drift report to find the "best inspection count"
    (the optimal number of samples to review based on F1-score). it then retrieves
    the metadata for exactly that many samples from the top of the report.

    Args:
        one_by_one_check_result_path: Path to the CSV/text report generated by
            the drift detection and PR-plotting module.

    Returns:
        A tuple containing three lists:
            - drift_samples_idx_list: Original indices of the samples in the test set.
            - drift_samples_real_labels: The ground-truth labels for these samples.
            - drift_samples_closest: The label of the closest historical family
              assigned to each sample.

    Raises:
        IndexError: If the 'best inspection count' pattern is not found in the file.
        FileNotFoundError: If the result path does not exist.
    """
    pattern = re.compile(r'best inspection count: \d+')
    with open(one_by_one_check_result_path) as f:
        inspect_cnt = int(
            re.findall(pattern, f.read())[0].replace('best inspection count: ', '')
        )

    drift_samples_idx_list, drift_samples_real_labels, drift_samples_closest = (
        [],
        [],
        [],
    )
    with open(one_by_one_check_result_path) as f:
        next(f)
        for idx, line in enumerate(f):
            if idx < inspect_cnt:
                line_data = line.strip().split(',')
                drift_samples_idx_list.append(int(line_data[0]))
                drift_samples_real_labels.append(int(line_data[1]))
                drift_samples_closest.append(int(line_data[2]))

    assert len(drift_samples_idx_list) == inspect_cnt
    assert len(drift_samples_closest) == inspect_cnt

    return drift_samples_idx_list, drift_samples_real_labels, drift_samples_closest


def load_encoder(cae_dims: list[int], cae_weights_path: str) -> Model:
    # be careful with this it may clean up previous loaded models.
    k.clear_session()
    ae = Autoencoder(cae_dims)
    _ae_model, encoder_model = ae.build()
    encoder_model.load_weights(cae_weights_path, by_name=True)
    return encoder_model


def load_training_info(
    training_info_for_detect_path: str, closest_family: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Extract training distribution statistics for a specific family from a saved report.

    This function loads pre-calculated latent space data, centroids, and Median
    Absolute Deviation (MAD) values, filtering them to provide metrics specific
    to the requested family ID.

    Args:
        training_info_for_detect_path: Path to the .npz file containing training info.
        closest_family: The integer label of the family to extract info for.

    Returns:
        A tuple containing:
            - z_train (np.ndarray): The full latent representation of the training set.
            - z_closest_family (np.ndarray): Latent vectors belonging only to the specified family.
            - centroid (np.ndarray): The mean/center vector of the specified family.
            - dis_to_centroid (np.ndarray): Array of distances from family members to the centroid.
            - mad (float): The Median Absolute Deviation of distances for this family.
    """  # noqa: E501
    info = np.load(training_info_for_detect_path)
    z_train = info['z_train']
    z_family = info['z_family']
    centroids = info['centroids']
    dis_family = info['dis_family']
    mad_family = info['mad_family']

    z_closest_family = z_family[closest_family]
    centroid = centroids[closest_family]
    dis_to_centroid = dis_family[closest_family]
    mad = mad_family[closest_family]

    logging.debug(f'z_closest_family shape: {z_closest_family.shape}')
    logging.debug(f'centroid-{closest_family}: {centroid}')
    logging.debug(f'dis_to_centroid median: {np.median(dis_to_centroid)}')
    logging.debug(f'mad-{closest_family}: {mad}')

    return z_train, z_closest_family, centroid, dis_to_centroid, mad


def explain_instance(
    x: np.ndarray,
    exp_method: Literal['distance_mm1', 'approximation_loose'],
    diff_idx: np.ndarray,
    centroid: np.ndarray,
    closest_to_centroid_sample: np.ndarray,
    distance_lowerbound: float,
    lambda_1: float,
    cae_dims: list[int],
    cae_weights_path: str,
) -> np.ndarray | None:
    optimizer = tf.train.AdamOptimizer
    initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
    lr = 1e-2  # learning rate
    # a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.  # noqa: E501
    regularizer = 'elasticnet'
    exp_epoch = 250
    exp_display_interval = 10  # print middle result every k epochs
    exp_lambda_patience = 20
    early_stop_patience = 10
    use_gumble_trick = True

    mask_shape = (x.shape[0],)
    latent_dim = cae_dims[-1]

    temp = 0.1
    m1 = np.zeros(shape=(x.shape[0],), dtype=np.float32)
    for i in diff_idx:
        m1[i] = 1

    logging.debug(f'MASK_SHAPE: {mask_shape}')
    logging.debug(f'latent_dim: {latent_dim}')
    logging.debug(f'distance lowerbound: {distance_lowerbound}')
    logging.debug(f'epoch: {exp_epoch}')
    logging.debug(f'temperature: {temp}')
    logging.debug(f'use gumble trick: {use_gumble_trick}')

    mask_best = None
    if exp_method == 'distance_mm1':
        import cade.mask_exp_by_distance_mask_m1 as mask_exp

        k.clear_session()
        model = load_encoder(cae_dims, cae_weights_path)

        exp_test = mask_exp.OptimizeExp(
            batch_size=10,
            mask_shape=mask_shape,
            latent_dim=latent_dim,
            model=model,
            optimizer=optimizer,
            initializer=initializer,
            lr=lr,
            regularizer=regularizer,
            temp=temp,
            normalize_choice='clip',
            use_concrete=use_gumble_trick,
            model_file=cae_weights_path,
        )

        mask_best = exp_test.fit_local(
            x=x,
            m1=m1,
            centroid=centroid,
            closest_to_centroid_sample=closest_to_centroid_sample,
            num_sync=50,
            num_changed_fea=1,
            epochs=exp_epoch,
            lambda_1=lambda_1,
            display_interval=exp_display_interval,
            exp_loss_lowerbound=distance_lowerbound,
            lambda_patience=exp_lambda_patience,
            early_stop_patience=early_stop_patience,
        )

        if mask_best is not None:
            logging.debug(f'M1 * mask == 1: {np.where(m1 * mask_best == 1.0)[0]}')
            return m1 * mask_best
    return None
