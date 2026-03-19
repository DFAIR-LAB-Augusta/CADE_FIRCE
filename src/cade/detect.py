"""
detect.py
~~~~~~~

Functions for detecting drifting samples, write the closest family for each sample in the testing set.

"""  # noqa: E501

import logging
import os
import random

import numpy as np
import tensorflow as tf
from keras import backend as k
from numpy.random import seed
from tqdm import tqdm

from cade.autoencoder import Autoencoder

os.environ['PYTHONHASHSEED'] = '0'


random.seed(1)
seed(1)

tf.random.set_seed(2)


def configure_tensorflow() -> None:
    tf.random.set_seed(2)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Memory growth must be set before GPUs are initialized
            pass


def detect_drift_samples(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    dims: list[int],
    margin: float,
    mad_threshold: float,
    best_weights_file: str,
    all_detect_path: str,
    simple_detect_path: str,
    training_info_for_detect_path: str,
) -> None:
    if os.path.exists(all_detect_path) and os.path.exists(simple_detect_path):
        logging.info(
            'Detection result files exist, no need to redo the detection')
    else:
        """get latent data for the entire training and testing set"""
        z_train, z_test = get_latent_representation_keras(
            dims, best_weights_file, x_train, x_test
        )

        """get latent data for each family in the training set"""
        n, n_family, z_family = get_latent_data_for_each_family(
            z_train, y_train)

        """get centroid for each family in the latent space"""
        centroids = [np.mean(z_family[i], axis=0) for i in range(n)]
        # centroids = [np.median(z_family[i], axis=0) for i in range(N)]
        logging.debug(f'centroids: {centroids}')

        """get distance between each training sample and their family's centroid in the latent space """  # noqa: E501
        dis_family = get_latent_distance_between_sample_and_centroid(
            z_family, centroids, margin, n, n_family
        )

        """get the MAD for each family"""
        mad_family = get_mad_for_each_family(dis_family, n, n_family)

        np.savez_compressed(
            training_info_for_detect_path,
            z_train=z_train,
            z_family=z_family,
            centroids=centroids,
            dis_family=dis_family,
            mad_family=mad_family,
        )

        """detect drifting in the testing set"""
        with open(all_detect_path, 'w') as f1:
            f1.write(
                'sample_idx,is_drift,closest_family,real_label,pred_label,min_distance,min_anomaly_score\n'
            )
            with open(simple_detect_path, 'w') as f2:
                f2.write(
                    'sample_idx,closest_family,real_label,pred_label,min_distance,min_anomaly_score\n'
                )

                for k in tqdm(range(len(x_test)), desc='detect', total=x_test.shape[0]):
                    z_k = z_test[k]
                    """get distance between each testing sample and each centroid"""
                    dis_k = [np.linalg.norm(z_k - centroids[i])
                             for i in range(n)]
                    anomaly_k = [
                        np.abs(dis_k[i] - np.median(dis_family[i])
                               ) / mad_family[i]
                        for i in range(n)
                    ]
                    logging.debug(f'sample-{k} - dis_k: {dis_k}')
                    logging.debug(f'sample-{k} - anomaly_k: {anomaly_k}')

                    closest_family = np.argmin(dis_k)
                    min_dis = np.min(dis_k)
                    min_anomaly_score = np.min(anomaly_k)

                    if min_anomaly_score > mad_threshold:
                        logging.debug(f'testing sample {k} is drifting')
                        f1.write(
                            f'{k},Y,{closest_family},{y_test[k]},{y_pred[k]},{min_dis},{min_anomaly_score}\n'
                        )
                        f2.write(
                            f'{k},{closest_family},{y_test[k]},{y_pred[k]},{min_dis},{min_anomaly_score}\n'
                        )
                    else:
                        f1.write(
                            f'{k},N,{closest_family},{y_test[k]},{y_pred[k]},{min_dis},{min_anomaly_score}\n'
                        )


def get_latent_representation_keras(
    dims: list[int], best_weights_file: str, x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trained encoder and extract latent features (embeddings) from input data.

    Args:
        dims: List of dimensions for the autoencoder architecture.
        best_weights_file: Path to the saved .h5 weights file.
        x_train: Training feature vectors to be encoded.
        x_test: Testing feature vectors to be encoded.

    Returns:
        A tuple containing:
            - z_train (np.ndarray): Latent representation of the training data.
            - z_test (np.ndarray): Latent representation of the testing data.
    """
    k.clear_session()
    ae = Autoencoder(dims)
    _ae_model, encoder_model = ae.build()
    encoder_model.load_weights(best_weights_file, by_name=True)

    z_train = encoder_model.predict(x_train)
    z_test = encoder_model.predict(x_test)

    logging.debug(f'z_train shape: {z_train.shape}')
    logging.debug(f'z_test shape: {z_test.shape}')
    logging.debug(f'z_train[0]: {z_train[0]}')

    return z_train, z_test


def get_latent_data_for_each_family(
    z_train: np.ndarray, y_train: np.ndarray
) -> tuple[int, list[int], list[np.ndarray]]:
    """
    Groups latent representations into a list of arrays based on family labels.

    Args:
        z_train: Feature vectors in the latent space (embeddings).
        y_train: Label array corresponding to the training samples.

    Returns:
        A tuple containing:
            - n (int): The number of unique families found in the labels.
            - n_family (list[int]): A list containing the count of samples
                per family.
            - z_family (list[np.ndarray]): A list where each element is an
                ndarray of latent vectors belonging to that specific family.
    """
    n = len(np.unique(y_train))
    n_family = [len(np.where(y_train == family)[0]) for family in range(n)]
    z_family = []
    for family in range(n):
        z_tmp = z_train[np.where(y_train == family)[0]]
        z_family.append(z_tmp)

    z_len = [len(z_family[i]) for i in range(n)]
    logging.debug(f'z_family length: {z_len}')

    return n, n_family, z_family


def get_latent_distance_between_sample_and_centroid(
    z_family: list[np.ndarray],
    centroids: list[np.ndarray],
    _margin: float,
    n: int,
    n_family: list[int],
) -> list[list[float]]:
    """
    Calculate Euclidean distances between family samples and their respective centroids.

    Args:
        z_family: List of arrays containing latent vectors for each family.
        centroids: List of mean vectors (centroids) for each family.
        _margin: The margin parameter (unused in this calculation but passed).
        n: The number of unique families.
        n_family: A list containing the count of samples in each family.

    Returns:
        dis_family: A nested list where dis_family[i][j] is the distance
            of the j-th sample of the i-th family to its centroid.
    """
    dis_family = []  # two-dimension list

    for i in range(n):  # i: family index
        dis = [
            np.linalg.norm(z_family[i][j] - centroids[i]) for j in range(n_family[i])
        ]
        dis_family.append(dis)

    dis_len = [len(dis_family[i]) for i in range(n)]
    logging.debug(f'dis_family length: {dis_len}')

    return dis_family


def get_mad_for_each_family(
    dis_family: list[list[float]], n: int, n_family: list[int]
) -> list[float]:
    """
    Calculate the Median Absolute Deviation (MAD) for each family's distances.

    The MAD is calculated as: 1.4826 * median(|x_i - median(x)|).
    The constant 1.4826 is used to make MAD a consistent estimator for
    the standard deviation of a Gaussian distribution.

    Args:
        dis_family: Nested list of distances from samples to centroids per family.
        n: The number of unique families.
        n_family: A list containing the count of samples in each family.

    Returns:
        A list of MAD values, one for each family.
    """
    mad_family = []
    for i in range(n):
        median = np.median(dis_family[i])
        logging.debug(f'family {i} median: {median}')
        diff_list = [np.abs(dis_family[i][j] - median)
                     for j in range(n_family[i])]
        mad = 1.4826 * np.median(
            diff_list
        )  # 1.4826: assuming the underlying distribution is Gaussian
        mad_family.append(mad)
    logging.debug(f'mad_family: {mad_family}')

    return mad_family
