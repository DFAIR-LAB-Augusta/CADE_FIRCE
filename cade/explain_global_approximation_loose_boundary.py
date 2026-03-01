"""
explain_global_approximation_loose_boundary.py
~~~~~~~

Functions for explaining why a sample is a drifting.
For each closest family of the testing samples, build a global approximation model.

For this version: we use in-dist and drift samples from the training set and drift samples from testing set, and synthesized drift based on
testing drift to build a loose approximation model (does not really reflect the exact boundary of the detection module).

"""  # noqa: E501

import logging
import os
import random
import re
import traceback
from functools import partial
from typing import Any

import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from numpy.random import seed
from sklearn.metrics import accuracy_score, pairwise_distances
from tensorflow import set_random_seed
from tqdm import tqdm
from utils import SimConfig

import cade.classifier as classifier
import cade.mask_exp_by_approximation as mask_exp
import cade.utils as utils
from cade.autoencoder import Autoencoder

os.environ['PYTHONHASHSEED'] = '0'


random.seed(1)
seed(1)

set_random_seed(2)


def explain_drift_samples_per_instance(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    _y_test: np.ndarray,
    config: SimConfig,
    one_by_one_check_result_path: str,
    training_info_for_detect_path: str,
    cae_weights_path: str,
    saved_exp_classifier_folder: str,
    mask_file_path: str,
) -> None:
    """
    Orchestrate the instance-level explanation process for all detected drift samples.

    This function executes the full explanation pipeline:
    1. Identifies samples flagged as drift from previous detection results.
    2. Projects test drift samples into latent space using a pre-trained CAE encoder.
    3. Builds or loads surrogate 'global' models for each historical family involved.
    4. Iterates through each drift instance to generate a feature importance mask
       that explains the deviation from its closest known family.
    5. Saves the resulting masks into a compressed .npz file.

    Args:
        x_train: Training feature vectors (raw input space).
        y_train: Ground truth labels for training data.
        x_test: Testing feature vectors (raw input space).
        _y_test: Ground truth labels for testing data (unused, prefixed with underscore).
        config: SimConfig containing hyperparameters (mad_threshold, cae_hidden, exp_lambda_1).
        one_by_one_check_result_path: Path to the CSV containing drift detection results.
        training_info_for_detect_path: Path to the .npz file containing training distribution stats.
        cae_weights_path: Path to the pre-trained weights for the Contrastive Autoencoder.
        saved_exp_classifier_folder: Directory to store/load surrogate MLP models.
        mask_file_path: Output path where the final explanation masks will be saved.

    Returns:
        None. Results are written to the file system at mask_file_path.
    """  # noqa: E501
    if os.path.exists(mask_file_path):
        logging.info(
            f'explanation result file {mask_file_path} exists, no need to run explanation module'  # noqa: E501
        )
        return
    drift_samples_idx_list, _drift_samples_real_labels, drift_samples_closest = (
        get_drift_samples_to_explain(one_by_one_check_result_path)
    )

    mad_threshold: float = config.mad_threshold

    cae_dims = utils.get_model_dims(
        'Contrastive AE', x_train.shape[1], config.cae_hidden, len(np.unique(y_train))
    )

    # load CAE encoder
    encoder_model = load_encoder(cae_dims, cae_weights_path)

    """get all the drift samples from the testing set, separated by their closest family"""  # noqa: E501
    test_z_drift_family = get_z_drift_from_testing_set_by_family(
        x_test, drift_samples_idx_list, drift_samples_closest, encoder_model
    )

    # build global target explanation model for each family (closest one to the testing samples)  # noqa: E501
    x_in_family = build_global_exp_model_for_each_closest_family(
        x_train,
        y_train,
        test_z_drift_family,
        drift_samples_closest,
        training_info_for_detect_path,
        mad_threshold,
        saved_exp_classifier_folder,
        cae_dims,
        cae_weights_path,
    )
    """explain drift per instance """
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

            logging.debug(f'idx-[{idx}] closest family: {closest_family}')

            logging.debug('[explanation] explain single instance...')
            final_model_path = os.path.join(
                saved_exp_classifier_folder,
                f'final_model_family_{closest_family}.h5',
            )
            os.path.join(
                saved_exp_classifier_folder, f'exp_mlp_family_{closest_family}.h5'
            )

            diff = x_target - x_in_family[closest_family][-1]
            diff_idx = np.where(diff != 0)[0]

            mask = explain_instance(
                x_target, config.exp_lambda_1, diff_idx, final_model_path
            )
            masks.append(mask)
            logging.debug('[explanation] explain single instance finished...')

        except Exception:
            logging.error(f'idx: {idx}, sample_idx: {sample_idx}')
            logging.error(traceback.format_exc())

    np.savez_compressed(mask_file_path, masks=masks)


def get_drift_samples_to_explain(
    one_by_one_check_result_path: str,
) -> tuple[list[int], list[int], list[int]]:
    """
    Extract drift samples from a check result file based on the best inspection count.

    This function parses a log file to find a specific 'best inspection count'
    pattern, then extracts that many samples from the subsequent CSV-formatted data.

    Args:
        one_by_one_check_result_path: Path to the file containing inspection
            counts and sample data.

    Returns:
        A tuple containing three lists:
            - drift_samples_idx_list: The indices of the identified drift samples.
            - drift_samples_real_labels: The ground-truth labels of these samples.
            - drift_samples_closest: The labels of the closest historical families.
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
    """
    Build the Autoencoder architecture and load specific weights for the encoder.

    Args:
        cae_dims: List of dimensions defining the Autoencoder layers.
        cae_weights_path: File path to the pre-trained .h5 weights.

    Returns:
        The instantiated Keras Model containing only the encoder layers.

    Note:
        This function calls k.clear_session(), which destroys the current
        TF graph and removes all existing models from memory.
    """
    k.clear_session()
    ae = Autoencoder(cae_dims)
    _ae_model, encoder_model = ae.build()
    encoder_model.load_weights(cae_weights_path, by_name=True)
    return encoder_model


def get_z_drift_from_testing_set_by_family(
    x_test: np.ndarray,
    drift_samples_idx_list: list[int],
    drift_samples_closest: list[int],
    encoder_model: Model,
) -> dict[int, np.ndarray]:
    """
    Extract latent representations for drift samples and group them by closest family.

    Args:
        x_test: The full testing dataset feature vectors.
        drift_samples_idx_list: Indices of the samples identified as drifting.
        drift_samples_closest: The label of the closest historical family for
            each drift sample.
        encoder_model: The trained Keras encoder model used for dimensionality reduction.

    Returns:
        A dictionary mapping family labels (int) to arrays of latent
        representations (np.ndarray) for the identified drift samples.
    """  # noqa: E501
    test_z_drift_family = {}

    x_test_drift = x_test[drift_samples_idx_list]
    z_test_drift = encoder_model.predict(x_test_drift)
    for family in np.unique(drift_samples_closest):
        test_z_drift_family[family] = z_test_drift[
            np.where(drift_samples_closest == family)[0]
        ]
        logging.debug(
            f'test_z_drift_family - {family}: {test_z_drift_family[family].shape}'
        )

    return test_z_drift_family


def build_global_exp_model_for_each_closest_family(
    x_train: np.ndarray,
    y_train: np.ndarray,
    test_z_drift_family: dict[int, np.ndarray],
    drift_samples_closest: list[int],
    training_info_for_detect_path: str,
    mad_threshold: float,
    saved_exp_classifier_folder: str,
    cae_dims: list[int],
    cae_weights_path: str,
) -> dict[Any, np.ndarray]:
    x_in_family = {}
    for family in np.unique(drift_samples_closest):
        """first need to synthesize more drift samples to balance in-dist and drift"""
        _z_train, z_closest_family, centroid, dis_to_centroid, mad = load_training_info(
            training_info_for_detect_path, family
        )

        lower_bound = mad * mad_threshold + np.median(dis_to_centroid)
        logging.critical(
            f'[family-{family}] distance lower bound (to be an drift): {lower_bound}'
        )

        x_train_family = x_train[np.where(y_train == family)[0]]
        z_in, z_drift, x_in = get_in_and_out_distribution_samples(
            x_train_family,
            z_closest_family,
            dis_to_centroid,
            centroid,
            mad,
            mad_threshold,
        )
        x_in_family[family] = x_in

        approximation_mlp_model_path = os.path.join(
            saved_exp_classifier_folder, f'exp_mlp_family_{family}.h5'
        )

        if os.path.exists(approximation_mlp_model_path):
            logging.info(
                f'approximation model file {approximation_mlp_model_path} exists, no need to rerun'  # noqa: E501
            )
        else:
            """
                only perturb test_z_drift a little bit to synthesize more samples,
                and put them into the detection module
            """
            test_z_drift = test_z_drift_family[family]

            cnt_syn_drift = (
                len(z_in) - len(test_z_drift)
            )  # for IDS data, it's better not to generate more drift data because testing drift are enough.  # noqa: E501
            if cnt_syn_drift > 0:
                z_syn_in, z_syn_drift = synthesize_local_samples(
                    test_z_drift,
                    cnt_syn_drift,
                    centroid,
                    dis_to_centroid,
                    mad,
                    mad_threshold,
                    'drift',
                )
                if len(z_syn_in) > 0:  # len(z_syn_in) == 0 can not be stacked
                    z_in = np.vstack((z_in, z_syn_in))
                if len(z_syn_drift) > 0:
                    z_drift = np.vstack((test_z_drift, z_syn_drift))
                else:
                    z_drift = test_z_drift
            else:  # no need to synthesize more drift samples.
                z_drift = test_z_drift
            logging.debug(f'test_z_drift.shape: {test_z_drift.shape}')
            logging.debug(f'[family-{family}]  z_drift.shape: {z_drift.shape}')
            logging.debug(f'[family-{family}]  z_in.shape: {z_in.shape}')

            y_in = np.zeros(shape=(z_in.shape[0],))
            y_drift = np.ones(shape=(z_drift.shape[0],))

            """ build a shallow classifier to distinguish in-distribution and drift samples """  # noqa: E501
            logging.info(
                f'[explantion] build a global classifier for family-{family}...'
            )
            z_weights = None  # DO not use weights at this time.
            num_latent_feats = z_in.shape[1]
            # there are only two classes: in-distribution and drift.
            num_classes = 2
            # NOTE: here use 8-15-2 for drebin, 3-15-2 for IDS
            mlp_dims = [num_latent_feats, 15, num_classes]
            dropout_ratio = 0  # do not use dropout here

            build_target_classifier(
                z_in,
                z_drift,
                y_in,
                y_drift,
                dropout_ratio,
                z_weights,
                mlp_dims,
                approximation_mlp_model_path,
            )
            logging.info(
                f'[explantion] build a global classifier for family-{family} finished'
            )

            """ combine the encoder and shallow MLP classifier (approximation model) as the final model to explain """  # noqa: E501
            final_model_path = os.path.join(
                saved_exp_classifier_folder, f'final_model_family_{family}.h5'
            )
            combine_encoder_and_approximation_model(
                cae_dims,
                mlp_dims,
                dropout_ratio,
                cae_weights_path,
                approximation_mlp_model_path,
                final_model_path,
            )

    return x_in_family


def load_training_info(
    training_info_for_detect_path: str, closest_family: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load saved training statistics and extract data for a specific family.

    Args:
        training_info_for_detect_path: Path to the .npz file containing
            z_train, z_family, centroids, dis_family, and mad_family.
        closest_family: The integer index of the family to extract.

    Returns:
        A tuple containing:
            - z_train (np.ndarray): The full latent representation of training data.
            - z_closest_family (np.ndarray): Latent vectors for the specific family.
            - centroid (np.ndarray): The mean latent vector for the family.
            - dis_to_centroid (np.ndarray): Distances of family samples to centroid.
            - mad (float): The Median Absolute Deviation for the family.
    """
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
    logging.critical(f'dis_to_centroid median: {np.median(dis_to_centroid)}')
    logging.critical(f'mad-{closest_family}: {mad}')

    return z_train, z_closest_family, centroid, dis_to_centroid, mad


def get_in_and_out_distribution_samples(
    x_train_family: np.ndarray,
    z_closest_family: np.ndarray,
    dis_to_centroid: np.ndarray,
    centroid: np.ndarray,
    mad: float,
    mad_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition training samples into in-distribution and out-distribution sets.

    This function sorts samples by their distance to the centroid and identifies
    the boundary where samples transition from drifting (outliers) to in-distribution.

    Args:
        x_train_family: Raw feature vectors for the specific family.
        z_closest_family: Latent representations (embeddings) for the family.
        dis_to_centroid: Pre-calculated distances of training samples to the centroid.
        centroid: The mean latent vector for the family.
        mad: The Median Absolute Deviation for the family.
        mad_threshold: The multiplier for MAD to define the drift boundary.

    Returns:
        A tuple containing:
            - all_in_distribution (np.ndarray): Latent vectors of ID samples.
            - all_out_distribution (np.ndarray): Latent vectors of OOD samples.
            - x_train_family_in_dist (np.ndarray): Raw features of ID samples.
    """
    # step 1: ranked by training samples' distance to the centroid
    dis_to_centroid_inds = np.array(dis_to_centroid).argsort()[
        ::-1
    ]  # dis descending order
    z_closest_family_sorted = z_closest_family[dis_to_centroid_inds]
    x_train_family_sorted = x_train_family[dis_to_centroid_inds]

    # step 2: only keep samples flagged as in-distribution by our detection module
    stop_idx = 0
    for idx, z in enumerate(z_closest_family_sorted):
        dis = np.linalg.norm(z - centroid)
        logging.critical(
            f'training set drift sample-{idx} latent distance to centroid: {dis}'
        )
        if not detect_if_sample_is_drift(
            z, centroid, dis_to_centroid, mad, mad_threshold
        ):
            stop_idx = idx
            break

    all_in_distribution = z_closest_family_sorted[stop_idx:, :]
    x_train_family_in_dist = x_train_family_sorted[stop_idx:, :]

    all_out_distribution = z_closest_family_sorted[0:stop_idx, :]

    logging.debug(f'all_in_distribution.shape: {all_in_distribution.shape}')
    logging.debug(f'all_out_distribution.shape: {all_out_distribution.shape}')
    logging.debug(
        f'training set drift ratio: {len(all_out_distribution) / len(z_closest_family):.3f}'  # noqa: E501
    )
    logging.debug(f'X_train_family_in_dist.shape: {x_train_family_in_dist.shape}')

    return all_in_distribution, all_out_distribution, x_train_family_in_dist


def synthesize_local_samples(
    z_group: np.ndarray,
    cnt_syn: int,
    centroid: np.ndarray,
    dis_to_centroid: np.ndarray,
    mad: float,
    mad_threshold: float,
    base_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize local latent samples using Gaussian noise and classify as ID or drift.

    Args:
        z_group: Original latent vectors to serve as the basis for synthesis.
        cnt_syn: Total number of samples to synthesize.
        centroid: The mean latent vector of the target family.
        dis_to_centroid: Training distances used for the drift detection boundary.
        mad: Median Absolute Deviation of the target family.
        mad_threshold: Multiplier for MAD to define the drift boundary.
        base_label: Label name for logging purposes.

    Returns:
        A tuple of (z_syn_in, z_syn_drift) as numpy arrays.
    """
    if len(z_group) == 0:
        return np.array([]), np.array([])

    # No. of times each sample is synthesized
    augment_times = max(1, round(cnt_syn / len(z_group)))
    sigma = mad
    logging.debug(f'noise sigma: {sigma} for {base_label}')

    all_syn_samples = []
    for z in z_group:
        noise = np.random.normal(0, sigma, size=(augment_times, len(z)))
        z_syn = z + noise  # Broadcasting: adds z to every row of noise
        all_syn_samples.append(z_syn)

    # Combine all synthesized batches into one array
    z_syn_total = np.vstack(all_syn_samples)

    # (≥cnt_syn, latent_dim)
    logging.debug(f'z_syn_total.shape: {z_syn_total.shape}')

    # use the detection module to determine the synthesized samples are in-dist or drift
    z_syn_drift, z_syn_in = [], []
    for i in range(len(z_syn_total)):
        is_drift = detect_if_sample_is_drift(
            z_syn_total[i], centroid, dis_to_centroid, mad, mad_threshold
        )
        if is_drift:
            z_syn_drift.append(z_syn_total[i])
            # logging.debug(f'synthesized drift sample to centroid dis - {i}: {np.linalg.norm(z_syn_total[i] - centroid)}')  # noqa: E501
        else:
            z_syn_in.append(z_syn_total[i])

    z_syn_drift = np.array(z_syn_drift)
    z_syn_in = np.array(z_syn_in)
    logging.debug(f'z_syn_drift.shape: {z_syn_drift.shape}')
    logging.debug(f'z_syn_in.shape: {z_syn_in.shape}')
    return z_syn_in, z_syn_drift


def detect_if_sample_is_drift(
    z: np.ndarray,
    centroid: np.ndarray,
    dis_to_centroid: np.ndarray,
    mad: float,
    mad_threshold: float,
) -> bool:
    dis = np.linalg.norm(z - centroid)
    anomaly = np.abs(dis - np.median(dis_to_centroid)) / mad
    return anomaly > mad_threshold


def assign_weights_based_on_dist(
    z_in: np.ndarray, z_drift: np.ndarray, z_target: np.ndarray
) -> np.ndarray:
    """
    Unused
    refer LIME's code to assign sample weights
    use an exponential kernel
    weight = e^(-D(z_syn, z_target)^2 / sigma^2),
    sigma is called the kernel's width, if not specified, use sqrt(#column) * 0.75

    whether it's in-dist or drift, all assigned weights based on their distance to the target drift sample.
    """  # noqa: E501
    z_all = np.vstack((z_in, z_drift))
    distances = pairwise_distances(
        z_all, z_target.reshape(1, -1), metric='euclidean'
    ).ravel()

    kernel_width = float(np.sqrt(z_all.shape[1]) * 0.75)

    kernel_fn = partial(
        kernel, kernel_width=kernel_width
    )  # partial: wrap the original function to have fewer arguments
    weights = kernel_fn(distances)

    # logging.debug(f'z distances in: {distances[:len(z_in)]}')
    # logging.debug(f'z distances drift: {list(distances[len(z_in):])}')
    # logging.debug(f'z weights in: {weights[:len(z_in)]}')
    # logging.debug(f'z weights drift: {list(weights[len(z_in):])}')

    return weights


def kernel(d: np.ndarray, kernel_width: float) -> np.ndarray:
    return np.sqrt(np.exp(-(d**2) / kernel_width**2))


def build_target_classifier(
    z_in: np.ndarray,
    z_drift: np.ndarray,
    y_in: np.ndarray,
    y_drift: np.ndarray,
    dropout_ratio: int,
    z_weights: None,
    mlp_dims: list[int],
    model_save_path: str,
) -> None:
    """
    Train and save a binary MLP classifier to distinguish between ID and drift samples.

    This "explanation" classifier acts as a surrogate model. By training it on
    latent representations of in-distribution versus out-distribution (drift)
    samples, we can later analyze the classifier's weights or use it to explain
    the characteristics of the detected drift.

    Args:
        z_in: Latent vectors of samples flagged as in-distribution.
        z_drift: Latent vectors of samples flagged as drift/out-distribution.
        y_in: Labels for in-distribution samples (typically 0).
        y_drift: Labels for drift samples (typically 1).
        dropout_ratio: The fraction of neurons to drop during training (0.0 to 1.0).
        z_weights: Optional sample weights for the training process; useful if
            classes are imbalanced.
        mlp_dims: A list of integers defining the neurons in each hidden layer.
        model_save_path: The file system path where the trained model (.h5) will be saved.

    Raises:
        TypeError: If the loaded model after training is not a valid Keras Model.
    """  # noqa: E501
    mlp_classifier = classifier.MLPClassifier(
        dims=mlp_dims, model_save_name=model_save_path, dropout=dropout_ratio, verbose=0
    )  # no logs

    logging.debug(f'Saving explanation MLP models to {model_save_path}...')
    retrain_flag = 1 if not os.path.exists(model_save_path) else 0

    x = np.vstack((z_in, z_drift))
    y = np.hstack((y_in, y_drift))

    logging.debug(f'x.shape: {x.shape}')
    logging.debug(f'y.shape: {y.shape}')

    epochs = 30

    mlp_classifier.train(
        x,
        y,
        lr=0.01,
        batch_size=32,
        epochs=epochs,
        loss='binary_crossentropy',
        class_weight=None,
        sample_weight=z_weights,
        train_val_split=False,  # do not split train and val, predict on all the training set  # noqa: E501
        retrain=bool(retrain_flag),
    )
    k.clear_session()  # to prevent load_model becomes slower and slower
    clf = load_model(model_save_path)
    if not isinstance(clf, Model):
        raise TypeError(
            f'Loaded model is invalid. Expected Keras Model, got {type(clf)}'
        )
    logging.debug(
        f'[build_target_classifier] prediction in: {list(np.argmax(clf.predict(x[0 : len(z_in)]), axis=1))}'  # noqa: E501
    )
    logging.debug(
        f'[build_target_classifier] prediction drift: {list(np.argmax(clf.predict(x[len(z_in) :]), axis=1))}'  # noqa: E501
    )

    y_pred = clf.predict(x)
    logging.debug(f'y_pred shape: {y_pred.shape}')
    y_pred = np.argmax(clf.predict(x), axis=1)
    logging.info(f'clf predict accuracy: {accuracy_score(y, y_pred)}')


def combine_encoder_and_approximation_model(
    cae_dims: list[int],
    mlp_dims: list[int],
    dropout_ratio: int,
    cae_weights_path: str,
    approximation_mlp_model_path: str,
    final_model_save_path: str,
) -> None:
    """
    Concatenate a pre-trained encoder and an MLP classifier into a single model.

    This function reconstructs the architecture of both the encoder and the
    surrogate MLP, then loads their respective weights. The resulting model
    takes raw feature vectors as input and outputs the ID vs. Drift probability.

    Args:
        cae_dims: Layer dimensions of the Convolutional/Dense Autoencoder.
        mlp_dims: Layer dimensions of the approximation (surrogate) MLP.
        dropout_ratio: The dropout rate used in the MLP hidden layers.
        cae_weights_path: Path to the file containing Autoencoder weights.
        approximation_mlp_model_path: Path to the file containing MLP weights.
        final_model_save_path: Destination path for the combined model file.
    """
    act = 'relu'
    init = 'glorot_uniform'
    n_stacks = len(cae_dims) - 1

    input_ = Input(shape=(cae_dims[0],), name='input')
    x = input_
    for i in range(n_stacks - 1):
        x = Dense(
            cae_dims[i + 1],
            activation=act,
            kernel_initializer=init,
            name=f'encoder_{i}',
        )(x)
    encoded = Dense(
        cae_dims[-1], kernel_initializer=init, name=f'encoder_{n_stacks - 1}'
    )(x)

    clf_stacks = len(mlp_dims) - 1
    x2 = encoded
    for i in range(clf_stacks - 1):
        x2 = Dense(mlp_dims[i + 1], activation='relu', name=f'clf_{i}')(x2)
        if dropout_ratio > 0:
            x2 = Dropout(dropout_ratio, seed=42)(x2)
    data = Dense(mlp_dims[-1], activation='softmax', name=f'clf_{clf_stacks - 1}')(x2)

    final_model = Model(inputs=input_, outputs=data)
    final_model.load_weights(cae_weights_path, by_name=True)
    final_model.load_weights(approximation_mlp_model_path, by_name=True)
    final_model.save(final_model_save_path)


def explain_instance(
    x: np.ndarray, lambda_1: float, diff_idx: np.ndarray, final_model_path: str
) -> np.ndarray | None:
    """
    Optimize a feature mask to explain why an instance was flagged as drift.

    Args:
        x: The feature vector of the sample to explain.
        lambda_1: Regularization strength for mask sparsity.
        diff_idx: Indices where the sample differs from the reference.
        final_model_path: Path to the combined encoder-surrogate model.

    Returns:
        The optimized feature mask as an ndarray, or None if the sample
        was not predicted as drift by the surrogate model.

    Raises:
        TypeError: If the loaded model is not an instance of a Keras Model.
    """
    optimizer = tf.train.AdamOptimizer
    initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
    learning_rate = 1e-2  # learning rate
    # a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.  # noqa: E501
    regularizer = 'elasticnet'
    exp_epoch = 250
    exp_display_interval = 10  # print middle result every k epochs
    exp_lambda_patience = 20
    early_stop_patience = 250

    mask_shape = (x.shape[0],)
    m1 = np.zeros(shape=(x.shape[0],), dtype=np.float32)
    for i in diff_idx:
        m1[i] = 1
    logging.debug(f'MASK_SHAPE: {mask_shape}')

    k.clear_session()
    model = load_model(final_model_path)
    if not isinstance(model, Model):
        raise TypeError(f'Model is not capable of predict: {type(model) = }')
    y = model.predict(x.reshape(1, -1))
    logging.debug(f'[explain_instance] y original: {y}')

    if np.argmax(y) != 1:  # don't explain wrongly classified target drift samples
        mask_best = None
        logging.error('[explain_instance] y is predicted as 0')
    else:
        mask_best = None
        exp_test = mask_exp.OptimizeExp(
            input_shape=x.shape,
            mask_shape=mask_shape,
            model=model,
            num_class=2,
            optimizer=optimizer,
            initializer=initializer,
            lr=learning_rate,
            regularizer=regularizer,
            model_file=final_model_path,
        )

        mask_best = exp_test.fit_local(
            x=x,
            y=y,
            epochs=exp_epoch,
            lambda_1=lambda_1,
            display_interval=exp_display_interval,
            lambda_patience=exp_lambda_patience,
            early_stop_patience=early_stop_patience,
        )
    return mask_best
