"""
Use different explanation methods to find important features for each drifting sample,
flip/modify the found important features, compare the distance (avg +/- std) between the perturbed sample's latent representation and the centroid.

command to run:
drebin:
    python -u evaluate_explanation_by_distance.py drebin_new_7 distance_mm1 0.001 1 0.1
    python -u evaluate_explanation_by_distance.py drebin_new_7 approximation_loose 0.001 0 0.1
    nohup python -u evaluate_explanation_by_distance.py drebin_new_7 gradient 0.001 0 0.1 > logs/nohup-drebin_new_7-gradient-exp.log &
    nohup python -u evaluate_explanation_by_distance.py drebin_new_7 random 0.001 0 0.1 > logs/nohup-drebin_new_7-random-100-exp.log &

IDS:
    nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration distance_mm1 0.001 1 0.1 > logs/nohup-IDS-distance-mm1-exp.log &
    python -u evaluate_explanation_by_distance.py IDS_new_Infilteration approximation_loose 0.001 0 0.1
    nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration gradient 0.001 0 0.1 > logs/nohup-IDS-gradient-exp.log &
    random 100 times: nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration random 0.001 0 0.1 > logs/nohup-IDS-random-exp.log &
"""  # noqa: E501

import logging
import os
import random
import sys
import traceback
from timeit import default_timer as timer
from typing import Any

import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.models import Model
from numpy.random import seed
from tensorflow import set_random_seed
from tqdm import tqdm

import cade.data as data
import cade.explain_by_distance as explain_dis
import cade.utils as utils
from cade.autoencoder import Autoencoder
from cade.logger import init_log

os.environ['PYTHONHASHSEED'] = '0'


random.seed(1)
seed(1)

set_random_seed(2)


families = [
    'FakeInstaller',
    'DroidKungFu',
    'Plankton',
    'GinMaster',
    'BaseBridge',
    'Iconosys',
    'Kmin',
    'FakeDoc',
]

RANDOM_TRY = 100


def load_necessary_model_and_data(
    x_train: np.ndarray,
    dataset: str,
    lambda_1: float,
    exp_method: str,
    new_label: int | None = None,
) -> tuple[np.ndarray | None, Model, list[str], list[int], str]:
    """
    Configure and load the Contrastive Autoencoder and associated metadata for explanation.

    This function dynamically determines the model architecture (dimensions), weight paths,
    and feature definitions based on the dataset being analyzed. It also retrieves
    pre-computed masks if the explanation method requires approximation or distance-based
    metrics.



    Args:
        x_train: Training feature matrix, used to determine input dimensionality.
        dataset: The name of the dataset (e.g., 'drebin', 'IDS', 'bluehex').
        lambda_1: The balance factor used during training, used here for result lookups.
        exp_method: The explanation method name (e.g., 'approximation', 'distance').
        new_label: The integer index of the new/drift family (required for Drebin).

    Returns:
        A tuple containing:
            * mask_list: Pre-computed explanation masks or None if not applicable.
            * encoder_model: The Keras model representing the trained encoder.
            * features: A list of feature names corresponding to the model input.
            * cae_dims: The list of integers defining the autoencoder layer sizes.
            * cae_weights_path: The file path string to the loaded .h5 weights.

    Raises:
        SystemExit: If the provided dataset name is not supported (exits with code -1).
    """  # noqa: E501
    if 'drebin' in dataset:
        cae_dims = [x_train.shape[1], 512, 128, 32, 7]
        cae_weights_path = f'models/{dataset}/cae_{x_train.shape[1]}-512-128-32-7_lr0.0001_b64_e250_m10.0_lambda0.1_weights.h5'  # noqa: E501
        feature_file = (
            f'data/{dataset}/drebin_new{new_label}_train_selected_features.txt'
        )
    elif 'IDS' in dataset:
        cae_dims = [x_train.shape[1], 64, 32, 16, 3]
        cae_weights_path = f'models/{dataset}/cae_83-64-32-16-3_lr0.0001_b512_e250_m10.0_lambda0.1_weights.h5'  # noqa: E501
        feature_file = 'data/IDS_83_features.txt'
    elif 'bluehex' in dataset:
        cae_dims = [x_train.shape[1], 1024, 256, 64, 5]
        cae_weights_path = f'models/{dataset}/cae_1857-1024-256-64-5_lr0.0001_b256_e250_m10.0_weights.h5'  # noqa: E501
        feature_file = '/home/liminyang/bluehex/cade_feature_names_setting5.txt'
    else:
        sys.exit(-1)

    # be careful with this it may clean up previous loaded models.
    k.clear_session()
    ae = Autoencoder(cae_dims)
    _ae_model, encoder_model = ae.build()
    encoder_model.load_weights(cae_weights_path, by_name=True)
    if 'approximation' in exp_method or 'distance' in exp_method:
        mask_list = np.load(f'reports/{dataset}/mask_{exp_method}_{lambda_1}.npz')[
            'masks'
        ]
    else:
        mask_list = None

    features = []
    if feature_file is not None:
        with open(feature_file) as fin:
            features = [line.strip() for line in fin]

    return mask_list, encoder_model, features, cae_dims, cae_weights_path


def load_training_info(
    training_info_for_detect_path: str, family: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load pre-computed latent space representations and distance metrics for a specific family.

    This function retrieves the statistical profile of a malware family from disk.
    It is used by the explanation module to compare drift samples against the
    known distribution of a training family.



    Args:
        training_info_for_detect_path: The directory containing the .npz files
            generated during the detection/training phase.
        family: The integer label of the malware family to load.

    Returns:
        A tuple containing:
            * z_train: Latent representations of all training samples in this family.
            * z_closest_family: Latent representations of the most representative
              cluster (often used for visualization).
            * centroid: The mathematical center (mean/median) of the family in latent space.
            * dis_to_centroid: An array of Euclidean distances from each training
              sample to the `centroid`.
            * mad: The Median Absolute Deviation of the distances, used for
              robust outlier thresholding.

    Raises:
        FileNotFoundError: If the .npz file for the specified family index
            does not exist in the path.
    """  # noqa: E501
    info = np.load(training_info_for_detect_path)
    z_train = info['z_train']
    z_family = info['z_family']
    centroids = info['centroids']
    dis_family = info['dis_family']
    mad_family = info['mad_family']

    z_closest_family = z_family[family]
    centroid = centroids[family]
    dis_to_centroid = dis_family[family]
    mad = mad_family[family]

    return z_train, z_closest_family, centroid, dis_to_centroid, mad


def get_important_fea_and_distance(  # noqa: C901
    dataset: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    drift_samples_idx_list: list[int],
    drift_samples_real_labels: list[int],
    drift_samples_closest: list[int],
    family_info_dict: dict[int, list[Any]],
    x_train_family_dict: dict[int, np.ndarray],
    closest_sample_family_dict: dict[int, np.ndarray],
    mask_list: np.ndarray | None,
    encoder_model: Model,
    exp_method: str,
    features: list[str],
    use_gumbel: int,
    save_distance_mm1_important_fea_len_file: str,
    save_result_path: str,
) -> None:
    """
    Evaluate explanation effectiveness by perturbing important features and measuring latent distance shift.

    This function identifies the most influential features (based on the provided mask),
    flips or replaces those features in the drift samples, and re-projects them into
    the encoder's latent space. A 'success' is recorded if the perturbation moves the
    sample back within the in-distribution threshold (lowerbound) of its closest
    training family.

    Args:
        dataset: Name of the dataset being evaluated.
        x_test: Testing feature matrix.
        y_test: Ground truth labels for the test set.
        drift_samples_idx_list: List of indices in x_test representing detected drift.
        drift_samples_real_labels: The true malware family labels for the drift samples.
        drift_samples_closest: The training family label identified as 'nearest' to the drift.
        family_info_dict: Dictionary containing [centroid, mad, lowerbound] for each family.
        x_train_family_dict: Dictionary mapping family labels to their training feature subsets.
        closest_sample_family_dict: Dictionary mapping family labels to their medoid sample.
        mask_list: Array of explanation masks (feature importance weights) for each drift sample.
        encoder_model: The trained Keras encoder used to project samples into latent space.
        exp_method: The specific explanation method used (e.g., 'distance_mm1', 'gradient').
        features: List of string names for each feature index.
        use_gumbel: Flag (1/0) indicating if Gumbel-Softmax was used for discrete masking.
        save_distance_mm1_important_fea_len_file: Path to save/load the count of important features.
        save_result_path: Path to save the final distance and success ratio reports.

    Returns:
        None. Results are logged to the console and written to the specified file paths.
    """  # noqa: E501

    if exp_method != 'distance_mm1':
        if os.path.exists(save_distance_mm1_important_fea_len_file):
            important_feas_len_list = read_feas_len_from_file(
                save_distance_mm1_important_fea_len_file
            )
            print(
                f'load important_feas_len_list len: {len(important_feas_len_list)}')
        else:
            logging.error(
                'you need to perform distance_mm1 method to get the length of important features first'  # noqa: E501
            )
            sys.exit(1)
    else:
        important_feas_len_list = []

    # number of sucessful perturbations from drift to in-distribution.
    success = 0

    lowerbound_list = []
    logging.debug(
        f'len(drift_samples_idx_list): {len(drift_samples_idx_list)}')

    # Declaration moved here to avoid unbound errors
    x_arr = []
    centroid_arr = []
    x_perturb_arr = []
    if mask_list is None:
        mask_list = np.empty((1, 1))

    for idx, sample_idx, real, closest_family in tqdm(
        zip(
            range(len(drift_samples_idx_list)),
            drift_samples_idx_list,
            drift_samples_real_labels,
            drift_samples_closest,
        ),
        total=len(drift_samples_idx_list),
    ):
        x = x_test[sample_idx]
        mask = mask_list[idx]

        lowerbound = family_info_dict[closest_family][2]

        lowerbound_list.append(lowerbound)

        if idx == 0:
            x_arr = np.copy(x)
            centroid_arr = np.copy(family_info_dict[closest_family][0])
        else:
            x_arr = np.vstack((x_arr, x))
            centroid_arr = np.vstack((
                centroid_arr,
                family_info_dict[closest_family][0],
            ))

        if 'approximation' in exp_method:
            tmp = np.sum(mask)
            if not np.isnan(tmp):
                prod = x * mask
                np.sort(prod, kind='mergesort', axis=None)[::-1]
                valid_n = len(np.where(prod > 0)[0])
                valid_n = min(valid_n, important_feas_len_list[idx])

                ranked_prod_idx = np.argsort(
                    prod, kind='mergesort', axis=None)[::-1]
                important_feas = ranked_prod_idx[: valid_n + 1]
            else:
                logging.debug(f'drift-{idx}: mask is None')
                important_feas = None
        elif exp_method == 'distance_mm1':
            if mask is not None:
                if use_gumbel:
                    # only when m = m1 = 1, it's important, we could also rank the rest of the features,  # noqa: E501
                    # but we keep m = m1 = 1 for simplicity (less features).
                    important_feas = np.where(mask == 1)[0]
                else:
                    np.sort(mask, kind='mergesort', axis=None)[
                        ::-1
                    ]  # bigger means more important feature
                    valid_n = len(np.where(mask == 1)[0])
                    ranked_mask_idx = np.argsort(mask, kind='mergesort', axis=None)[
                        ::-1
                    ]
                    important_feas = ranked_mask_idx[: valid_n + 1]
            else:
                logging.debug(f'drift-{idx}: mask is None')
                important_feas = None
        else:
            important_feas = None

        x_test_family = x_test[np.where(y_test == closest_family)[0]]
        x_closest_family_all = np.vstack((
            x_train_family_dict[closest_family],
            x_test_family,
        ))
        x_real_family = x_test[np.where(y_test == real)[0]]

        if important_feas is None:
            raise ValueError(f'important_feas is None; {important_feas = }')
        if exp_method != 'distance_mm1':
            raise ValueError(f'xp_method is not distance_mm1; {exp_method = }')
        important_feas_len_list.append(len(important_feas))

        # case study
        if 'drebin' in dataset:
            # idx-2: closest to Gin Master, idx-1: closest to DroidKungfu (most FakeDoc closer to DroidKungfu).  # noqa: E501
            cases = [1, 2]
        elif 'IDS' in dataset:
            # the first 5 cases are closer to SSH, SSH, Hulk, Hulk, Hulk
            cases = range(5)
        elif 'bluehex' in dataset:
            cases = range(5)
        else:
            raise ValueError(
                f'drebin, IDS or bluehex not in {dataset = }\n'
                f'Please use valid dataset name!'
            )
        if idx in cases:
            utils.create_folder('reports/explanation_case_study/')
            with open(
                f'reports/explanation_case_study/{dataset}-{exp_method}-drifting-{idx}-temp-0.1.txt',
                'w',
            ) as f:
                f.write(
                    f'feature index,sample {idx} important feature,original value,avg value in testing set(real family),avg value in training set(closest family),avg value in both train and test set(closest family),closest sample value\n'  # noqa: E501
                )
                f.writelines(
                    f'{fea},{features[fea]},{x[fea]:e},{np.mean(x_real_family[:, fea]):e},'  # noqa: E501
                    f'{np.mean(X_train_family_dict[closest_family][:, fea]):e},'
                    f'{np.mean(x_closest_family_all[:, fea]):e},{closest_sample_family_dict[closest_family][fea]:e}\n'  # noqa: E501
                    for fea in important_feas
                )

        """ the chosen method: perturb the important features and craft a new sample """
        x_new = np.copy(x)
        for i in important_feas:
            """ NOTE: flip important features:
                for baseline 2: important features all have a feature value = 1, so there is only 1 -> 0.
                for distance based methods: both 1-> 0 and 0->1 are possible"""  # noqa: E501
            if 'drebin' in dataset:
                x_new[i] = 1 if x[i] == 0 else 0
            elif 'IDS' in dataset:
                """ use the sample (closest to centroid) feature value"""
                perturbed_value = closest_sample_family_dict[closest_family][i]

                x_new[i] = perturbed_value
            elif 'bluehex' in dataset:
                perturbed_value = closest_sample_family_dict[closest_family][i]
                x_new[i] = perturbed_value
        if idx == 0:
            x_perturb_arr = np.copy(x_new)
        else:
            x_perturb_arr = np.vstack((x_perturb_arr, x_new))

    latent_x = encoder_model.predict(x_arr)
    latent_x_perturb = encoder_model.predict(x_perturb_arr)
    original_dis = np.sqrt(np.sum(np.square(latent_x - centroid_arr), axis=1))
    perturbed_dis = np.sqrt(
        np.sum(np.square(latent_x_perturb - centroid_arr), axis=1))

    success_idx = np.where(perturbed_dis <= lowerbound_list)[0]
    success = len(success_idx)

    if exp_method == 'distance_mm1':
        write_result_to_file(
            original_dis, 'original distance', save_result_path, 'w')
        write_result_to_file(
            important_feas_len_list,
            f'{exp_method} important feas len',
            save_result_path,
            'a',
        )

    write_result_to_file(
        perturbed_dis, f'{exp_method} perturbed distance', save_result_path, 'a'
    )
    with open(save_result_path, 'a') as f:
        ratio = success / len(perturbed_dis)
        f.write(f'{exp_method} success idx: {success_idx}\n\n')
        print(
            f'{exp_method} success from drifting to in-dist: {success}, ratio: {ratio * 100:.2f}%'  # noqa: E501
        )
        f.write(
            f'{exp_method} success from drifting to in-dist: {success}, ratio: {ratio * 100:.2f}%\n'  # noqa: E501
        )

    with open(save_distance_mm1_important_fea_len_file, 'w') as f:
        logging.debug(
            f'important_feas_len_list len: {len(important_feas_len_list)}')
        f.writelines(f'{fea_len}\n' for fea_len in important_feas_len_list)


def preprocess_training_info(
    x_train: np.ndarray,
    y_train: np.ndarray,
    drift_samples_closest: list,
    training_info_for_detect_path: str,
) -> tuple[dict[int, list[Any]], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Pre-calculate family statistics and reference samples for drift explanation.

    This function aggregates training metadata for each family involved in the
    drift detection. It calculates the outlier 'lowerbound' using a Median Absolute
    Deviation (MAD) threshold and identifies the 'Medoid' (the sample closest
    to the cluster centroid) to serve as a reference point for local explanations.



    Args:
        x_train: The training feature matrix (binary vectors).
        y_train: The labels for the training set.
        drift_samples_closest: An array of labels representing the training families
            closest to each detected drift sample.
        training_info_for_detect_path: Path to the directory containing
            pre-saved `.npz` files with latent space (Z) and distance metrics.

    Returns:
        A tuple containing three dictionaries:
            * family_info_dict: Maps family label to [centroid, mad, lowerbound].
            * x_train_family_dict: Maps family label to its corresponding
              subset of X_train.
            * closest_sample_family_dict: Maps family label to the specific
              X_train sample closest to that family's centroid.
    """
    family_info_dict = {}
    x_train_family_dict = {}
    closest_sample_family_dict = {}

    # the load_training_info() is actually very time consuming, so just load it once for each closest family here.  # noqa: E501
    for family in np.unique(drift_samples_closest):
        _z_train, _z_closest_family, centroid, dis_to_centroid, mad = (
            load_training_info(training_info_for_detect_path, family)
        )
        lowerbound = mad * 3.5 + np.median(dis_to_centroid)
        dis_to_centroid_inds = np.array(dis_to_centroid).argsort()
        x_train_family = x_train[np.where(y_train == family)[0]]
        closest_to_centroid_sample = x_train_family[dis_to_centroid_inds][0]

        family_info_dict[family] = [centroid, mad, lowerbound]
        x_train_family_dict[family] = x_train_family
        closest_sample_family_dict[family] = closest_to_centroid_sample

    return family_info_dict, x_train_family_dict, closest_sample_family_dict


def write_result_to_file(
    single_list: list[int], name: str, filepath: str, mode: str
) -> None:
    """
    Calculates the average and standard deviation of a list and writes to a file.

    Computes statistical metrics using NumPy and appends/writes a formatted
    string containing the 'name' label, average, and standard deviation to
    the specified file.

    Args:
        single_list: A list of numeric values to analyze.
        name: A descriptive label for the data (e.g., 'Success Rate').
        filepath: The system path where the result should be saved.
        mode: File opening mode (e.g., 'a' for append, 'w' for write).

    Returns:
        None. Output is written to the file and printed to the console.

    Raises:
        IOError: If the file cannot be opened or written to.
    """
    try:
        with open(filepath, mode) as f:
            if not single_list:
                logging.warning(
                    f"Result '{name}' not written: single_list is empty.")
                return

            avg = np.average(single_list)
            std = np.std(single_list)

            result = f'{name}  avg: {avg:.3f}, std: {std:.3f}'
            print(result)

            f.write(result + '\n')
            f.write('=' * 80 + '\n')

    except (ZeroDivisionError, TypeError, ValueError) as e:
        logging.error(
            f'Calculation error for {name}: {e}\n{traceback.format_exc()}')
    except OSError as e:
        logging.error(f'File access error for {filepath}: {e}')


def get_backpropagation_important_features(
    dataset: str,
    _x_train: np.ndarray,
    x_test: np.ndarray,
    _y_train: np.ndarray,
    _y_test: np.ndarray,
    drift_samples_idx_list: list[int],
    drift_samples_closest: list[int],
    family_info_dict: dict[int, list[Any]],
    encoder_model: Model,
    cae_dims: list[int],
    closest_sample_family_dict: dict[int, np.ndarray],
    features: list[str],
    cae_weights_path: str,
    important_feas_len_list: list[int],
    save_result_path: str,
) -> None:
    """
    Identifies important features via backpropagation gradients and evaluates
    explanation effectiveness through sample perturbation.

    The function calculates the gradient of the displacement in latent space
    with respect to the input features:

    $$G = \frac{\\partial (f(x) - f(c))}{\\partial x}$$

    where $f(x)$ is the latent projection of the drift sample and $f(c)$ is the
    latent centroid of the closest training family. Features are ranked by the
    magnitude of these gradients to identify 'important' features for perturbation.

    Args:
        dataset: Name of the dataset (e.g., 'drebin', 'IDS').
        _x_train: Training feature matrix (unused, kept for signature consistency).
        x_test: Testing feature matrix.
        _y_train: Training labels (unused).
        _y_test: Testing labels (unused).
        drift_samples_idx_list: Indices of samples identified as out-of-distribution.
        drift_samples_closest: The training family labels closest to each drift sample.
        family_info_dict: Mapping of family labels to [centroid, mad, lowerbound].
        encoder_model: The Keras Model used for latent projection.
        cae_dims: Architecture dimensions of the Contractive Autoencoder.
        closest_sample_family_dict: Mapping of family labels to their medoid samples.
        features: Human-readable names for each feature index.
        cae_weights_path: File path to the trained model weights.
        important_feas_len_list: Pre-calculated counts of features to perturb per sample.
        save_result_path: File path where statistical reports will be appended.

    Returns:
        None. Results are saved to disk and logged.
    """  # noqa: E501
    lowerbound_list = []
    s = timer()

    """ construct the tf nodes to calculate the gradients.
        the tensors should be put outside the for loop so that we only add these nodes
        to the graph once instead of multiple times, the latter would make the graph bigger and bigger and get slower"""  # noqa: E501
    input_tensor = encoder_model.get_input_at(0)
    centroid_tensor = tf.placeholder(tf.float32, shape=(None, cae_dims[-1]))
    latent_input = encoder_model(input_tensor)
    g = tf.gradients((latent_input - centroid_tensor), input_tensor)

    gradient_valid_important_feas_len_list = []
    x_perturb_arr = []
    centroid_arr = []
    for idx, sample_idx, family in tqdm(
        zip(
            range(len(drift_samples_idx_list)),
            drift_samples_idx_list,
            drift_samples_closest,
        ),
        total=len(drift_samples_idx_list),
    ):
        x = x_test[sample_idx]
        centroid = family_info_dict[family][0]
        lowerbound_list.append(family_info_dict[family][2])

        start = timer()
        # original_importance could be positive or negative
        important_feas_idx, _abs_importance, original_importance = (
            backpropagation_gradients(
                idx,
                x,
                centroid,
                encoder_model,
                cae_weights_path,
                features,
                input_tensor,
                centroid_tensor,
                g,
            )
        )
        end = timer()
        logging.debug(
            f'{idx} - backpropagation_gradients time: {(end - start):.3f}s')

        distance_method_important_feas_len = important_feas_len_list[idx]

        x_new = np.copy(x)
        valid_n = 0
        for i in important_feas_idx:
            if valid_n < distance_method_important_feas_len:
                if 'drebin' in dataset:
                    if original_importance[i] > 0:
                        x_new[i] = 1 if x[i] == 0 else 0
                        valid_n += 1
                elif 'IDS' in dataset:
                    """ use the sample(closest to centroid) feature value"""
                    if original_importance[i] > 0:
                        closest_sample_family_dict[family][i]
                        valid_n += 1

        gradient_valid_important_feas_len_list.append(valid_n)

        if idx == 0:
            x_perturb_arr = np.copy(x_new)
            centroid_arr = np.copy(family_info_dict[family][0])
        else:
            x_perturb_arr = np.vstack((x_perturb_arr, x_new))
            centroid_arr = np.vstack(
                (centroid_arr, family_info_dict[family][0]))

    encoder_model.load_weights(cae_weights_path, by_name=True)
    latent_x_perturb = encoder_model.predict(x_perturb_arr)
    perturbed_dis = np.sqrt(
        np.sum(np.square(latent_x_perturb - centroid_arr), axis=1))
    success = len(np.where(perturbed_dis <= lowerbound_list)[0])
    write_result_to_file(
        perturbed_dis, 'gradient perturbed distance', save_result_path, 'a'
    )
    write_result_to_file(
        gradient_valid_important_feas_len_list,
        'gradient valid important features len',
        save_result_path,
        'a',
    )
    with open(save_result_path, 'a') as f:
        ratio = success / len(perturbed_dis)
        f.write(
            f'baseline 2: gradient success from drifting to in-dist: {success}, ratio: {ratio * 100:.2f}%\n'  # noqa: E501
        )

    e = timer()
    logging.debug(
        f'get_backpropagation_important_features time: {(e - s):.3f}s')


def backpropagation_gradients(
    idx: int,
    x: np.ndarray,
    centroid: np.ndarray,
    model: Model,
    model_file: str,
    features: list[str],
    input_tensor: tf.Tensor,
    centroid_tensor: tf.Tensor,
    g: list[tf.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates feature importance using backpropagated gradients.

    This function executes a TensorFlow session to compute the gradient of the
    latent difference (f(x) - f(c)) with respect to the input features. It
    identifies which input features most significantly contribute to the
    sample's displacement from the target family centroid.

    Args:
        idx: The current iteration index for the drift sample.
        x: The input feature vector (1D array of shape [n_features]).
        centroid: The target family centroid in the latent space.
        model: The Keras/TF model instance used for encoding.
        model_file: Path to the saved model weights or configuration.
        features: A list of human-readable feature names corresponding to indices.
        input_tensor: The pre-defined input placeholder/tensor for the model.
        centroid_tensor: The pre-defined placeholder for the target latent centroid.
        g: The pre-computed gradient operation(s) from tf.gradients.

    Returns:
        A tuple containing:
            - important_feas_idx (np.ndarray): Indices of features ranked by importance.
            - abs_importance (np.ndarray): The absolute values of the gradients.
            - original_importance (np.ndarray): The raw gradient values (signed).
    """

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_weights(model_file, by_name=True)

        feed_dict = {
            input_tensor: x[None],  # type: ignore
            centroid_tensor: centroid[None,],  # type: ignore
        }
        g_matrix = sess.run(g, feed_dict=feed_dict)[0]

        # rank by importance descending order, the output importance could be negative,
        # so we rank by their absolute value.
        g_matrix = g_matrix.reshape(
            -1,
        )
        ordered_g_abs = np.sort(np.abs(g_matrix))[::-1]

        ordered_g_abs_index = np.argsort(np.abs(g_matrix))[::-1]
        if idx == 2:
            logging.critical(f'ordered g: {list(ordered_g_abs)}')
            important_feas_list = []
            logging.debug(
                f'backpropagation important features for drifting-{idx}: \n####################################'  # noqa: E501
            )
            for i in range(50):
                logging.debug(f'{features[ordered_g_abs_index[i]]}')
                important_feas_list.append(features[ordered_g_abs_index[i]])
            logging.debug('####################################')

        return ordered_g_abs_index, ordered_g_abs, g_matrix


def read_feas_len_from_file(save_distance_mm1_important_fea_len_file: str) -> list[int]:
    """
    Read the counts of important features from a baseline results file.

    This function retrieves the number of features identified as "important" by
    the baseline method (usually 'distance_mm1'). These counts are used to
    standardize the number of features perturbed across different explanation
    methods, ensuring a fair comparison of their effectiveness.

    Args:
        save_distance_mm1_important_fea_len_file: Path to the text file
            containing integer feature counts (one per line).

    Returns:
        A list of integers representing the number of important features
        to be considered for each drift sample.
    """
    with open(save_distance_mm1_important_fea_len_file) as f:
        important_feas_len_list = [int(line.strip()) for line in f]

    return important_feas_len_list


def eval_random_select_important_feas(
    dataset: str,
    save_distance_mm1_important_fea_len_file: str,
    drift_samples_idx_list: list[int],
    drift_samples_closest: list[int],
    x_test: np.ndarray,
    y_test: np.ndarray,
    family_info_dict: dict[int, list[Any]],
    closest_sample_family_dict: dict[int, np.ndarray],
    encoder_model: Model,
    save_result_path: str,
) -> None:
    """baseline 3: randomly choose the same number of important features and craft a new sample"""  # noqa: E501
    x_random_arr = []
    centroid_arr = []
    if not os.path.exists(save_distance_mm1_important_fea_len_file):
        logging.error(
            'you need to perform distance_mm1 method to get the length of important features first'  # noqa: E501
        )
        sys.exit(1)
    s = timer()
    important_feas_len_list = read_feas_len_from_file(
        save_distance_mm1_important_fea_len_file
    )
    random_dis_array_list = []
    total_success_random = 0
    for _random_cnt in tqdm(range(RANDOM_TRY)):
        lowerbound_list = []
        for idx, sample_idx, family in zip(
            range(len(drift_samples_idx_list)),
            drift_samples_idx_list,
            drift_samples_closest,
        ):
            x = x_test[sample_idx]

            lowerbound_list.append(family_info_dict[family][2])

            fea_len = important_feas_len_list[idx]

            x_random = np.copy(x)
            random_important_feas = np.random.choice(
                x.shape[0], size=fea_len, replace=False
            )
            for i in random_important_feas:
                if 'drebin' in dataset:
                    x_random[i] = 1 if x[i] == 0 else 0
                elif 'IDS' in dataset:
                    perturbed_value = closest_sample_family_dict[family][i]
                    x_random[i] = perturbed_value
            if idx == 0:
                x_random_arr = np.copy(x_random)
                centroid_arr = np.copy(family_info_dict[family][0])
            else:
                x_random_arr = np.vstack((x_random_arr, x_random))
                centroid_arr = np.vstack((
                    centroid_arr,
                    family_info_dict[family][0],
                ))

        latent_x_random = encoder_model.predict(x_random_arr)
        random_dis = np.sqrt(
            np.sum(np.square(latent_x_random - centroid_arr), axis=1))

        success_random = len(np.where(random_dis <= lowerbound_list)[0])
        total_success_random += success_random
        random_dis_array_list.append(random_dis)

    write_result_to_file(
        random_dis_array_list,
        f'random perturbed distance (n = {RANDOM_TRY})',
        save_result_path,
        'w',
    )

    with open(save_result_path, 'a') as f:
        total_try = len(random_dis_array_list) * len(random_dis_array_list[0])
        random_ratio = total_success_random / total_try
        print(
            f'random success perturbed from drifting to in-dist: {random_ratio * 100:.2f}'  # noqa: E501
        )
        f.write(
            f'random success from drifting to in-dist: {total_success_random}, total_try: {total_try}, \
                    ratio: {random_ratio * 100:.2f}%\n'  # noqa: E501
        )

    e = timer()
    logging.debug(f'eval_random_select_important_feas: {(e - s):.3f} seconds')


if __name__ == '__main__':
    if len(sys.argv) != 6:
        logging.error(
            'usage example: python -u evaluate_explanation_by_distance.py drebin_new_7 distance_mm1 0.001 1 0.1'  # noqa: E501
        )
        sys.exit(-1)

    dataset = sys.argv[1]  # drebin_new_7 or IDS_new_Infilteration
    # distance_mm1, approximation_loose, random, gradient
    exp_method = sys.argv[2]
    # lambda_1 for baseline methods needs to be the same as distance_mm1 to keep the same important features length  # noqa: E501
    lambda_1 = float(sys.argv[3])  # 0.001
    use_gumbel = int(sys.argv[4])  # 1 or 0, use gumbel when distance_mm1
    temp = float(
        sys.argv[5]
    )  # temp for baseline methods needs to be the same as distance_mm1

    REPORT_FOLDER = f'reports/exp_evaluation/{dataset}'
    utils.create_folder(REPORT_FOLDER)
    LOG_FOLDER = f'logs/exp_evaluation/{dataset}'
    utils.create_folder(LOG_FOLDER)

    log_path = (
        f'./{LOG_FOLDER}/{dataset}-{exp_method}-lambda-{lambda_1}-gumble-{use_gumbel}'
    )
    if os.path.exists(log_path + '.log'):
        os.remove(log_path + '.log')
        logging.info('log file removed')

    init_log(log_path, level=logging.INFO)

    gumble_flag = 'with' if use_gumbel == 1 else 'without'

    save_result_path = f'{REPORT_FOLDER}/{dataset}-{exp_method}-lambda-{lambda_1}-temp-{temp}-{gumble_flag}-gumble.txt'  # noqa: E501
    # other explanation method would load this file to determine how many important features to pick  # noqa: E501
    save_distance_mm1_important_fea_len_file = os.path.join(
        REPORT_FOLDER,
        f'{dataset}-distance-mm1-important-feas-len-temp-{temp}-lambda-{lambda_1}.txt',
    )

    if 'drebin' in dataset:
        new_label = 7
    elif 'IDS' in dataset:
        new_label = 3
    elif 'bluehex_top_5' in dataset:
        new_label = 5
    else:
        logging.error(f'dataset {dataset} not supported')
        sys.exit(-1)

    one_by_one_check_result_path = f'reports/{dataset}/dist_mlp_one_by_one_check_pr_value_m10.0_mad3.5_lambda0.1.csv'  # noqa: E501

    x_train, y_train, x_test, y_test = data.load_features(dataset, new_label)

    drift_samples_idx_list, drift_samples_real_labels, drift_samples_closest = (
        explain_dis.get_drift_samples_to_explain(one_by_one_check_result_path)
    )

    training_info_for_detect_path = os.path.join(
        'reports',
        dataset,
        'intermediate',
        'mlp_training_info_for_detect_m10.0_lambda0.1.npz',
    )
    family_info_dict, X_train_family_dict, closest_sample_family_dict = (
        preprocess_training_info(
            x_train, y_train, drift_samples_closest, training_info_for_detect_path
        )
    )

    s1 = timer()
    mask_list, encoder_model, features, cae_dims, cae_weights_path = (
        load_necessary_model_and_data(x_train, dataset, lambda_1, exp_method)
    )
    e1 = timer()
    logging.debug(f'load_necessary_model_and_data time: {(e1 - s1):.2f}')

    """main logic"""
    if 'approximation' in exp_method or 'distance' in exp_method:
        s2 = timer()
        get_important_fea_and_distance(
            dataset,
            x_test,
            y_test,
            drift_samples_idx_list,
            drift_samples_real_labels,
            drift_samples_closest,
            family_info_dict,
            X_train_family_dict,
            closest_sample_family_dict,
            mask_list,
            encoder_model,
            exp_method,
            features,
            use_gumbel,
            save_distance_mm1_important_fea_len_file,
            save_result_path,
        )
        e2 = timer()
        logging.debug(f'get_important_fea_and_distance time: {(e2 - s2):.2f}')
    elif exp_method == 'gradient':
        """ try Dr. Xing's mathematical baseline: backpropagate the gradients from low-d to high-d and get feature importance. """  # noqa: E501
        if os.path.exists(save_distance_mm1_important_fea_len_file):
            important_feas_len_list = read_feas_len_from_file(
                save_distance_mm1_important_fea_len_file
            )
            get_backpropagation_important_features(
                dataset,
                x_train,
                x_test,
                y_train,
                y_test,
                drift_samples_idx_list,
                drift_samples_closest,
                family_info_dict,
                encoder_model,
                cae_dims,
                closest_sample_family_dict,
                features,
                cae_weights_path,
                important_feas_len_list,
                save_result_path,
            )
        else:
            logging.error(
                'you need to perform distance_mm1 method to get the length of important features first'  # noqa: E501
            )
            sys.exit(1)
    elif exp_method == 'random':
        eval_random_select_important_feas(
            dataset,
            save_distance_mm1_important_fea_len_file,
            drift_samples_idx_list,
            drift_samples_closest,
            x_test,
            y_test,
            family_info_dict,
            closest_sample_family_dict,
            encoder_model,
            save_result_path,
        )

    else:
        logging.error(f'explanation method {exp_method} not supported')
        sys.exit(-1)
