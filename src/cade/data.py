"""
data.py
~~~~~~~

Functions for caching and loading data.

"""

import logging
import os
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from timeit import default_timer as timer

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from utils import SimConfig

import cade.utils as utils
from cade.config import config

random.seed(1)


def load_features(
    dataset: str, newfamily: int, folder: str = 'data/'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load feature vectors and labels from an NPZ file and perform dataset-specific relabeling.

    This function normalizes class labels using a LabelEncoder for the training set and
    maps the testing set labels accordingly. It handles 'unseen' or 'drift' families
    by assigning them a persistent label index based on the specific dataset being used.

    Args:
        dataset: The name of the dataset to load (e.g., 'drebin', 'IDS', 'bluehex').
        newfamily: The label index to use for the unseen family if 'bluehex' is selected.
        folder: The directory path where the .npz data files are stored.

    Returns:
        A tuple containing:
            * x_train: Training feature vectors.
            * y_train_prime: Training labels transformed to continuous integers.
            * x_test: Testing feature vectors.
            * y_test_prime: Testing labels, where known families match training indices
              and unknown families are assigned to `persistent_new_family`.

    Raises:
        SystemExit: If the provided `dataset` name is not recognized (exits with code -4).
    """  # noqa: E501
    logging.info(f'Loading {dataset} feature vectors and labels...')
    filepath = os.path.join(folder, dataset + '.npz')

    with np.load(filepath) as data:
        x_train = np.array(data['X_train'])
        y_train = np.array(data['y_train'])
        x_test = np.array(data['X_test'])
        y_test = np.array(data['y_test'])

    logging.debug(
        f'before label adjusting: y_train: {Counter(y_train)}\n  y_test: {Counter(y_test)}'  # noqa: E501
    )

    # Dataset-specific persistent label logic
    if 'drebin' in dataset:
        persistent_new_family = 7
    elif 'IDS' in dataset:
        persistent_new_family = 3
    elif 'bluehex' in dataset:
        persistent_new_family = newfamily
    else:
        logging.error(f'dataset {dataset} not supported')
        sys.exit(-4)

    le = LabelEncoder()
    y_train_prime = np.asarray(le.fit_transform(y_train)).astype(np.int32)

    unique_classes = np.asarray(le.classes_)
    encoded_values = np.asarray(le.transform(unique_classes))

    mapping = {label: int(val) for label, val in zip(unique_classes, encoded_values)}

    logging.debug(f'LabelEncoder mapping: {mapping}')

    # Relabel test set: known labels get mapped, unknown get persistent_new_family
    y_test_prime_list = [mapping.get(label, persistent_new_family) for label in y_test]
    y_test_prime = np.array(y_test_prime_list, dtype=np.int32)

    y_train_prime = np.array(y_train_prime, dtype=np.int32)

    logging.debug(f'after relabeling training: {Counter(y_train_prime)}')
    logging.debug(f'after relabeling testing: {Counter(y_test_prime)}')

    return x_train, y_train_prime, x_test, y_test_prime


def prepare_dataset(config: SimConfig) -> None:
    """
    Validate and initiate the data preparation pipeline for the Drebin dataset.

    This function serves as a wrapper to ensure that the experimental configuration
    is strictly set to 'drebin' before calling the underlying data processing
    utilities. It verifies the dataset name and passes the specified
    'newfamily_label' to the preparation module.

    Args:
        config: A SimConfig object containing experiment hyperparameters,
            specifically 'data' (dataset name) and 'newfamily_label'.

    Raises:
        ValueError: If `config.data` does not contain 'drebin', ensuring that
            incompatible preprocessing logic is not applied to other datasets.

    Returns:
        None. Resulting data is typically saved to disk by the underlying
        `prepare_drebin_data` call.
    """
    if 'drebin' in config.data:
        prepare_drebin_data(config.data, newfamily=config.newfamily_label)
    else:
        raise ValueError('Should only be run in drebin dataset config')


def prepare_drebin_data(
    dataset_name: str,
    folder: str = 'data/',
    test_ratio: float = 0.2,
    newfamily: int = 7,
) -> None:
    saved_data_file = os.path.join(folder, f'{dataset_name}.npz')
    if os.path.exists(saved_data_file):
        logging.info(f'{saved_data_file} exists, no need to re-generate')
    else:
        """Train fit test, use only 7 of top 8 families, sort by timestamp, samples do not have timestamp would be removed"""  # noqa: E501
        logging.info('Preparing Drebin malware data...')
        raw_feature_vectors_folder = config['drebin']

        intermediate_folder = os.path.join(
            'data', dataset_name
        )  # for saving intermediate data files.
        utils.create_folder(intermediate_folder)
        sha_sorted_by_time, label_sorted_by_time, newfamily_sha_list = (
            sort_drebin_7family_by_time(intermediate_folder, newfamily)
        )
        logging.debug(f'sha_sorted_by_time len: {len(sha_sorted_by_time)}')

        """split 8 families to training and testing set by timestamp, insert the new family to the testing set"""  # noqa: E501
        train_shas, test_shas, train_labels, test_labels = split_drebin_train_and_test(
            sha_sorted_by_time,
            label_sorted_by_time,
            newfamily_sha_list,
            test_ratio,
            newfamily,
        )

        """get all the feature names in the training set"""
        train_feature_names = get_training_full_feature_names(
            intermediate_folder, newfamily, raw_feature_vectors_folder, train_shas
        )

        """save all the training set feature vectors"""
        saved_train_vectors = save_training_full_feature_vectors(
            intermediate_folder,
            raw_feature_vectors_folder,
            train_shas,
            train_feature_names,
            train_labels,
            newfamily,
        )

        """feature selection on the training set"""
        selected_features, saved_selected_vectors_file = get_selected_features(
            intermediate_folder, saved_train_vectors, newfamily, train_feature_names
        )

        """ generate the final data by saving feature vectors of both training and testing set"""  # noqa: E501
        samples = len(test_shas)
        feature_map = {fea: i for i, fea in enumerate(selected_features)}
        num_feas = len(selected_features)

        x_test = np.zeros((samples, num_feas))
        for sample_idx, sha in enumerate(test_shas):
            file_path = os.path.join(raw_feature_vectors_folder, sha)
            try:
                with open(file_path) as f:
                    for line in f:
                        feature = line.strip()
                        if feature:
                            try:
                                fea_idx = feature_map[feature]
                                x_test[sample_idx][fea_idx] = 1
                            except KeyError:
                                pass
            except FileNotFoundError:
                logging.warning(f'Test sample file not found: {sha}')

        y_test = np.array([int(label) for label in test_labels])
        train_data = np.load(saved_selected_vectors_file)
        x_train, y_train = train_data['X_train'], train_data['y_train']
        logging.info(f'X_train: {x_train.shape}, y_train: {y_train.shape}')
        logging.info(f'X_test: {x_test.shape}, y_test: {y_test.shape}')
        np.savez_compressed(
            saved_data_file,
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
        )
        for idx, x in enumerate(x_test):
            if np.all(x == 0):
                logging.warning(f'X_test {idx} all 0')

        logging.info('Preparing Drebin malware data finished')


def sort_drebin_7family_by_time(
    intermediate_folder: str, newfamily: int
) -> tuple[list[str], list[int], list[str]]:
    """
    Filter and sort Drebin malware families by their latest modification timestamp.

    This function processes the Drebin metadata to select the top 8 malware families
    (excluding 'Opfake' to avoid confusion with 'FakeInstaller'). It separates one
    family as the 'new' (drift) family based on the provided index and sorts the
    remaining 7 families chronologically.

    Args:
        intermediate_folder: Path to the directory where the processed CSV
            mapping will be saved.
        newfamily: The index (0-7) of the family in `top8` to be treated as
            the unseen/drift family.

    Returns:
        A tuple containing:
            * sha_sorted_by_time: List of SHAs for the 7 known families,
              sorted oldest to newest.
            * label_sorted_by_time: List of integer labels corresponding to
              `sha_sorted_by_time`.
            * newfamily_sha_list: List of all SHAs belonging to the
              designated 'new' family.
    """
    top8: list[str] = [
        'FakeInstaller',
        'DroidKungFu',
        'Plankton',
        'GinMaster',
        'BaseBridge',
        'Iconosys',
        'Kmin',
        'FakeDoc',
    ]

    sha_family_map: dict[str, int] = {}
    known_family_timestamps: dict[str, datetime] = {}
    newfamily_sha_list: list[str] = []
    newfamily_timestamps: dict[str, datetime] = {}

    with open('data/drebin_metadata.csv') as f:
        next(f)  # Skip header
        for line in f:
            sha, family, latest_modify_time = line.strip().split(',')

            if family in top8 and latest_modify_time != 'None':
                family_idx = top8.index(family)
                timestamp = datetime.strptime(
                    latest_modify_time, '%Y-%m-%d %H:%M:%S'
                ).replace(tzinfo=UTC)

                if family_idx == newfamily:
                    newfamily_sha_list.append(sha)
                    newfamily_timestamps[sha] = timestamp
                else:
                    sha_family_map[sha] = family_idx
                    known_family_timestamps[sha] = timestamp

    sorted_known = sorted(known_family_timestamps.items(), key=lambda x: x[1])
    sorted_new = sorted(newfamily_timestamps.items(), key=lambda x: x[1])

    sha_sorted_by_time: list[str] = []
    label_sorted_by_time: list[int] = []

    saved_file = os.path.join(
        intermediate_folder, f'drebin_new{newfamily}_sha_timestamp_family.csv'
    )

    with open(saved_file, 'w') as f:
        f.write('sha256,timestamp,family\n')

        for sha, ts in sorted_known:
            sha_sorted_by_time.append(sha)
            label_sorted_by_time.append(sha_family_map[sha])
            f.write(f'{sha},{ts},{sha_family_map[sha]}\n')

        for sha, ts in sorted_new:
            f.write(f'{sha},{ts},{newfamily}\n')

    return sha_sorted_by_time, label_sorted_by_time, newfamily_sha_list


def split_drebin_train_and_test(
    sha_sorted_by_time: list[str],
    label_sorted_by_time: list[int],
    newfamily_sha_list: list[str],
    test_ratio: float,
    newfamily: int,
) -> tuple[list[str], list[str], list[int], list[int]]:
    r"""
    Split Drebin malware samples into training and testing sets based on temporal order.

    This function performs a chronological split where the most recent samples of
    known families are moved to the test set. Crucially, it injects 100% of the
    'new family' (drift) samples into the test set to simulate an
    out-of-distribution detection scenario.

    The split logic follows:
    1. The first $(1 - test\_ratio)$ of known samples go to training.
    2. The last $(test\_ratio)$ of known samples + all new family samples go to testing.

    Args:
        sha_sorted_by_time: List of sample SHAs for known families,
            ordered from oldest to newest.
        label_sorted_by_time: List of integer labels corresponding to
            `sha_sorted_by_time`.
        newfamily_sha_list: List of sample SHAs belonging to the
            unseen/drift family.
        test_ratio: Fraction of the known family samples to reserve for testing.
        newfamily: The integer label index assigned to the new family.

    Returns:
        A tuple containing four lists:
            * train_shas: SHAs for the training set.
            * test_shas: SHAs for the testing set (includes drift).
            * train_labels: Labels for the training set.
            * test_labels: Labels for the testing set (includes drift).
    """
    test_num = int(len(sha_sorted_by_time) * test_ratio)
    train_shas = sha_sorted_by_time[0:-test_num]
    train_labels = label_sorted_by_time[0:-test_num]
    test_shas = sha_sorted_by_time[-test_num:] + newfamily_sha_list
    test_labels = label_sorted_by_time[-test_num:] + [newfamily] * len(
        newfamily_sha_list
    )
    logging.debug(f'train_shas: {len(train_shas)}, test_shas: {len(test_shas)}')

    return train_shas, test_shas, train_labels, test_labels


def get_training_full_feature_names(
    intermediate_folder: str,
    newfamily: int,
    raw_feature_vectors_folder: str,
    train_shas: list[str],
) -> list[str]:
    """
    Extract and aggregate the unique feature space from all training samples.

    This function scans the raw Drebin feature files for the specified training
    samples to build a master list of all unique features (e.g., API calls,
    permissions, URLs). It implements a caching mechanism by saving the
    resulting feature list to a text file to avoid redundant I/O operations.

    Args:
        intermediate_folder: Path to the directory for saving/loading the feature cache.
        newfamily: The index of the drift family (used for cache file naming).
        raw_feature_vectors_folder: Path to the directory containing raw Drebin
            feature files named by their SHA256.
        train_shas: List of sample SHAs that constitute the training set.

    Returns:
        A sorted list of all unique feature names found across the training set.
    """
    saved_train_feature_file = os.path.join(
        intermediate_folder, f'drebin_new{newfamily}_full_training_features.txt'
    )

    if os.path.exists(saved_train_feature_file):
        logging.debug(f'Loading cached feature names from {saved_train_feature_file}')
        with open(saved_train_feature_file) as f:
            train_feature_names = [line.strip() for line in f]
    else:
        logging.info('Extracting unique features from raw Drebin files...')
        unique_features: set[str] = set()

        for sha in train_shas:
            file_path = os.path.join(raw_feature_vectors_folder, sha)
            try:
                with open(file_path) as f:
                    for line in f:
                        feature = line.strip()
                        if feature:
                            unique_features.add(feature)
            except FileNotFoundError:
                logging.warning(f'Feature file not found for SHA: {sha}')

        train_feature_names = sorted(unique_features)

        logging.info(
            f'[drebin-new{newfamily}] # of unique features in training set: {len(train_feature_names)}'  # noqa: E501
        )

        with open(saved_train_feature_file, 'w') as f:
            f.writelines(fea + '\n' for fea in train_feature_names)

    return train_feature_names


def save_training_full_feature_vectors(
    intermediate_folder: str,
    raw_feature_vectors_folder: str,
    train_shas: list[str],
    train_feature_names: list[str],
    train_labels: list[int],
    newfamily: int,
) -> str:
    """
    Vectorize training samples into a binary feature matrix and save to a compressed NPZ file.

    This function transforms raw text-based feature lists into a sparse-like dense
    binary matrix ($X \\in \\{0, 1\\}^{n \times m}$). It maps each sample's features
    to their corresponding indices in the global `train_feature_names` list.

    Args:
        intermediate_folder: Path to store the resulting .npz file.
        raw_feature_vectors_folder: Path to the directory containing raw feature files.
        train_shas: SHAs of the samples to be included in the training matrix.
        train_feature_names: Master list of unique feature names defining the vector space.
        train_labels: Numerical labels for each training sample.
        newfamily: Index of the drift family (used for file naming).

    Returns:
        The file path to the saved compressed NumPy archive (.npz).
    """  # noqa: E501
    saved_train_vectors_path = os.path.join(
        intermediate_folder, f'drebin_new{newfamily}_train_full_feature_vectors.npz'
    )

    if not os.path.exists(saved_train_vectors_path):
        logging.info(f'Vectorizing {len(train_shas)} training samples...')

        num_samples = len(train_shas)
        num_features = len(train_feature_names)

        x = np.zeros((num_samples, num_features), dtype=np.float32)

        feature_to_idx = {name: i for i, name in enumerate(train_feature_names)}

        for sample_idx, sha in enumerate(train_shas):
            file_path = os.path.join(raw_feature_vectors_folder, sha)
            try:
                with open(file_path) as f:
                    for line in f:
                        feature_name = line.strip()
                        if feature_name in feature_to_idx:
                            x[sample_idx, feature_to_idx[feature_name]] = 1
            except FileNotFoundError:
                logging.warning(f'File missing during vectorization: {sha}')

        y = np.array(train_labels, dtype=np.int32)

        np.savez_compressed(saved_train_vectors_path, X_train=x, y_train=y)
        logging.info(f'Saved training vectors to {saved_train_vectors_path}')

    return saved_train_vectors_path


def get_selected_features(
    intermediate_folder: str,
    saved_train_vectors: str,
    newfamily: int,
    train_feature_names: list[str],
) -> tuple[np.ndarray, str]:
    """
    Perform feature selection on training vectors using a variance threshold.

    This function identifies and removes low-variance (near-constant) features
    from the binary feature matrix. This dimensionality reduction helps
    improve model generalization and reduces the computational overhead
    for the explanation module. Resulting features and reduced vectors
    are cached to disk.



    Args:
        intermediate_folder: Path to store selected features and reduced vectors.
        saved_train_vectors: Path to the .npz file containing 'X_train' and 'y_train'.
        newfamily: Index of the drift family (used for file naming).
        train_feature_names: The full list of feature names corresponding to X_train columns.

    Returns:
        A tuple containing:
            * selected_features: A NumPy array of strings containing the names
              of features that passed the variance threshold.
            * saved_selected_vectors_file: The file path to the saved compressed
              NPZ containing the reduced feature matrix.
    """  # noqa: E501
    train_data = np.load(saved_train_vectors)
    x, y = train_data['X_train'], train_data['y_train']
    logging.debug(
        f'[drebin_new_{newfamily}] before feature selection X shape: {x.shape}'
    )
    selector = VarianceThreshold(0.003)
    x_select = selector.fit_transform(x)
    logging.debug(
        f'[drebin_new_{newfamily}] after feature selection X_select shape: {x_select.shape}'  # noqa: E501
    )

    selected_feature_indices = selector.get_support(indices=True)
    # logging.debug(f'selected_feature_indices: {list(selected_feature_indices)}')
    selected_features = np.array(train_feature_names)[selected_feature_indices]

    """ save selected features and corresponding feature vectors of training set """
    saved_selected_feature_file = os.path.join(
        intermediate_folder, f'drebin_new{newfamily}_train_selected_features.txt'
    )
    if not os.path.exists(saved_selected_feature_file):
        with open(saved_selected_feature_file, 'w') as fout:
            fout.writelines(f'{fea}\n' for fea in selected_features)
    saved_selected_vectors_file = os.path.join(
        intermediate_folder, f'drebin_new{newfamily}_train_selected_feature_vectors.npz'
    )
    if not os.path.exists(saved_selected_vectors_file):
        np.savez_compressed(saved_selected_vectors_file, X_train=x_select, y_train=y)

    return selected_features, saved_selected_vectors_file


def epoch_batches(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    similar_samples_ratio: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Generate batches optimized for Contrastive Autoencoder training via pair mining.

    This function transforms a standard dataset into a series of "pair-ready" batches.
    Each batch is structured so that the first half contains unique samples, and the
    second half contains a mix of similar (same class) and dissimilar (different class)
    counterparts to facilitate contrastive loss calculation.



    Args:
        x_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Training labels of shape (n_samples,). Must be continuous integers
            starting from 0.
        batch_size: The total number of samples per batch. Must be a multiple of 4.
            Note: The effective number of unique training samples used per batch
            is `batch_size / 2`.
        similar_samples_ratio: The fraction of the second half of the batch that
            should consist of similar (positive) pairs.

    Returns:
        A tuple containing:
            * batch_count (int): Total number of full batches generated.
            * b_out_x (np.ndarray): 3D feature array of shape
                (batch_count, batch_size, n_features).
            * b_out_y (np.ndarray): 2D label array of shape (batch_count, batch_size).

    Raises:
        SystemExit: If `batch_size` is not a multiple of 4.
    """
    if batch_size % 4 == 0:
        half_size = int(
            batch_size / 2
        )  # the really used batch_size for each batch. Another half data is filled by similar and dissimilar samples.  # noqa: E501
    else:
        logging.error('batch_size should be a multiple of 4.')
        sys.exit(-1)

    # Divide data into batches. # TODO: ignore the last batch for now, maybe there is a better way to address this.  # noqa: E501
    batch_count = int(x_train.shape[0] / half_size)
    logging.debug(f'batch_count: {batch_count}')  # -> 118
    num_sim = int(batch_size * similar_samples_ratio)  # 64 * 0.25 = 16
    b_out_x = np.zeros([batch_count, batch_size, x_train.shape[1]])
    b_out_y = np.zeros([batch_count, batch_size], dtype=int)
    logging.debug(f'b_out_x: {b_out_x.shape}, b_out_y: {b_out_y.shape}')

    random_idx = np.random.permutation(x_train.shape[0])  # random shuffle the batches
    # split the random shuffled X_train and y_train to batch_count shares
    b_out_x[:, :half_size] = np.split(
        x_train[random_idx[: batch_count * half_size]], batch_count
    )
    b_out_y[:, :half_size] = np.split(
        y_train[random_idx[: batch_count * half_size]], batch_count
    )

    tmp = random_idx[half_size]

    # NOTE: if error here, it's because we didn't convert X_train and X_test as np.float32 when generating the npz file.  # noqa: E501
    # to check if the split is correct
    assert np.all(x_train[tmp] == b_out_x[1, 0])

    # Sort data by label
    index_cls, index_no_cls = [], []
    """ NOTE: if we want to adapt to training label non-continuing, e.g., [0,1,2,3,4,5,7], but this would cause
    b_out_y[b, m] list index out of range. So we should convert [0,1,2,3,4,5,7] to [0,1,2,3,4,5,6] in the training set."""  # noqa: E501
    for label in range(len(np.unique(y_train))):
        index_cls.append(
            np.where(y_train == label)[0]
        )  # each row shows the index of y_train where y_train == label
        index_no_cls.append(np.where(y_train != label)[0])

    index_cls_len = [len(e) for e in index_cls]
    logging.debug(f'index_cls len: {index_cls_len}')
    index_no_cls_len = [len(e) for e in index_no_cls]
    logging.debug(f'index_no_cls len: {index_no_cls_len}')

    logging.debug('generating the batches and pairs...')
    # Generate the pairs
    logging.debug(f'num_sim: {num_sim}')
    logging.debug(f'half_size: {half_size}')
    start = timer()
    for b in range(batch_count):
        # Get similar samples
        for m in range(num_sim):
            # random sampling without replacement, randomly pick an index from y_train
            # where y_train[index] = b_out_y[b, m]
            # NOTE: list() operation is very slow, random.sample is also slower than np.random.choice()  # noqa: E501
            # ## pair = random.sample(list(index_cls[b_out_y[b, m]]), 1) would take 80s for each b,  # noqa: E501
            # np.random.choice() and list() would lead to 130s for each b
            # using only np.random.choice() would be 0.06s for each b
            pair = np.random.choice(index_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + half_size] = x_train[
                pair[0]
            ]  # pick num_sim samples with the same label
            b_out_y[b, m + half_size] = y_train[pair[0]]
        # pick (half_size - num_sim) dissimilar samples
        for m in range(num_sim, half_size):
            # randomly pick an index from y_train where y_train[index] != b_out_y[b, m]
            pair = np.random.choice(index_no_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + half_size] = x_train[pair[0]]
            b_out_y[b, m + half_size] = y_train[pair[0]]
        # DEBUG
        # if b == 1:
        # b_out_y[0] should looks like this (for simplicity assuming batch_size = 32, half_size = 16)  # noqa: E501
        # The first half is similar, the second half is dissimilar
        # 1, 2, 4, 8 | 2, 3, 5, 6
        # 1, 2, 4, 8 | 3, 4, 1, 7
        # logging.debug(f'b_out_x[1, 0, :20]: {b_out_x[b, 0, :20]}')
        # logging.debug(f'b_out_y[1]: {b_out_y[b]}')
    end = timer()

    logging.debug(f'split batch finished: {end - start} seconds')  # ~10s
    return batch_count, b_out_x, b_out_y
