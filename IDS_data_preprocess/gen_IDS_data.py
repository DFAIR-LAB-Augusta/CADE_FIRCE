"""
Note: the final generated data has a minor mismatch of numbers, this is mainly because I downsampling 10% on training and testing respectively instead of 10% on training+testing

for the generated data:
IDS_new_Infilteration.npz: Counter({0: 66245, 2: 43486, 1: 11731, 3: 9238})
IDS_new_SSH.npz: Counter({0: 66245, 2: 43486, 1: 11732, 3: 9237})
IDS_new_Hulk.npz: Counter({0: 66245, 2: 43487, 1: 11731, 3: 9237})
"""  # noqa: E501

import argparse
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass
from pprint import pformat
from timeit import default_timer as timer

import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from cade.config import config

os.environ['PYTHONHASHSEED'] = '0'


random.seed(1)
seed(1)


# On our own lab server
DATA_FOLDER = config['IDS2018_clean']
UNNORMALIZED_SAVE_FOLDER = os.path.join(config['IDS2018'], 'unnormalized')

cwd = os.getcwd()  # IDS_data_preprocess
root_dir = os.path.dirname(cwd)  # CADE
SAVE_FOLDER = os.path.join(root_dir, 'data/')  # CADE/data

TRAFFIC_TYPE_LIST = np.array([
    'Benign',
    'FTP-BruteForce',
    'SSH-Bruteforce',
    'DoS attacks-GoldenEye',
    'DoS attacks-Slowloris',
    'DoS attacks-SlowHTTPTest',
    'DoS attacks-Hulk',
    'DDoS attacks-LOIC-HTTP',
    'DDOS attack-LOIC-UDP',
    'DDOS attack-HOIC',
    'Brute Force -Web',
    'Brute Force -XSS',
    'SQL injection',
    'Infilteration',
    'Bot',
])
ALL_FILES_LIST = [
    '02_14_2018',
    '02_15_2018',
    '02_16_2018',
    '02_21_2018',
    '02_22_2018',
    '02_23_2018',
    '02_28_2018',
    '03_01_2018',
    '03_02_2018',
    '02_20_2018',
]


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a specific dataset generation run."""

    name: str | None
    benign: str | None
    mal: str | None
    new_mal: str | None
    sampling_ratio: float


def create_folder(name: str) -> None:
    if not os.path.exists(name):
        os.makedirs(name)


def main() -> None:
    config = parse_args()

    """ parse required training and testing files, concatenate and resplit them. """
    saved_unnormalized_path = os.path.join(
        UNNORMALIZED_SAVE_FOLDER, f'{config.name}_unnormalized.npz'
    )
    x_train, x_test, y_train, y_test = split_data(
        config, saved_unnormalized_path)

    """ normalize train, test and save them to file. """
    save_path = os.path.join(SAVE_FOLDER, f'{config.name}.npz')
    normalize(x_train, x_test, y_train, y_test,
              config.sampling_ratio, save_path)


def parse_args() -> RunConfig:
    """
    Parse command line configuration and return a typed configuration object.

    Returns:
        RunConfig: A validated, immutable data object containing run parameters.
    """
    p = argparse.ArgumentParser()

    p.add_argument(
        '--name', help='The name of the generated dataset would be as name.npz.'
    )
    p.add_argument(
        '--benign', help='Specify which day of benign data will be used.')
    p.add_argument(
        '--mal',
        help='The date and type of malicious traffic for training/testing. '
        'Separated by "/". e.g., "02_14_2018,SSH-Bruteforce/02_16_2018,DoS attacks-Hulk"',  # noqa: E501
    )
    p.add_argument(
        '--new-mal',
        help='The date and type of malicious traffic (new family) for the testing set.',
    )
    p.add_argument(
        '--sampling-ratio',
        type=float,
        default=1.0,  # Changed from True to 1.0 (float) for type consistency
        help='The ratio of downsampling.',
    )

    args = p.parse_args()

    # Convert Namespace to our Dataclass
    config = RunConfig(**vars(args))

    logging.warning(f'Running with configuration: \n{pformat(vars(config))}')

    return config


def split_data(
    config: RunConfig, saved_unnormalized_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads or generates the training and testing datasets based on configuration.

    If a pre-existing dataset is found at `saved_unnormalized_path`, it is loaded
    directly. Otherwise, the function extracts benign and malicious data categories,
    performs a train-test split (80/20) for seen families, and appends a 'new'
    malicious family exclusively to the testing set to simulate drift.

    Args:
        config: Typed configuration object containing data paths and sampling ratios.
        saved_unnormalized_path: File path to the .npz archive for caching.

    Returns:
        A tuple of (x_train, x_test, y_train, y_test) as NumPy arrays.
        - x_train, x_test: Feature matrices.
        - y_train, y_test: Label vectors.

    Raises:
        ValueError: If 'mal', 'new_mal' or 'benign' configurations are missing when
            re-generating data.
    """
    if os.path.exists(saved_unnormalized_path):
        raw_data = np.load(saved_unnormalized_path)
        return (
            raw_data['X_train'],
            raw_data['X_test'],
            raw_data['y_train'],
            raw_data['y_test'],
        )

    # 1. Validation
    if config.mal is None or config.new_mal is None or config.benign is None:
        raise ValueError(
            '`mal`, `new_mal`, and `benign` args must all be set.')

    # 2. Extract Data Categories
    seen_mal_dict = get_needed_file_types_dict(config.mal)
    new_mal_dict = get_needed_file_types_dict(config.new_mal)

    # Benign
    x_benign, y_benign = extract_data_by_category(config.benign, 'Benign')

    # Build lists for training/testing components
    x_train_list, x_test_list = [], []
    y_train_list, y_test_list = [], []

    # 3. Process Benign + Seen Malicious (80/20 split)
    all_seen_data = [(x_benign, y_benign)]
    for day, cat in seen_mal_dict.items():
        all_seen_data.append(extract_data_by_category(day, cat))

    for x_cat, y_cat in all_seen_data:
        xt, xv, yt, yv = train_test_split(
            x_cat, y_cat, test_size=0.2, shuffle=False)
        x_train_list.append(xt)
        x_test_list.append(xv)
        y_train_list.append(yt)
        y_test_list.append(yv)

    # 4. Process New Malicious (Testing set only)
    for day, cat in new_mal_dict.items():
        x_new, y_new = extract_data_by_category(day, cat)
        x_test_list.append(x_new)
        y_test_list.append(y_new)

    # 5. Final Concatenation (Single memory allocation)
    x_train = np.concatenate(x_train_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Reporting and Caching
    print(f'Final shapes - X_train: {x_train.shape}, X_test: {x_test.shape}')
    print(f'y_test labels: {Counter(y_test)}')

    np.savez_compressed(
        saved_unnormalized_path,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
    )

    return x_train, x_test, y_train, y_test


def get_needed_file_types_dict(mal_str: str) -> dict[str, str]:
    """
    Parses a delimited string of malware files and categories into a dictionary.

    The input string should follow the format 'filename,category/filename,category'.
    Each '/' separates different data files, and each ',' separates the date-based
    filename from the specific traffic type (e.g., 'DoS attacks-Hulk').

    Args:
        mal_str: A formatted string containing malicious traffic metadata.
            Example: '02_15_2018,DoS attacks-GoldenEye/02_16_2018,DoS attacks-Hulk'

    Returns:
        A dictionary where keys are filenames (dates) and values are the
        corresponding traffic types.
        Example: {'02_15_2018': 'DoS attacks-GoldenEye'}

    Raises:
        ValueError: If the input string does not follow the 'key,value'
            comma-separated format.
    """
    # return a dict with filename as key, list of needed traffic types as value.
    # e.g., {'02_15_2018': ['DoS attacks-GoldenEye']}
    data_list = mal_str.split('/')
    data_file_types_dict = {}
    for data in data_list:
        filename, traffic_type = data.split(',')
        data_file_types_dict[filename] = traffic_type
    print(f'data_file_types_dict: {data_file_types_dict}')
    return data_file_types_dict


def extract_data_by_category(
    single_day: str, category: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts and sorts feature vectors and labels for a specific traffic category.

    Loads a compressed NumPy archive for a given day, filters the dataset to
    isolate samples matching the specified category name, and sorts the
    resulting subset based on the third feature column (index 2).

    Args:
        single_day: The filename/date string (e.g., '02_15_2018') representing
            the source data file.
        category: The specific traffic label to extract (e.g., 'Benign', 'SSH-Bruteforce').

    Returns:
        A tuple containing:
            - sorted_data (np.ndarray): The filtered and sorted feature matrix.
            - sorted_label (np.ndarray): The corresponding sorted label vector.

    Raises:
        FileNotFoundError: If the .npz file for 'single_day' does not exist in DATA_FOLDER.
        KeyError: If 'X', 'y', or 'y_name' are missing from the loaded archive.
    """  # noqa: E501
    data_file_path = os.path.join(DATA_FOLDER, single_day + '.npz')
    raw_data = np.load(data_file_path)
    x, y, y_name = raw_data['X'], raw_data['y'], raw_data['y_name']
    data_filter = np.where(y_name == category)[0]
    data = x[data_filter]
    label = y[data_filter]
    sort_idx = np.argsort(data[:, 2], kind='mergesort')
    sorted_data = data[sort_idx]
    sorted_label = label[sort_idx]
    print(f'sorted_data {category}.shape: {sorted_data.shape}')
    print(f'sorted_label {category}.shape: {sorted_label.shape}')
    return sorted_data, sorted_label


def normalize(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ratio: float,
    save_path: str,
) -> None:
    """
    Performs comprehensive feature engineering and normalization on the dataset.

    This function executes a sequential preprocessing pipeline:
    1. Downsamples the training and testing sets based on the provided ratio.
    2. Transforms the 'Destination Port' (index 0) into frequency-based
       categories (High/Medium/Low) before One-Hot Encoding.
    3. One-Hot Encodes the 'Protocol' feature (index 1).
    4. Applies MinMaxScaler to all remaining numeric features (indices 2+),
       fitting on training data and transforming testing data.
    5. Saves the final processed arrays to a compressed .npz file.

    Args:
        x_train: Training feature matrix.
        x_test: Testing feature matrix.
        y_train: Training labels.
        y_test: Testing labels.
        ratio: The downsampling ratio to apply to both sets.
        save_path: System path where the normalized .npz file will be stored.

    Returns:
        None. Processed data is saved to disk and summary statistics are logged.

    Notes:
        The function handles 'unseen' ports or protocols in the test set by
        ignoring them during encoding (resulting in all-zero vectors).
    """
    print(f'y_train unique: {np.unique(y_train)}')

    # 1. Downsampling
    x_train, y_train = downsampling(x_train, y_train, ratio, _phase='train')
    x_test, y_test = downsampling(x_test, y_test, ratio, _phase='test')

    # 2. Port Frequency Binning Logic
    training_ports = x_train[:, 0]
    training_ports_counter = Counter(training_ports)

    high_freq_ports = []
    med_freq_ports = []
    low_freq_ports = []

    for port, count in training_ports_counter.items():
        if count >= 10000:
            high_freq_ports.append(port)
        elif count >= 1000:
            med_freq_ports.append(port)
        else:
            low_freq_ports.append(port)

    # 3. Fit Port Encoder
    port_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_ports_cat = np.array(
        transform_ports_to_categorical(
            training_ports, high_freq_ports, med_freq_ports, low_freq_ports
        )
    ).reshape(-1, 1)

    # Cast to ndarray to fix Pylance index/getitem issues
    train_ports_enc = np.asarray(port_encoder.fit_transform(train_ports_cat))

    # 4. Fit Protocol Encoder
    proto_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_protos = x_train[:, 1].reshape(-1, 1)

    # Cast to ndarray to fix Pylance index/getitem issues
    train_protos_enc = np.asarray(proto_encoder.fit_transform(train_protos))

    # 5. Fit Scaler (Numeric features index 2 onwards)
    scaler = MinMaxScaler()
    x_train_scale = scaler.fit_transform(x_train[:, 2:])

    # 6. Combine Training Data
    # Pylance is happy here because all inputs are confirmed as np.ndarray
    x_old = np.concatenate(
        [train_ports_enc, train_protos_enc, x_train_scale], axis=1)
    y_old = y_train.astype('int32')

    # 7. Transform Test Data
    test_ports_cat = np.array(
        transform_ports_to_categorical(
            x_test[:, 0], high_freq_ports, med_freq_ports, low_freq_ports
        )
    ).reshape(-1, 1)
    test_ports_enc = np.asarray(port_encoder.transform(test_ports_cat))

    test_protos = x_test[:, 1].reshape(-1, 1)
    test_protos_enc = np.asarray(proto_encoder.transform(test_protos))

    x_test_scale = scaler.transform(x_test[:, 2:])

    # 8. Combine Testing Data
    x_new = np.concatenate(
        [test_ports_enc, test_protos_enc, x_test_scale], axis=1)
    y_new = y_test.astype('int32')

    # 9. Save and Report
    print(f'X_old: {x_old.shape}, X_new: {x_new.shape}')
    np.savez_compressed(
        save_path, X_train=x_old, y_train=y_old, X_test=x_new, y_test=y_new
    )

    # Optional stats helpers
    stats(x_old, x_new, y_old, y_new)
    stats_data_helper(x_new, 'normalized x_test')


def downsampling(
    x_train: np.ndarray, y_train: np.ndarray, ratio: float, _phase: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs stratified random downsampling across all traffic families.

    This function maintains the relative class distribution by sampling each unique
    family label independently. It collects subsets in lists before a single
    concatenation to optimize memory usage and performance.

    Args:
        x_train: The input feature matrix (samples, features).
        y_train: The corresponding label vector.
        ratio: The fraction of samples to retain for each family (0.0 to 1.0).
        _phase: Descriptive string indicating the current pipeline phase
            (e.g., 'train' or 'test').

    Returns:
        A tuple containing:
            - x_sampled (np.ndarray): The downsampled feature matrix.
            - y_sampled (np.ndarray): The corresponding downsampled label vector.
        If no samples meet the ratio threshold, returns a tuple of empty arrays.
    """
    x_collect = []
    y_collect = []

    for family in np.unique(y_train):
        family_indices = np.where(y_train == family)[0]
        sample_size = int(ratio * len(family_indices))

        if sample_size == 0:
            continue

        filter_idx = np.random.choice(
            family_indices, size=sample_size, replace=False)

        x_collect.append(x_train[filter_idx])
        y_collect.append(y_train[filter_idx])

    if not x_collect:
        return np.array([]), np.array([])

    # Single allocation of memory for the final arrays
    x_sampled = np.concatenate(x_collect, axis=0)
    y_sampled = np.concatenate(y_collect, axis=0)

    return x_sampled, y_sampled


def transform_ports_to_categorical(
    ports: np.ndarray,
    high_freq_port_list: list[float | int],
    medium_freq_port_list: list[float | int],
    _low_freq_port_list: list[float | int],
) -> list[int]:
    """
    Maps raw destination port numbers to categorical frequency tiers.

    This transformation reduces the dimensionality of the port feature by grouping
    ports based on their prevalence in the training set:
    - 0: High frequency (e.g., common services like 80, 443)
    - 1: Medium frequency
    - 2: Low frequency (ephemeral or rare ports)

    Args:
        ports: A 1D array of port numbers to be transformed.
        high_freq_port_list: List of ports identified as high frequency.
        medium_freq_port_list: List of ports identified as medium frequency.
        _low_freq_port_list: List of ports identified as low frequency (unused).

    Returns:
        A list of categorical integers (0, 1, or 2) corresponding to each input port.
    """
    ports_transform = []
    for port in ports:
        if port in high_freq_port_list:
            ports_transform.append(0)
        elif port in medium_freq_port_list:
            ports_transform.append(1)
        else:
            ports_transform.append(2)
    return ports_transform


def stats_data_helper(x: np.ndarray, data_type: str) -> None:
    print('==================')
    print(f'feature stats for {data_type}')
    print(f'min: {np.min(x, axis=0)}')
    print(f'avg: {np.average(x, axis=0)}')
    print(f'max: {np.max(x, axis=0)}')


def stats_label_helper(y: np.ndarray, data_type: str) -> None:
    print(f'label stats for {data_type}')
    print(f'{Counter(y)}')
    print('==================')


def stats(
    x_old: np.ndarray, x_new: np.ndarray, y_old: np.ndarray, y_new: np.ndarray
) -> None:
    stats_data_helper(x_old, 'old')
    stats_label_helper(y_old, 'old')
    stats_data_helper(x_new, 'new')
    stats_label_helper(y_new, 'new')


if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    print(f'time elapsed: {end - start}')
