import os
import re
import sys
from collections import Counter
from typing import Literal

import numpy as np

import cade.data as data


def main(
    dataset: Literal['drebin', 'IDS'],
    use_pure_ae: int,
    families_cnt: Literal[8, 4],
    last_label: Literal[7, 3],
    margin: float,
    mad: float,
    cae_lambda: float,
) -> None:
    """
    Aggregates and reports drift detection performance across multiple malware families.

    This function iterates through the results of individual family drift experiments,
    extracts metrics using regular expressions, calculates the mean and standard
    deviation across all families, and saves a summary report. It specifically
    handles logic differences between the 'drebin' (Android malware) and
    'IDS' (Intrusion Detection) datasets.

    Args:
        dataset: The dataset name, either 'drebin' or 'IDS'.
        use_pure_ae: Flag (1/0) indicating if a Pure Autoencoder was used
            instead of a Contractive Autoencoder (CAE).
        families_cnt: Total number of families to iterate through.
        last_label: The integer label representing the 'new' or 'drifting'
            family in the classification task.
        margin: The margin hyperparameter used during the distance-based check.
        mad: The Median Absolute Deviation scaling factor used in evaluation.
        cae_lambda: The regularization lambda used for the CAE.

    Returns:
        None. Outputs results to the console and writes a detailed summary
        to a text file in the designated report directory.

    Raises:
        FileNotFoundError: If the expected result CSVs are missing for any family.
        IndexError: If the regex patterns (p1-p4) fail to find a match in the
            report files.
    """
    report_dir = 'reports' if use_pure_ae == 0 else 'pure_ae_reports'

    families = range(families_cnt) if dataset == 'drebin' else range(1, families_cnt)

    # Only used by IDS, outside to prevent unbound errors
    name_dict = {1: 'SSH', 2: 'Hulk', 3: 'Infilteration'}

    p1 = re.compile(r'precision: \d+\.\d+')
    p2 = re.compile(r'recall: \d+\.\d+')
    p3 = re.compile(r'f1: \d+\.\d+')
    p4 = re.compile(r'best inspection count: \d+')

    precision_list, recall_list, f1_list, inspect_cnt_list = [], [], [], []
    involved_families_list = []
    normalized_inspect_cnt_list = []

    for i in families:
        """calc how many new family samples in the testing set"""
        if dataset == 'drebin':
            single_dataset = f'drebin_new_{i}'
            name = i
        else:
            single_dataset = f'IDS_new_{name_dict[i]}'
            name = name_dict[i]

        _x_train, _y_train, _x_test, y_test = data.load_features(single_dataset, i)

        total_new_family = len(np.where(y_test == last_label)[0])

        """record results for each family"""
        result_path = os.path.join(
            f'{report_dir}',
            single_dataset,
            f'dist_mlp_one_by_one_check_pr_value_m{margin}_mad{mad}_lambda{cae_lambda}.csv',
        )
        with open(result_path) as f:
            content = f.read()
        precision = float(re.findall(p1, content)[0].replace('precision: ', '')) / 100
        recall = float(re.findall(p2, content)[0].replace('recall: ', '')) / 100
        f1 = float(re.findall(p3, content)[0].replace('f1: ', '')) / 100
        inspect_cnt = int(
            re.findall(p4, content)[0].replace('best inspection count: ', '')
        )
        print(
            f'family {name}: precision: {precision * 100}%, recall: {recall * 100}%, f1: {f1 * 100}%, inspect: {inspect_cnt}'  # noqa: E501
        )
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        inspect_cnt_list.append(inspect_cnt)
        normalized_inspect_cnt_list.append(inspect_cnt / total_new_family)

        # check the involved families in the drifting samples when we get the best results  # noqa: E501
        involved_families = []
        with open(result_path) as f:
            next(f)
            for idx, line in enumerate(f):
                if idx < inspect_cnt:
                    line = line.strip().split(',')
                    real = line[1]
                    involved_families.append(real)

        involved_families_list.append(involved_families)

    print('============================================')
    print('avg +/- std (final result in Table 3): ')
    print(
        f'precision: {np.average(precision_list) * 100:.2f}% +/- {np.std(precision_list):.2f}'  # noqa: E501
    )
    print(f'recall: {np.average(recall_list) * 100:.2f}% +/- {np.std(recall_list):.2f}')
    print(f'f1: {np.average(f1_list) * 100:.2f}% +/- {np.std(f1_list):.2f}')
    print(
        f'inspect_cnt: {np.average(inspect_cnt_list):.2f} +/- {np.std(inspect_cnt_list):.2f}'  # noqa: E501
    )
    print(
        f'normalized inspect_cnt: {np.average(normalized_inspect_cnt_list):.2f} '
        f'+/- {np.std(normalized_inspect_cnt_list):.2f}'
    )
    print('============================================')

    saved_report_folder = f'{report_dir}/average_{dataset}'
    os.makedirs(saved_report_folder, exist_ok=True)
    with open(
        f'{saved_report_folder}/average_{dataset}_result_margin{margin}_mad{mad}_lambda{cae_lambda}.txt',
        'w',
    ) as f:
        f.write('family_idx,precision,recall,f1,insepct_cnt,normalized_inspect_cnt\n')
        for i in range(len(precision_list)):
            name = i if dataset == 'drebin' else name_dict[i + 1]
            f.write(
                f'{name},{precision_list[i]:.4f},{recall_list[i]:.4f},{f1_list[i]:.4f},'
                f'{inspect_cnt_list[i]:.2f},{normalized_inspect_cnt_list[i]:.2f}\n'
            )

        f.write('============================================\n')
        f.write('avg +/- std (final result in Table 3): \n')
        f.write(
            f'precision: {np.average(precision_list) * 100:.2f}% +/- {np.std(precision_list):.2f}\n'  # noqa: E501
        )
        f.write(
            f'recall: {np.average(recall_list) * 100:.2f}% +/- {np.std(recall_list):.2f}\n'  # noqa: E501
        )
        f.write(f'f1: {np.average(f1_list) * 100:.2f}% +/- {np.std(f1_list):.2f} \n')
        f.write(
            f'inspect_cnt: {np.average(inspect_cnt_list):.2f} +/- {np.std(inspect_cnt_list):.2f}\n'  # noqa: E501
        )
        f.write(
            f'normalized inspect_cnt: {np.average(normalized_inspect_cnt_list):.2f} '
            f'+/- {np.std(normalized_inspect_cnt_list):.2f}\n'
        )

        f.write('============================================\n')
        for i in range(len(involved_families_list)):
            name = i if dataset == 'drebin' else name_dict[i + 1]
            f.write(
                f'family {name}:\t families detected as drifting: {Counter(involved_families_list[i])}\n'  # noqa: E501
            )


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(
            'usage: "python -u average_all_detection_results.py drebin 0", '
            'where 0 for CADE, 1 for vanilla autoencoder. You may also specify to use drebin or IDS.'  # noqa: E501
        )
        sys.exit(-1)

    dataset = sys.argv[1]
    use_pure_ae = int(sys.argv[2])

    if dataset == 'drebin':
        families_cnt = 8
        last_label = 7
    elif dataset == 'IDS':
        families_cnt = 4
        last_label = 3
    else:
        print('dataset could only be "drebin" or "IDS"')
        sys.exit(-1)

    mad = 0.0 if use_pure_ae else 3.5

    margin = 10.0
    cae_lambda = 0.1

    main(dataset, use_pure_ae, families_cnt, last_label, margin, mad, cae_lambda)
