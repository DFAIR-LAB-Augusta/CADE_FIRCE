"""
evaluate.py
~~~~~~~

Functions for evaluating drifting detection and report classification results.

"""

import logging
import os
import pickle
import random
import sys
import traceback
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as k
from keras.models import Model, load_model
from numpy.random import seed
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import set_random_seed
from tqdm import tqdm

import cade.utils as utils

os.environ['PYTHONHASHSEED'] = '0'


random.seed(1)
seed(1)

set_random_seed(2)


def report_classification_results(
    model_path: str,
    x_new: np.ndarray,
    y_new: np.ndarray,
    classify_results_all_path: str,
    classify_results_simple_path: str,
) -> None:
    """
    Report wrongly classified samples and probabilities for classification model.

    Args:
        model_path: File path for the target MLP model.
        x_new: Feature vectors of new data samples.
        y_new: Groundtruth labels of new data samples.
        classify_results_all_path: File path for saving all new samples (prediction, real, prob).
        classify_results_simple_path: File path for saving only wrongly classified samples.
    """  # noqa: E501
    report_classification_results_helper(
        model_path, x_new, y_new, classify_results_all_path, only_wrongly_samples=False
    )
    report_classification_results_helper(
        model_path,
        x_new,
        y_new,
        classify_results_simple_path,
        only_wrongly_samples=True,
    )


def report_classification_results_helper(
    model_path: str,
    x_new: np.ndarray,
    y_new: np.ndarray,
    report_file_path: str,
    *,
    only_wrongly_samples: bool,
) -> None:
    """
    Generate a CSV report of classification results, optionally filtering for errors.

    Args:
        model_path: Path to the saved model (.h5 for Keras, .pkl for Scikit-learn).
        x_new: Feature vectors of the samples to classify.
        y_new: Groundtruth labels for the samples.
        report_file_path: Destination path for the CSV report.
        only_wrongly_samples: If True, only log samples where prediction != real_label.
    """
    if 'h5' in model_path:
        k.clear_session()
        clf_model = load_model(model_path)
        if not isinstance(clf_model, Model):
            raise TypeError(
                f'Loaded model is invalid. Expected Keras Model, got {type(clf_model)}'
            )
        preds = clf_model.predict(x_new)
        y_new_pred = np.argmax(preds, axis=1)
        y_new_prob = np.max(preds, axis=1)
    elif 'pkl' in model_path:
        with open(model_path, 'rb') as f:
            clf_model = pickle.load(f)
        y_new_pred = clf_model.predict(x_new)
        y_new_prob = np.max(clf_model.predict_proba(x_new), axis=1)
    else:
        logging.error(
            f'saved model name {model_path} is neither h5 or pkl format')
        sys.exit(-1)

    utils.create_parent_folder(report_file_path)
    with open(report_file_path, 'w') as f:
        f.write('sample_idx,real_label,pred_label,pred_prob\n')
        for idx, real_label in tqdm(enumerate(y_new), desc='MLP classified'):
            if only_wrongly_samples:
                if y_new_pred[idx] != real_label:
                    f.write(
                        f'{idx},{real_label},{y_new_pred[idx]},{y_new_prob[idx]}\n')
            else:
                f.write(
                    f'{idx},{real_label},{y_new_pred[idx]},{y_new_prob[idx]}\n')
    if only_wrongly_samples:
        logging.info('Reported wrongly classified samples.')
    else:
        logging.info('Reported the classification for all new samples.')


def combine_classify_and_detect_result(
    classify_results_all_path: str,
    detect_results_all_path: str,
    combined_report_path: str,
) -> None:
    """
    Merge classification and drift detection logs into a single summary report.

    Args:
        classify_results_all_path: Path to the CSV with classification outcomes.
        detect_results_all_path: Path to the CSV with drift/anomaly outcomes.
        combined_report_path: Destination path for the merged CSV.
    """
    if os.path.exists(combined_report_path):
        logging.info(
            f'Report already exists at {combined_report_path}. Skipping.')
        return

    header = 'sample_idx,real_label,pred_label,closest_label,is_drift,pred_prob,min_distance,min_anomaly_score\n'  # noqa: E501

    try:
        with (
            open(classify_results_all_path) as f_cls,
            open(detect_results_all_path) as f_det,
            open(combined_report_path, 'w') as f_out,
        ):
            f_out.write(header)
            next(f_cls)
            next(f_det)

            for line_cls, line_det in zip(f_cls, f_det):
                idx, real, pred, pred_prob = line_cls.strip().split(',')

                det_data = line_det.strip().split(',')
                is_drift, closest = det_data[1], det_data[2]
                min_dis, min_score = det_data[5], det_data[6]

                f_out.write(
                    f'{idx},{real},{pred},{closest},{is_drift},{pred_prob},{min_dis},{min_score}\n'
                )
    except FileNotFoundError as e:
        logging.error(f'Failed to combine reports: {e}')
        if os.path.exists(combined_report_path):
            os.remove(combined_report_path)


def evaluate_newfamily_as_drift_by_distance(
    dataset_name: str,
    newfamily: int,
    combined_report_path: str,
    mad_threshold: float,
    save_ordered_dis_path: str,
    dist_effort_pr_value_fig_path: str,
    dist_one_by_one_check_result_path: str,
) -> None:
    if 'drebin' in dataset_name:
        newfamily = (
            # since we adjust all the new family to label 7, no matter it is 0~7.
            7
        )
    elif 'IDS' in dataset_name:
        newfamily = 3

    total_new_family = 0
    sample_result_dict = {}
    y_closest = []
    y_real = []
    y_pred = []
    with open(combined_report_path) as f:
        next(f)
        for _idx, line in enumerate(f):
            sample_idx, real, pred, closest, _is_drift, _prob, min_dis, min_score = (
                read_combined_report_line(line)
            )
            y_closest.append(closest)
            y_real.append(real)
            y_pred.append(pred)

            if real == newfamily:
                total_new_family += 1
            if min_score > mad_threshold:
                sample_result_dict[sample_idx] = [
                    real,
                    pred,
                    closest,
                    min_dis,
                    min_score,
                ]

    ordered_sample_result_dict = OrderedDict(
        sorted(sample_result_dict.items(), key=lambda x: x[1][3], reverse=True)
    )
    with open(save_ordered_dis_path, 'w') as f:
        f.write('sample_idx,real_label,min_dis\n')
        f.writelines(
            f'{k},{v[0]},{v[3]}\n' for k, v in ordered_sample_result_dict.items()
        )

    plot_inspection_effort_pr_value_by_dist(
        ordered_sample_result_dict,
        newfamily,
        total_new_family,
        dist_effort_pr_value_fig_path,
        dist_one_by_one_check_result_path,
    )

    """get the drift closest family confusion matrix"""
    acc_classifier = float(accuracy_score(y_real, y_pred))
    acc_closest = float(accuracy_score(y_real, y_closest))
    cm = confusion_matrix(y_real, y_closest)

    append_accuracy_result_to_final_report(
        acc_classifier, acc_closest, dist_one_by_one_check_result_path
    )

    logging.debug(
        f'use drift closest family as prediction accuracy:\n {acc_closest}')
    logging.debug(
        f'use drift closest family as prediction confusion matrix:\n {cm}')


def plot_inspection_effort_pr_value_by_dist(
    sorted_samples: OrderedDict,
    newfamily: int,
    total_new_family: int,
    fig_path: str,
    pr_result_path: str,
) -> None:
    """
    Calculate PR metrics over varying inspection efforts and generate a performance plot.

    This function simulates a manual inspection process where samples are reviewed one
    by one (ordered by a metric like anomaly score). It calculates cumulative
    Precision and Recall, identifies the optimal inspection threshold, and saves
    both a detailed CSV report and a visualization of the PR curve.

    Args:
        sorted_samples: An OrderedDict where keys are sample identifiers and
            values are a list/tuple (at minimum [real_label, ..., closest_label]).
        newfamily: The integer label representing the 'drift' or 'target' class
            (True Positive condition).
        total_new_family: Total number of actual drift samples in the entire
            testing set (used for Recall denominator).
        fig_path: File system path where the resulting PNG/PDF plot will be saved.
        pr_result_path: File system path for the CSV report containing
            per-step metrics.

    Returns:
        None. Output is persisted to the file system.
    """  # noqa: E501
    tp, fp = 0, 0
    precision_list, recall_list = [], []
    inspection_cnt_list = range(1, len(sorted_samples) + 1)

    with open(pr_result_path, 'w') as f:
        f.write('sample_idx,real,closest,TP,FP,precision,recall\n')
        for sample_idx, values in sorted_samples.items():
            if values[0] == newfamily:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp)
            recall = tp / total_new_family
            precision_list.append(precision)
            recall_list.append(recall)
            f.write(
                f'{sample_idx},{values[0]},{values[2]},{tp},{fp},{precision},{recall}\n'
            )

        best_inspection_cnt, best_precision, best_recall, best_f1 = get_best_result(
            precision_list, recall_list
        )
        best_inspection_percent = best_inspection_cnt / len(precision_list)
        f.write(f'\n\nTotal: {len(sorted_samples)}\n')
        f.write(
            f'best inspection count: {best_inspection_cnt}, percent: {best_inspection_percent}\n'  # noqa: E501
        )
        f.write(
            f'best performance -- precision: {best_precision * 100:.2f}%, recall: {best_recall * 100:.2f}%\
                f1: {best_f1 * 100:.2f}%\n'  # noqa: E501
        )

    annotation_text = f'inspect {best_inspection_cnt} samples\nP:{best_precision * 100:.2f}%, R:{best_recall * 100:.2f}%\nF1:{best_f1 * 100:.2f}%'  # noqa: E501

    fig, ax = plt.subplots()
    ax.plot(inspection_cnt_list, precision_list, label='precision', color='g')
    ax.plot(inspection_cnt_list, recall_list, label='recall', color='b')
    ax.annotate(
        annotation_text, (best_inspection_cnt, best_precision), fontsize=6, color='red'
    )
    ax.set_title(
        'Precision Recall value as the change of inspection efforts', fontsize=12
    )
    ax.set_xticks(np.around(np.linspace(
        0, len(inspection_cnt_list), 10), decimals=0))
    ax.set_xlabel('Inspection Effort (# of Samples)', fontsize=16)
    ax.set_ylabel('Rate', fontsize=16)
    ax.legend(loc='best')
    fig.savefig(fig_path, dpi=200)
    plt.clf()


def append_accuracy_result_to_final_report(
    acc_classifier: float, acc_closest: float, dist_one_by_one_check_result_path: str
) -> None:
    """
    Appends classification and drift-based accuracy metrics to a text report.

    Args:
        acc_classifier: Accuracy of the primary classifier on the testing set.
        acc_closest: Accuracy achieved by using the closest historical family
            as the prediction for drifted samples.
        dist_one_by_one_check_result_path: Path to the log file where results
            should be appended.

    Returns:
        None
    """
    with open(dist_one_by_one_check_result_path, 'a') as f:
        f.write('\n====================================\n')
        f.write(f'classifier acc on testing set: {acc_classifier}\n')
        f.write(
            f'use drift closest family as prediction accuracy on testing set: {acc_closest}\n'  # noqa: E501
        )


def get_best_result(
    precision_list: list[float], recall_list: list[float]
) -> tuple[int, float, float, float]:
    """
    Find the optimal inspection threshold based on the maximum F1-score.

    Iterates through cumulative precision and recall values to calculate the
    F1-score at each step, identifying the point that best balances the
    trade-off between precision and recall.

    Args:
        precision_list: Cumulative precision values at each inspection step.
        recall_list: Cumulative recall values at each inspection step.

    Returns:
        A tuple containing:
            - best_inspection_cnt (int): The number of samples to inspect for
                the best F1.
            - best_precision (float): Precision at the optimal point.
            - best_recall (float): Recall at the optimal point.
            - best_f1 (float): The maximum F1-score achieved.
    """
    best_inspection_cnt = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    for i in range(len(precision_list)):
        try:
            f1 = (
                2
                * precision_list[i]
                * recall_list[i]
                / (precision_list[i] + recall_list[i])
            )
            if f1 > best_f1:
                best_f1 = f1
                best_inspection_cnt = i + 1
                best_precision = precision_list[i]
                best_recall = recall_list[i]
        except ZeroDivisionError:
            logging.debug(f'list-{i}\n {traceback.format_exc()}')
            continue
    return best_inspection_cnt, best_precision, best_recall, best_f1


def read_combined_report_line(
    line: str,
) -> tuple[str, int, int, int, str, float, float, float]:
    """
    Parse and type-cast a single line from the combined report CSV.
    Magic tuple ah return; should be a dataclass, or better yet a pydantic model

    Args:
        line: A comma-separated string containing sample metadata and metrics.

    Returns:
        A tuple containing:
            - sample_idx (str): The unique identifier for the sample.
            - real (int): The ground-truth label.
            - pred (int): The label predicted by the classifier.
            - closest (int): The label of the closest historical family.
            - is_drift (str): 'True' or 'False' indicating if drift was detected.
            - prob (float): The prediction probability/confidence.
            - min_dis (float): Euclidean distance to the closest centroid.
            - min_score (float): The calculated anomaly or non-conformity score.
    """
    (
        sample_idx,
        real_str,
        pred_str,
        closest_str,
        is_drift,
        prob_str,
        min_dis_str,
        min_score_str,
    ) = line.strip().split(',')

    real = int(real_str)
    pred = int(pred_str)
    closest = int(closest_str)
    prob = float(prob_str)
    min_dis = float(min_dis_str)
    min_score = float(min_score_str)

    return sample_idx, real, pred, closest, is_drift, prob, min_dis, min_score
