import logging
import os
import random
import sys
from collections import Counter
from pprint import pformat
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed

import cade.classifier as classifier
import cade.data as data
import cade.detect as detect
import cade.evaluate as evaluate
import cade.explain_by_distance as explain_dis
import cade.utils as utils
from cade.autoencoder import Autoencoder, ContrastiveAE
from cade.logger import init_log

os.environ['PYTHONHASHSEED'] = '0'


random.seed(1)
seed(1)

set_random_seed(2)


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # so the IDs match nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
# OLD & Bad
# k.tensorflow_backend.set_session(tf.Session(config=config))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def main() -> None:
    # ---------------------------------------- #
    # 0. Init log path and parse args          #
    # ---------------------------------------- #

    args = utils.parse_args()

    log_path = './logs/main'
    if args.quiet:
        init_log(log_path, level=logging.INFO)
    else:
        init_log(log_path, level=logging.DEBUG)
    logging.warning('Running with configuration:\n' + pformat(vars(args)))
    logging.getLogger('matplotlib.font_manager').disabled = True

    logging.debug(f'available GPUs: {tf.config.list_physical_devices("GPU")}')

    # ----------------------------------------------- #
    # 1. Prepare the dataset                          #
    # ----------------------------------------------- #

    data.prepare_dataset(args)

    # ---------------------------------------- #
    # 2. Load the feature vectors and labels   #
    # ---------------------------------------- #

    logging.info(f'Loading {args.data} dataset')

    x_train, y_train, x_test, y_test = data.load_features(
        args.data, args.newfamily_label
    )

    logging.info(f'Loaded train: {x_train.shape}, {y_train.shape}')
    logging.info(f'Loaded test: {x_test.shape}, {y_test.shape}')
    logging.info(f'y_train labels: {np.unique(y_train)}')
    logging.info(f'y_test labels: {np.unique(y_test)}')
    logging.info(f'y_train: {Counter(y_train)}')
    logging.info(f'y_test: {Counter(y_test)}')

    # ----------------------------------------------- #
    # 3. Train classifier and evaluate on test set    #
    # ----------------------------------------------- #

    # some commonly used variables.
    saved_model_dir = 'models/'
    if args.pure_ae == 0:
        report_dir = 'reports/'
        fig_dir = 'fig/'
    else:
        report_dir = 'pure_ae_reports/'
        fig_dir = 'pure_ae_fig/'
    num_features = x_train.shape[1]
    num_classes = len(np.unique(y_train))

    logging.info(
        f'Number of features: {num_features}; Number of classes: {num_classes}'
    )

    logging.info(
        'train on the training set and predict on the validation and testing set...'
    )

    if args.classifier == 'mlp':
        mlp_dims = utils.get_model_dims(
            'MLP', num_features, args.mlp_hidden, num_classes
        )

        class_weight = None

        mlp_model_name = (
            f'{args.data}_{args.classifier}_'
            f'{mlp_dims}_lr{args.mlp_lr}_'
            f'b{args.mlp_batch_size}_e{args.mlp_epochs}_'
            f'd{args.mlp_dropout}.h5'
        )

        mlp_model_path = os.path.join(saved_model_dir, args.data, mlp_model_name)

        mlp_classifier = classifier.MLPClassifier(
            dims=mlp_dims, model_save_name=mlp_model_path, dropout=args.mlp_dropout
        )

        # incase args.mlp_retrain = 0 while there is no Model file
        logging.debug(f'Saving MLP models to {mlp_model_path}...')
        retrain_flag = 1 if not os.path.exists(mlp_model_path) else args.mlp_retrain
        logging.debug(f'retrain? {retrain_flag}')

        mlp_classifier.train(
            x_train,
            y_train,
            lr=args.mlp_lr,
            batch_size=args.mlp_batch_size,
            epochs=args.mlp_epochs,
            class_weight=class_weight,
            retrain=bool(retrain_flag),
        )

        saved_confusion_matrix_fig_path = os.path.join(
            fig_dir, args.data, 'intermediate', 'MLP_confusion_matrix.png'
        )

        y_pred, _new_acc = mlp_classifier.predict(
            x_test,
            y_test,
            args.data,
            args.newfamily_label,
            saved_confusion_matrix_fig_path,
        )

    elif args.classifier == 'rf':
        rf_model_name = f'{args.data}_{args.classifier}_{args.tree}.pkl'
        rf_model_path = os.path.join(saved_model_dir, args.data, rf_model_name)
        rf_classifier = classifier.RFClassifier(rf_model_path, args.tree)

        # incase args.rf_retrain = 0 while there is no Model file
        retrain_flag = 1 if not os.path.exists(rf_model_path) else args.rf_retrain
        saved_confusion_matrix_fig_path = os.path.join(
            fig_dir, args.data, 'RF_confusion_matrix.png'
        )
        y_pred, _val_acc, _new_acc = rf_classifier.fit_and_predict(
            x_train,
            y_train,
            x_test,
            y_test,
            args.data,
            args.newfamily_label,
            saved_confusion_matrix_fig_path,
            retrain=retrain_flag,
        )
    else:
        logging.error(f'classifier {args.classifier} NOT supported.')
        sys.exit(-2)

    # --------------------------------------------------------- #
    # 4. Report the classification results                      #
    # --------------------------------------------------------- #
    logging.info(
        'Report classification results for the wrongly classified and all the new samples...'  # noqa: E501
    )

    name_tmp = f'{args.classifier}'
    # ALL: contains all the classification results in the testing set
    classify_results_all_path = os.path.join(
        report_dir,
        f'{args.data}',
        'intermediate',
        name_tmp + '_classification_results_all.csv',
    )
    # SIMPLE: only contains misclassified samples in the testing set
    classify_results_simple_path = os.path.join(
        report_dir,
        f'{args.data}',
        'intermediate',
        name_tmp + '_classification_results_simple.csv',
    )
    if args.classifier == 'mlp':
        if (
            os.path.exists(classify_results_all_path)
            and os.path.exists(classify_results_simple_path)
            and not args.mlp_retrain
        ):
            logging.debug(
                f'{classify_results_all_path} and {classify_results_simple_path} already exists.'  # noqa: E501
            )
        else:
            evaluate.report_classification_results(
                mlp_model_path,
                x_test,
                y_test,
                classify_results_all_path,
                classify_results_simple_path,
            )
    elif args.classifier == 'rf':
        if (
            os.path.exists(classify_results_all_path)
            and os.path.exists(classify_results_simple_path)
            and not args.rf_retrain
        ):
            logging.debug(
                f'{classify_results_all_path} and {classify_results_simple_path} already exists.'  # noqa: E501
            )
        else:
            evaluate.report_classification_results(
                rf_model_path,
                x_test,
                y_test,
                classify_results_all_path,
                classify_results_simple_path,
            )

    # --------------------------------------------------------- #
    # 5. Train the Contrastive Autoencoder                      #
    # --------------------------------------------------------- #
    logging.info('Training contrastive autoencoder...')
    cae_dims = utils.get_model_dims(
        'Contrastive AE', num_features, args.cae_hidden, num_classes
    )
    optimizer = tf.train.AdamOptimizer
    ckpt_dir = os.path.join(saved_model_dir, args.data)
    utils.create_folder(ckpt_dir)
    cae_dims_str = (
        str(cae_dims)
        .replace(' ', '')
        .replace(',', '-')
        .replace('[', '')
        .replace(']', '')
    )  # remove extra symbols

    s1 = timer()

    if args.pure_ae == 0:
        """ Our method: use contrastive autoencoder"""
        ae_weights_path = os.path.join(
            ckpt_dir,
            f'cae_{cae_dims_str}_lr{args.cae_lr}'
            f'_b{args.cae_batch_size}_e{args.cae_epochs}_m{args.margin}_lambda{args.cae_lambda_1}_weights.h5',
        )
        cae = ContrastiveAE(cae_dims, optimizer, args.cae_lr)
        cae.train(
            x_train,
            y_train,
            args.cae_lambda_1,
            args.cae_batch_size,
            args.cae_epochs,
            args.similar_ratio,
            args.margin,
            ae_weights_path,
            args.display_interval,
        )
    else:
        """baseline: use vanilla autoencoder"""
        ae_weights_path = os.path.join(
            ckpt_dir,
            f'pure_ae_{cae_dims_str}_lr{args.cae_lr}'
            f'_b{args.cae_batch_size}_e{args.cae_epochs}_m{args.margin}_weights.h5',
        )
        pure_ae = Autoencoder(cae_dims)
        batch_size = int(
            args.cae_batch_size / 2
        )  # CAE need the pair comparison, so we adjust it to half the CAE batch_size.
        pure_ae.train_and_save(
            x_train,
            ae_weights_path,
            args.cae_lr,
            batch_size,
            args.cae_epochs,
            loss='mse',
        )

    e1 = timer()
    logging.info(f'Training contrastive autoencoder time: {(e1 - s1):.3f} seconds')
    logging.info('Training contrastive autoencoder finished')

    # --------------------------------------------------------- #
    # 6. Detect drifting samples in the testing set                  #
    # --------------------------------------------------------- #
    logging.info('Detect drifting samples in the testing set...')
    postfix_no_mad = f'm{args.margin}_lambda{args.cae_lambda_1}'
    # ALL: contains the closest family for all the testing set, use this to compare with classifier's prediction  # noqa: E501
    all_detect_path = os.path.join(
        report_dir,
        args.data,
        'intermediate',
        f'{args.classifier}_detect_results_all_{postfix_no_mad}.csv',
    )
    utils.create_parent_folder(all_detect_path)
    # SIMPLE: only contains drift samples flagged by the MAD # NOTE: this is just for quickly viewing the results  # noqa: E501
    simple_detect_path = os.path.join(
        report_dir,
        args.data,
        'intermediate',
        f'{args.classifier}_detect_results_simple_{postfix_no_mad}.csv',
    )

    # training info for detect: contains all the needed info to determine a new testing sample  # noqa: E501
    # is an drifting sample for a particular family
    training_info_for_detect_path = os.path.join(
        report_dir,
        args.data,
        'intermediate',
        f'{args.classifier}_training_info_for_detect_{postfix_no_mad}.npz',
    )

    s2 = timer()
    detect.detect_drift_samples(
        x_train,
        y_train,
        x_test,
        y_test,
        y_pred,
        cae_dims,
        args.margin,
        args.mad_threshold,
        ae_weights_path,
        all_detect_path,
        simple_detect_path,
        training_info_for_detect_path,
    )
    e2 = timer()
    logging.debug(f'detect_odd_samples time: {(e2 - s2):.3f} seconds')
    logging.info('Detect drifting samples in the testing set finished')

    # --------------------------------------------------------- #
    # 7. Evaluate the detection performance                     #
    # --------------------------------------------------------- #
    logging.info('Evaluate the detection performance...')
    postfix_with_mad = (
        f'm{args.margin}_mad{args.mad_threshold}_lambda{args.cae_lambda_1}'
    )

    name_tmp = (
        f'{args.classifier}_combined_classify_detect_results_{postfix_no_mad}.csv'
    )
    combined_report_path = os.path.join(report_dir, args.data, 'intermediate', name_tmp)

    evaluate.combine_classify_and_detect_result(
        classify_results_all_path, all_detect_path, combined_report_path
    )

    saved_ordered_dis_path = os.path.join(
        report_dir,
        args.data,
        'intermediate',
        f'ordered_sample_real_mindis_{postfix_with_mad}.txt',
    )
    # final result
    dist_effor_pr_val_fig_path = os.path.join(
        fig_dir,
        args.data,
        f'dist_{args.classifier}_inspection_effort_pr_value_{postfix_with_mad}.png',
    )
    dist_one_by_one_check_result_path = os.path.join(
        report_dir,
        args.data,
        f'dist_{args.classifier}_one_by_one_check_pr_value_{postfix_with_mad}.csv',
    )

    evaluate.evaluate_newfamily_as_drift_by_distance(
        args.data,
        args.newfamily_label,
        combined_report_path,
        args.mad_threshold,
        saved_ordered_dis_path,
        dist_effor_pr_val_fig_path,
        dist_one_by_one_check_result_path,
    )
    logging.info('Evaluate the detection performance finished')

    # --------------------------------------------------------- #
    # 8. Explain why it's an drifting sample                    #
    # --------------------------------------------------------- #
    if args.stage == 'explanation':
        logging.info('Explain the detected drifting samples...')
        lambda1 = args.exp_lambda_1
        exp_method = args.exp_method
        mask_file_path = os.path.join(
            report_dir, args.data, f'mask_{exp_method}_{lambda1}.npz'
        )  # final explanation
        if exp_method == 'approximation_loose':
            import cade.explain_global_approximation_loose_boundary as explain

            saved_exp_classifier_dir = os.path.join(
                saved_model_dir, args.data, f'exp_{exp_method}'
            )

            explain.explain_drift_samples_per_instance(
                x_train,
                y_train,
                x_test,
                y_test,
                args,
                dist_one_by_one_check_result_path,
                training_info_for_detect_path,
                ae_weights_path,
                saved_exp_classifier_dir,
                mask_file_path,
            )
        elif exp_method == 'distance_mm1':
            """explain by minimizing latent distance to centroid """
            explain_dis.explain_drift_samples_per_instance(
                x_train,
                y_train,
                x_test,
                y_test,
                args,
                dist_one_by_one_check_result_path,
                training_info_for_detect_path,
                ae_weights_path,
                mask_file_path,
            )
        else:
            logging.error(f'exp_method {exp_method} not supported')
            sys.exit(-3)
        logging.info('Explain the detected drifting samples finished')


if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start}')
