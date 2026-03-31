"""
mask_exp_by_approximation.py
~~~~~~~

Functions for mask explanation: why a sample is an drifting (approximation-based exp only).

exp = x * mask

Only use the target x to solve the mask, didn't use the x + noise (as the perturbation might not be a good choice).

"""  # noqa: E501

import logging
import os
import random
import warnings

import numpy as np
import tensorflow as tf
from numpy.random import seed
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as k  # type: ignore

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # so the IDs match nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


random.seed(1)
seed(1)


tf.random.set_seed(2)


gpus = tf.config.list_physical_devices('GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


warnings.filterwarnings('ignore')


class OptimizeExp:
    def __init__(
        self,
        input_shape: tuple[int, ...],
        mask_shape: tuple[int, ...],
        model: tf.keras.Model,
        num_class: int,
        optimizer: type[tf.train.Optimizer],
        initializer: tf.keras.initializers.Initializer,
        lr: float,
        regularizer: str,
        model_file: str,
    ) -> None:
        """
        Initialize the optimization-based explanation generator.

        This class facilitates the creation of a feature mask that explains a
        model's prediction by optimizing a perturbation that minimizes
        the target class probability while maintaining mask sparsity.

        Args:
            input_shape: The shape of the input feature vector (e.g., (n_features,)).
            mask_shape: The shape of the mask to be optimized (usually matches input_shape).
            model: The pre-trained Keras model instance to be explained.
            num_class: Total number of distinct output labels in the target model.
            optimizer: A TensorFlow optimizer class (uninstantiated) for the mask update.
            initializer: A Keras initializer for the initial mask weights.
            lr: The learning rate for the mask optimization process.
            regularizer: The type of penalty to apply to the mask (e.g., 'elasticnet').
            model_file: System path to the saved model file for weight reloading.
        """  # noqa: E501

        self.model = model
        self.num_class = num_class
        self.lambda_1 = tf.placeholder(
            tf.float32
        )  # placeholder is similar to cin of C++
        self.optimizer = optimizer(lr)
        self.initializer = initializer
        self.regularizer = regularizer
        self.build_opt_func(input_shape, mask_shape)
        self.model_file = model_file

    @staticmethod
    def elasticnet_loss(tensor: tf.Tensor) -> tf.Tensor:
        """
        Calculates the Elastic Net regularization loss for a given tensor.

        This penalty is a linear combination of the L1 norm (Sparsity) and
        the L2 norm (Smoothness). It is particularly useful for feature
        selection when multiple features may be correlated.

        The loss is calculated as:
        $$\text{Loss} = \\sum |x| + \\sqrt{\\sum x^2}$$

        Args:
            tensor: The input TensorFlow tensor (usually the raw mask weights).

        Returns:
            A scalar TensorFlow tensor representing the combined regularization loss.
        """
        loss_l1 = tf.reduce_sum(tf.abs(tensor))
        loss_l2 = tf.sqrt(tf.reduce_sum(tf.square(tensor)))
        return loss_l1 + loss_l2

    def build_opt_func(
        self, _input_shape: tuple[int, ...], mask_shape: tuple[int, ...]
    ) -> None:
        """
        Constructs the TensorFlow computational graph for mask optimization.

        This method defines the symbolic tensors, loss functions, and training
        operations required to find an optimal feature mask. It sets up two
        parallel paths (explanation and remainder) and implements a combined
        loss function to encourage sparsity while maintaining prediction fidelity.

        The total loss is defined as:
        $$L = \text{loss\\_exp} - \text{loss\\_remain} + \\lambda_1 \\cdot \text{loss\\_reg\\_mask}$$

        Args:
            _input_shape: The shape of the input sample (unused).
            mask_shape: The shape of the trainable mask variable.

        Returns:
            None. Attributes like `self.train_op`, `self.loss`, and `self.mask`
            are initialized within the class instance.
        """  # noqa: E501
        # use tf.variable_scope and tf.get_variable() to achieve "variable sharing"
        # AUTO_REUSE: we create variables if they do not exist, and return them otherwise  # noqa: E501
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            self.mask = tf.get_variable(
                'mask', shape=mask_shape, initializer=self.initializer
            )

        self.mask_reshaped = self.mask

        self.mask_normalized = tf.minimum(1.0, tf.maximum(self.mask_reshaped, 0.0))

        # get_input_at(node_index): Retrieves the input tensor(s) of a layer at a given node. node_index = 0 corresponds to the first time the layer was called.  # noqa: E501
        self.input = self.model.get_input_at(0)

        self.x_exp = (
            self.input * self.mask_normalized
        )  # + self.fused_image * reverse_mask  # the explanation we are looking for, which contributes the most to the final prediction  # noqa: E501
        reverse_mask = tf.ones_like(self.mask_normalized) - self.mask_normalized
        self.x_remain = (
            self.input * reverse_mask
        )  # + self.fused_image * self.mask_normalized

        """
            because it's symbolic tensor with no actual value, so can't use model.predict()
            see: https://stackoverflow.com/questions/51515253/optimizing-a-function-involving-tf-kerass-model-predict-using-tensorflow-op
        """  # noqa: E501
        self.output_exp = self.model(
            self.x_exp
        )  # self.x_exp is the input to the self.model
        self.output_remain = self.model(self.x_remain)

        self.y_target = tf.placeholder(tf.float32, shape=(None, self.num_class))
        self.loss_exp = k.mean(k.binary_crossentropy(self.y_target, self.output_exp))
        self.loss_remain = k.mean(
            k.binary_crossentropy(self.y_target, self.output_remain)
        )

        if self.regularizer == 'l1':
            self.loss_reg_mask = tf.reduce_sum(tf.abs(self.mask_reshaped))
        elif self.regularizer == 'elasticnet':
            self.loss_reg_mask = self.elasticnet_loss(
                self.mask_reshaped
            )  # minimize mask
        elif self.regularizer == 'l2':
            self.loss_reg_mask = tf.sqrt(tf.reduce_sum(tf.square(self.mask_reshaped)))
        else:
            self.loss_reg_mask = tf.constant(0)

        self.loss = (
            self.loss_exp - self.loss_remain + self.lambda_1 * self.loss_reg_mask
        )

        # trainable variable
        self.var_train = tf.trainable_variables(
            scope='mask'
        )  # only one trainable variable: mask/mask: 0

        # training function
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.var_train)

    def fit_local(  # noqa: C901
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lambda_1: float,
        display_interval: int = 10,
        exp_acc_lowerbound: float = 0.8,
        iteration_threshold: float = 1e-4,
        lambda_patience: int = 100,
        lambda_multiplier: float = 1.5,
        early_stop_patience: int = 10,
    ) -> np.ndarray | None:
        """
        Explains a local model prediction by optimizing a feature mask.

        This method iteratively updates a mask to find the minimal set of features
        that maintain the model's prediction. It uses dynamic lambda scaling
        and early stopping to balance explanation accuracy and mask sparsity.

        Args:
            x: A single input sample (feature vector) to be explained.
            y: The model's original predicted probability distribution for x.
            epochs: Maximum number of optimization iterations.
            lambda_1: Initial regularization strength for mask sparsity.
            display_interval: Frequency (in epochs) to log optimization progress.
            exp_acc_lowerbound: Target threshold for explanation fidelity.
            iteration_threshold: Minimum loss improvement required to reset early stopping.
            lambda_patience: Epochs to wait before adjusting the lambda multiplier.
            lambda_multiplier: Factor by which to scale lambda when patience is met.
            early_stop_patience: Epochs to wait for improvement before halting.

        Returns:
            The optimized feature mask as an ndarray, or None if optimization fails.
        """  # noqa: E501
        input_ = np.expand_dims(x, axis=0).reshape(1, -1)
        logging.debug(f'input_ shape: {input_.shape}')

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=0)
        logging.debug(f'y shape: {y.shape}')

        loss_best = float('inf')
        loss_last = float('inf')
        loss_sparse_mask_best = float('inf')
        loss_sparse_mask_last = float('inf')

        mask_best = None
        early_stop_counter = 0
        lambda_up_counter = 0
        lambda_down_counter = 0

        # start training...
        with (
            tf.Session() as sess
        ):  # WARNING: it has to be like this, or the weights of the model could not be really loaded.  # noqa: E501
            sess.run(self.mask.initializer)
            sess.run(tf.variables_initializer(self.optimizer.variables()))
            sess.run(tf.global_variables_initializer())

            for step in range(epochs):
                feed_dict = {
                    self.input: input_,
                    self.y_target: y,
                    self.lambda_1: lambda_1,
                }

                """
                debugging with tensorboard
                """
                # logging.debug('debugging*******************')
                # writer = tf.summary.FileWriter("/tmp/mnist/", self.sess.graph)

                sess.run(self.train_op, feed_dict)

                self.model.load_weights(self.model_file, by_name=True)
                self.output_exp = self.model(self.x_exp)
                # logging.debug('current weights: ', self.model.get_layer('encoder_0').get_weights()[0][0][:5])  # noqa: E501

                """
                debugging if the weights of the target model are correctly loaded
                """
                # x_exp_value = sess.run(self.x_exp, feed_dict)
                # expected_exp = self.model.predict(x_exp_value)

                pred_exp = sess.run([self.output_exp], feed_dict)[0]
                [loss, loss_sparse_mask] = sess.run(
                    [self.loss, self.loss_reg_mask], feed_dict
                )

                acc_2 = accuracy_score(
                    np.argmax(y, axis=1), np.argmax(pred_exp, axis=1)
                )

                # check cost modification
                if acc_2 >= exp_acc_lowerbound:
                    lambda_up_counter += 1
                    if lambda_up_counter >= lambda_patience:
                        lambda_1 = lambda_1 * lambda_multiplier
                        lambda_up_counter = 0
                        logging.debug(
                            'Updating lambda1 to {:.8f} to {:.8f}'.format(
                                *self.lambda_1
                            )
                        )
                else:
                    lambda_down_counter += 1
                    if lambda_down_counter >= lambda_patience:
                        lambda_1 = lambda_1 / lambda_multiplier
                        lambda_down_counter = 0
                        logging.debug(
                            'Updating lambda1 to {:.8f} to {:.8f}'.format(
                                *self.lambda_1
                            )
                        )

                if (np.abs(loss - loss_last) < iteration_threshold) or (
                    np.abs(loss_sparse_mask - loss_sparse_mask_last)
                    < iteration_threshold
                ):
                    early_stop_counter += 1

                if (acc_2 > exp_acc_lowerbound) and (
                    early_stop_counter >= early_stop_patience
                ):
                    logging.info(
                        f'Reach the threshold and stop training at iteration {step + 1}/{epochs}.'  # noqa: E501
                    )
                    mask_best = sess.run([self.mask_normalized])[0]
                    break

                loss_last = loss
                loss_sparse_mask_last = loss_sparse_mask

                if (step + 1) % display_interval == 0:
                    mask = sess.run(self.mask)

                    if np.isnan(mask).any():
                        mask[np.isnan(mask)] = 1e-16
                        sess.run(self.mask.assign(mask))
                    feed_dict = {
                        self.input: input_,
                        self.y_target: y,
                        self.lambda_1: lambda_1,
                    }
                    [pred_remain, pred_exp] = sess.run(
                        [self.output_remain, self.output_exp], feed_dict
                    )
                    [loss, loss_exp, loss_remain, loss_sparse_mask] = sess.run(
                        [
                            self.loss,
                            self.loss_exp,
                            self.loss_remain,
                            self.loss_reg_mask,
                        ],
                        feed_dict,
                    )

                    acc_1 = accuracy_score(
                        np.argmax(y, axis=1), np.argmax(pred_remain, axis=1)
                    )
                    acc_2 = accuracy_score(
                        np.argmax(y, axis=1), np.argmax(pred_exp, axis=1)
                    )

                    # loss_sparse_mask: minimize mask

                    if loss_best > loss or loss_sparse_mask_best > loss_sparse_mask:
                        logging.debug(f'updating best loss from {loss_best} to {loss}')
                        logging.debug(
                            f'updating best sparse mask loss from {loss_sparse_mask_best} to {loss_sparse_mask}'  # noqa: E501
                        )
                        logging.debug(
                            f'Epoch {step + 1}/{epochs}: {loss:.5f = } explanation_loss = {loss_exp:.5f} '  # noqa: E501
                            f'remain_loss = {loss_remain:.5f} mask_sparse_loss = {loss_sparse_mask:.5f} '  # noqa: E501
                            f'acc_remain = {acc_1:.5f} acc_exp = {acc_2:.5f}'
                        )
                        loss_best = loss
                        loss_sparse_mask_best = loss_sparse_mask
                        mask_best = sess.run([self.mask_normalized])[0]
        if mask_best is None:
            logging.info('did NOT find the best mask')

        return mask_best
