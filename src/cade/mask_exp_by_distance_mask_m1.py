r"""
Our method: use m * m1 as explanation, only when m = m1 = 1, feature is important.
Explaining drift: minimize the difference between a drift $x$ and an in distribution centroid $c$ by swapping a
                small proportion of features.
Loss function:  \min E_{m \sim Bern(p)} ||f(x * (1 - m * m1) + (1-x)*(m * m1)), f(centroid)||_2 + \lambda * ||m * m1||_{1+2}
"""  # noqa: E501

import logging
import os
import random
import warnings
from typing import Literal

import numpy as np
import tensorflow as tf
from keras.models import Model
from numpy.random import seed

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # so the IDs match nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


random.seed(1)
seed(1)


tf.random.set_seed(2)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'Available GPUs: {gpus}')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

warnings.filterwarnings('ignore')


class OptimizeExp:
    def __init__(
        self,
        batch_size: int,
        mask_shape: tuple[int, ...],
        latent_dim: int,
        model: Model,
        optimizer: type[tf.train.Optimizer],
        initializer: tf.keras.initializers.Initializer,
        lr: float,
        regularizer: Literal['elasticnet'],
        temp: float,
        normalize_choice: Literal['sigmoid', 'tanh', 'clip'],
        *,  # Ruff Fix: Forces subsequent arguments to be keyword-only
        use_concrete: bool,
        model_file: str,
    ) -> None:
        """
        Explaining drift by minimizing the difference between a drift sample $x$ and
        an in-distribution centroid $c$ via feature swapping.

        The objective is to find a minimal mask $m$ such that the masked sample $x'$
        approaches the distribution of $c$ in the latent space.

        Args:
            batch_size: Training batch size for the optimization loop.
            mask_shape: Dimensionality of the feature mask (matches input space).
            latent_dim: Dimensionality of the encoder's latent representation.
            model: The pre-trained Keras encoder model.
            optimizer: The TensorFlow optimizer class (e.g., tf.train.AdamOptimizer).
            initializer: The Keras initializer instance for the mask weights.
            lr: Initial learning rate for the mask optimization.
            regularizer: Regularization technique to enforce mask sparsity.
            temp: Temperature parameter for the Gumbel-Softmax (Concrete) distribution.
            normalize_choice: Normalization method for the mask values.
            use_concrete: Whether to use the Concrete (Gumbel-Softmax) distribution trick.
            model_file: Path to the saved model weights for initialization.
        """  # noqa: E501

        self.model = model
        self.lambda_1 = tf.placeholder(
            tf.float32
        )  # placeholder is similar to cin of C++
        self.optimizer = optimizer(lr)
        self.initializer = initializer
        self.regularizer = regularizer
        self.batch_size = batch_size
        self.temp = temp
        self.normalize_choice = normalize_choice
        self.use_concrete = use_concrete
        self.build_opt_func(mask_shape, latent_dim)
        self.model_file = model_file

    @staticmethod
    def concrete_transformation(
        p: tf.Tensor, mask_shape: tuple[int, ...], batch_size: int, temp: float = 0.1
    ) -> tf.Tensor:
        """
        Apply the Concrete (Gumbel-Softmax) distribution trick to approximate binary variables.

        This transformation provides a differentiable approximation to a Bernoulli
        distribution. By adding Gumbel noise to the log-probabilities of the mask
        parameters, we can sample masks during training while allowing gradients
        to flow back to the mask parameters `p`.

        Args:
            p: The Bernoulli distribution parameters (mask probabilities), range [0, 1].
            mask_shape: The dimensions of the feature mask.
            batch_size: Number of synthetic samples to generate in the neighborhood.
            temp: The relaxation temperature. As $temp \to 0$, the output becomes
                discrete (0 or 1).

        Returns:
            A tensor of shape (batch_size, mask_shape[0]) containing
            continuous values that approximate binary mask selections.
        """  # noqa: E501
        epsilon = np.finfo(float).eps  # 1e-16

        unif_noise = tf.random_uniform(
            shape=(batch_size, mask_shape[0]), minval=0, maxval=1
        )
        reverse_theta = tf.ones_like(p) - p
        reverse_unif_noise = tf.ones_like(unif_noise) - unif_noise

        appro = (
            tf.log(p + epsilon)
            - tf.log(reverse_theta + epsilon)
            + tf.log(unif_noise)
            - tf.log(reverse_unif_noise)
        )
        logit = appro / temp

        return tf.sigmoid(logit)

    @staticmethod
    def elasticnet_loss(tensor: tf.Tensor) -> tf.Tensor:
        """
        Calculate the Elastic Net regularization loss for a given tensor.

        This combines $L_1$ (Lasso) and $L_2$ (Ridge) penalties to encourage
        both sparsity and group stability in the feature mask.

        Args:
            tensor: The input tensor (typically the feature mask) to regularize.

        Returns:
            A scalar tensor representing the sum of the $L_1$ and $L_2$ norms.
        """
        loss_l1 = tf.reduce_sum(tf.abs(tensor))
        loss_l2 = tf.sqrt(tf.reduce_sum(tf.square(tensor)))
        return loss_l1 + loss_l2

    def build_opt_func(self, mask_shape: tuple[int, ...], latent_dim: int) -> None:
        """
        Construct the TensorFlow computational graph for the mask optimization.

        This method defines the trainable mask variable, applies the chosen
        normalization, handles the Gumbel-Softmax (Concrete) transformation,
        and sets up the combined loss function (Distance Loss + Regularization).

        The optimization objective is:
        $$\\min_{m} \\| f(x \\odot (1-m \\cdot m_1) + x_{rev} \\odot (m \\cdot m_1)) - c \\|_2 + \\lambda_1 \\cdot \text{Reg}(m \\cdot m_1)$$

        Args:
            mask_shape: The shape of the input feature vector.
            latent_dim: The dimensionality of the encoder's latent space.

        Returns:
            None. Attributes like `self.train_op` and `self.loss` are attached to the instance.
        """  # noqa: E501
        # define and prepare variables.
        with tf.variable_scope('p', reuse=tf.AUTO_REUSE):
            self.p = tf.get_variable(
                'p', shape=mask_shape, initializer=self.initializer
            )

        # normalize variables
        if self.normalize_choice == 'sigmoid':
            logging.debug('Using sigmoid normalization.')
            self.p_normalized = tf.sigmoid(self.p)
        elif self.normalize_choice == 'tanh':
            logging.debug('Using tanh normalization.')
            self.p_normalized = (tf.tanh(self.p + 1)) / (2 + tf.keras.backend.epsilon())
        else:
            logging.debug('Using clip normalization.')
            self.p_normalized = tf.minimum(1.0, tf.maximum(self.p, 0.0))

        # discrete variables to continuous variables.
        if self.use_concrete:
            self.mask = self.concrete_transformation(
                self.p_normalized, mask_shape, self.batch_size, self.temp
            )
        else:
            self.mask = self.p_normalized
        self.reverse_p = tf.ones_like(self.p_normalized) - self.p_normalized

        # get input and reverse input.
        self.input = self.model.get_input_at(0)
        self.reverse_x = tf.placeholder(tf.float32, shape=(None, mask_shape[0]))

        self.m1 = tf.placeholder(tf.float32, shape=mask_shape)
        self.reverse_mask = tf.ones_like(self.mask) - self.mask * self.m1
        # if flip their feature value, it would be closer to the centroid
        self.x_exp = (
            self.input * self.reverse_mask + self.reverse_x * self.mask * self.m1
        )
        self.centroid = tf.placeholder(tf.float32, shape=(None, latent_dim))
        self.output_exp = self.model(self.x_exp)

        # l2 norm distance.
        self.loss_exp = tf.reduce_mean(
            tf.sqrt(tf.reduce_sum(tf.square(self.output_exp - self.centroid), axis=1))
        )

        if self.regularizer == 'l1':
            self.loss_reg_mask = tf.reduce_sum(tf.abs(self.p_normalized * self.m1))
        elif self.regularizer == 'elasticnet':
            self.loss_reg_mask = self.elasticnet_loss(
                self.p_normalized * self.m1
            )  # minimize mask
        elif self.regularizer == 'l2':
            self.loss_reg_mask = tf.sqrt(
                tf.reduce_sum(tf.square(self.p_normalized * self.m1))
            )
        else:
            self.loss_reg_mask = tf.constant(0)

        self.loss = self.loss_exp + self.lambda_1 * self.loss_reg_mask

        # trainable variable
        self.var_train = tf.trainable_variables(scope='p')

        # training function
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.var_train)

    def fit_local(  # noqa: C901
        self,
        x: np.ndarray,
        m1: np.ndarray,
        centroid: np.ndarray,
        closest_to_centroid_sample: np.ndarray,
        num_sync: int,
        num_changed_fea: int,
        epochs: int,
        lambda_1: float,
        display_interval: int = 10,
        exp_loss_lowerbound: float = 0.17,
        iteration_threshold: float = 1e-4,
        lambda_patience: int = 100,
        lambda_multiplier: float = 1.5,
        early_stop_patience: int = 10,
    ) -> np.ndarray | None:
        """
        Fit the local explanation mask by optimizing the feature swap between a drift sample
        and the target family distribution.

        This method iteratively updates a feature mask to minimize the distance to the
        latent `centroid` while maintaining sparsity via the `lambda_1` penalty.

        Args:
            x: The target drift sample feature vector.
            m1: Binary mask indicating which features are different from the reference.
            centroid: The latent space center vector of the closest known family.
            closest_to_centroid_sample: The training sample used as the in-distribution baseline.
            num_sync: Number of synchronized samples used for gradient stability.
            num_changed_fea: Minimum number of features expected to change in the explanation.
            epochs: Maximum number of optimization iterations.
            lambda_1: Initial sparsity loss penalty coefficient.
            display_interval: Frequency (in epochs) to log optimization progress.
            exp_loss_lowerbound: The distance threshold ($dist < lowerbound$) required
                before increasing sparsity pressure.
            iteration_threshold: The minimum change in loss required to consider
                the optimization as "progressing."
            lambda_patience: Number of epochs to wait before multiplying lambda if
                the lowerbound is not met.
            lambda_multiplier: Factor by which to increase lambda_1 during optimization.
            early_stop_patience: Number of epochs to wait for improvement before
                terminating the search.
        """  # noqa: E501

        # assuming the shape of X (p,)
        # swap a small number of features from the target drift sample to synthesize new drift sample,  # noqa: E501
        # so we can have more drift samples for the concrete distribution gumbel trick.
        sync_idx = np.random.choice(x.shape[0], (num_sync, num_changed_fea))
        sync_x = np.repeat(x[None, :], num_sync, axis=0).reshape(num_sync, x.shape[0])
        for i in range(num_sync):
            sync_x[i, sync_idx[i]] = 1 - x[sync_idx[i]]

        input_ = np.vstack((x, sync_x))
        logging.debug(f'input_ shape: {input_.shape}')

        sync_lowd = self.model.predict(input_)
        dis = np.square(sync_lowd - centroid)
        dis = np.mean(np.sqrt(np.sum(dis, axis=1)))
        logging.debug(
            f'x_target + synthesized sample average distance to centroid: {dis}'
        )

        if input_.shape[0] % self.batch_size != 0:
            num_batch = (input_.shape[0] // self.batch_size) + 1
        else:
            num_batch = input_.shape[0] // self.batch_size
        idx = np.arange(input_.shape[0])

        loss_best = float('inf')
        loss_sparse_mask_best = float('inf')
        loss_last = float('inf')
        loss_sparse_mask_last = float('inf')

        mask_best = None
        early_stop_counter = 0
        lambda_up_counter = 0
        lambda_down_counter = 0

        # start training...
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.model.load_weights(self.model_file, by_name=True)

            for step in range(epochs):
                loss_tmp = []
                loss_exp_tmp = []
                loss_sparse_mask_tmp = []
                for i in range(num_batch):
                    feed_dict = {
                        self.input: input_[
                            idx[
                                i * self.batch_size : min(
                                    (i + 1) * self.batch_size, input_.shape[0]
                                )
                            ],
                        ],
                        self.lambda_1: lambda_1,
                        self.centroid: centroid[None,],
                        self.reverse_x: closest_to_centroid_sample[None,],
                        self.m1: m1,
                    }
                    sess.run(self.train_op, feed_dict)
                    # NOTE: we don't need to load weights every batch. this is really time consuming. 5x time  # noqa: E501
                    # self.model.load_weights(self.model_file, by_name=True)
                    [loss, loss_sparse_mask, loss_exp] = sess.run(
                        [self.loss, self.loss_reg_mask, self.loss_exp], feed_dict
                    )
                    loss_tmp.append(loss)
                    loss_exp_tmp.append(loss_exp)
                    loss_sparse_mask_tmp.append(loss_sparse_mask)

                loss = sum(loss_tmp) / len(loss_tmp)
                loss_exp = sum(loss_exp_tmp) / len(loss_exp_tmp)
                loss_sparse_mask = sum(loss_sparse_mask_tmp) / len(loss_sparse_mask_tmp)

                if loss_exp <= exp_loss_lowerbound:
                    lambda_up_counter += 1
                    if lambda_up_counter >= lambda_patience:
                        lambda_1 = lambda_1 * lambda_multiplier
                        lambda_up_counter = 0
                else:
                    lambda_down_counter += 1
                    if lambda_down_counter >= lambda_patience:
                        lambda_1 = lambda_1 / lambda_multiplier
                        lambda_down_counter = 0

                if (np.abs(loss - loss_last) < iteration_threshold) or (
                    np.abs(loss_sparse_mask - loss_sparse_mask_last)
                    < iteration_threshold
                ):
                    early_stop_counter += 1

                if (loss_exp <= exp_loss_lowerbound) and (
                    early_stop_counter >= early_stop_patience
                ):
                    logging.info(
                        f'Reach the threshold and stop training at iteration {step + 1}/{epochs}.'  # noqa: E501
                    )
                    mask_best = sess.run([self.p_normalized])[0]
                    break

                if (step + 1) % display_interval == 0:
                    mask = sess.run(self.p)
                    if np.isnan(mask).any():
                        mask[np.isnan(mask)] = 1e-16
                        sess.run(self.mask.assign(mask))

                    if loss_best > loss or loss_sparse_mask_best > loss_sparse_mask:
                        logging.debug(f'updating best loss from {loss_best} to {loss}')
                        logging.debug(
                            f'updating best sparse mask loss from {loss_sparse_mask_best} to {loss_sparse_mask}'  # noqa: E501
                        )
                        logging.debug(
                            f'Epoch {step + 1}/{epochs}: loss = {loss:.5f} '
                            f'explanation_loss = {loss_exp:.5f} '
                            f'mask_sparse_loss = {loss_sparse_mask:.5f}'
                        )
                        loss_best = loss
                        loss_sparse_mask_best = loss_sparse_mask
                        mask_best = sess.run([self.p_normalized])[0]

                loss_last = loss
                loss_sparse_mask_last = loss_sparse_mask

        if mask_best is None:
            logging.info('did NOT find the best mask')

        return mask_best
