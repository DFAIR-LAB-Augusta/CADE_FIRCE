"""
autoencoder.py
~~~~~~~

Functions for training a unified autoencoder or individual autoencoders for each family.

"""

import logging
import math
import os
import time
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans

import cade.data as data
import cade.logger as logger
import cade.utils as utils

# tf.compat.v1.disable_eager_execution()
# os.environ['PYTHONHASHSEED'] = '0'


# random.seed(1)
# seed(1)

# tf.random.set_seed(2)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # TensorFlow wizardry
# config = tf.ConfigProto()
# # Don't pre-allocate memory; allocate as-needed
# config.gpu_options.allow_growth = True
# # Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 0.5


class Autoencoder:
    def __init__(
        self,
        dims: list[int],
        activation: str = 'relu',
        init: str = 'glorot_uniform',
        verbose: int = 1,
    ) -> None:
        """
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
        The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        activation: activation, not applied to Input, last layer of the encoder, and Output layers

        """  # noqa: E501
        self.dims = dims
        self.act = activation
        self.init = init
        self.verbose = verbose

    def build(self) -> tuple[Model, Model]:
        """Fully connected auto-encoder model, symmetric.

        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        dims = self.dims
        act = self.act
        init = self.init

        n_stacks = len(dims) - 1
        # input
        input_img = Input(shape=(dims[0],), name='input')
        x = input_img
        # internal layers in encoder
        for i in range(n_stacks - 1):
            x = Dense(
                dims[i + 1],
                activation=act,
                kernel_initializer=init,
                name=f'encoder_{i}',
            )(x)
            # kernel_initializer is a fancy term for which statistical distribution or function to use for initializing the weights. Neural network needs to start with some weights and then iteratively update them  # noqa: E501

        # hidden layer, features are extracted from here, no activation is applied here, i.e., "linear" activation: a(x) = x  # noqa: E501
        encoded = Dense(
            dims[-1], kernel_initializer=init, name=f'encoder_{n_stacks - 1}'
        )(x)
        self.encoded = encoded

        x = encoded
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            x = Dense(
                dims[i],
                activation=act,
                kernel_initializer=init,
                name=f'decoder_{i}',  # Fixed UP031
            )(x)

        # output
        x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
        decoded = x
        self.out = decoded

        ae = Model(inputs=input_img, outputs=decoded, name='AE')
        encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
        return ae, encoder

    def train_and_save(
        self,
        x: np.ndarray,
        weights_save_name: str,
        lr: float = 0.001,
        batch_size: int = 32,
        epochs: int = 250,
        _loss: str = 'mse',
    ) -> None:
        x = np.asarray(x, dtype=np.float32)
        if os.path.exists(weights_save_name):
            logging.info('weights file exists, no need to train pure AE')
        else:
            logging.debug(f'AE train_and_save lr: {lr}')
            logging.debug(f'AE train_and_save batch_size: {batch_size}')
            logging.debug(f'AE train_and_save epochs: {epochs}')

            verbose = self.verbose

            autoencoder, _encoder = self.build()

            pretrain_optimizer = Adam(learning_rate=lr)

            autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')

            utils.create_parent_folder(weights_save_name)

            mcp_save = ModelCheckpoint(
                weights_save_name,
                monitor='loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose,
                mode='min',
            )

            autoencoder.fit(
                x,
                x,
                epochs=epochs,
                batch_size=batch_size,
                verbose=str(self.verbose),
                callbacks=[mcp_save, logger.LoggingCallback(logging.debug)],
            )

    def evaluate_quality(
        self, x_old: np.ndarray, y_old: np.ndarray, model_save_name: str
    ) -> float | Literal[0]:
        if not os.path.exists(model_save_name):
            self.train_and_save(x_old, model_save_name)

        k.clear_session()
        _autoencoder, encoder = self.build()
        encoder.load_weights(model_save_name, by_name=True)
        logging.debug(f'Load weights from {model_save_name}')
        latent = encoder.predict(x_old)

        best_acc = 0
        best_n_init = 10
        num_classes = len(np.unique(y_old))
        logging.debug(f'KMeans k = {num_classes}')

        warnings.filterwarnings('ignore')

        for n_init in range(10, 110, 10):
            kmeans = KMeans(n_clusters=num_classes, n_init=n_init, random_state=42)
            y_pred = kmeans.fit_predict(latent)
            acc = utils.get_cluster_acc(y_old, y_pred)
            logging.debug(f'KMeans n_init: {n_init}, acc: {acc}')
            if acc > best_acc:
                best_n_init = best_n_init
                best_acc = acc
        logging.info(
            f'best accuracy of KMeans on latent data: {best_acc} with n_init {best_n_init}'  # noqa: E501
        )
        return best_acc


class ContrastiveAE:
    def __init__(
        self,
        dims: list[int],
        optimizer: Callable[..., Any],
        lr: float,
        verbose: int = 1,
    ) -> None:
        self.dims = dims
        self.optimizer_factory = optimizer
        self.lr = lr
        self.verbose = verbose

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        lambda_1: float,
        batch_size: int,
        epochs: int,
        similar_ratio: float,
        margin: float,
        weights_save_name: str,
        display_interval: int,
    ) -> None:
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int32)

        utils.create_parent_folder(weights_save_name)
        if os.path.exists(weights_save_name):
            logging.info('weights file exists, no need to train contrastive AE')
            return

        if batch_size < 4 or batch_size % 4 != 0:
            raise ValueError('batch_size must be a multiple of 4 and >= 4')

        ae = Autoencoder(self.dims, verbose=self.verbose)
        ae_model, encoder_model = ae.build()

        optimizer = self.optimizer_factory(self.lr)

        min_loss = np.inf
        half_size = batch_size // 2

        for epoch in range(epochs):
            epoch_time = time.time()

            batch_count, batch_x, batch_y = data.epoch_batches(
                x_train,
                y_train,
                batch_size,
                similar_ratio,
            )

            loss_batch: list[float] = []
            contrastive_loss_batch: list[float] = []
            ae_loss_batch: list[float] = []
            same_pair_batch: list[float] = []
            diff_pair_batch: list[float] = []

            for b in range(batch_count):
                xb = tf.convert_to_tensor(batch_x[b], dtype=tf.float32)
                yb = tf.convert_to_tensor(batch_y[b], dtype=tf.int32)

                with tf.GradientTape() as tape:
                    reconstructed = ae_model(xb, training=True)
                    z = encoder_model(xb, training=True)

                    z_left = z[:half_size]
                    z_right = z[half_size:]
                    y_left = yb[:half_size]
                    y_right = yb[half_size:]

                    is_same = tf.cast(tf.equal(y_left, y_right), tf.float32)

                    dist = tf.sqrt(
                        tf.reduce_sum(tf.square(z_left - z_right), axis=1) + 1e-10
                    )

                    contrastive_vec = is_same * dist + (1.0 - is_same) * tf.nn.relu(
                        margin - dist
                    )
                    contrastive_loss = tf.reduce_mean(contrastive_vec)

                    reconstruction_loss = tf.reduce_mean(
                        tf.keras.losses.mse(xb, reconstructed)
                    )

                    total_loss = reconstruction_loss + lambda_1 * contrastive_loss

                grads = tape.gradient(total_loss, ae_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, ae_model.trainable_variables))

                loss_batch.append(float(total_loss.numpy()))
                contrastive_loss_batch.append(float(contrastive_loss.numpy()))
                ae_loss_batch.append(float(reconstruction_loss.numpy()))
                same_pair_batch.append(float(tf.reduce_sum(is_same).numpy()))
                diff_pair_batch.append(float(tf.reduce_sum(1.0 - is_same).numpy()))

            current_loss = float(np.mean(loss_batch))

            if math.isnan(current_loss):
                logging.error('NaN value in loss')
                raise RuntimeError('NaN encountered during CADE contrastive training')

            if epoch % display_interval == 0:
                logging.info(
                    f'Epoch {epoch}: loss {current_loss} -- '
                    f'contrastive {np.mean(contrastive_loss_batch)} -- '
                    f'ae {np.mean(ae_loss_batch)} -- '
                    f'pairs {np.mean(same_pair_batch)} : {np.mean(diff_pair_batch)} -- '
                    f'time {time.time() - epoch_time}'
                )

            if current_loss < min_loss:
                logging.info(f'updating best loss from {min_loss} to {current_loss}')
                min_loss = current_loss
                ae_model.save_weights(weights_save_name)
