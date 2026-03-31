"""
classifier.py
~~~~~~~

Functions for building a target classifier.

"""

import logging
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import cade.utils as utils
from cade.logger import LoggingCallback

os.environ['PYTHONHASHSEED'] = '0'


random.seed(1)
seed(1)

tf.random.set_seed(2)


class MLPClassifier:
    """a MLP classifier only for multi-class classification."""

    def __init__(
        self,
        dims: list[int],
        model_save_name: str,
        dropout: float = 0.2,
        activation: str = 'relu',
        verbose: int = 1,
    ) -> None:
        self.dims = dims  # e.g., [1347, 100, 30, 7]
        self.model_save_name = model_save_name
        self.act = activation
        self.dropout = dropout
        self.verbose = verbose  # 1 print logs, 0 no logs.

    def build(self) -> Model:
        """
        Build an MLP model using the Keras functional API.

        This method constructs a multi-layer perceptron based on the dimensions
        specified in `self.dims`. It automatically adds Dense layers with the
        specified activation and optional Dropout layers.

        Returns:
            Model: A compiled or uncompiled Keras Functional API Model instance.
        """
        n_stacks = len(self.dims) - 1
        input_tensor = Input(shape=(self.dims[0],), name='input')
        x = input_tensor

        for i in range(n_stacks - 1):
            x = Dense(self.dims[i + 1], activation=self.act, name=f'clf_{i}')(x)

            if self.dropout > 0:
                x = Dropout(self.dropout, seed=42)(x)

        x = Dense(self.dims[-1], activation='softmax', name=f'clf_{n_stacks - 1}')(x)

        output_tensor = x
        model = Model(inputs=input_tensor, outputs=output_tensor, name='MLP')

        if self.verbose:
            import io

            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            logging.debug(f'MLP classifier summary:\n{stream.getvalue()}')

        return model

    def train(
        self,
        x_old: np.ndarray,
        y_old: np.ndarray,
        *,
        test_size: float = 0.2,
        _validation_split: float = 0.1,
        lr: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        loss: str = 'categorical_crossentropy',
        class_weight: dict | None = None,
        sample_weight: np.ndarray | None = None,
        train_val_split: bool = True,
        retrain: bool = True,
    ) -> float:
        """
        Train the MLP classifier on historical data.

        Args:
            x_old: Feature vectors for the old samples.
            y_old: Groundtruth labels for the old samples.
            test_size: Proportion of the dataset to include in the test split.
            _validation_split: Deprecated.
            lr: Learning rate for the Adam optimizer.
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train the model.
            loss: Name of the loss function.
            class_weight: Optional dictionary mapping class indices to weights.
            sample_weight: Optional Numpy array of weights for the training samples.
            train_val_split: Whether to split data into training and validation sets.
            retrain: Whether to train a new model or load a saved one.

        Returns:
            The classifier's accuracy on the validation or training set.

        Raises:
            AttributeError: If the loaded model is missing expected Keras methods.
        """
        if train_val_split:
            x_train, x_val, y_train, y_val = train_test_split(
                x_old, y_old, test_size=test_size, random_state=42, shuffle=True
            )

            y_train_onehot = to_categorical(y_train)
            y_val_onehot = to_categorical(y_val)

            if retrain:
                model = self.build()
                model.compile(
                    loss=loss, optimizer=Adam(learning_rate=lr), metrics=['accuracy']
                )

                utils.create_parent_folder(self.model_save_name)
                mcp_save = ModelCheckpoint(
                    self.model_save_name,
                    monitor='val_acc',
                    save_best_only=True,
                    mode='max',
                )

                callbacks: list[Callback] = [mcp_save]
                if self.verbose:
                    callbacks.append(LoggingCallback(logging.debug))

                history = model.fit(
                    x_train,
                    y_train_onehot,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val_onehot),
                    verbose=str(self.verbose),
                    class_weight=class_weight,
                    sample_weight=sample_weight,
                    callbacks=callbacks,
                )
                val_acc = np.max(
                    history.history.get(
                        'val_acc', history.history.get('val_accuracy', 0)
                    )
                )

                plt.figure()
                plt.plot(history.history['loss'], '-b', label='Training')
                if 'val_loss' in history.history:
                    plt.plot(history.history['val_loss'], '--r', label='Testing')
                plt.legend()
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.savefig(
                    f'{self.model_save_name}_lr{lr}_epoch{epochs}_loss.png', dpi=200
                )
                plt.close()

            k.clear_session()
            clf = load_model(self.model_save_name)

            if isinstance(clf, Model):
                val_score = clf.evaluate(x_val, y_val_onehot)
                val_acc = val_score[1]

                metric_name = clf.metrics_names[1]
                logging.info(f'MLP validation {metric_name}: {val_acc * 100:.2f}%')
            else:
                raise AttributeError(
                    f'Expected a Keras Model, but loaded {type(clf)}. '
                    'Check if the model path is correct or if the file is corrupted.'
                )

        else:
            y_old_onehot = to_categorical(y_old)
            if retrain:
                model = self.build()
                model.compile(
                    loss=loss, optimizer=Adam(learning_rate=lr), metrics=['accuracy']
                )
                utils.create_parent_folder(self.model_save_name)

                model.fit(
                    x_old,
                    y_old_onehot,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=str(self.verbose),
                    callbacks=[LoggingCallback(logging.debug)] if self.verbose else [],
                )

            k.clear_session()
            clf = load_model(self.model_save_name)
            if isinstance(clf, Model):
                val_score = clf.evaluate(x_old, y_old_onehot)
                val_acc = val_score[1]
            else:
                raise AttributeError(
                    f'Loaded model is invalid. Expected Keras Model, got {type(clf)}'
                )

        return float(val_acc)

    def predict(
        self,
        x_new: np.ndarray,
        y_new: np.ndarray,
        dataset_name: str,
        newfamily: int,
        saved_cm_fig_path: str,
    ) -> tuple[np.ndarray, float]:
        """
        Predict labels for new data using the saved MLP model and evaluate performance.

        Args:
            x_new: Feature vectors for the new samples.
            y_new: Groundtruth labels for the new samples.
            dataset_name: Name of the dataset (used for plot titles).
            newfamily: Label index representing the 'unseen' or new family.
            saved_cm_fig_path: File path where the confusion matrix plot will be saved.

        Returns:
            A tuple containing:
                - y_pred (np.ndarray): The predicted class labels.
                - new_acc (float): The accuracy score on the new data.

        Raises:
            TypeError: If the loaded model is not a valid Keras Model instance.
        """
        k.clear_session()
        clf = load_model(self.model_save_name)
        if not isinstance(clf, Model):
            raise TypeError(
                f'Loaded model is invalid. Expected Keras Model, got {type(clf)}'
            )
        y_pred = np.argmax(clf.predict(x_new), axis=1)
        new_acc = float(accuracy_score(y_new, y_pred))
        cm = confusion_matrix(y_new, y_pred)

        logging.info(f'MLP testing set {clf.metrics_names[1]}: {new_acc * 100:.2f}%')
        logging.info(f'MLP confusion matrix: \n {cm}')

        utils.plot_confusion_matrix(
            cm, y_pred, y_new, dataset_name, newfamily, saved_cm_fig_path
        )

        return y_pred, new_acc


class RFClassifier:
    """RandomForest classifier wrapper.
    It internally supports multi-class classification. So don't need to one-hot encode the labels.
    """  # noqa: E501

    def __init__(self, rf_save_path: str, tree: int = 100) -> None:
        self.rf_save_path = rf_save_path
        self.tree = tree

    def fit_and_predict(
        self,
        x_old: np.ndarray,
        y_old: np.ndarray,
        x_new: np.ndarray,
        y_new: np.ndarray,
        dataset_name: str,
        newfamily: int,
        saved_cm_fig_path: str,
        *,
        retrain: bool,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, float, float]:
        """
        Train a Random Forest on old data and evaluate on both old and new data.

        Args:
            x_old: Feature vectors for the historical samples.
            y_old: Groundtruth labels for the historical samples.
            x_new: Feature vectors for the new/unseen samples.
            y_new: Groundtruth labels for the new/unseen samples.
            dataset_name: Name of the dataset for plotting logic.
            newfamily: Label index for the 'new family' class.
            saved_cm_fig_path: Path to save the confusion matrix visualization.
            retrain: If True, fits a new model; otherwise, loads from disk.
            test_size: Proportion of old data to use for testing.

        Returns:
            A tuple containing:
                - y_new_pred (np.ndarray): Predictions for the new samples.
                - test_acc (float): Accuracy on the old data test split.
                - new_acc (float): Accuracy on the new samples.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x_old, y_old, test_size=test_size, random_state=42, shuffle=True
        )

        if retrain:
            model = RandomForestClassifier(n_estimators=self.tree, random_state=0)
            model.fit(x_train, y_train)
            utils.create_parent_folder(self.rf_save_path)
            with open(self.rf_save_path, 'wb') as f:
                pickle.dump(model, f)

        with open(self.rf_save_path, 'rb') as f:
            model = pickle.load(f)

        y_test_pred = model.predict(x_test)
        test_acc = float(accuracy_score(y_test, y_test_pred))
        logging.info(f'RF test samples acc: {test_acc * 100:.2f}%')

        y_new_pred = model.predict(x_new)
        logging.debug(f'y_new_pred: {y_new_pred}')
        new_acc = float(accuracy_score(y_new, y_new_pred))
        logging.info(f'RF new samples acc: {new_acc * 100:.2f}%')

        cm = confusion_matrix(y_new, y_new_pred)
        utils.plot_confusion_matrix(
            cm, y_new_pred, y_new, dataset_name, newfamily, saved_cm_fig_path
        )

        return y_new_pred, test_acc, new_acc
