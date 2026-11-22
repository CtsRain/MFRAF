import numpy as np
import pandas as pd
import tensorflow as tf


def binary_focal_loss(alpha=0.25, gamma=2.0):
    """
    binary classification Focal Loss
    alpha: positive sample weight
    gamma: hard sample focused parameter
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.reshape(y_true, (-1, 1))

        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        focal = alpha_t * tf.pow(1.0 - p_t, gamma) * bce

        return tf.reduce_mean(focal)

    return loss

def get_loss_function(loss_mode='bce', focal_alpha=0.25, focal_gamma=2.0):
    """
    return the loss function according to loss_mode
    loss_mode:
        'bce'            : binary_crossentropy
        'focal'          : Focal Loss
    """

    if loss_mode == 'bce':
        return 'binary_crossentropy'

    elif loss_mode == 'focal':
        return binary_focal_loss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )

    else:
        raise ValueError(
            "loss_mode can only be 'bce', 'focal', 'ranking', 'focal_ranking'"
        )


""" balanced batch """
class BalancedBatchGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        X,
        y,
        batch_size=64,
        pos_fraction=0.5,
        steps_per_epoch=None,
        shuffle=True,
        seed=42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seed = seed
        np.random.seed(self.seed)

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y).astype(np.float32)

        self.batch_size = batch_size
        self.pos_fraction = pos_fraction
        self.shuffle = shuffle

        self.pos_idx = np.where(self.y == 1)[0]
        self.neg_idx = np.where(self.y == 0)[0]

        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError("The number of positive or negative samples is 0, so a balanced batch cannot be used")

        if steps_per_epoch is None:
            self.steps_per_epoch = int(np.ceil(len(self.y) / batch_size))
        else:
            self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        n_pos = int(self.batch_size * self.pos_fraction)
        n_neg = self.batch_size - n_pos

        batch_pos_idx = np.random.choice(
            self.pos_idx,
            size=n_pos,
            replace=True
        )

        batch_neg_idx = np.random.choice(
            self.neg_idx,
            size=n_neg,
            replace=True
        )

        batch_idx = np.concatenate([batch_pos_idx, batch_neg_idx])

        if self.shuffle:
            np.random.shuffle(batch_idx)

        X_batch = self.X[batch_idx].astype(np.float32)
        y_batch = self.y[batch_idx].astype(np.float32)

        return X_batch, y_batch