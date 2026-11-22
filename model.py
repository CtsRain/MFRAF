import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Concatenate,
    Add,
    Reshape,
    GlobalAveragePooling1D,
    Multiply,
    Lambda,
    Activation,
    GlobalMaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from function import *


# =========================================================
# 1. improved SE
# =========================================================
def squeeze_excitation_1d(input_tensor, reduction_ratio=4):
    channels = int(input_tensor.shape[-1])
    avg_pool = GlobalAveragePooling1D()(input_tensor)
    max_pool = GlobalMaxPooling1D()(input_tensor)
    squeeze = Concatenate(axis=-1)([avg_pool, max_pool])

    excitation = Dense(units=max(1, channels // reduction_ratio), activation='relu')(squeeze)

    excitation = Dense(units=channels, activation='sigmoid')(excitation)

    excitation = Reshape((1, channels))(excitation)

    scaled_output = Multiply()([input_tensor, excitation])

    return scaled_output

# =========================================================
# 2. Two-Stage Iterative Fusion Mechanism (TSIFM)
# =========================================================
def ms_cam_3branch(x1, x2, x3, channels, reduction_ratio=4, name='mscam_3branch'):
    hidden = max(1, channels // reduction_ratio)

    merged = Concatenate(axis=-1, name=name + '_concat')([x1, x2, x3])

    local_att = Conv1D(filters=hidden, kernel_size=1, padding='same', activation='relu',)(merged)

    local_att = Conv1D( filters=3 * channels, kernel_size=1, padding='same', activation=None,)(local_att)

    local_att = Reshape((1, 3, channels))(local_att)

    global_att = Flatten()(merged)

    global_att = Dense(units=hidden, activation='relu',)(global_att)

    global_att = Dense(units=3 * channels, activation=None,)(global_att)

    global_att = Reshape((1, 3, channels),)(global_att)

    att = Add()([local_att, global_att])

    weights = Lambda(lambda t: tf.nn.softmax(t, axis=2),)(att)

    return weights


def weighted_sum_3branch(x1, x2, x3, weights, name='weighted_sum_3branch'):
    stacked = Lambda(
        lambda tensors: tf.stack(tensors, axis=2),
        name=name + '_stack'
    )([x1, x2, x3])

    out = Lambda(
        lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=2),
        name=name + '_reduce_sum'
    )([stacked, weights])

    return out


def iaff_3branch(x1, x2, x3, channels, reduction_ratio=4, name='iaff_3branch'):
    w1 = ms_cam_3branch(x1, x2, x3, channels=channels, reduction_ratio=reduction_ratio,)

    xi = weighted_sum_3branch(x1, x2, x3, w1,)

    x1_refine = Add(name=name + '_x1_refine')([x1, xi])
    x2_refine = Add(name=name + '_x2_refine')([x2, xi])
    x3_refine = Add(name=name + '_x3_refine')([x3, xi])

    w2 = ms_cam_3branch(x1_refine, x2_refine, x3_refine, channels=channels, reduction_ratio=reduction_ratio,)

    out = weighted_sum_3branch(x1, x2, x3, w2,)

    return out


# =========================================================
# 3. Period-Aware Representation (PAR) and Coupled Recalibration Mechanism (CRM)
# =========================================================

def PAR_and_CRM(x_in, patch_len):
    """Period-Aware Representation"""
    x1 = Conv1D(filters=32, kernel_size=patch_len, strides=patch_len, padding='same', activation='relu',)(x_in)

    """Coupled Recalibration Mechanism"""
    x1 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu',)(x1)
    x1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',)(x1)

    x1 = squeeze_excitation_1d(x1, reduction_ratio=2,)

    x = Flatten()(x1)

    x = Dense(128, activation='relu',)(x)

    return x

# =========================================================
# 4. main model
# =========================================================

def Hybrid_Model(data_len, lr=0.001, fusion_mode='iaff', loss_mode='focal', focal_alpha=0.5, focal_gamma=0,):
    """
    fusion_mode optional:
        fusion_mode optional:
        'concat' : Concatenate
        'iaff'   : iAFF

    loss_mode optional:
        'bce'     : binary_crossentropy
        'focal'   : Focal Loss
    """

    # input layer
    inputs = Input(shape=(data_len,),)

    x_in = Reshape((data_len, 1),)(inputs)

    # patch length
    patch_len_1 = 7
    patch_len_2 = 14
    patch_len_3 = 28

    # three branches
    x1 = PAR_and_CRM(x_in=x_in, patch_len=patch_len_1,)

    x2 = PAR_and_CRM(x_in=x_in, patch_len=patch_len_2,)

    x3 = PAR_and_CRM(x_in=x_in, patch_len=patch_len_3,)

    # feature fusion
    if fusion_mode == 'concat':
        x = Concatenate()([x1, x2, x3])

    elif fusion_mode == 'iaff':
        x1_r = Reshape((1, 128))(x1)

        x2_r = Reshape((1, 128))(x2)

        x3_r = Reshape((1, 128))(x3)

        x = iaff_3branch(x1_r, x2_r, x3_r, channels=128, reduction_ratio=16)

        x = Flatten()(x)

    else:
        raise ValueError("fusion_mode can only be 'concat' or 'iaff'")

    # classification head
    x = Dense(64, activation='relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=pred)

    loss_fn = get_loss_function(
        loss_mode=loss_mode,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=loss_fn
    )

    return model