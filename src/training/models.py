import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from tensorflow.keras import layers

import utils

charset_base = utils.charset_base

def Conventional_CNN(img_width=128, img_height=32):

    input_img = layers.Input(shape=(img_width, img_height, 1),
                             name="image",
                             dtype="float32")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    shape = x.get_shape()
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    output = layers.Dense(len(charset_base) + 1, activation="softmax")(x)

    return keras.models.Model(inputs=input_img, outputs=output)


def inception_model(img_width=128, img_height=32):

    input_img = layers.Input(shape=(img_width, img_height, 1),
                             name="image",
                             dtype="float32")

    x1 = layers.Conv2D(32, (1, 1), activation="relu",
                       padding="same")(input_img)

    x2 = layers.Conv2D(32, (1, 1), activation="relu",
                       padding="same")(input_img)

    x2 = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
    )(x2)

    x3 = layers.Conv2D(16, (1, 1), activation="relu",
                       padding="same")(input_img)

    x3 = layers.Conv2D(
        32,
        (5, 5),
        activation="relu",
        padding="same",
    )(x3)

    x4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)

    x4 = layers.Conv2D(
        32,
        (5, 5),
        activation="relu",
        padding="same",
    )(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    x1 = layers.Conv2D(
        64,
        (1, 1),
        activation="relu",
        padding="same",
    )(x)

    x2 = layers.Conv2D(64, (1, 1), activation="relu", padding="same")(x)
    x2 = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
    )(x2)

    x3 = layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x)
    x3 = layers.Conv2D(
        64,
        (5, 5),
        activation="relu",
        padding="same",
    )(x3)

    x4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    x4 = layers.Conv2D(
        64,
        (5, 5),
        activation="relu",
        padding="same",
    )(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    shape = x.get_shape()
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    output = layers.Dense(len(charset_base) + 1, activation="softmax")(x)

    return keras.models.Model(inputs=input_img, outputs=output)


def build_model(model_choice):

    model = model_choice

    opt = keras.optimizers.Adam()

    model.compile(optimizer=opt, loss=utils.ctc_loss_lambda_func)
    return model