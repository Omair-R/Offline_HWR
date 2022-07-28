import os
import tensorflow as tf
import data_handler

charset_base_size = len(data_handler.charset_base)


def Conventional_CNN(img_width=128, img_height=32):

    input_img = tf.keras.layers.Input(shape=(img_width, img_height, 1),
                                      name="image",
                                      dtype="float32")

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                               padding="same")(input_img)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                               padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    shape = x.get_shape()
    x = tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    output = tf.keras.layers.Dense(charset_base_size + 1,
                                   activation="softmax")(x)

    return tf.keras.models.Model(inputs=input_img, outputs=output)


def inception_model(img_width=128, img_height=32):

    input_img = tf.keras.layers.Input(shape=(img_width, img_height, 1),
                                      name="image",
                                      dtype="float32")

    x1 = tf.keras.layers.Conv2D(32, (1, 1), activation="relu",
                                padding="same")(input_img)

    x2 = tf.keras.layers.Conv2D(32, (1, 1), activation="relu",
                                padding="same")(input_img)

    x2 = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
    )(x2)

    x3 = tf.keras.layers.Conv2D(16, (1, 1), activation="relu",
                                padding="same")(input_img)

    x3 = tf.keras.layers.Conv2D(
        32,
        (5, 5),
        activation="relu",
        padding="same",
    )(x3)

    x4 = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                      padding='same')(input_img)

    x4 = tf.keras.layers.Conv2D(
        32,
        (5, 5),
        activation="relu",
        padding="same",
    )(x4)

    x = tf.keras.layers.concatenate([x1, x2, x3, x4], axis=-1)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                               padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Conv2D(
        64,
        (1, 1),
        activation="relu",
        padding="same",
    )(x)

    x2 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu",
                                padding="same")(x)
    x2 = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
    )(x2)

    x3 = tf.keras.layers.Conv2D(32, (1, 1), activation="relu",
                                padding="same")(x)
    x3 = tf.keras.layers.Conv2D(
        64,
        (5, 5),
        activation="relu",
        padding="same",
    )(x3)

    x4 = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                      padding='same')(x)
    x4 = tf.keras.layers.Conv2D(
        64,
        (5, 5),
        activation="relu",
        padding="same",
    )(x4)

    x = tf.keras.layers.concatenate([x1, x2, x3, x4], axis=-1)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu",
                               padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shape = x.get_shape()
    x = tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    x = tf.keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    output = tf.keras.layers.Dense(charset_base_size + 1,
                                   activation="softmax")(x)

    return tf.keras.models.Model(inputs=input_img, outputs=output)


def build_and_compile_model(model_choice):

    model = model_choice

    opt = tf.keras.optimizers.Adam()

    model.compile(optimizer=opt, loss=ctc_loss_lambda_func)
    return model


def ctc_loss_lambda_func(y_true, y_pred):

    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true)

    input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

    label_length = tf.math.count_nonzero(y_true,
                                         axis=-1,
                                         keepdims=True,
                                         dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length,
                                           label_length)
    loss = tf.reduce_mean(loss)

    return loss