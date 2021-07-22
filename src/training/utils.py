import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2 as cv

import matplotlib.pyplot as plt
import numpy as np
import string
import re
import random

charset_base = " " + string.printable[:93]

encode = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(charset_base), num_oov_indices=0, mask_token=None)

decode = layers.experimental.preprocessing.StringLookup(
    vocabulary=encode.get_vocabulary(), mask_token=None, invert=True)


def data_augmentation(img):

 
    if random.random() < 0.3:
            img = img * (0.25 + random.random() * 0.75)

    if random.random() < 0.3:
        img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=1)

    if random.random() < 0.3:
        img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=1)

    if random.random() < 0.3:
        rand = random.randint(4, 8)
        img = cv.GaussianBlur(img, (5, 5), rand)

    if random.random() < 0.3:
        gauss = np.random.normal(0.2, 0.5, img.size)
        gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
        img = img + img * gauss

    if random.random() < 0.3:
        img = np.clip(
            img +
            (np.random.random(img.shape) - 0.5) * random.randint(1, 50), 0,
            255)

    return img


def prepare_image(img,
                  img_width=128,
                  img_height=32,
                  sharpen=False,
                  augment=False):

    if img is None:
        img = np.zeros((img_height, img_width))

    if sharpen:
        img = cv.GaussianBlur(img, (5, 5), 0)

        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 7, 2)

        img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=1)

    if augment:
        img = data_augmentation(img)

    img = img.astype(np.float)

    f = min(img_width / img.shape[1], img_height / img.shape[0])

    x = (img_width / 2 - img.shape[1] * f / 2)
    y = (img_height / 2 - img.shape[0] * f / 2)

    M = np.float32([[f, 0, x],
                     [0, f, y]])

    if augment:
        M = M = np.float32([[f, 0, random.randint(0, int(f * img_width))],
                            [0, f, y]])

    dst = np.zeros((img_height, img_width))

    img = cv.warpAffine(img,
                        M,
                        dsize=(img_width, img_height),
                        dst=dst,
                        borderMode=cv.BORDER_CONSTANT,
                        borderValue=200)

    img = cv.transpose(img)

    img = img / 255 - 0.5
    return img


def extract_data_labels(data, path, num_data=0, img_width=128, img_height=32):

    loaded_data = []

    print("LOG:  Loading datafrom database ... ")

    if num_data == 0:
        num_data = len(data)

    label = np.array(data)[:num_data, 1]

    label = [l.replace(" ", "") for l in label]

    for i in range(num_data):

        path_img = path + data[i][0]

        img = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
        img_aug = prepare_image(img, img_width, img_height)
        loaded_data.append(np.array(img_aug))

    print("LOG:  Loading is over.")
    return np.array(loaded_data), label


def prepare_labels(label):

    labels = [
        encode(tf.strings.unicode_split(y, input_encoding="UTF-8"))
        for y in label
    ]

    labels = tf.keras.preprocessing.sequence.pad_sequences(labels,
                                                           padding="post")
    return labels


def import_dataset(textpath):
    f = open(textpath)

    data = []
    fileName = ''
    label = ''
    for line in f:
        if line[0] == '#' or not line:
            continue

        separatedline = line.strip().split(' ')

        if separatedline[0] == 'a01-117-05-02' or separatedline[
                0] == 'r06-022-03-05':
            continue

        separatedFileName = separatedline[0].split('-')

        fileName = '/' + separatedFileName[0] + '/' + '{}-{}'.format(
            separatedFileName[0],
            separatedFileName[1]) + '/' + separatedline[0] + '.png'

        label = ' '.join(separatedline[8:])

        data.append((fileName, label))

    return data


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


def decode_batch_predictions(pred, max_length=16):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = keras.backend.ctc_decode(pred,
                                       input_length=input_len,
                                       greedy=True)[0][0][:, :max_length]

    decoded = []

    for res in results:
        res = tf.strings.reduce_join(decode(res)).numpy().decode("utf-8")
        decoded.append(res)

    output = decoded[0]
    output = re.sub(r"\[UNK\]", "", output)

    return output


def process_image_prediction(img, img_height=32, img_width=128, verbose=False):

    # img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    img_aug = np.array(prepare_image(img, img_width, img_height))

    if verbose:
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.subplot(122)
        plt.imshow(cv.transpose(img_aug), cmap='gray')
        plt.show()

    img_a = img_aug.reshape(-1, img_width, img_height, 1)

    return img_a


