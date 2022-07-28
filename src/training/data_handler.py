import numpy as np
import cv2 as cv
import image_processing
import tensorflow as tf
import string
import re

charset_base = " " + string.printable[:93]

encode = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=list(charset_base), num_oov_indices=0, mask_token=None)

decode = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=encode.get_vocabulary(), mask_token=None, invert=True)


def import_dataset(textpath):

    print("LOG:  Loading datafrom database ... ")

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


def extract_data_labels(data, path, num_data=0, img_width=128, img_height=32):

    loaded_data = []

    print("LOG:  Processing data ... ")

    if num_data == 0:
        num_data = len(data)

    label = np.array(data)[:num_data, 1]

    label = [l.replace(" ", "") for l in label]

    for i in range(num_data):

        path_img = path + data[i][0]

        img = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
        img_aug = image_processing.prepare_image(img, img_width, img_height)
        loaded_data.append(np.array(img_aug))

    print("LOG:  Data was processed")
    return np.array(loaded_data), label


def encode_labels(label):

    labels = [
        encode(tf.strings.unicode_split(y, input_encoding="UTF-8"))
        for y in label
    ]

    labels = tf.keras.preprocessing.sequence.pad_sequences(labels,
                                                           padding="post")
    return labels


def decode_batch_predictions(pred, max_length=16):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = tf.keras.backend.ctc_decode(pred,
                                          input_length=input_len,
                                          greedy=True)[0][0][:, :max_length]

    decoded = []

    for res in results:
        res = tf.strings.reduce_join(decode(res + 1)).numpy().decode("utf-8")
        decoded.append(res)

    output = decoded[0]
    output = re.sub(r"\[UNK\]", "", output)

    return output
