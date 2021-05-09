import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
from training import utils
import cv2 as cv

model = tf.keras.models.load_model('inception_model_aug_80K.h5', compile=False)

img = cv.imread('sddsf.jpeg', cv.IMREAD_GRAYSCALE)

img_a = utils.process_image_prediction(img, verbose=True)

preds = model.predict(img_a)

pred_texts = utils.decode_batch_predictions(preds)

print(pred_texts)