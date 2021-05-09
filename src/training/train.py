import utils
import models
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2 as cv

num_labels = 400

img_width = 128

img_height = 32

batch_size = 16

epochs = 50

datafiles = utils.import_dataset('src/words.txt')

data, labels = utils.extract_data_labels(datafiles,
                                         'src/words',
                                         num_data=num_labels)

max_length = max([len(l) for l in labels])

labels = utils.prepare_labels(labels)

print("LOG:  Dataset was loaded with size of", data.shape)

data = data.reshape(-1, img_width, img_height, 1)

print("LOG:  Dataset was loaded with size of", data.shape)

model = models.build_model(models.inception_model())
# model.summary()

from tensorflow.keras.utils import plot_model
import graphviz
import pydot_ng as pydot

plot_model(model, show_shapes=True, to_file='my_model.png')

patience = 3

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=patience,
                                               restore_best_weights=True)

model.fit(x=data,
          y=labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1,
          callbacks=[early_stopping],
          shuffle=True)

model.save("inception_model_aug_80K.h5")
