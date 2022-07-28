import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from training import utils, models
from tensorflow import keras
from tensorflow.keras.utils import plot_model

import argparse

NUM_LABELS = 400

IMG_WIDTH = 128

IMG_HEIGHT = 32


def main():

    parser = argparse.ArgumentParser(
        description=""" """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "data",
        type=str,
        help=
        "The path to the data folder, on of the mentioned above datasets, where the structure of the directory is left as is. Note that a generic method data importing, might be introduced in furture versions."
    )

    parser.add_argument(
        "-d",
        "--dataset",
        default="IAM",
        type=str,
        choices={"IAM"},
        help="The dataset names inputted, so far, IAM is only used.")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model.h5",
        help="The and the name of the model to be saved. it should end with .h5"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        choices={0, 1, 2},
        help=
        "Verbose levels, where 0 is no verbose, 1 shows the tensorflow training logs, and 2 adds the model summary, and saves a plotted image of the model."
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices={"conv_cnn", "inception"},
        default="inception",
        help=
        "choose the model structure, so far there is only a typical cnn, and an inception inspired model."
    )

    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=32,
                        help="The batch size for training.")

    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=50,
                        help="The epochs number for training.")

    parser.add_argument(
        "-n",
        "--data_num",
        type=int,
        default=0,
        help=
        "Number of datapoint to be used from the training, default is 0 where all the dataset is used."
    )

    parser.add_argument("-p",
                        "--patience",
                        type=int,
                        default=8,
                        help="Patience number for earlystopping callback.")

    args = parser.parse_args()

    batch_size = args.batch_size

    epochs = args.epochs

    if args.dataset == "IAM":
        datafiles = utils.import_dataset(args.data + '.txt')
        data, labels = utils.extract_data_labels(datafiles,
                                                 args.data,
                                                 num_data=args.data_num)

    labels = utils.prepare_labels(labels)

    print("LOG:  Dataset was loaded with size of", data.shape)
    data = data.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)

    model_type = {
        "conv_cnn": models.Conventional_CNN(),
        "inception": models.inception_model()
    }

    model = models.build_model(model_type[args.type])

    if args.verbose == 2:
        model.summary()

    patience = args.patience

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience,
                                                   restore_best_weights=True)

    verbose = 1 if args.verbose == 2 else args.verbose

    model.fit(x=data,
              y=labels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_split=0.1,
              callbacks=[early_stopping],
              shuffle=True)

    model.save(args.output)


if __name__ == "__main__":
    main()
