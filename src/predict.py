import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from training import data_handler, image_processing
import cv2 as cv
import time
import argparse


def main():

    parser = argparse.ArgumentParser(
        description="""  """,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=str, help="input image for prediction.")
    parser.add_argument(
        "-m",
        "--model",
        default="inception_model_aug_80K.h5",
        help="path to the prediction model, this will change the defualt model."
    )
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="show the image, and the prediction time.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    img = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
    img_a = image_processing.process_image_prediction(img,
                                                      verbose=args.verbose)

    start = time.time()
    preds = model.predict(img_a)
    end = time.time()

    pred_texts = data_handler.decode_batch_predictions(preds)

    print(f"The detected word is: {pred_texts}")
    if args.verbose:
        print(f"The time it took to predict: {end-start:.2f}s")


if __name__ == "__main__":
    main()