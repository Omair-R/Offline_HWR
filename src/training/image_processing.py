import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def data_augmentation(img, seed: int = 42, probability: float = 0.3):

    randomizer = np.random.default_rng(seed)

    if randomizer.random() < probability:
        img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=1)

    if randomizer.random() < probability:
        img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=1)

    if randomizer.random() < probability:
        gauss = np.random.normal(0.2, 0.5, img.size)
        gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
        img = img + img * gauss

    if randomizer.random() < probability:
        img = np.clip(
            img +
            (np.random.random(img.shape) - 0.5) * randomizer.integers(1, 50),
            0, 255)

    return img


def prepare_image(img,
                  img_width=128,
                  img_height=32,
                  sharpen=True,
                  augment=True,
                  seed=42):

    if img is None:
        raise Exception()

    randomizer = np.random.default_rng(seed)

    if sharpen:
        img = cv.GaussianBlur(img, (3, 3), 0)

        # img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                            cv.THRESH_BINARY, 11, 2)

        _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=1)

    if augment:
        img = data_augmentation(img)

    img = img.astype(np.float)

    f = min(img_width / img.shape[1], img_height / img.shape[0])

    x = (img_width / 2 - img.shape[1] * f / 2)
    y = (img_height / 2 - img.shape[0] * f / 2)

    M = np.float32([[f, 0, x], [0, f, y]])

    if augment:
        M = np.float32([[f, 0,
                         randomizer.integers(0, int(f * img_width))],
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
