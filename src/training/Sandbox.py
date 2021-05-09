import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

img = cv.imread('sddsf.jpeg', cv.IMREAD_GRAYSCALE)

img_width = 128

img_height = 32


def data_augmentation(img):

    if random.random() < 0.3:
        img = img * (0.5 + random.uniform(0, 0.4))

    if random.random() < 0.3:
        img = cv.dilate(img, np.ones((2, 2), np.uint8), iterations=1)

    if random.random() < 0.3:
        img = cv.erode(img, np.ones((2, 2), np.uint8), iterations=1)

    if random.random() < 0.3:
        rand = random.randint(2, 4)
        img = cv.GaussianBlur(img, (5, 5), rand)

    if random.random() < 0.9:
        gauss = np.random.normal(0.2, 0.5, img.size)
        gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
        img = img + img * gauss

    return img


def prepare_image(img, img_width=128, img_height=32, sharpen=False):

    if img is None:
        img = np.zeros((img_height, img_width))

    if sharpen:
        img = cv.GaussianBlur(img, (5, 5), 0)

        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 11, 2)

        img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=1)

    img = img.astype(np.float)

    f = min(img_width / img.shape[1], img_height / img.shape[0])

    x = (img_width / 2 - img.shape[1] * f / 2)
    y = (img_height / 2 - img.shape[0] * f / 2)

    M = np.float32([[f, 0, x], [0, f, y]])

    dst = np.zeros((img_height, img_width))
    dst.fill(255)

    img = cv.warpAffine(img,
                        M,
                        dsize=(img_width, img_height),
                        dst=dst,
                        borderMode=cv.BORDER_TRANSPARENT)

    img = cv.transpose(img)

    img = img / 255 - 0.5
    return img


img = cv.imread('a04-010-00-01.png', cv.IMREAD_GRAYSCALE)

# img_aug = np.array(prepare_image(img, 128, 32))
img = data_augmentation(img)
img_aug = np.array(prepare_image(img, 128, 32))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(cv.transpose(img_aug), cmap='gray')
plt.show()