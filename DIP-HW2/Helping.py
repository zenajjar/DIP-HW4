import cv2
import numpy as np
from PIL import Image
from numpy import fft


def read_GIF(path: str):
    if path.split('.')[-1].lower() != 'gif':
        return cv2.imread(path)[:, :, 0].astype('float64')
    gif = cv2.VideoCapture(path)
    if gif is None:
        return None
    ret, frame = gif.read()
    img = Image.fromarray(frame)
    open_cv_image = np.array(img, dtype='float64')
    open_cv_image = open_cv_image[:, :, ::-1]
    return open_cv_image[:, :, 0]


def mean(images: np.ndarray):
    return images.mean(axis=0)


def median(images: np.ndarray):
    return np.median(images, axis=0)


def percentile(images: np.ndarray, percent: int):
    images.sort(axis=0)
    index1 = int(np.floor(percent * (images.shape[0] + 1) / 100))
    index2 = int(np.ceil(percent * (images.shape[0] + 1) / 100))
    return (images[index1 - 1] + images[index2 - 1]) / 2


def PSNR(image1: np.ndarray, image2: np.ndarray):
    height, width = image1.shape
    differenceImage = image2 - image1
    differenceImage **= 2
    difference = differenceImage.sum().sum()
    PSNR_ = 10 * np.log10(((255 ** 2) * height * width) / difference)
    return PSNR_


def c_log(image: np.ndarray):
    # noinspection PyArgumentList
    mx = image.max()
    return 255 / np.log10(1 + mx)


def c_inverse_log(image: np.ndarray):
    # noinspection PyArgumentList
    mx = image.max()
    return np.log10(1 + 255) / mx


def c_linear(image: np.ndarray):
    if image.max() == 0:
        return 1
    # noinspection PyArgumentList
    return 255 / image.max()


def get_shifted_image_magnitude(ft):
    scaled = log_scale_image(ft)
    shifted = fft.fftshift(scaled)
    return shifted


def log_scale_image(image):
    image = abs(image)
    log_image = np.log(image + 1)
    linear_image = linear_scale_image(log_image)
    return linear_image


def linear_scale_image(image):
    if image.min() != image.max():
        image -= image.min()
    image = c_linear(image) * image
    return image


def show_image_and_wait(image, label='image'):
    cv2.imshow(label, image.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_middle(image: np.ndarray):
    height, width = image.shape[0:2]
    return height / 2, width / 2


def get_dft_components(ft: np.ndarray):
    shifted_magnitude = fft.fftshift(np.abs(ft))
    phase = np.arctan2(ft.imag, ft.real)

    return shifted_magnitude, phase


def zero_pad_image(image: np.ndarray):
    height, width = image.shape[:2]
    padded_image = np.pad(image, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
    return padded_image


def remove_zero_padding(image: np.ndarray):
    height, width = image.shape[:2]
    return image[: height // 2, :width // 2]


def get_inverse_ft(shifted_magnitude, phase):
    original_magnitude = fft.fftshift(shifted_magnitude)
    ift_image = np.real(fft.ifft2(original_magnitude * np.exp(1j * phase)))
    return ift_image
