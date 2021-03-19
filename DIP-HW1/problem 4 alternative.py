from math import floor

from Helping import *
from matplotlib import pyplot as plt

L = 256


def hist_eq(image: np.ndarray):
    height, width = image.shape
    freq = get_hist(image)
    for y in range(height):
        for x in range(width):
            image[y, x] = floor(((L - 1) / (height * width)) * freq[int(image[y, x])])
    return image


def get_hist(block: np.ndarray):
    height, width = block.shape
    freq = [0] * L

    for y in range(height):
        for x in range(width):
            freq[int(block[y, x])] += 1

    for i in range(1, len(freq)):
        freq[i] += freq[i - 1]

    return freq


def process(image: np.ndarray, label: str):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()
    hist_eq_image = hist_eq(image).astype('uint8')
    cv2.imwrite(f'out/{label}_equalized.png', hist_eq_image)
    cv2.imshow(label, hist_eq_image)
    cv2.waitKey()
    cv2.destroyWindow(label)
    plt.hist(hist_eq_image.ravel(), 256, [0, 256])
    plt.show()
    print(f'absolute difference in mean in {label}:', abs(image.mean() - hist_eq_image.mean()).round(2))


clock = read_GIF("res/clock.gif")
moon = read_GIF("res/moonOriginal.gif")
toys = read_GIF("res/toys.GIF")

process(clock, 'clock')
process(moon, 'moon')
process(toys, 'toys')
