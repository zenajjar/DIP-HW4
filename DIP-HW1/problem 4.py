from Helping import *
from matplotlib import pyplot as plt

L = 256


def hist_eq(image: np.ndarray):
    height, width = image.shape
    values, freq = np.unique(image, return_counts=True)
    cum_sum = np.cumsum(freq)
    # maps each value to its new value depending on its cum_sum
    replace = dict(zip(values, (cum_sum * ((L - 1) / (height * width)))))
    image = np.vectorize(replace.get)(image)
    return image


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
