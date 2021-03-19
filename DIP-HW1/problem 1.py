from Helping import *


def part_1(image: np.ndarray):
    return image, 'Original'


def part_A(image: np.ndarray):
    log_img = image - image.min()
    c = c_log(log_img)
    log_img += 1
    log_img = c * (np.log10(log_img))
    return log_img, 'Log Transformation'


def part_B(image: np.ndarray):
    invLog_img = image - image.min()
    c = c_inverse_log(invLog_img)
    invLog_img = (10 ** (c * invLog_img)) - 1
    return invLog_img, 'Inverse Log Transformation'


def part_C(image: np.ndarray):
    # noinspection PyArgumentList
    linear_image = image - image.min()
    c = c_linear(linear_image)
    linear_image *= c
    return linear_image, 'Linear Transformation'


def measure(method, image):
    return_img, title = method(image)
    return_img = return_img.round().astype('uint8')
    print(f'{title}:')
    print(f"The min is : {int(return_img.min())}")
    print(f"The max is : {int(return_img.max())}")
    print(f"The mean is : {return_img.mean().round(5)}")
    print(f"The std is : {return_img.std().round(5)}\n")
    cv2.imwrite(f'out/{title}.png', return_img)
    cv2.imshow('image', return_img.astype('uint8'))
    cv2.setWindowTitle('image', title)
    cv2.waitKey()


F16_PATH = 'res/f16.gif'
f16_image = read_GIF(F16_PATH)

measure(part_1, f16_image.copy())
measure(part_A, f16_image.copy())
measure(part_B, f16_image.copy())
measure(part_C, f16_image.copy())
