from Helping import *

ORIGINAL_IMAG_PATH = 'res/Peppers.tiff'


def rgb_equalize(image):
    blue, green, red = cv2.split(image)

    red_equalized = cv2.equalizeHist(red)
    blue_equalized = cv2.equalizeHist(blue)
    green_equalized = cv2.equalizeHist(green)

    return cv2.merge([blue_equalized, green_equalized, red_equalized])


def hsv_equalize(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_img)
    value = cv2.equalizeHist(value)
    hsv_img = cv2.merge([hue, saturation, value])
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


original_image = cv2.imread(ORIGINAL_IMAG_PATH)

show_image_and_wait(original_image)

rgb_equalized_image = rgb_equalize(original_image)
show_image_and_wait(rgb_equalized_image)
cv2.imwrite('out/rgb_equalized.png', rgb_equalized_image)

hsv_equalized_image = hsv_equalize(original_image)
show_image_and_wait(hsv_equalized_image)
cv2.imwrite('out/hsv_equalized.png', hsv_equalized_image)
