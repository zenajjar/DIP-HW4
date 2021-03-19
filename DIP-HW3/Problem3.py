from Helping import *

ORIGINAL_IMAGE_PATH = 'res/Truck.jpg'


def adjust_image_to_RGB(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float64')
    image /= 255
    return image


def adjust_image_to_BGR(image):
    return cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_RGB2BGR)


def find_color_distance(image, color):
    distance = image - color
    distance **= 2
    distance = distance.sum(axis=2)
    distance = distance ** 0.5
    distance = np.repeat(distance, 3)
    distance = distance.reshape(image.shape)

    return distance


def replace_color_with(image, center, radius, color):
    distance = find_color_distance(image, center)
    color_replaced_image = np.full_like(image, color)
    return np.where(distance <= radius, color_replaced_image, image)


original_image = cv2.imread(ORIGINAL_IMAGE_PATH)
original_image = adjust_image_to_RGB(original_image)

replace_color = [0.2, 0.3, 0.6]

center1 = [179 / 255, 40 / 255, 50 / 255]
radius1 = 0.25
sliced_image = replace_color_with(original_image, center1, radius1, replace_color)

center2 = [225 / 255, 89 / 255, 103 / 255]
radius2 = 0.2
sliced_image = replace_color_with(sliced_image, center2, radius2, replace_color)
#
center3 = [100 / 255, 20 / 255, 30 / 255]
radius3 = 0.1
sliced_image = replace_color_with(sliced_image, center3, radius3, replace_color)

show_image_and_wait(adjust_image_to_BGR(sliced_image))
cv2.imwrite('out/sliced.png', adjust_image_to_BGR(sliced_image))
