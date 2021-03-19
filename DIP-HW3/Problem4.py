from Helping import *

ORIGINAL_IMAGE_PATH = 'res/Jet.gif'
NOISY_IMAGE_PATH = 'res/JetNoisy.gif'

mask_size = 5


def evaluate_filter(filtered_image, filter_name):
    show_image_and_wait(filtered_image, filter_name)
    cv2.imwrite(f'out/{filter_name} filtered.png', filtered_image)
    print(f'PSNR for the {filter_name} filter = ', PSNR(filtered_image, original_image))


def arithmetic_mean_filter_image(image):
    arithmetic_mean_filter = np.vectorize(arithmetic_mean_filter_pixel, excluded=[0])
    y, x = np.indices(noisy_image.shape)
    filtered_image = arithmetic_mean_filter(image, y, x, mask_size)
    return filtered_image


def arithmetic_mean_filter_pixel(image, y, x, segment_size):
    segment = get_segment(image, y, x, segment_size)

    local_arithmetic_mean = np.mean(segment)

    return local_arithmetic_mean


def geometric_mean_filter_image(image):
    arithmetic_mean_filter = np.vectorize(geometric_mean_filter_pixel, excluded=[0])
    y, x = np.indices(noisy_image.shape)
    filtered_image = arithmetic_mean_filter(image, y, x, mask_size)
    return filtered_image


def geometric_mean_filter_pixel(image, y, x, segment_size):
    segment = get_segment(image, y, x, segment_size)
    height, width = segment.shape

    local_geometric_mean = segment.prod() ** (1.0 / (height * width))

    return local_geometric_mean


def adaptive_filter_image(image, noise_variance):
    arithmetic_mean_filter = np.vectorize(adaptive_filter_pixel, excluded=[0])
    y, x = np.indices(noisy_image.shape)
    filtered_image = arithmetic_mean_filter(image, y, x, mask_size, noise_variance)
    return filtered_image


def adaptive_filter_pixel(image, y, x, segment_size, noise_variance):
    segment = get_segment(image, y, x, segment_size)

    pixel_value = image[y, x]
    local_mean = np.mean(segment)
    local_variance = np.var(segment)

    return pixel_value - (noise_variance / local_variance) * (pixel_value - local_mean)


def get_segment(image, y, x, segment_size):
    height, width = image.shape[:2]
    sze = segment_size // 2

    y_start = max(y - sze, 0)
    y_end = min(y + sze + 1, height)

    x_start = max(x - sze, 0)
    x_end = min(x + sze + 1, width)

    return image[y_start:y_end, x_start:x_end]


original_image = read_GIF(ORIGINAL_IMAGE_PATH)
noisy_image = read_GIF(NOISY_IMAGE_PATH)

evaluate_filter(arithmetic_mean_filter_image(noisy_image), 'arithmetic mean')
evaluate_filter(geometric_mean_filter_image(noisy_image), 'geometric mean')
evaluate_filter(adaptive_filter_image(noisy_image, 100), 'adaptive')
