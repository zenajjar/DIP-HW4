from Helping import *

BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)

colors_names = {
    BLACK: 'black',
    BLUE: 'blue',
    GREEN: 'green',
    RED: 'red',
    YELLOW: 'yellow',
    MAGENTA: 'magenta',
    CYAN: 'cyan'
}

BLUR_DIMENSION = 9
COLOR_DISTANCE = 150
CIRCLE_EROSION_DIMENSION = 3
CIRCLE_EROSION_ITERATIONS = 1
CIRCLE_DILATION_DIMENSION = 5
CIRCLE_DILATION_ITERATIONS = 3
HOUGH_CIRCLE_DP = 1
HOUGH_CIRCLE_MIN_DIST = 100
HOUGH_CIRCLE_CANNY_THRESH = 20
HOUGH_CIRCLE_ACCUMULATOR_THRESH = 80
COLUMN_WIDTH = 20
CIRCLE_THICKNESS = 6

LINE_DILATION_DIMENSION = 4
LINE_DILATION_ITERATIONS = 1
LINE_EROSION_DIMENSION = 3
LINE_EROSION_ITERATIONS = 1
HOUGH_LINES_RHO = 10
HOUGH_LINES_THETA = np.pi / 180
HOUGH_LINES_THRESH = 100
HOUGH_LINES_MIN_LINE = 200
HOUGH_LINES_LINE_GAP = 60
LINE_SHIFT = 0
LINE_THICKNESS = 6


def remove_circles(image, circle_color, output_image):
    filtered_image = filter_color(image, circle_color)
    blurred_image = cv2.blur(filtered_image, (BLUR_DIMENSION, BLUR_DIMENSION))

    circles = get_circles(blurred_image)

    for circle in circles:
        ellipse = find_ellipse(filtered_image, circle, circle_color)
        draw_ellipse(output_image, ellipse, BLACK)


def fill_largest_circle(image, circle_color, fill_color, output_image):
    filtered_image = filter_color(image, circle_color)
    blurred_image = cv2.blur(filtered_image, (BLUR_DIMENSION, BLUR_DIMENSION))

    save_image(blurred_image, f'{colors_names[circle_color]}_blurred')

    circles = get_circles(blurred_image)

    largest_circle = get_largest_circle(circles)
    ellipse = find_ellipse(filtered_image, largest_circle, circle_color)
    draw_ellipse(output_image, ellipse, fill_color)


def get_circles(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,
                               dp=HOUGH_CIRCLE_DP,
                               minDist=HOUGH_CIRCLE_MIN_DIST,
                               param1=HOUGH_CIRCLE_CANNY_THRESH,
                               param2=HOUGH_CIRCLE_ACCUMULATOR_THRESH)

    if circles is None:
        print('no circles were found')
        return

    circles = np.round(circles[0, :]).astype("int")

    return circles


def filter_color(image, color):
    color_image = get_color(image, color)
    save_image(color_image, f'{colors_names[color]}_sliced')
    color_image = erode(color_image, CIRCLE_EROSION_DIMENSION, CIRCLE_EROSION_ITERATIONS)
    save_image(color_image, f'{colors_names[color]}_eroded')
    color_image = dilate(color_image, CIRCLE_DILATION_DIMENSION, CIRCLE_DILATION_ITERATIONS)
    save_image(color_image, f'{colors_names[color]}_dilated')

    return color_image


def get_color(image, color):
    distance_from_color = find_color_distance(image, color)
    color_pixels = np.where(distance_from_color < COLOR_DISTANCE, 255, 0)
    return color_pixels.astype('uint8')


def find_color_distance(image, color):
    distance = image - color
    distance **= 2
    distance = distance.sum(axis=2)
    distance = distance ** 0.5

    return distance


def erode(image, dimension, iterations):
    kernel = np.ones((dimension, dimension), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return eroded_image


def dilate(image, dimension, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dimension, dimension))
    dilation_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilation_image


def get_largest_circle(circles):
    largest_circle = circles[0]
    for x, y, r in circles:
        if r > largest_circle[2]:
            largest_circle = (x, y, r)
    return largest_circle


def find_ellipse(image, circle, color):
    x, y, radius = circle
    radius += 20
    y_start = y - radius
    y_end = y + radius
    x_start = x - radius
    x_end = x + radius
    segment = image[y_start:y_end, x_start: x_end]

    save_image(segment, f'{colors_names[color]}_ellipse_segment')

    center = radius
    diff = (CIRCLE_DILATION_DIMENSION * CIRCLE_DILATION_ITERATIONS
            - CIRCLE_EROSION_DIMENSION * CIRCLE_EROSION_ITERATIONS
            + CIRCLE_THICKNESS) // 2

    left_border = find_first_in_x_direction(segment, (center, center), -1) - diff
    right_border = find_first_in_x_direction(segment, (center, center), 1) + diff
    top_border = find_first_in_y_direction(segment, (center, center), -1) - diff
    bottom_border = find_first_in_y_direction(segment, (center, center), 1) + diff

    major_radius = (right_border - left_border) // 2
    minor_radius = (bottom_border - top_border) // 2

    x_center = x_start + left_border + major_radius
    y_center = y_start + top_border + minor_radius

    return x_center, y_center, major_radius, minor_radius


def find_first_in_x_direction(segment, center, direction):
    y_center, x_center = center
    inf = segment.shape[1]
    row_width = COLUMN_WIDTH // 2
    x = x_center
    while True:
        x += direction
        if x not in range(0, inf):
            return None
        for y in range(y_center - row_width, y_center + row_width):
            if segment[y, x] != 0:
                return x


def find_first_in_y_direction(segment, center, direction):
    x_center, y_center = center
    inf = segment.shape[0]
    column_width = COLUMN_WIDTH // 2
    y = y_center
    while True:
        y += direction
        if y not in range(0, inf):
            return None
        for x in range(x_center - column_width, x_center + column_width):
            if segment[y, x] != 0:
                return y


def draw_ellipse(image, ellipse, fill_color):
    x_center, y_center, major_length, minor_length = ellipse
    print(f'{colors_names[fill_color]} circle center = ({x_center}, {y_center}), '
          f'major length = {major_length} and minor length = {minor_length}')
    center_coordinates = (x_center, y_center)
    axesLength = (major_length, minor_length)
    angle = 0
    startAngle = 0
    endAngle = 360
    thickness = -1

    cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, fill_color, thickness)


def fill_longest_line(image, line_color, fill_color, output_image):
    filtered_image = image
    filtered_image = get_color(filtered_image, line_color)

    filtered_image = dilate(filtered_image, LINE_DILATION_DIMENSION, LINE_DILATION_ITERATIONS)
    filtered_image = erode(filtered_image, LINE_EROSION_DIMENSION, LINE_EROSION_ITERATIONS)
    save_image(filtered_image, f'{colors_names[line_color]}_filtered_lines')

    lines = cv2.HoughLinesP(filtered_image,
                            rho=HOUGH_LINES_RHO,
                            theta=HOUGH_LINES_THETA,
                            threshold=HOUGH_LINES_THRESH,
                            minLineLength=HOUGH_LINES_MIN_LINE,
                            maxLineGap=HOUGH_LINES_LINE_GAP)

    longest_line = get_longest_line(lines)
    print(f'longest {colors_names[line_color]} line is {round(find_length(longest_line), 2)} pixels long')
    draw_line(output_image, longest_line, fill_color)


def get_longest_line(lines):
    longest = lines[0][0]
    for sub_lines in lines:
        line = sub_lines[0]
        if find_length(line) > find_length(longest):
            longest = line
    return longest


def find_length(line):
    x1, y1, x2, y2 = line
    return ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5


def draw_line(image, line, fill_color):
    x1, y1, x2, y2 = line
    cv2.line(image, (x1, y1), (x2, y2), fill_color, LINE_THICKNESS)


original_image = cv2.imread('res/test.jpg')
output = original_image.copy()
lines_image = original_image.copy()

fill_largest_circle(original_image, BLUE, BLUE, output)
fill_largest_circle(original_image, GREEN, GREEN, output)
fill_largest_circle(original_image, RED, RED, output)
save_image(output, 'result_circles')

remove_circles(original_image, BLUE, lines_image)
remove_circles(original_image, GREEN, lines_image)
remove_circles(original_image, RED, lines_image)
save_image(lines_image, 'lines_image')

fill_longest_line(lines_image, BLUE, YELLOW, output)
fill_longest_line(lines_image, GREEN, MAGENTA, output)
fill_longest_line(lines_image, RED, BLACK, output)

save_image(output, 'result')
