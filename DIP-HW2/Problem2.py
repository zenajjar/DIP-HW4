from Helping import *

ORIGINAL_IMAGE_PATH = 'res/compressed sky.jpeg'


def part_a():
    image = read_GIF(ORIGINAL_IMAGE_PATH)
    show_image_and_wait(image, 'Original Image')
    return image


def part_b():
    ft = fft.fft2(original_image)
    magnitude, phase = get_dft_components(ft)
    log_scaled_magnitude = log_scale_image(magnitude)
    show_image_and_wait(log_scaled_magnitude, 'Image Magnitude')
    cv2.imwrite('out/Compressed Image Magnitude.png', log_scaled_magnitude)

    return ft, magnitude, phase


def part_c():
    total_power = np.sum(image_magnitude ** 2)
    dc_power = np.real(image_ft[0, 0]) ** 2
    percentage = ((dc_power / total_power) * 100)

    print(f'total power = {total_power.round(2)}')
    print(f'DC power component = {dc_power.round(2)}')
    print(f'The DC power component is {percentage.round(2)}% of the total power.')


def part_d():
    phase_inverse = np.real(fft.ifft2(np.exp(1j * image_phase)))
    scaled = linear_scale_image(phase_inverse)
    show_image_and_wait(scaled, 'Phase Component')
    cv2.imwrite('out/Compressed Phase Component.png', scaled)


def part_e():
    reconstructed_phase = get_inverse_ft(image_magnitude, 0.1j * image_phase)
    show_image_and_wait(reconstructed_phase, 'Reconstructed Phase')
    cv2.imwrite('out/Reconstructed Phase.png', reconstructed_phase)


original_image = part_a()
image_ft, image_magnitude, image_phase = part_b()
part_c()
part_d()
part_e()
