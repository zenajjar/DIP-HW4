from Helping import *
import cv2

F16_ORIGINAL_PATH = 'res/f16.gif'
F16_NOISY_PATH = 'res/f16noisy.gif'
MEDIAN_OUTPUT_PATH = 'out/median_filter.png'
MAX_OUTPUT_PATH = 'out/max_filter.png'
MIN_OUTPUT_PATH = 'out/min_filter.png'

f16_original = read_GIF(F16_ORIGINAL_PATH)
f16_noisy = read_GIF(F16_NOISY_PATH)

kernel = np.ones((3, 3), 'uint8')
f16_min = cv2.erode(f16_noisy.astype('uint8'), kernel)
f16_max = cv2.dilate(f16_noisy.astype('uint8'), kernel)
f16_median = cv2.medianBlur(f16_noisy.astype('uint8'), 3)

print(f'PSNR for minimum filter mask = {PSNR(f16_original, f16_min)}')
print(f'PSNR for maximum filter mask = {PSNR(f16_original, f16_max)}')
print(f'PSNR for median filter mask = {PSNR(f16_original, f16_median)}')

cv2.imwrite(MEDIAN_OUTPUT_PATH, f16_median)
cv2.imwrite(MAX_OUTPUT_PATH, f16_max)
cv2.imwrite(MIN_OUTPUT_PATH, f16_min)
