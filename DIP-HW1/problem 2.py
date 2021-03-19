from Helping import *

NOISY_PATH = 'res/moonNoisy{}.gif'
ORIGINAL_PATH = 'res/moonOriginal.gif'
FRAME_COUNT = 10


def reduce_noise(frames: np.ndarray, count: int):
    noise_reduced = frames[:count].mean(axis=0)
    print(f'Using {count} noisy frames PSNR = {PSNR(noise_reduced, original_image).round(2)}')
    cv2.imwrite(f'out/noise_reduced_{count}.png', noise_reduced)
    return noise_reduced


original_image = read_GIF(ORIGINAL_PATH)
height, width = original_image.shape

frames = np.ndarray((FRAME_COUNT, height, width))
for i in range(0, FRAME_COUNT):
    frames[i] = read_GIF(NOISY_PATH.format(i + 1))

results = (original_image,)
results += (reduce_noise(frames, 7),)
results += (reduce_noise(frames, 5),)
results += (reduce_noise(frames, 3),)

concatenated = np.concatenate(results, axis=1).astype('uint8')
cv2.imshow('image', concatenated)
cv2.waitKey()
