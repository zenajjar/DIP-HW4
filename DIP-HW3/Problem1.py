from matplotlib import pyplot as plt

from Helping import *

original_image = read_GIF('res/HouseNoisy.gif')

consistent_intensity_region = original_image[2:96, 383:508]

show_image_and_wait(consistent_intensity_region)
plt.hist(consistent_intensity_region.ravel(), 256, [0, 256])
plt.show()

consistent_intensity_region_mean = np.mean(consistent_intensity_region)

print(f"mean before any mathematical computations = {consistent_intensity_region_mean}")
print(f"var of the noisy image = {np.var(consistent_intensity_region)}")
print(f"std of the noisy image = {np.std(consistent_intensity_region)}")

smoothed_region = consistent_intensity_region

for i in range(10000):
    smoothed_region = cv2.blur(smoothed_region, (50, 50))

print(f"The mean of excessive smoothed image = {np.mean(smoothed_region)}")
print(f"Mean of the noise = {consistent_intensity_region_mean - np.mean(smoothed_region)}")

show_image_and_wait(smoothed_region, 'smoothed image')
