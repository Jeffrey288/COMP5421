from alignChannels import alignChannels
# Problem 1: Image Alignment
# start at 15:15

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# https://www.cs.toronto.edu/~guerzhoy/411/lec/W01/numpy/NumpyImgs.html#:~:text=Images%20can%20be%20read%20into,stored%20as%20multi%2Ddimensional%20arrays.&text=The%20pixel%20intensity%20values%20are,range%20from%200%20to%20255%20.
import os

# 1. Load images (all 3 channels)
red = np.load('../data/red.npy')
green = np.load('../data/green.npy')
blue = np.load('../data/blue.npy')
# print(red.shape, green.shape, blue.shape)

# img_noshift = np.array([red, green, blue])
# img_noshift = np.transpose(img_noshift, [1, 2, 0])
# plt.imshow(img_noshift)
# plt.show()

# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
from PIL import Image
im = Image.fromarray(rgbResult)
im.save("../results/rgb_output.jpeg")
