'''
Adam Forestier
May 15, 2023
'''

import matplotlib.image as mpimg # NOTE: Amazing. Converts .jpg, .png, etc. to numpy array
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read in image
arr = mpimg.imread('palm_trees.jpg')
print(arr.shape) # NOTE: Height, Width, Color (1401 pixels in height, 934 pixels in width, 3 colors)
print(arr) # Shows red, green, blue count for each row

# Show the image
plt.imshow(arr) # NOTE: imshow needs numpy array, cannot take .img file path
plt.show()

# convert 3d to 2d (H,W,C to 2d) --> H * W, C
(h, w, c) = arr.shape
two_d_arr = arr.reshape(h * w, c)

# Create model to seperate colors into 6 labels
model = KMeans(n_clusters=6)
labels = model.fit_predict(two_d_arr)

# Show cluster centers as rgb_codes
rgb_codes = model.cluster_centers_.round(0).astype(int)

# Assign points to 1 of 6 rgb code (whichever it is closest to). Reshape back to original shape
# NOTE: This makes final quantized image!
quantized_img = np.reshape(rgb_codes[labels], (h,w,c))

# Show quantized image
plt.imshow(quantized_img)
plt.show()

# NOTE: WHY DO THIS???
# Much smaller. Takes up wayyyyy less space, will load faster