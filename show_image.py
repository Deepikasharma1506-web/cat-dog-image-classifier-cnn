import numpy as np
import matplotlib.pyplot as plt

image = np.array([
    [255, 0, 255],
    [0, 255, 0],
    [255, 0, 255]
])

plt.imshow(image, cmap='gray')
plt.show()