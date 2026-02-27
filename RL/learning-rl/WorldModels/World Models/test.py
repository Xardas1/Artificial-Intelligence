# Your PyTorch format: (3, 64, 64)
# Matplotlib wants: (64, 64, 3)
import numpy as np


batch = np.load('D:/WorldModels/batches/batch_1.npy')
print(f"Original shape: {batch.shape}")

# Take first image and transpose it
img = batch[0]  # Shape: (3, 64, 64)
img_for_plot = np.transpose(img, (1, 2, 0))  # Shape: (64, 64, 3)

print(f"Image for matplotlib: {img_for_plot.shape}")
print(f"Image min/max: {img_for_plot.min()}/{img_for_plot.max()}")

import matplotlib.pyplot as plt
plt.imshow(img_for_plot)
plt.show()