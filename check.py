import numpy as np
import matplotlib.pyplot as plt

img = np.load("./processed/Training/glioma/Te-gl_1.npy")

print(f"Shape: {img.shape}")
print(f"Min: {img.min()}")
print(f"Max: {img.max()}")

plt.imshow(img, cmap='gray')
plt.title("Image prétraitée")
plt.axis('off')
plt.show()
