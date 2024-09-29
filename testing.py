import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fastai.vision.all import *

img = mpimg.imread('ISIC-images-Test/ISIC_0034524.jpg')
label,_,probs = learn.predict(img)
plt.imshow(img)
plt.axis('off')
plt.show()
print(f"This is a: {label}.")
print(f"Probability it's a malignant: {probs[1]:.4f}")