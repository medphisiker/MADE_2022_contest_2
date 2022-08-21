import cv2
import numpy as np
  
img = cv2.imread("data/train_data/1.png")

# converting BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(np.unique(img, axis=2))

cv2.imwrite('image-1.png', img)