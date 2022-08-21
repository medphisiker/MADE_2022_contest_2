import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_blank_image(shape):
    return np.ones(shape=shape, dtype=np.int16) * 255


def add_rectangle(img, x0, y0, dx, dy, color, thickness):
    x1 = int(x0 - dx / 2)
    y1 = int(y0 - dy / 2)
    x2 = x1 + dx
    y2 = y1 + dy
    img = cv2.rectangle(img,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=color,
                        thickness=thickness)
    return img


shape = (320, 320, 3)
img = generate_blank_image(shape)
img = add_rectangle(img, 100, 100, 50, 100, color=(255, 0, 0), thickness=4)
print(img.shape)

# converting BGR to RGB
img = np.float32(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('image-2.png', img)
