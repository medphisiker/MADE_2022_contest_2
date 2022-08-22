import os
import random
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def get_images_colors(imges_folder_path, images_ext):
    colors = set()
    search_path = os.path.join(imges_folder_path, f'*{images_ext}')
    images_paths = glob(search_path)

    for image_path in images_paths:
        colors_count = Image.open(image_path).getcolors()
        for color_i in colors_count:
            color_tmp = color_i[1]
            if color_tmp != (255, 255, 255):
                colors.add(color_tmp)

    return colors


def generate_blank_image(shape):
    return np.ones(shape=shape, dtype=np.int16) * 255


def add_rectangle(img, x0, y0, dx, dy, color, thickness):
    x1 = int(x0 - dx / 2)
    y1 = int(y0 - dy / 2)
    x2 = x1 + dx
    y2 = y1 + dy
    status = False
    if (min_coord < x1 < max_coord) and \
        (min_coord < y1 < max_coord) and \
        (min_coord < x2 < max_coord) and \
            (min_coord < y2 < max_coord):

        img = cv2.rectangle(img,
                            pt1=(x1, y1),
                            pt2=(x2, y2),
                            color=color,
                            thickness=thickness)

    return img


def add_semi_square(img, x0, y0, dx, color, thickness):
    delta1 = np.random.randint(2, 20)
    delta2 = np.random.randint(2, 20)
    x1 = int(x0 - dx / 2)
    y1 = int(y0 - dx / 2)
    x2 = x1 + dx + delta1
    y2 = y1 + dx + delta2

    if (min_coord < x1 < max_coord) and \
        (min_coord < y1 < max_coord) and \
        (min_coord < x2 < max_coord) and \
            (min_coord < y2 < max_coord):
        img = cv2.rectangle(img,
                            pt1=(x1, y1),
                            pt2=(x2, y2),
                            color=color,
                            thickness=thickness)
    return img


def add_square(img, x0, y0, dx, color, thickness):

    x1 = int(x0 - dx / 2)
    y1 = int(y0 - dx / 2)
    x2 = x1 + dx
    y2 = y1 + dx

    status = False
    if (min_coord < x1 < max_coord) and \
        (min_coord < y1 < max_coord) and \
        (min_coord < x2 < max_coord) and \
            (min_coord < y2 < max_coord):
        img = cv2.rectangle(img,
                            pt1=(x1, y1),
                            pt2=(x2, y2),
                            color=color,
                            thickness=thickness)
        status = True

    return img, status


def add_circle(img, x0, y0, r, color, thickness):
    if (min_coord < x0 + r < max_coord) and \
        (min_coord < y0 + r < max_coord) and \
        (min_coord < x0 - r < max_coord) and \
        (min_coord < y0 - r < max_coord) and \
        (min_coord < x0 + r < max_coord) and \
        (min_coord < y0 - r < max_coord) and \
        (min_coord < x0 - r < max_coord) and \
            (min_coord < y0 + r < max_coord):
        img = cv2.circle(img, (x0, y0), r, color, thickness)
    return img


def add_paralelogram(img, x0, y0, dx, dy, delta, orient, color, thickness):
    x1 = int(x0 - dx / 2)
    y1 = int(y0 - dy / 2)
    x2 = x1 + dx
    y2 = y1
    x3 = x2
    y3 = y2 + dy
    x4 = x1
    y4 = y1 + dy

    if orient == 0:
        y1 = y1 - delta
        y4 = y4 - delta
    elif orient == 1:
        y1 = y1 + delta
        y4 = y4 + delta
    elif orient == 2:
        x4 = x4 + delta
        x3 = x3 + delta
    elif orient == 3:
        x4 = x4 - delta
        x3 = x3 - delta

    pts = np.array([[x1, y1],
                    [x2, y2],
                    [x3, y3],
                    [x4, y4]], np.int32)

    if (min_coord < x1 < max_coord) and (min_coord < y1 < max_coord) and \
        (min_coord < x2 < max_coord) and (min_coord < y2 < max_coord) and \
        (min_coord < x3 < max_coord) and (min_coord < y3 < max_coord) and \
            (min_coord < x4 < max_coord) and (min_coord < y4 < max_coord):
        img = cv2.polylines(img, [pts],
                            True, color, thickness)

    return img


def add_rand_rectangle(img, color, min_coord, max_coord):
    x0 = np.random.randint(min_coord, max_coord)
    y0 = np.random.randint(min_coord, max_coord)
    dx = np.random.randint(10, 200)
    dy = np.random.randint(10, 200)
    thickness = np.random.randint(1, 8)
    img = add_rectangle(img, x0, y0, dx, dy, color, thickness)
    return img


def add_rand_square(img, color, min_coord, max_coord):
    x0 = np.random.randint(min_coord, max_coord)
    y0 = np.random.randint(min_coord, max_coord)
    dx = np.random.randint(10, 200)
    thickness = np.random.randint(1, 8)
    img, status = add_square(img, x0, y0, dx, color, thickness)
    return img, status


def add_rand_semi_square(img, color, min_coord, max_coord):
    x0 = np.random.randint(min_coord, max_coord)
    y0 = np.random.randint(min_coord, max_coord)
    dx = np.random.randint(10, 200)
    thickness = np.random.randint(1, 8)
    img = add_semi_square(img, x0, y0, dx, color, thickness)
    return img


def add_rand_circle(img, color, min_coord, max_coord):
    x0 = np.random.randint(min_coord, max_coord)
    y0 = np.random.randint(min_coord, max_coord)
    r = np.random.randint(10, 200)
    thickness = np.random.randint(1, 8)
    img = add_circle(img, x0, y0, r, color, thickness)
    return img


def add_rand_paralelogram(img, color, min_coord, max_coord):
    x0 = np.random.randint(min_coord, max_coord)
    y0 = np.random.randint(min_coord, max_coord)
    dx = np.random.randint(10, 200)
    dy = np.random.randint(10, 200)
    delta = np.random.randint(10, 70)
    orient = np.random.randint(4)
    thickness = np.random.randint(1, 8)
    img = add_paralelogram(img, x0, y0, dx, dy, delta,
                           orient, color, thickness)
    return img


shape = (320, 320, 3)
min_coord, max_coord = 10, 310

imges_train_path = 'data/train_data'
imges_test_path = 'data/test_data'
images_ext = '.png'

save_folder = 'data/gen_data'
imgs_number = 10000


colors_train = get_images_colors(imges_train_path, images_ext)
colors_test = get_images_colors(imges_test_path, images_ext)
colors = tuple(colors_train | colors_test)

img_path, label = [], []
for img_num in range(1, imgs_number + 1):
    img = generate_blank_image(shape)
    
    img_squares_cnt = 0
    for i in range(np.random.randint(1, 15)):
        try:
            color = random.choice(colors)
            img, status = add_rand_square(img, color, min_coord, max_coord)
            img_squares_cnt += status
        except:
            pass
    
    label.append(img_squares_cnt)

    for i in range(np.random.randint(1, 5)):
        try:
            color = random.choice(colors)
            img = add_rand_rectangle(img, color, min_coord, max_coord)
        except:
            pass

    for i in range(np.random.randint(1, 5)):
        try:
            color = random.choice(colors)
            img = add_rand_semi_square(img, color, min_coord, max_coord)
        except:
            pass

    for i in range(np.random.randint(1, 5)):
        try:
            color = random.choice(colors)
            img = add_rand_circle(img, color, min_coord, max_coord)
        except:
            pass

    for i in range(np.random.randint(1, 5)):
        try:
            color = random.choice(colors)
            img = add_rand_paralelogram(img, color, min_coord, max_coord)
        except:
            pass
    
    # converting BGR to RGB
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_path_i = os.path.join(save_folder, f'{img_num}.png')
    img_path.append(img_path_i)
    print(img_path_i)
    cv2.imwrite(img_path_i, img)
    

df = pd.DataFrame([img_path, label], index=['img_path', 'label']).T
df.to_csv('data/gen.csv', index=False)
print('Закончили')
