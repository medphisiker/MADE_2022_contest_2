import math
import os
import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon
import cv2
import numpy as np
from scipy.spatial import distance
from math import atan2, degrees
from functools import reduce
import operator
from glob import glob
import pandas as pd
from natsort import natsorted


def mask_to_polygons_layer(mask):
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask > 0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        return shapely.geometry.shape(shape)
        # all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons


def AngleBtw2Points(pointA, pointB):
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    # remove degrees if you want your answer in radians
    return degrees(atan2(changeInY, changeInX))


def sort_coords(coords):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    sort_coodinate = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    return sort_coodinate

def find_squares(image_path, dest_folder):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    squares_num = 0
    colors = np.unique(gray)
    for color_i in colors:
        if color_i != 255:
            polygons = mask_to_polygons_layer((gray == color_i))
            polygons = polygons.simplify(tolerance=2)
            x, y = polygons.exterior.coords.xy

            pts = []
            for xi, yi in zip(x, y):
                xi, yi = int(xi), int(yi)
                pts.append((xi, yi))

            pts = list(set(pts))
            pts = sort_coords(pts)

            if len(pts) == 4:
                d1 = distance.euclidean(pts[0], pts[1])
                d2 = distance.euclidean(pts[1], pts[2])
                d3 = distance.euclidean(pts[2], pts[3])
                d4 = distance.euclidean(pts[3], pts[0])

                max_side = max(d1, d2, d3, d4)
                side = min(d1, d2, d3, d4) / max_side > 0.90
                max_side_flag = max_side > 8
                angle1 = np.min(
                    np.array([0, 90, 180]) - np.abs(AngleBtw2Points(pts[0], pts[1]))) < 1
                angle2 = np.min(
                    np.array([0, 90, 180]) - np.abs(AngleBtw2Points(pts[1], pts[2]))) < 1
                angle3 = np.min(
                    np.array([0, 90, 180]) - np.abs(AngleBtw2Points(pts[2], pts[3]))) < 1
                angle4 = np.min(
                    np.array([0, 90, 180]) - np.abs(AngleBtw2Points(pts[3], pts[1]))) < 1

                if side and angle1 and angle2 and angle3 and angle4 and max_side_flag:
                    print('Найден квадрат!')
                    squares_num += 1

                    pts = np.array(pts, np.int32)
                    img = cv2.polylines(img, [pts], True, color=[
                                        255, 0, 0], thickness=1)
                    img = cv2.putText(img, f'Square_{squares_num}', pts[0], cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(
                        0, 0, 0), thickness=1)
                    
                    for (xi, yi) in pts:
                        img = cv2.circle(img, (xi, yi), radius=4,
                                 color=(0, 0, 0), thickness=-1)

    filename = os.path.split(image_path)[-1]
    save_path = os.path.join(dest_folder, filename)

    cv2.imwrite(save_path, img)
    
    return squares_num


imges_folder_path = 'data/train_data'
images_ext = '.png'
search_path = os.path.join(imges_folder_path, f'*{images_ext}')
images_paths = natsorted(glob(search_path))

dest_folder = 'data/train_predict'

# img_paths, labels = [], []
# for image_path in images_paths:
#     squares_num = find_squares(image_path, dest_folder)
#     img_paths.append(image_path)
#     labels.append(squares_num)

squares_num = find_squares('data/train_data/2.png', dest_folder)
print(squares_num)

# df = pd.DataFrame([img_paths, labels], index=['img_path', 'label']).T
# df.to_csv('data/algo_test_pred.csv', index=False)
print('Закончил!')