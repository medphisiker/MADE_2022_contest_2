import os
from PIL import Image
from glob import glob


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


imges_folder_path = 'data/train_data'
images_ext = '.png'

colors = get_images_colors(imges_folder_path, images_ext)
print(colors)
print(len(colors))
