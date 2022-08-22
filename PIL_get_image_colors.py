from PIL import Image


def get_images_colors(image_path):
    colors = set()

    colors_count = Image.open(image_path).getcolors()
    for color_i in colors_count:
        color_tmp = color_i[1]
        if color_tmp != (255, 255, 255):
            colors.add(color_tmp)

    return colors


image_path = 'gray_sample.png'

colors = get_images_colors(image_path)
print(colors)
print(len(colors))
