import numpy as np
import cv2
from collections import abc as container_abc
import io
from PIL import Image
import matplotlib.pyplot as plt

COLOR_POOL = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 0)]


def plt_show(x, title=None):
    import matplotlib.pyplot as plt
    plt.imshow(x)
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()


def pltAxes2npArray(plt_ax):
    canvas = plt_ax.figure.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    arr = Image.open(buffer)
    arr = np.asarray(arr)
    return arr


def vis_cellular_img(in_img):
    re_img = in_img.transpose(1, 2, 0)
    re_img = cv2.resize(re_img, (512, 512))
    re_img = re_img / 16

    # plt.figure(figsize=(12, 3))

    re_img = re_img[:, :, [0, 1, 2]]
    # re_img[:, :, 0][re_img[:, :, 0] < 0.2] = 0
    # re_img[:, :, 1][re_img[:, :, 1] < 0.3] = 0
    #
    # re_img[:, :, 2][re_img[:, :, 2] < 0.1] = 0
    # re_img = np.power(re_img, 0.8)
    re_img = (255 * re_img / re_img.max()).astype(np.uint8)
    return re_img
