# Count the number of pixels brighter than <X>
# Show the thresholded image and the percent of
# the image that meets the threshold

from turtle import width
import cv2
import numpy as np

def threshold(img):
    # Convert the image from RGB to HSV colorspace
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    (h_l, s_l, v_l) = (0, 0, 0)  # HSV lower bounds
    (h_u, s_u, v_u) = (255, 255, 128)  # HSV upper bounds. 255 is the max 8bit value.

    # Transform the image into a mask of pixels that are or aren't in range.
    img = cv2.inRange(img, (h_l, s_l, v_l), (h_u, s_u, v_u))
    return img

def count_pixels(img):
    (height, width ) = img.shape

    count = 0
    for h in height:
        for w in width:
            if img[h, w] != 0:
                count = count + 1

    return count


def count_np(img):
    pass

def calc_percent(total_pixels, count):
    return count / total_pixels

if __name__ == '__main__':
    img = cv2.imread('hi.png')

    img = threshold(img)
    count = count_pixels(img)

    pixels = img.shape[0] * img.shape[1]
    perc = calc_percent(pixels, count)

    print("Image is {} percent in range".format(perc * 100))


