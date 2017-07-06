import cv2
import numpy as np
import logging

BLACK = 0
WHITE = 255

def create_empty_img(h, w):
    return np.full((h, w, 1), WHITE, dtype=np.uint8)


def resize_to_height(img, resize_height):
    h, w = img.shape[:2]
    resize_width = int(w * resize_height / h)
    return cv2.resize(img, (resize_width, resize_height))


def to_binary(img):
    return cv2.threshold(img, 150, WHITE, cv2.THRESH_BINARY)[1]


def get_height_histogram(img):
    h, w = img.shape[:2]
    return list(len([j for j in range(w) if img[i, j] == 0]) for i in range(h))


def get_width_histogram(img):
    h, w = img.shape[:2]
    return list(len([i for i in range(h) if img[i, j] == 0]) for j in range(w))


def draw_height_histogram(hist):
    res = create_empty_img(len(hist), max(hist))
    for i in range(len(hist)):
        for j in range(hist[i]):
            res[i, j] = BLACK
    return res


def draw_width_histogram(hist):
    res = create_empty_img(max(hist), len(hist))
    for j in range(len(hist)):
        for i in range(hist[j]):
            res[i, j] = BLACK
    return res


def crop(img, start, size):
    return img[start[0]:start[0]+size[0], start[1]:start[1]+size[1]]

