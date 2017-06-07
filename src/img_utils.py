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

class ConnectedAreaInfo:
    def __init__(self, points, weight):
        self.points = points
        self.weight = weight
        rows = list(map(lambda pr: pr[0], self.points))
        self.min_i, self.max_i = min(rows), max(rows)
        cols = list(map(lambda pr: pr[1], self.points))
        self.min_j, self.max_j = min(cols), max(cols)

    def height_dist(self, other):
        return abs(self.max_i - other.max_i) + abs(self.min_i - other.min_i)

    def split_chars(self, img_h, img_w):
        pixels_to_take = set([p[1] for p in self.points if p[0] < 0.60*img_h])
        chars = list()
        cur_char = list()
        for j in sorted(set([p[1] for p in self.points])) + [img_w]:
            if j in pixels_to_take:
                cur_char.append(j)
            else:
                if len(cur_char) > 0.2*img_h:
                    for digit_set in self._separate_digits(img_h, cur_char):
                        char_points = list(filter(lambda p: p[1] in digit_set, self.points))
                        chars.append(ConnectedAreaInfo(char_points, self.weight))
                cur_char = list()
        return chars

    def _separate_digits(self, img_h, cols):
        j_from, j_to = min(cols), max(cols)+1
        dist_k = (j_to - j_from) / img_h
        cnt = 1
        for i, w in enumerate([1.1, 1.7, 2.4]):
            if dist_k > w:
                cnt = i+2
        res = [set() for i in range(cnt)]
        for j in cols:
            res[int(cnt*(j-j_from)/(j_to-j_from))].add(j)
        return res



def find_connected_areas(img, min_height_func):
    h, w = img.shape[:2]
    connected_areas = list()
    was = np.full((h, w, 1), False, dtype=bool)
    def enter(i, j):
        if i < 0 or i >= h or j < 0 or j >= w or img[i, j] == WHITE or was[i, j]:
            return False
        was[i, j] = True
        return True

    directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    for init_i in range(h):
        for init_j in range(w):
            if enter(init_i, init_j):
                cur_area = list()
                q = [(init_i, init_j)]
                while len(q) > 0:
                    cur = q.pop()
                    cur_area.append(cur)
                    for d in directions:
                        next_i, next_j = cur[0]+d[0], cur[1]+d[1]
                        if enter(next_i, next_j):
                            q.append((next_i, next_j))
                connected_areas.append(cur_area)

    filtered_areas_info = list()
    for i, area in enumerate(sorted(connected_areas, key=lambda l: len(l), reverse=True)):
        connected_area = ConnectedAreaInfo(area, 1)
        area_height = connected_area.max_i - connected_area.min_i
        min_height = min_height_func(h, i)
        if area_height >= min_height:
            filtered_areas_info.append(connected_area)

    return filtered_areas_info


def crop_to_field_height(img):
    h, w = img.shape[:2]
    filtered_areas_info = find_connected_areas(img, lambda h, i: h/8 if i < 6 else h/5);

    best_score, best_area_info = 0, None
    for base_area_info in filtered_areas_info:
        max_deviation = (base_area_info.max_i - base_area_info.min_i)/10
        score = 0
        for cur_area_info in filtered_areas_info:
            if base_area_info.height_dist(cur_area_info) < max_deviation:
                score += cur_area_info.weight
        if score > best_score:
            best_score, best_area_info = score, base_area_info

    margin = int(0.0*(best_area_info.max_i - best_area_info.min_i))
    res = img[max(best_area_info.min_i-margin, 0):min(best_area_info.max_i+margin, h), 0:w]
    return res

def check_minus_sign(img, sign_from, sign_to):
    h, w = img.shape[:2]
    if sign_to - sign_from < 0.4*h:
        return False
    sign_img = img[0:h, sign_from:sign_to]
    for area in find_connected_areas(sign_img, lambda h, i: 0):
        if area.min_i > 0.4*h and area.max_i < 0.75*h and area.max_j-area.min_j > 0.25*h:
            return True
    return False


def extract_chars(img):
    h, w = img.shape[:2]
    areas = find_connected_areas(img, lambda h, i: int(0.8*h));
    filtered_img = create_empty_img(h, w)
    minus = False
    was_dollar = False
    chars = list()
    for area in sorted(find_connected_areas(img, lambda h, i: int(0.75*h)), key=lambda a: a.min_j):
        for char_area in area.split_chars(h, w):
            if was_dollar:
                for p in char_area.points:
                    filtered_img[p[0], p[1]] = BLACK

            if not was_dollar and char_area.max_j - char_area.min_j > 0.4*h:
                was_dollar = True
                minus = check_minus_sign(img, max(char_area.min_j-int(1.2*h), 0), char_area.min_j)
            #chars.append(img[0:h, char_area.min_j:char_area.max_j])

    return (
        cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=3),
        minus,
        chars,
    )


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

