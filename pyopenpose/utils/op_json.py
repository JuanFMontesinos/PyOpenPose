import os
import sys

import cv2
import numpy as np
import torch
from classdict import ClassDict

CODECOLOR = {
    0: [(220, 13, 255), (250, 0, 250), (0, 0, 255)],
    1: [(10, 105, 235), (14, 207, 130), (0, 0, 255)],
    2: [(20, 142, 228)],
    3: [(10, 206, 242)],
    4: [],
    5: [(11, 209, 17)],
    6: [(10, 206, 242)],
    7: [],
    8: [(11, 209, 102), (210, 145, 10)],
    9: [(120, 209, 11)],
    10: [(179, 209, 11)],
    11: [(179, 209, 11), (179, 209, 11)],
    12: [(209, 11, 26)],
    13: [(209, 11, 26)],
    14: [(209, 11, 26), (209, 11, 26)],
    15: [(192, 11, 194)],
    16: [(209, 11, 142)],
    17: [],
    18: [],
    19: [(209, 11, 26)],
    20: [],
    21: [],
    22: [(179, 209, 11)],
    23: [],
    24: []}

CONECTIVITY = {0: [16, 15, 1],
               1: [2, 5, 8],
               2: [3],
               3: [4],
               4: [],
               5: [6],
               6: [7],
               7: [],
               8: [9, 12],
               9: [10],
               10: [11],
               11: [24, 22],
               12: [13],
               13: [14],
               14: [21, 19],
               15: [17],
               16: [18],
               17: [],
               18: [],
               19: [20],
               20: [],
               21: [],
               22: [23],
               23: [],
               24: []

               }
FACE_CONECTIVITY = {
    0: [1],
    1: [2],
    2: [3],
    3: [4],
    4: [5],
    5: [6],
    6: [7],
    7: [8],
    8: [9],
    9: [10],
    10: [11],
    11: [12],
    12: [13],
    13: [14],
    14: [15],
    15: [16],
    16: [],
    17: [18],
    18: [19],
    19: [20],
    20: [21],
    21: [],
    22: [23],
    23: [24],
    24: [25],
    25: [26],
    26: [],
    27: [28],
    28: [29],
    29: [30],
    30: [],
    31: [32],
    32: [33],
    33: [34],
    34: [35],
    35: [],
    36: [37],
    37: [38],
    38: [39],
    39: [40],
    40: [41],
    41: [36],
    42: [43],
    43: [44],
    44: [45],
    45: [46],
    46: [47],
    47: [48],
    48: [49],
    49: [50],
    50: [51],
    51: [52],
    52: [53],
    53: [54],
    54: [55],
    55: [56],
    56: [57],
    57: [58],
    58: [59],
    59: [48],
    60: [61],
    61: [62],
    62: [63],
    63: [64],
    64: [65],
    65: [66],
    66: [67],
    67: [60],
    68: [],
    69: []

}
HAND_CONECTIVITY = {
    0: [1, 5, 9, 13, 17],
    1: [2],
    2: [3],
    3: [4],
    4: [],
    5: [6],
    6: [7],
    7: [8],
    8: [],
    9: [10],
    10: [11],
    11: [12],
    12: [],
    13: [14],
    14: [15],
    15: [16],
    16: [],
    17: [18],
    18: [19],
    19: [20],
    20: []
}
HAND_CODECOLOR = {
    0: [(0, 0, 255), (0, 255, 155), (125, 255, 0), (255, 0, 15), (255, 0, 215)],
    1: [(0, 0, 255)],
    2: [(0, 0, 255)],
    3: [(0, 0, 255)],
    4: [],
    5: [(0, 255, 155)],
    6: [(0, 255, 155)],
    7: [(0, 255, 155)],
    8: [],
    9: [(125, 255, 0)],
    10: [(125, 255, 0)],
    11: [(125, 255, 0)],
    12: [],
    13: [(255, 0, 15)],
    14: [(255, 0, 15)],
    15: [(255, 0, 15)],
    16: [],
    17: [(255, 0, 215)],
    18: [(255, 0, 215)],
    19: [(255, 0, 215)],
    20: [(255, 0, 215)]
}


def json2tensor(dic):
    hand_list = []
    body_list = []
    face_list = []
    for person in dic['people']:
        r_hand = []

        l_hand = []

        body = []

        face = []

        r_hand.append(torch.tensor(person['hand_right_keypoints_2d'][0::3]).float())
        r_hand.append(torch.tensor(person['hand_right_keypoints_2d'][1::3]).float())
        r_hand.append(torch.tensor(person['hand_right_keypoints_2d'][2::3]).float())

        l_hand.append(torch.tensor(person['hand_left_keypoints_2d'][0::3]).float())
        l_hand.append(torch.tensor(person['hand_left_keypoints_2d'][1::3]).float())
        l_hand.append(torch.tensor(person['hand_left_keypoints_2d'][2::3]).float())

        body.append(torch.tensor(person['pose_keypoints_2d'][0::3]).float())
        body.append(torch.tensor(person['pose_keypoints_2d'][1::3]).float())
        body.append(torch.tensor(person['pose_keypoints_2d'][2::3]).float())

        face.append(torch.tensor(person['face_keypoints_2d'][0::3]).float())
        face.append(torch.tensor(person['face_keypoints_2d'][1::3]).float())
        face.append(torch.tensor(person['face_keypoints_2d'][2::3]).float())

        r_hand = torch.stack(r_hand).float()
        l_hand = torch.stack(l_hand).float()

        hand_list.append(torch.stack([r_hand, l_hand]))

        body_list.append(torch.stack(body).float())

        face_list.append(torch.stack(face).float())

    if len(dic['people']) == 1:
        return hand_list[0].unsqueeze(0), body_list[0].unsqueeze(0), face_list[0].unsqueeze(0)
    elif len(dic['people']) == 0:
        hand_list = torch.zeros(2,2, 3, 21)
        body_list = torch.zeros(2, 3, 25)
        face_list = torch.zeros(2, 3, 70)

    else:
        try:
            hand_list = torch.stack(hand_list)
        except RuntimeError as ex:
            raise type(ex)(str(ex) + 'Could not stack hand list of shape: %s' % str(hand_list[0].shape)).with_traceback(
                sys.exc_info()[2])
        try:
            body_list = torch.stack(body_list)
        except RuntimeError as ex:
            raise type(ex)(str(ex) + 'Could not stack body list of shape: %s' % str(body_list[0].shape)).with_traceback(
                sys.exc_info()[2])
        try:
            face_list = torch.stack(face_list)
        except RuntimeError as ex:
            raise type(ex)(str(ex) + 'Could not stack face list of shape: %s' % str(face_list[0].shape)).with_traceback(
                sys.exc_info()[2])
    return hand_list, body_list, face_list


def file2tensor(path):
    assert os.path.isfile(path)
    x = ClassDict()
    x.load(path)
    return json2tensor(x)


def folder2tensor(path):
    assert os.path.isdir(path)
    files = sorted(os.listdir(path))
    N = len(files)
    files = [str(x) + '_keypoints.json' for x in range(N)]
    paths = [os.path.join(path, x) for x in files]

    return [file2tensor(path) for path in paths]

