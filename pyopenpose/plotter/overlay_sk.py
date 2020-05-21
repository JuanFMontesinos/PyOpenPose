import cv2
import torch
import numpy as np
from .opencv import Window, opencv_writer


class Skeleton_Plotter(object):
    def __init__(self, graph, th=0.):
        self.th = th
        self.graph = graph

    @property
    def img(self):
        if torch.is_tensor(self._img):
            return self._img.clone()
        else:
            return self._img.copy()

    @img.setter
    def img(self, img):
        self._img = img

    def _set_sk(self, skeleton):
        self.img, self.X_disp, self.Y_disp = define_img(skeleton)
        self.skeleton = skeleton
        self.skeleton[:, 0, :] += self.X_disp
        self.skeleton[:, 1, :] += self.Y_disp

        return skeleton

    def __len__(self):
        return len(self.skeleton)

    def __call__(self, skeleton):
        self._set_sk(skeleton)
        wd = self.plot(self.th)
        wd.main()
        return skeleton

    def as_image(self, idx, th, img=None):
        if img is None:
            img = self.img
        return plot(self.skeleton[idx, ..., :], img, self.graph, th)

    def as_video(self, th, frame_iter=None):
        """

        :param th: threshold value to display skeleton edges
        :param frame_iter: Optional iterable which provides frames onto display skeletons.
        :return: Genrator of RGB images (np.ndarray)
        """
        obj = range(len(self.skeleton))
        if frame_iter is None:
            for idx in obj:
                yield self.as_image(idx, th)
        else:
            for idx, frame in zip(obj, frame_iter):
                yield self.as_image(idx, th, frame)

    def interactive(self, th):
        return Window(self, 'Image', th)

    def save_video(self, dst, fps, stream):
        opencv_writer(dst, fps, stream)
        # Imageio backend available



def overlay_sk(img, tensor, graph, th):
    colorcode = (255, 0, 255)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    miss_colorcode = (0, 255, 0)
    for edge in graph:
        j0, j1 = edge
        x0, y0, c0 = tensor[:, j0]
        x1, y1, c1 = tensor[:, j1]
        c = min(c0, c1).item()
        if th == 0 and c == 0:
            cv2.line(img, (x0, y0), (x1, y1), miss_colorcode, 4)
        elif c >= th:
            colorcode = (round(255 * (1 - c)), 0, round(255 * c))
            cv2.line(img, (x0, y0), (x1, y1), colorcode, 4)

    return img


def define_img(tensor):
    X_max = int(tensor.max(dim=0)[0].max().item())
    Y_max = int(tensor.max(dim=1)[0].max().item())
    X_min = int(tensor.min(dim=0)[0].min().item())
    Y_min = int(tensor.min(dim=1)[0].min().item())

    print('Image range (x,y): (%d,%d) to (%d,%d)' % (X_min, Y_min, X_max, Y_max))

    X_disp, Y_disp = 0, 0
    if X_min < 0:
        X_disp += int(abs(X_min) * 1.1)
    if Y_min < 0:
        Y_disp += int(abs(Y_min) * 1.1)
    print('Image range (x,y): (%d,%d) to (%d,%d)' % (X_min + X_disp, Y_min + Y_disp, X_max + X_disp, Y_max + Y_disp))

    img = np.zeros((int(1.2 * (X_max + X_disp)), int(1.2 * (Y_max + Y_disp)), 3), dtype=np.uint8) + 255
    return img, X_disp, Y_disp


def plot(tensor, img, graph, th=0.2):
    img = overlay_sk(img, tensor, graph, th)
    return img
