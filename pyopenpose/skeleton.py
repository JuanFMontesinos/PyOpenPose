import numpy as np
import torch
from .plotter.opencv import Window
from .utils import op_json
from typing import Union


class Skeleton(object):
    def __init__(self, skeleton: Union[str, np.ndarray]):
        if isinstance(skeleton, np.ndarray):
            self.skeleton = skeleton
        elif isinstance(skeleton, str):
            self.skeleton = skeleton_from_path(skeleton)
            self.npeople = self.skeleton.size()[1] if n_people is None else n_people
            self.order_sk()
        self.nframes = self.skeleton.size()[0]

    def order_sk(self):
        optimal = torch.zeros_like(self.skeleton)
        optimal[0, ...] = self.skeleton[0, ...]

        for i in range(1, self.nframes):
            scores = torch.zeros(self.npeople, self.npeople)
            sk_ = optimal[i - 1, ...]
            sk = self.skeleton[i, ...]
            for p in range(self.npeople):
                for q in range(self.npeople):
                    sc = (sk_[p, :2, :] - sk[q, :2, :]).norm(dim=0)
                    coef = torch.min(sk_[p, 2, :], sk[q, 2, :])
                    sc = sc * coef
                    scores[p, q] = sc.sum()
            indices = torch.argmin(scores, dim=0)
            for p in range(self.npeople):
                optimal[i, p, ...] = sk[indices[p], ...]
        self.skeleton = optimal
    def __len__(self):
        return len(self.skeleton)