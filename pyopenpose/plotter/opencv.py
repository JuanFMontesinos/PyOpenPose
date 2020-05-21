import cv2
import numpy as np


class Window():
    def __init__(self, flow, name, th):
        self.flow = flow
        self.name = name
        self.N = len(self.flow)
        self.idx = 0
        self.th = th
        self._reset()
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        # keep looping until the 'q' key is pressed

    def main(self):
        self._reset()
        while True:
            # display the image and wait for a keypress
            cv2.imshow(self.name, self.image)
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            flag = self.run_key(key)
            if flag.all() == self.key2mask(ord('q')).all():
                break
        cv2.destroyAllWindows()
        return self.flow

    def key2mask(self, key):
        ascii_ = ord('a')
        tmp = np.zeros(25, dtype=bool)
        tmp[key - ascii_] = True
        return tmp

    def run_key(self, key):
        if key == ord('r') and self.allow_key(key):
            self._reset()
        # if the 'c' key is pressed, break from the loop
        elif key == ord('q') and self.allow_key(key):
            return self.key2mask(key)
        elif key == ord('a') and self.allow_key(key):
            self.overflow_idx(-1)
            self.display(self.idx)
        elif key == ord('s') and self.allow_key(key):
            self.overflow_idx(+1)
            self.display(self.idx)
        elif key == ord('l') and self.allow_key(key):
            speed = 125
            pause = False
            while True:
                cv2.imshow(self.name, self.image)
                if not pause:
                    self.overflow_idx(1)
                    self.display(self.idx)
                key = cv2.waitKey(int(speed)) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    speed *= 1.1
                elif key == ord('s'):
                    speed *= 0.9
                elif key == ord('p'):
                    pause = not pause

        return np.ones(25, dtype=bool)

    def overflow_idx(self, i):
        idx_ = self.idx + i
        if idx_ >= self.N:
            self.idx = 0
        elif idx_ < 0:
            self.idx = self.N - 1
        else:
            self.idx += i

    def allow_key(self, key):
        k = self.key2mask(key)
        return k.all() == (k.all() and self.state.all()).all()

    def _reset_state(self):
        self.state = np.ones(25, dtype=bool)

    def _reset(self):
        self.refPt = []
        self.cropping = False
        self.idx = 0
        self.display(self.idx)
        self._reset_state()

    def display(self, idx):
        self.image = self.flow.as_image(idx, self.th)
        self.clone = self.flow.as_image(idx, self.th)


def opencv_writer(dst, fps, stream):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    shape = tuple(stream.shape[-2:])
    writer = cv2.VideoWriter(dst, fourcc, fps, shape)
    for frame in stream:
        writer.write(frame)
    writer.release()
    return True
