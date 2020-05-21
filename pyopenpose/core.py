from .utils import ClassDict
import os
import sys
import cv2
import fire

OPENPOSE_CONF = ClassDict()
OPENPOSE_CONF.load('/home/jfm/GitHub/OpenposeWrapper/openpose_wrapper/openpose_config.json')
MODEL_PATH = OPENPOSE_CONF['MODELS_PATH']
PYTHON_PATH = OPENPOSE_CONF['PYTHON_LIBRARY']

sys.path.append(PYTHON_PATH)
from openpose import pyopenpose as op

__all__ = ['OPose']


class OPose(object):
    def __init__(self, opt=None):
        self.opWrapper = op.WrapperPython()
        self._init(opt)
        self._add_default_opt()
        self.image_formats = ['png', 'jpg', 'jpeg', 'bmp']
        self.video_formats = ['mp4', 'avi', 'mkv', 'web']

    def _add_default_opt(self):
        self.opt.model_folder = MODEL_PATH

    @staticmethod
    def opt_from_string(path):
        opt = ClassDict()
        return opt.load(path)

    @property
    def opt(self):
        return self._opt

    @opt.setter
    def opt(self, value):
        self._opt = value
        self.opt['model_folder'] = MODEL_PATH
        if 'self' in self.opt:
            del self.opt['self']
        self.opWrapper.configure(self.opt)
        self.opWrapper.start()

    def _init(self, opt):
        if isinstance(opt, dict):
            self.opt = ClassDict(opt)
        elif isinstance(opt, str):
            if os.path.isfile(opt):
                if os.path.splitext(opt)[1] == '.json':
                    self.opt = self.opt_from_string(opt)
                else:
                    raise TypeError('Unsupported extension %s. Use json files' % os.path.splitext(opt)[1])
            else:
                raise FileExistsError('File %s does not exist' % opt)
        elif opt is None:
            self.opt = ClassDict()
        else:
            raise NotImplementedError(
                'Only python dictionaries nor string inputs are supported, %s passed' % type(opt))

    def config(self, logging_level=3, disable_multi_thread=False, profile_speed=1000, camera=-1,
               camera_resolution='-1x-1', flir_camera=False, flir_camera_index=-1,
               ip_camera='',
               frame_first=0, frame_step=1, frame_last=-1, frame_flip=False, frame_rotate=0, frames_repeat=False,
               process_real_time=False, num_gpu=-1, num_gpu_start=0, keypoint_scale=0, number_people_max=-1,
               maximize_positives=False, fps_max=-1, body=1, model_pose='BODY_25', net_resolution='-1x368',
               scale_number=1, scale_gap=0.25, heatmaps_add_parts=False, heatmaps_add_bkg=False,
               heatmaps_add_PAFs=False, heatmaps_scale=2, part_candidates=False, upsampling_ratio=0., face=False,
               face_detector=0, face_net_resolution='368x368', hand=False, hand_detector=0,
               hand_net_resolution='368x368', hand_scale_number=1, hand_scale_range=0.4, render_threshold=0.05,
               render_pose=-1, alpha_pose=0.6, alpha_heatmap=0.7, write_images='', write_images_format="png",
               write_video='', write_video_fps='-1', write_video_with_audio=False, write_json='', write_coco_json='',
               write_heatmaps='', write_heatmaps_format='png'):
        self.opt = locals()

    def config_essential(self, keypoint_scale=0, body=1, face=False, hand=False, write_images='',
                         write_images_format='png', write_json='', number_people_max=-1):
        self.opt = locals()

    def save_config(self, path):
        self.opt.write(path)
        self.opWrapper.configure(self.opt)

    def load_config(self, path):
        self.opt.load(path)

    def process_images(self, path, display=False, keep_in_ram=False):
        assert isinstance(path, str)
        if os.path.isfile(path):
            imagePaths = [path]
        elif os.path.isdir(path):
            imagePaths = op.get_images_on_directory(path)

        imageOutput = []
        for imagePath in imagePaths:  # JFM seems sorted
            datum = op.Datum()
            imageToProcess = cv2.imread(imagePath)
            datum.cvInputData = imageToProcess
            self.opWrapper.emplaceAndPop([datum])

            # print("Body keypoints: \n" + str(datum.poseKeypoints))

            if display:
                cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
                key = cv2.waitKey(15)
                if key == 27: break
            if keep_in_ram:
                imageOutput.append(datum.cvOutputData)
        return imageOutput


if __name__ == '__main__':
    fire.Fire(OPose)
