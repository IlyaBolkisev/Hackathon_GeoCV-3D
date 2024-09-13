import numpy as np
import cv2
import os

class ImageRectifier:
    def __init__(self, input_folder):
        #: Camera matrices (M)
        self.cam_mats = {"left": None, "right": None}
        #: Distortion coefficients (D)
        self.dist_coefs = {"left": None, "right": None}
        #: Rotation matrix (R)
        self.rot_mat = None
        #: Translation vector (T)
        self.trans_vec = None
        #: Essential matrix (E)
        self.e_mat = None
        #: Fundamental matrix (F)
        self.f_mat = None
        #: Rectification transforms (3x3 rectification matrix R1 / R2)
        self.rect_trans = {"left": None, "right": None}
        #: Projection matrices (3x4 projection matrix P1 / P2)
        self.proj_mats = {"left": None, "right": None}
        #: Disparity to depth mapping matrix (4x4 matrix, Q)
        self.disp_to_depth_mat = None
        #: Bounding boxes of valid pixels
        self.valid_boxes = {"left": None, "right": None}
        #: Undistortion maps for remapping
        self.undistortion_map = {"left": None, "right": None}
        #: Rectification maps for remapping
        self.rectification_map = {"left": None, "right": None}

        self.load(input_folder)

    def load(self, input_folder):
        """Load values from ``*.npy`` files in ``input_folder``."""
        self._interact_with_folder(input_folder, 'r')
    
    def _interact_with_folder(self, output_folder, action):
        """
        Export/import matrices as *.npy files to/from an output folder.

        ``action`` is a string. It determines whether the method reads or writes
        to disk. It must have one of the following values: ('r', 'w').
        """
        if not action in ('r', 'w'):
            raise ValueError("action must be either 'r' or 'w'.")
        for key, item in self.__dict__.items():
            if isinstance(item, dict):
                for side in ("left", "right"):
                    filename = os.path.join(output_folder,
                                            "{}_{}.npy".format(key, side))
                    if action == 'w':
                        np.save(filename, self.__dict__[key][side])
                    else:
                        self.__dict__[key][side] = np.load(filename)
            else:
                filename = os.path.join(output_folder, "{}.npy".format(key))
                if action == 'w':
                    np.save(filename, self.__dict__[key])
                else:
                    self.__dict__[key] = np.load(filename)
    
    def rectify(self, pair):
        new_pair = []
        for i, side in enumerate(('left', 'right')):
            new_pair.append(cv2.remap(pair[i],
                                      self.undistortion_map[side],
                                      self.rectification_map[side],
                                      cv2.INTER_NEAREST))
        
        return new_pair
    