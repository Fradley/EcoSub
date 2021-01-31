import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class LandmarkDetector:
    """
    Module to detect landmarks in a stereoscopic image
    """
    def __init__(self, left: str, right: str, sensor_distance: float = .08):
        """
        Initializes Landmark Detector
        :param left: left image
        :param right: right image
        :param sensor_distance: distance between image sensors
        """
        self.imgL = cv2.imread(left, 0)
        self.edgeL = cv2.Canny(self.imgL, 100, 200)
        self.imgR = cv2.imread(right, 0)
        self.edgeR = cv2.Canny(self.imgR, 100, 200)
        self.sensor_distance = sensor_distance

    def find_objects(self):
        pass

    def show(self):
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=25)
        disparity = stereo.compute(self.imgL, self.imgR)
        plt.imshow(disparity, 'gray')
        plt.show()
        plt.imshow(self.edgeL)
        plt.show()
        plt.imshow(self.edgeR)
        plt.show()

    def get_landmarks(self):
        pass
