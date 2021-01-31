import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from sklearn.cluster import *
import hdbscan
import seaborn as sns


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
        self.imgL = cv2.imread(left, 1)
        self.greyL = cv2.imread(left, 0)
        self.edgeL = cv2.Canny(self.imgL, 100, 200)
        self.imgR = cv2.imread(right, 1)
        self.greyR = cv2.imread(right, 0)
        self.edgeR = cv2.Canny(self.imgR, 100, 200)
        self.sensor_distance = sensor_distance

    @staticmethod
    def index_img(A):
        A_ind = list()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A_ind.append([i, j, A[i, j][0], A[i, j][1], A[i, j][2]])
        return np.array(A_ind, dtype=int)

    @staticmethod
    def cluster(arr):
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(arr)
        return clusterer.labels_

    def find_objects(self):
        img_tab = self.index_img(self.imgL)
        labels = self.cluster(img_tab)
        num_labels = max(labels)
        palette = sns.color_palette(None, num_labels)
        print(palette)


    def show(self):
        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=25)
        # disparity = stereo.compute(self.imgL, self.imgR)
        # plt.imshow(disparity, 'gray')
        # plt.show()
        plt.imshow(self.imgL)
        plt.show()
        plt.imshow(self.edgeR)
        plt.show()

    def get_landmarks(self):
        pass
