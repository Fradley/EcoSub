import os
import glob
from Landmark.detection import LandmarkDetector


class DataLoader(object):
    def __init__(self, fpath):
        os.chdir('..')
        self.left_dir = os.path.join(fpath, 'Left')
        self.right_dir = os.path.join(fpath, 'Right')
        self.left_file_iter = glob.glob(self.left_dir + '\\*.png')
        self.right_file_iter = glob.glob(self.right_dir + '\\*.png')
        self.iter = zip(self.left_file_iter, self.right_file_iter)


def main():
    dl = DataLoader('Data')
    l, r = list(dl.iter)[0]
    ld = LandmarkDetector(l, r)
    ld.show()


if __name__ == "__main__":
    main()
