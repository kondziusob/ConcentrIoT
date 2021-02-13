import os
import cv2 
from Config import Config


class ResizePhotos(Config):
    def __init__(self):
        print("starting resize of user photos")
        for i in os.listdir(self.FG_DIR)[:-1]:
            img_name = os.path.join(self.FG_DIR, i)
            print(img_name)
            image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, self.IMG_SIZE, interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(img_name, image)
ResizePhotos()