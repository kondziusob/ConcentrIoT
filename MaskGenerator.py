import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from Config import Config
import tensorflow as tf


# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append("..") 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

class MaskGenerator(Config):
    def __init__(self):
        self.config = InferenceConfig()
        #config.display()
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']
    
    def generate_masks(self, photo_dir=None, mask_dir=None):
        if photo_dir==None: 
            photo_dir = self.FG_DIR 
        if mask_dir == None: 
            mask_dir = self.MASK_DIR
        for file_name in os.listdir(photo_dir)[:-1]:
            self.create_mask(photo_dir, mask_dir, file_name)

    def create_mask(self, photo_dir, mask_dir, photo_name=None):
    
        if os.path.isdir(photo_dir) and photo_name != None:
            file_dir = photo_dir + '/' + photo_name
            if not os.path.isfile(file_dir):
                raise ValueError("File does not exists!!! :" + file_dir)
        elif os.path.isfile(photo_dir):
            file_dir = photo_dir

        image = skimage.io.imread(file_dir)
        # Run detection
        results = self.model.detect([image], verbose=1)[0]

        index = np.argmax([np.sum(results['masks'][:,:,x]) for x in  range(results['masks'].shape[-1])])
        mask = results['masks'][:,:,index].astype(int)
        mask = cv2.resize(mask, self.IMG_SIZE, interpolation = cv2.INTER_LINEAR)

        file_name = os.path.split(file_dir)[-1].split('/')[-1]
        file_name = os.path.split(file_name)[-1].split('.')[0]+".jpg"
        mask_dir = mask_dir +'/'+file_name
        
        cv2.imwrite(mask_dir, mask)
        return


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

