import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import random
from tensorflow.python.keras.utils import data_utils
from Config import Config
import imageio
import cv2
from PIL import Image



class DataGenerator(tf.data.Dataset):
    
    def _generator(num_samples, size = (224,224), SAMPLE_DIR = "Database/cars_train" , FG_DIR="Database/VW/tmp3", MASK_DIR = "Database/VW_Masks2", BG_DIR="Database/Backgrounds"):
        
        data_augmentation = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.RandomZoom(-0.8, -0.5),
          tf.keras.layers.experimental.preprocessing.RandomRotation(0.5),
          tf.keras.layers.experimental.preprocessing.RandomContrast(0.5)
        ])

        #it's custom arumnted tf's layer to diversyfied db. in color dimension
        color_augmentation = tf.keras.Sequential([
          RandomColorDistortion(), # colour distoriton 
          tf.keras.layers.experimental.preprocessing.RandomContrast(0.5) #contrast distortion 
        ])
        FG_DIR = FG_DIR.decode("utf-8")
        fg_list = os.listdir(FG_DIR)[:-1]
        fg_img = [tf.expand_dims(imageio.imread(os.path.join(FG_DIR, i)),0) for i in fg_list]          
        fg_size = len(fg_list)
        
        MASK_DIR = MASK_DIR.decode("utf-8")
        mask_img = [np.repeat(imageio.imread(os.path.join(MASK_DIR, i))[:, :, np.newaxis], 3, axis=2)  for i in fg_list]
        
        BG_DIR = BG_DIR.decode("utf-8")
        bg_list = os.listdir(BG_DIR)[:-1]
        bg_img = [tf.expand_dims(imageio.imread(os.path.join(BG_DIR, i)),0) for i in bg_list]          
        bg_size = len(bg_list)
        
        SAMPLE_DIR = SAMPLE_DIR.decode("utf-8")
        sample_list = os.listdir(SAMPLE_DIR)[:50]
        sample_img = [tf.expand_dims(imageio.imread(os.path.join(SAMPLE_DIR, i)),0) for i in sample_list]          
        sample_size = len(sample_list)
        img_shape = size + (3,)

        for sample_idx in range(num_samples):
            class_index = random.randint(0,1) 
            if class_index:
                
                fg_index = random.randint(0, fg_size-1)
                foreground =  fg_img[fg_index] 
                #foreground = data_augmentation(background, training = True)
                foreground = color_augmentation(foreground, training = True).numpy()[0,:,:,:]
                
                bg_index = random.randint(0, bg_size - 1)
                background = bg_img[bg_index]
                
                #background = tf.expand_dims(background, 0)
                background = data_augmentation(background, training = True)
                background = color_augmentation(background, training = True)    
                background = background.numpy()[0,:,:,:]

                mask = mask_img[fg_index]
                
                foreground = np.multiply(mask, foreground)
                background = np.multiply(1 - mask, background)

                img = np.add(foreground, background)
                img = tf.expand_dims(img, 0)
                img = data_augmentation(img, training=True).numpy()[0,:,:,:]
                
            else: 
                img_index = random.randint(0, sample_size-1)
                img = sample_img[img_index]                
                #img = tf.expand_dims(img, 0)
                img = data_augmentation(img, training = True)
                img = color_augmentation(img, training = True).numpy()[0,:,:,:]
                

            yield (img, class_index,)
    
    def __new__(cls,num_samples, img_size, samle_dir, fg_dir, mask_dir, bg_dir):
        img_shape = img_size + (3, )
        
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_shapes = (img_shape, ()),
            output_types = (tf.float32, tf.int32),
            args=(
                num_samples, 
                img_size, 
                samle_dir, 
                fg_dir, 
                mask_dir, 
                bg_dir
            )
        )

class RandomColorDistortion(tf.keras.layers.Layer):
    def __init__(self, contrast_range=[0.5, 1.5], 
                 brightness_delta=[-0.5, 0.5], **kwargs):
        super(RandomColorDistortion, self).__init__(**kwargs)
        self.contrast_range = contrast_range
        self.brightness_delta = brightness_delta
    
    def call(self, images, training=None):
        if not training:
            return images
        
        contrast = np.random.uniform(
            self.contrast_range[0], self.contrast_range[1])
        brightness = np.random.uniform(
            self.brightness_delta[0], self.brightness_delta[1])
        images = tf.image.adjust_contrast(images, contrast)
        images = tf.image.adjust_brightness(images, brightness)

        return images
"""
    
    
    


class DataGenerator(data_utils.Sequence):

    def __init__(self, sample_dir, fg_dir, mask_dir, bg_dir, output_size =None, shuffle=False, batch_size=32):

        #self.df = pd.read_csv(csv_file)
        super(DataGenerator, self).__init__()
        self.sample_dir = sample_dir
        self.sample_list = os.listdir(sample_dir)[:-1]
        self.sample_size = len(self.sample_list)

        self.fg_dir = fg_dir 
        self.fg_list = os.listdir(fg_dir)[:-1]
        self.fg_size = len(self.fg_list)

        self.mask_dir = mask_dir 
        self.mask_list = os.listdir(mask_dir)[:-1]

        self.bg_dir = bg_dir
        self.bg_list = os.listdir(bg_dir)[:-1]
        self.bg_size = len(self.bg_list)

        if output_size == None: 
            self.output_size = (224,224)
        else: 
            self.output_size = output_size

        self.shuffle = shuffle
        self.batch_size = batch_size

        # define augmanted tf's layer to diversyfied our dataset
        #This layer performs random zooming, rotation, and contrast distortion
        self.training_augmentation = tf.keras.Sequential([
              tf.keras.layers.experimental.preprocessing.RandomZoom(-0.2, -0.2),
              tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
              tf.keras.layers.experimental.preprocessing.RandomContrast(0.5),
              tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),

            ])

        #it's custom arumnted tf's layer to diversyfied db. in color dimension
        self.color_augmentation = tf.keras.Sequential([
              RandomColorDistortion(), # colour distoriton 
              tf.keras.layers.experimental.preprocessing.RandomContrast(0.25) #contrast distortion 
            ])

        self.on_epoch_end()

    def on_epoch_end(self):
        self.sample_indices = np.arange(len(self.sample_list))
        self.fg_indices = np.arange(len(self.fg_list))
        self.bg_indices = np.arange(len(self.bg_list))

        if self.shuffle:
          np.random.shuffle(self.sample_indices)
          np.random.shuffle(self.fg_indices)
          np.random.shuffle(self.bg_indices)

        #tf.keras.backend.clear_session()

    def __len__(self):
        return int(self.fg_size*10 / self.batch_size)

    def augment_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = tf.expand_dims(img, 0)
        img = self.training_augmentation(img, training = True).numpy()[0,:,:,:]
        return img

    def blend_image_mask_bg(self, img_path, mask_path, bg_path):

        foreground = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        foreground = tf.expand_dims(foreground, 0)
        foreground = self.color_augmentation(foreground, training = True).numpy()[0,:,:,:]

        background = cv2.imread(bg_path,  cv2.IMREAD_UNCHANGED)
        background = tf.expand_dims(background, 0)
        background = self.training_augmentation(background, training = True)
        background = self.color_augmentation(background, training = True).numpy()[0,:,:,:]

        mask = cv2.imread(mask_path)

        foreground = np.multiply(mask, foreground)
        background = np.multiply(1 - mask, background)

        img_blend = np.add(foreground, background)
        #img_blend = tf.expand_dims(img_blend, 0)
        #img_blend = self.training_augmentation(img_blend, training = True)

        return img_blend


    def __data_generation(self):

      #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self.output_size, 3))
      y = np.empty((self.batch_size), dtype=int)

      # Generate data
      for i in range(self.batch_size):

            class_index = random.randint(0,1) 
            if class_index:
                fg_index = random.randint(0, self.fg_size-1)
                fg_name = os.path.join(self.fg_dir, self.fg_list[fg_index])

                mask_name = os.path.join(self.mask_dir, self.fg_list[fg_index].split('.')[0] +"_mask.jpg")

                bg_index = random.randint(0, self.bg_size - 1)
                bg_name = os.path.join(self.bg_dir, self.bg_list[bg_index])

                img = self.blend_image_mask_bg(fg_name, mask_name, bg_name)
            else: 
                img_index = random.randint(0, self.sample_size -1)
                img_name =os.path.join(self.sample_dir, self.sample_list[img_index])
                img = self.augment_image(img_name)

            X[i,] = img
            y[i] = class_index

      #return tf.convert_to_tensor(X), tf.convert_to_tensor(y, dtype=np.int32)
      return X, tf.convert_to_tensor(y, dtype=np.int32)
      #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getitem__(self, index):
        ## Initializing Batch
        #  that one in the shape is just for a one channel images
        # if you want to use colored images you might want to set that to 3
        'Generate one batch of data'
        # Generate indexes of the batch

        X, y = self.__data_generation()

        return (X,y)
"""


