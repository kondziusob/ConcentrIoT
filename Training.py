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
          tf.keras.layers.experimental.preprocessing.RandomZoom(-0.3, -0.3),
          tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
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
                #foreground = tf.expand_dims(foreground, 0)
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
                 brightness_delta=[-0.2, 0.2], **kwargs):
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

tr = TrainOnCustomObject()
model = tr.train_model("result_plot")

tflite_models_dir = pathlib.Path("Models/")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"vwmodelv1.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)