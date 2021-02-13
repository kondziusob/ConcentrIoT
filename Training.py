import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import random
from tensorflow.python.keras.utils import data_utils
from Config import Config
from DatabaseGenerator import DataGenerator
import imageio
import pathlib

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class TrainOnCustomObject(Config):
    def __init__(self):
        
        db_gen = DataGenerator()
        self.train_dataset = db_gen.batch(self.BATCH_SIZE)
        self.validation_dataset = db_gen.batch(self.BATCH_SIZE)
        self.test_dataset = db_gen.batch(self.BATCH_SIZE)  
        """ 
        dataset = tf.data.Dataset.from_generator(
         gen,
         output_signature=(
             tf.TensorSpec(shape=(), dtype=tf.int32),
             tf.RaggedTensorSpec(shape=(224,224,3,None), dtype=tf.int32)))
        
        self.train_dataset = DataGenerator(
            "Database/cars_train", 
            "Database/VW/tmp3", 
            "Database/VW_Masks2", 
            "Database/Backgrounds")
        self.validation_dataset = DataGenerator(
            "Database/cars_train", 
            "Database/VW/tmp3", 
            "Database/VW_Masks2", 
            "Database/Backgrounds")
        self.test_dataset = DataGenerator(
            "Database/cars_train", 
            "Database/VW/tmp3", 
            "Database/VW_Masks2", 
            "Database/Backgrounds")
        """
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        IMG_SHAPE = self.IMG_SIZE + (3,)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
        
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train_dataset = self.train_dataset.prefetch(buffer_size=self.BATCH_SIZE)

        image_batch, label_batch = next(iter(self.train_dataset))
        feature_batch = self.base_model(image_batch)
        self.base_model.trainable = False
        
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        
        prediction_layer = tf.keras.layers.Dense(1, activation = "tanh")
        prediction_batch = prediction_layer(feature_batch_average)

        
        inputs = tf.keras.Input(shape=IMG_SHAPE)
        #x = self.training_augmentation(inputs)
        x = preprocess_input(inputs)
        x = self.base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.LR),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return 

    def get_model(self):
        return self.model
       #from Config import Config
       
    def create_folder(self, path):
        if not os.path.exists(path):
            print("Creating of new directory: " + path )
            os.makedirs(path)
        
    def train_model(self, save_dir = None):
        
        history = self.model.fit(
            self.train_dataset,
            epochs=self.INITIAL_EPOCHS,
            validation_data = self.validation_dataset)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # fine tuning 

        self.base_model.trainable = True
        for layer in self.base_model.layers[:self.FINE_TUNE_AT]:
            layer.trainable = False

        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=self.FINE_TUNE_LR),
              metrics=['accuracy'])
        

        fine_tune_epochs = self.INITIAL_EPOCHS + self.FINE_TUNE_EPOCHS

        total_epochs =  self.INITIAL_EPOCHS + self.FINE_TUNE_EPOCHS

        history_fine = self.model.fit(self.train_dataset,
                         initial_epoch=history.epoch[-1],
                         epochs=fine_tune_epochs,
                         validation_data=self.validation_dataset)

        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        if save_dir:

            self.create_folder(save_dir)

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.ylim([0.3, 1])
            plt.plot([self.INITIAL_EPOCHS-1,self.INITIAL_EPOCHS-1],
                    plt.ylim(), label='Start Fine Tuning')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.ylim([0, 1.0])
            plt.plot([self.INITIAL_EPOCHS-1,self.INITIAL_EPOCHS-1],
                    plt.ylim(), label='Start Fine Tuning')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')
            plt.savefig(save_dir+"/trainig2.jpg")

        return self.model

tr = TrainOnCustomObject()
model = tr.train_model("result_plot")

tflite_models_dir = pathlib.Path("Models/")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"vwmodelv1.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)