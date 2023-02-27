import random

import numpy as np
from utils.utils import HEURISTICS
from utils.Agent import *
import matplotlib.pyplot as plt
import keras
import matplotlib.animation as animation
from multiprocessing import Process, Pipe
import copy
from keras.callbacks import CSVLogger


tf.compat.v1.enable_eager_execution()
import tensorflow_datasets as tfds

#@tf.function
def antirectifier(x):
    #x = tf.where(
    #    x <= tf.constant(np.full(fill_value=HEURISTICS['max_freq'], shape=(192,), dtype=np.float32), shape=(192,)),
    #    tf.ones_like(x), x)
    x = tf.where(x <= tf.constant(np.full(fill_value=HEURISTICS['min_freq'],shape=(192,),dtype=np.float32),shape=(192,))
                 , tf.ones_like(x),  tf.zeros_like(x))
    return x



class NeuralNetworkApproximation(Agent):
    def __init__(self, inputs, outputs, data_schema_input, data_schema_output, class_num):
        self.model = None
        self.func_map = {}
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.init_neural_network(inputs, outputs, data_schema_input, latent_dim=(32 * 3), class_num=class_num)  # 8*4*3
        self.calc_map_plot_counter = 0
        self.confusion_matrix = {}
        self.csv_logger = CSVLogger('log_norm_'+str(HEURISTICS['min_freq'])+'.csv', append=True, separator=';')


    def init_neural_network(self, inputs, outputs, data_schema, latent_dim, class_num):

        self.latent_dim = latent_dim


        self.model_input =  tf.keras.layers.Input(shape=(28, 28,1))
        self.model_middle =   tf.keras.layers.Conv2D(
                filters=3, kernel_size=3, strides=(2, 2), activation='relu')(self.model_input)
        self.model_middle = tf.keras.layers.Conv2D(
                filters=3, kernel_size=3, strides=(2, 2), activation='relu')(self.model_middle)
        self.model_middle = tf.keras.layers.Flatten()(self.model_middle)
        self.model_middle = tf.keras.layers.Dense(192, activation='relu')(self.model_middle)
        self.model_middle = tf.keras.layers.Dense(192, activation='relu')(self.model_middle)
       # self.model_middle = tf.keras.layers.Lambda(antirectifier)(self.model_middle)
        self.model_middle = tf.keras.layers.Dense(10, activation='softmax')(self.model_middle)
        self.model = tf.keras.Model(inputs=self.model_input,outputs= self.model_middle)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',run_eagerly=True)

    def prepare_data(self, images, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []

        return np.array(local_x_train_arr), np.array(local_x_train_arr)

    def train(self, images, force_train=False):
        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        def normalize_img(image, label):
            return tf.cast(image, tf.float32) / 255.,tf.one_hot(label,10)

        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        self.model.fit(
            ds_train,
            epochs=200,
            validation_data=ds_test,
         callbacks=[self.csv_logger])




    def predict(self, image):
        x_train, y_train = self.prepare_data([image], in_train=True)

        return None
