import matplotlib.pyplot as plt

from utils.Agent import *
from matplotlib import pyplot

import cv2
import numpy as np
import random
from PIL import Image
import os
import shutil
from utils.utils import DataCollection
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')


class BERT(Agent):

    def __init__(self, inputs, outputs, data_schema, class_num):
        self.model = None
        self.init_neural_network(inputs, outputs, data_schema, class_num)
        self.total_tested = 0
        self.good_tested = 0
        self.model_input = inputs[0]
        self.model_output = outputs[0]

    def init_neural_network(self, inputs, outputs, data_schema, class_num):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                                             name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer('https://tfhub.dev/google/experts/bert/pubmed/2')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        self.model = tf.keras.Model(text_input, net)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics = tf.metrics.BinaryAccuracy()

        self.epochs = 5
        steps_per_epoch = 10
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.1 * num_train_steps)

        init_lr = 3e-5
        self.optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                       num_train_steps=num_train_steps,
                                                       num_warmup_steps=num_warmup_steps,
                                                       optimizer_type='adamw')
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

    def prepare_data(self, x: DataCollection, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []

        for x_element in zip(x):
            local_x_train_arr.append(x_element[0].get_by_name(self.model_input.name))
            local_y_train_arr.append(x_element[0].get_by_name(self.model_output.name))

        return np.array(local_x_train_arr), np.array(local_y_train_arr)

    def train(self, collection, force_train=False):
        x_train, y_train = self.prepare_data(collection, in_train=True)
        history = self.model.fit(x=x_train, y=x_train,
                                 validation_data=[],
                                 epochs=self.epochs)

    def predict(self, x):
        x = self.prepare_data(x, in_train=False)
        return self.model.predict(x)
