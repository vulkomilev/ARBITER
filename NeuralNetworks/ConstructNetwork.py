import matplotlib.pyplot as plt

from utils.Agent import *
from matplotlib import  pyplot

import cv2
import numpy as np
import random
from PIL import Image

class ConstructNetwork(object):

    def __init__(self):
        pass

    def init_neural_network(self):

        branch_data_input = tf.keras.Input(shape=(128,128,1))
        branch_data = tf.keras.layers.Covn2d((3,3),(3,3))(branch_data_input)
        branch_data = tf.keras.layers.GlobalPool()(branch_data)
        branch_data = tf.keras.layers.Flatten()(branch_data)
        branch_data = tf.keras.layers.Dense(42*42)(branch_data)
        branch_data = tf.keras.layers.Dense(42*42)(branch_data)

        branch_connect_input = tf.keras.Input(shape=(128,128,1))
        branch_connect = tf.keras.layers.Covn2d((3,3),(3,3))(branch_connect_input)
        branch_connect = tf.keras.layers.GlobalPool()(branch_connect)
        branch_connect = tf.keras.layers.Flatten()(branch_connect)
        branch_connect = tf.keras.layers.Dense(42*42)(branch_connect)
        branch_connect = tf.keras.layers.Dense(42*42)(branch_connect)

        agregate_network = tf.keras.layers.concatenate([branch_data,branch_connect])
        agregate_network = tf.keras.layers.Dense(42*42*2)(agregate_network)
        agregate_network = tf.keras.layers.Dense(42*42*2)(agregate_network)

        agregate_network_type_vertex, agregate_network_connection_network = tf.split(agregate_network, [10,10*10], 0)
        agregate_network_type_vertex = tf.keras.layers.Dense(10)(agregate_network_type_vertex)
        agregate_network_type_vertex = tf.keras.layers.Dense(10)(agregate_network_type_vertex)
        agregate_network_connection_network = tf.keras.layers.Dense(100)(agregate_network_connection_network)
        agregate_network_connection_network = tf.keras.layers.Dense(100)(agregate_network_connection_network)
        #connection shape [a_vetrex,b_vertex,a_properties,b_properties]
    def prepare_data(self):
        pass

    def train(self, images, force_train=False):
        pass

    def evaluate(self):
        pass
