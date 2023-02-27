from utils.Agent import *
from matplotlib import pyplot as plt
from keras import backend as K

class Conv(Agent):

    def __init__(self,local_inputs,local_outputs,data_schema_input,data_schema_output,class_num):
        self.init_neural_network(local_inputs,local_outputs,data_schema_input,data_schema_output,class_num)

    def init_neural_network(self,local_inputs,local_outputs,data_schema_input,data_schema_output,class_num):
        local_input_shape =  ()
        for element in data_schema_input:
            if element.name == 'Image' :
                local_input_shape = element.shape
                break

        self.model_input  = tf.keras.Input(shape = local_input_shape)
        self.model = tf.keras.layers.Conv2D(shape = (3,3),strides = (1,1))(self.model_input)
        self.model = tf.keras.layers.GlobalPool()(self.model)
        self.model = tf.keras.layers.Conv2D(shape = (3,3),strides = (1,1))(self.model_input)
        self.model = tf.keras.layers.GlobalPool()(self.model)
        self.model = tf.keras.layers.Conv2D(shape = (3,3),strides = (1,1))(self.model_input)
        self.model = tf.keras.layers.GlobalPool()(self.model)
        

    def train(self, images, force_train=False):
        pass

    def predict(self, image):
        pass