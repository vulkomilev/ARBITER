import matplotlib.pyplot as plt

from utils.Agent import *
from matplotlib import  pyplot

import keras
import numpy as np
import random
from PIL import Image

class LinearBinary(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(LinearBinary, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        result = tf.matmul(inputs, self.w) + self.b
        #result = tf.squeeze(result)
        #print("tf.shape(result)",tf.print(inputs))
        #exit(0)
        #result = tf.map_fn(lambda x : tf.ones(3) if tf.math.greater(x[0], 0.5) else tf.zeros(2), result, fn_output_signature=tf.float32)
        result = tf.map_fn(lambda x: x+1 if tf.math.greater(x[0], 0.5) else x-1, result,
                           fn_output_signature=tf.float32)

        return result


class ConstructNetwork(object):

    def __init__(self,local_inputs,local_outputs,data_schema_input,data_schema_output,class_num):
        self.init_neural_network()

    def init_neural_network(self):

        branch_data_input = tf.keras.Input(shape=(64,64,3))
        branch_data = tf.keras.layers.Conv2D(2,(3,3),strides=(3,3))(branch_data_input)
        branch_data = tf.keras.layers.GlobalAveragePooling2D()(branch_data)
        branch_data = tf.keras.layers.Flatten()(branch_data)
        branch_data = tf.keras.layers.Dense(42*42)(branch_data)
        branch_data = tf.keras.layers.Dense(42*42)(branch_data)


        agregate_network = tf.keras.layers.concatenate([branch_data])#,branch_connect])
        agregate_network = tf.keras.layers.Dense(42*42*2)(agregate_network)
        agregate_network = tf.keras.layers.Dense(42*42*2)(agregate_network)

        agregate_network = tf.keras.layers.Dense((42*42*2)/4)(agregate_network)
        agregate_network = tf.keras.layers.Dense((42*42*2)/2)(agregate_network)
        agregate_network = tf.keras.layers.Dense(110)(agregate_network)

        agregate_network_type_vertex, agregate_network_connection_network = tf.split(agregate_network, [10,10*10], 1)
        agregate_network_type_vertex = tf.keras.layers.Dense(10)(agregate_network_type_vertex)
        agregate_network_type_vertex = tf.keras.layers.Dense(10)(agregate_network_type_vertex)
        agregate_network_type_vertex = tf.keras.layers.Reshape((1,10))(agregate_network_type_vertex)
        agregate_network_connection_network = tf.keras.layers.Dense(100)(agregate_network_connection_network)
        agregate_network_connection_network = tf.keras.layers.Dense(100)(agregate_network_connection_network)
        agregate_network_connection_network = tf.keras.layers.Reshape((1,100))(agregate_network_connection_network)
        #connection shape [a_vetrex,b_vertex,a_properties,b_properties]
        self.model = tf.keras.Model(inputs = [branch_data_input],outputs = {'agregate_network_type_vertex':agregate_network_type_vertex,
                                                                            'agregate_network_connection_network':agregate_network_connection_network})
        self.model.compile(optimizer = 'Adam',loss='MSE')

        input_type = tf.keras.Input(shape=(10,))
        type_model = tf.keras.layers.Dense(10)(input_type)
        type_model = tf.keras.layers.Dense(10)(type_model)

        input_connection = tf.keras.Input(shape=(100,))
        connection_model = tf.keras.layers.Dense(100)(input_connection)
        connection_model = tf.keras.layers.Dense(100)(connection_model)

        agregate_network_output = tf.keras.layers.concatenate([type_model,connection_model])
        agregate_network_output = tf.keras.layers.Dense(110,activation='sigmoid')(agregate_network_output)
        agregate_network_output = tf.keras.layers.Dropout(0.2)(agregate_network_output)
        agregate_network_output = tf.keras.layers.Dense(120,activation='sigmoid')(agregate_network_output)
        agregate_network_output = LinearBinary(units=120,input_dim=120)(agregate_network_output)
        #agregate_network_output = tf.keras.layers.Dense(120,activation='sigmoid')(agregate_network_output)
        agregate_network_output = tf.keras.layers.Reshape((8,5,3))(agregate_network_output)
        agregate_network_output = tf.keras.layers.Conv2DTranspose(3,(20,20))(agregate_network_output)
        agregate_network_output = tf.keras.layers.Conv2DTranspose(3,(20,20))(agregate_network_output)
        agregate_network_output = tf.keras.layers.Conv2DTranspose(3,(19,22))(agregate_network_output)

        self.model_output = tf.keras.Model(inputs = {'input_type':input_type,
                                                     'input_connection':input_connection}
                                           ,outputs = [agregate_network_output] )
        self.model_output.compile(optimizer = 'Adam',loss='MSE')

    def prepare_data(self,images):
        x = []
        y = []

       # random.shuffle(images)

        for element  in images:
            print('element  in', element)
            local_image = element.get_by_name('Image')
            local_int = element.get_by_name('number')
            #if local_int not in [2,1,7,3]:
            #    continue
            M_type = [0]*10
            M = np.zeros((5,5,4))
            if local_int  == 0:
                M_type[1] = 2
                M_type[2] = 2
                M[1][2][1] = 1
                M[2][1][1] = 1
                M[1][2][0] = 1
                M[2][1][0] = 1

            # ----------------
            # 1
            if local_int  == 1:
                M_type[1] = 1
                M_type[2] = 1
                M[1][2][0] = 1
                M[2][1][0] = 1
            # ----------------
            # 2
            if local_int == 2:
                M_type[1] = 2
                M_type[2] = 1
                M[1][2][1] = 1
                M[2][1][1] = 1
            # ----------------
            # 3
            if local_int == 3:
                M_type[1] = 2
                M_type[2] = 2
                M[2][1][1] = 1
                M[1][2][1] = 1
            # ----------------
            # 4
            if local_int == 4:
                M_type[1] = 1
                M_type[2] = 1
                M_type[3] = 1
                M_type[4] = 1

                M[1][2][0] = 1
                M[2][1][0] = 1
                M[2][3][1] = 1
                M[2][4][1] = 1
                M[3][2][1] = 1
                M[3][4][1] = 1
                M[4][2][1] = 1
                M[4][3][1] = 1
            # ----------------
            # 5
            if local_int == 5:
                M_type[1] = 1
                M_type[2] = 1
                M_type[3] = 2
                M[1][2][1] = 1
                M[2][1][1] = 1
                M[2][3][0] = 1
                M[3][2][0] = 1
            # ----------------
            # 6
            if local_int == 6:
                M_type[1] = 2
                M_type[2] = 2
                M_type[3] = 2
                M[2][1][1] = 1
                M[3][1][1] = 1
                M[1][2][1] = 1
                M[3][2][0] = 1
                M[1][3][1] = 1
                M[2][3][0] = 1
            # ----------------
            # 7
            if local_int == 7:
                M_type[1] = 1
                M_type[2] = 1
                M_type[3] = 1
                M[1][2][1] = 1
                M[2][1][1] = 1
                M[2][3][0] = 1
                M[3][2][0] = 1

            # ----------------
            # 8
            if local_int == 8:
                M_type[1] = 2
                M_type[2] = 2
                M_type[3] = 2
                M_type[4] = 2

                M[2][1][1] = 1
                M[3][1][1] = 1
                M[3][1][0] = 1
                M[1][2][1] = 1
                M[3][2][1] = 1
                M[4][2][0] = 1
                M[4][2][1] = 1
                M[1][3][0] = 1
                M[1][3][1] = 1
                M[2][3][1] = 1
                M[4][3][1] = 1
                M[2][4][0] = 1
                M[2][4][1] = 1
                M[3][4][1] = 1

            # ----------------
            # 9
            if local_int == 9:
                M_type[1] = 2
                M_type[2] = 2
                M_type[3] = 2

                M[1][2][0] = 1
                M[1][2][1] = 1
                M[1][3][1] = 1
                M[2][1][0] = 1
                M[2][1][1] = 1
                M[2][3][1] = 1
                M[3][1][1] = 1
                M[3][2][1] = 1
            x.append(local_image)
            y.append((M_type,M))
        return x,y
    def train(self, images, force_train=False):
        x_arr,y_arr = self.prepare_data(images)
        local_counter = 0
        for x,y in zip(x_arr,y_arr):
            y_t ,y_c = y
            y_c = np.array([y_c.flatten()])
            y_t = np.array([y_t])
            x = np.array([x])
            try:
                self.model.fit(x=x,y={'agregate_network_type_vertex':y_t,
                                    'agregate_network_connection_network':y_c})
                self.model_output.fit(x={'input_type':y_t,
                                                     'input_connection':y_c},y=x)
            except Exception as e:
                print(e)
                pass
            local_counter += 1
            if local_counter > 900:
                return

    def predict(self,image):
        x_arr, y_arr = self.prepare_data([image])

        for x,y in zip(x_arr,y_arr):
            #result = self.model.predict(np.array([x]))
            #result = self.model_output({'input_type':result['agregate_network_type_vertex'],
            #                                'input_connection':result['agregate_network_connection_network']})
            try:
                result = self.model.predict(np.array([x]))
                result = self.model_output({'input_type': result['agregate_network_type_vertex'][0],
                                            'input_connection': result['agregate_network_connection_network'][0]})
                result = result.numpy()
                print('=======')
                result = result[0]
                result = ((result - result.min()) * (1 / (result.max() - result.min()) * 255)).astype('uint8')
                plt.imshow(x)
                plt.show()
                plt.imshow(result)
                plt.show()
            except Exception as e:
                print(e)
                return [None]
            return result[0]


    def evaluate(self):
        pass
