import matplotlib.pyplot as plt
from utils.utils import is_float,is_float_arr,normalize_list,flatten_list
from utils.Agent import *
from matplotlib import  pyplot
import collections
import keras
import numpy as np
import random
from PIL import Image
from NeuralNetworks.TransformerNoEncoding import  TransformerNoEncoding
import functools
import operator

class TransformerExpSmartphoneD2022(object):

    def __init__(self,local_inputs,local_outputs,data_schema_input,data_schema_output,class_num):
        self.init_neural_network(local_inputs,local_outputs,data_schema_input,data_schema_output,class_num)
        self.local_inputs = local_inputs
        self.local_outputs_keys = []
        for element in local_outputs:
            self.local_outputs_keys.append(element['name'])

    def init_neural_network(self,local_inputs,local_outputs,data_schema_input,data_schema_output,class_num):
        #num_layers, key_element_size, num_heads,output_size, dff, rate=0.1

        input_model = tf.keras.Input(shape=(331, 331,3))
        #self.model =  TransformerNoEncoding(num_layers=10,key_element_size=len(local_inputs),num_heads=2,dff=20,output_size=len(local_outputs))

        self.model = TransformerNoEncoding(num_layers=10, key_element_size=48, num_heads=3, dff=20, output_size=6)

    def get_data_from_schema(self,date_unit,keys,transfrm_data):

        if type(date_unit) == dict:
            for element in date_unit.keys():
               self.get_data_from_schema(date_unit[element],keys+[element],transfrm_data)

        elif type(date_unit) == list:
            #for element in date_unit:
            #    local_keys.append( element.name)
            transfrm_data['_'.join(keys)] = date_unit

    def prepare_data(self,images):
        x = []
        y = []

       # random.shuffle(images)

        for element  in images:
            local_x = {}
            local_y = {}
            transfrm_data_norm = {}
            for sub_element, k in enumerate(element):
                transfrm_data_norm[k] = {}
            millis_key = 0
            for sub_element,k in enumerate(element):
                  transfrm_data = {}



                  for sub_sub_element ,k_s in enumerate(element[k]):

                    #self.get_data_from_schema(element,[],transfrm_data)


                    key = k_s
                    local_arr = element[k][k_s]
                    if  'Millis' in key:
                            millis_key = key
                    if is_float_arr(local_arr):
                            #print(transfrm_data[key][0])
                            #print(key, max(transfrm_data[key]))
                            #print(key, min(transfrm_data[key]))
                            transfrm_data_norm[k][key] =normalize_list(local_arr,max(local_arr),min(local_arr),target_max=1,target_min=-1)
                            print(len(transfrm_data_norm[k][key] ),k,key)
                            #print(transfrm_data_norm[key][-1])
                    elif key == 'MessageType':
                            local_list = []
                            for element_second in local_arr:
                                if element_second == 'UncalGyro':
                                    local_list.append([0,0, 0])
                                elif element_second== 'UncalAccel':
                                    local_list.append([0,0, 1])
                                elif element_second == 'UncalMag':
                                    local_list.append( [0,1, 0])

                                elif element_second == 'Fix':
                                    local_list.append([0,1, 1])

                                elif element_second == 'Raw':
                                    local_list.append([1,0, 0])
                            transfrm_data_norm[k][key] = local_list

                    elif key == 'CodeType':
                            local_list = []
                            for element_second in local_arr:
                                 local_list.append( tf.keras.preprocessing.text.one_hot(element_second, 16))
                            transfrm_data_norm[k][key] = local_list
                    elif key == 'SignalType':
                            local_list = []
                            for element_second in local_arr:
                                 if type(element_second) is not type('text'):
                                     element_second = 'nan'
                                 local_list.append( tf.keras.preprocessing.text.one_hot(element_second, 11))
                            transfrm_data_norm[k][key] = local_list
                    elif 'agent_local' in key:
                            continue
                    else:
                            print('ERROR VALL')

                            for element_second in transfrm_data[key]:
                                    try:
                                     float(float(element_second))
                                    except Exception as e:
                                        print (element_second)


                            print(key)
                            print(key[0])
                            print(transfrm_data[key][0])
                            print( 'agent_local' in key)
                            print( 'agent_local' in key[0])
                            print (key)
                            exit(0)

            for sub_element, k in enumerate(element):
             for sub_sub_element, k_s in enumerate(element[k]):

                    # self.get_data_from_schema(element,[],transfrm_data)

                    key = k_s
                    local_arr = element[k][k_s]
                    if 'Millis' in key:
                        millis_key = key
             for i_dict in range(len(element[k][list(element[k].keys())[0]])):
                        if i_dict > 100:
                            break
                        for i_second ,key_second in enumerate(self.local_inputs):
                            if str(transfrm_data_norm[k][millis_key][i_dict]) not in local_x.keys():
                                local_x[str(transfrm_data_norm[k][millis_key][i_dict])]  = [0]*len(self.local_inputs)
                            if key_second not in transfrm_data_norm[k].keys():
                                pass
                                #local_x[str(transfrm_data_norm[k][millis_key][i_dict])][i_second] = 0
                            else:
                             #print('k',k)
                             #print('self.local_inputs',self.local_inputs)
                             #print('key_second',key_second,transfrm_data_norm[k][key_second][i_dict])
                             #print('i_second',i_second)
                             local_x[str(transfrm_data_norm[k][millis_key][i_dict])][i_second] = transfrm_data_norm[k][key_second][i_dict]

                             local_x[str(transfrm_data_norm[k][millis_key][i_dict])] =  local_x[str(transfrm_data_norm[k][millis_key][i_dict])]
                             local_x[str(transfrm_data_norm[k][millis_key][i_dict])] = list(flatten_list(local_x[str(transfrm_data_norm[k][millis_key][i_dict])]))

                        for i_second ,key_second  in enumerate(self.local_outputs_keys):
                            if str(transfrm_data_norm[k][millis_key][i_dict]) not in local_y.keys():
                                local_y[str(transfrm_data_norm[k][millis_key][i_dict])] = [0]*len(self.local_outputs_keys)
                            if key_second not in transfrm_data_norm[k].keys():
                                pass
                                #local_y[str(transfrm_data_norm[k][millis_key][i_dict])][i_second] = 0
                            else:
                                local_y[str(transfrm_data_norm[k][millis_key][i_dict])][i_second] = transfrm_data_norm[k][key_second][i_dict]
                                local_y[str(transfrm_data_norm[k][millis_key][i_dict])] = local_y[str(transfrm_data_norm[k][millis_key][i_dict])]
                                local_y[str(transfrm_data_norm[k][millis_key][i_dict])] = list(flatten_list(local_y[str(transfrm_data_norm[k][millis_key][i_dict])]))

            local_x = collections.OrderedDict(sorted(local_x.items()))
            local_y = collections.OrderedDict(sorted(local_y.items()))
            local_x_arr = []
            local_y_arr = []
            for k, v in local_x.items():
                local_x_arr.append(v)
            for k, v in local_y.items():
                local_y_arr.append(v)

            x.append(local_x_arr)

            y.append(local_y_arr)
        return x,y
    def train(self, images, force_train=False):
        x_arr,y_arr = self.prepare_data(images)
        #x_arr = x_arr[:10]
        #y_arr = y_arr[:10]
        local_counter = 0

        temp_input = tf.random.uniform((46,), dtype=tf.float32, minval=0, maxval=1)
        temp_target = tf.random.uniform((46,), dtype=tf.float32, minval=0, maxval=1)

        for x,y in zip(x_arr,y_arr):
          for x_s,y_s in zip(x,y):
            try:
                    print(x_s[5],x_s[5] != 0,np.array(x_s).shape[0],x_s[0])
                    if np.array(x_s).shape[0] == 48 and x_s[5] != 0:
                       print(x_s,x_s[5] )
                       self.model.train(x_s, y_s)

                #    self.model.train(x,y)
            except Exception as e:
                     print('e',e)
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
                result = self.model.predict([x])
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
