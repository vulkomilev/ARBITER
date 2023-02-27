import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from PIL import Image
from matplotlib import pyplot
from utils.utils import REGRESSION, REGRESSION_CATEGORY, IMAGE, TIME_SERIES,CATEGORY
from utils.utils import normalize_list,one_hot
from utils.Agent import *
import datetime
class DenseScrable(Agent):
    '''
    data_schema_input = {
        'games': [
            DataUnit('int', (), None, 'game_id', is_id=True),
            DataUnit('str', (), None, 'first', is_id=False),
            DataUnit('str', (), None, 'time_control_name', is_id=False),
            DataUnit('str', (), None, 'game_end_reason', is_id=False),
            DataUnit('int', (), None, 'winner', is_id=False),
            DataUnit('int', (), None, 'created_at', is_id=False),
            DataUnit('str', (), None, 'lexicon', is_id=False),
            DataUnit('int', (), None, 'initial_time_seconds', is_id=False),
            DataUnit('int', (), None, 'increment_seconds', is_id=False),
            DataUnit('str', (), None, 'rating_mode', is_id=False),
            DataUnit('int', (), None, 'max_overtime_minutes', is_id=False),
            DataUnit('float', (), None, 'game_duration_seconds', is_id=False)
        ],
        'train': [
            DataUnit('int', (), None, 'game_id', is_id=True),
            DataUnit('str', (), None, 'nickname', is_id=False),
            DataUnit('int', (), None, 'score', is_id=False),
            DataUnit('int', (), None, 'rating', is_id=False)],
        'turns': [
            DataUnit('int', (), None, 'game_id', is_id=True),
            DataUnit('int', (), None, 'turn_number', is_id=False),
            DataUnit('str', (), None, 'nickname', is_id=False),
            DataUnit('str', (), None, 'rack', is_id=False),
            DataUnit('str', (), None, 'location', is_id=False),
            DataUnit('str', (), None, 'move', is_id=False),
            DataUnit('int', (), None, 'points', is_id=False),
            DataUnit('int', (), None, 'score', is_id=False),
            DataUnit('str', (), None, 'turn_type', is_id=False)]}
'''
    def __init__(self, local_inputs, local_outputs, data_schema_input, data_schema_output, class_num):
        self.model = None
        local_input = None
        self.data_schema_input = data_schema_input
        self.data_schema_output = data_schema_output
        self.init_neural_network()
        self.total_tested = 0
        self.good_tested = 0

    def unlist(self,element):
        local_list = []
        if type(element) == list :
            if type(element[0]) == list:
                for local_element in element:
                    local_list.append(self.unlist(local_element))
            else:
                local_list = element
        elif type(element) == dict:
          for key, value in element.items():
            #local_list.append(key)
            local_list.append(self.unlist(value))
        return local_list
    def init_neural_network(self):


        input_model = tf.keras.Input(shape=(2))

        model_mid = tf.keras.layers.Dense((2))(input_model)
        model_mid = tf.keras.layers.Dense((12))(model_mid)
        model_mid = tf.keras.layers.Dense((6))(model_mid)
        model_mid = tf.keras.layers.Dense((3))(model_mid)
        model_mid = tf.keras.layers.Dense((1))(model_mid)
        self.model = tf.keras.Model(inputs=input_model, outputs=model_mid)

        self.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())#,metrics=[tf.keras.metrics.CategoricalAccuracy()])

    def normalize_date(self,date_list):
        return_list = []
        for element in date_list:
             return_list.append(float(datetime.datetime.strptime(element, '%Y-%m-%d %H:%M:%S').strftime("%s")))
        return_list = normalize_list(return_list,max(return_list),min(return_list),1.0, -1.0)
        return return_list

    def normalize(self,data,data_schema,path):
        print(type(data_schema))
        return_dict = {}
        if type(data_schema) is dict:
            for element_key in data_schema.keys():
                local_element = data[element_key]
                return_dict[element_key] = self.normalize(local_element,data_schema[element_key],path+[element_key])
        elif type(data_schema) is list:
            for element_key in data_schema:
                local_data = None
                local_data_path = ''
                for element_path in path:
                    local_data_path += '['+element_path+']'
                print('data'+'['+element_key.name+']')
                local_data = eval('data'+'[\''+element_key.name+'\']')
                if type(local_data) == list:
                  if  element_key.type == 'int' and element_key.is_id == False:
                      return_dict[element_key.name] = normalize_list(local_data,max(local_data),min(local_data),1.0, -1.0)
                  elif element_key.type == 'date' and element_key.is_id == False:
                      return_dict[element_key.name] = self.normalize_date(local_data)
                  elif element_key.type == 'str' and element_key.is_id == False:
                      local_data=one_hot(local_data,element_key)
                      if len(local_data) >0:
                        return_dict[element_key.name]  = local_data
                  elif element_key.type == 'float' and element_key.is_id == False:
                      return_dict[element_key.name] = normalize_list(local_data,max(local_data),min(local_data),1.0, -1.0)
                  elif element_key.is_id == False:
                      print('!!!!!!!!!!!!!!!!!!!!!!')
                      print(element_key.type)
                      exit(0)

        return return_dict


    def prepare_data(self, data, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []
        local_y_labels = []

        maxes = []
        mins = []
        arrays_x = []
        arrays_y = []
        #for i in range(len(data['games']['game_id'])):
        print('=======================')
        normalized_data_input = self.normalize(data,self.data_schema_input,[])
        normalized_data_output = self.normalize(data, self.data_schema_output, [])

        local_unlist = self.unlist(normalized_data_input)
        normalized_data_input = np.array(self.unlist(normalized_data_input))[0]
        normalized_data_output = np.array(self.unlist(normalized_data_output))[0]

        normalized_data_input = np.swapaxes(normalized_data_input, 0, 1)
        normalized_data_output = np.swapaxes(normalized_data_output, 0, 1)

        norm_arr_x = normalized_data_input

        norm_arr_y = normalized_data_output

        return norm_arr_x,norm_arr_y

    def train(self, images, force_train=False):
        x_train, y_train = self.prepare_data(images, in_train=True)

        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(0)
        if Path('./checkpoints/' + ckpt_name).exists() and not force_train:
            self.model = tf.keras.models.load_model('./checkpoints/' + ckpt_name)
        else:

            self.model.fit(x_train, y_train,batch_size=32, epochs=30,validation_split=0.1)
            self.model.save('./checkpoints/' + ckpt_name)
    def save(self):
        pass
    def predict(self, image):
        x_train, y_train = self.prepare_data([image], in_train=True)

        return self.model.predict(x_train)



