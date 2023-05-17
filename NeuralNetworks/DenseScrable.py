import numpy as np

from utils.utils import normalize_list, one_hot, DataUnit
from utils.Agent import *
import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
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
class DenseScrable(Agent):

    def __init__(self,input_dict,output_dict):
        self.model = None

        self.total_tested = 0
        self.good_tested = 0
        self.local_bucket = []
        self.reg_input = []
        self.reg_output = []
        for element in input_dict:
         if element.shape == ():
             self.reg_input.append(DataUnit(str(element.type), (), None, '', is_id=element.is_id))
        for element in output_dict:
         if element.shape == ():
             self.reg_output.append(DataUnit(str(element.type), (), None, '', is_id=element.is_id))
        #self.reg_output = [
        #    DataUnit('int', (), None, '', is_id=True),
        #    DataUnit('float', (), None, '', is_id=False),
        #    DataUnit('float', (), None, '', is_id=False)]

        self.init_neural_network()
    def register(self, arbiter):
        arbiter.register_neural_network(self, self.reg_input, self.reg_output)

    def unlist(self, element):
        local_list = []
        if type(element) == list:
            if type(element[0]) == list:
                for local_element in element:
                    local_list.append(self.unlist(local_element))
            else:
                local_list = element
        elif type(element) == dict:
            for key, value in element.items():
                # local_list.append(key)
                local_list.append(self.unlist(value))
        return local_list

    def init_neural_network(self):

        input_model = tf.keras.Input(shape=(6))

        model_mid = tf.keras.layers.Dense((6))(input_model)
        model_mid = tf.keras.layers.Dense((12))(model_mid)
        model_mid = tf.keras.layers.Dense((6))(model_mid)
        model_mid = tf.keras.layers.Dense((3))(model_mid)
        model_mid = tf.keras.layers.Dense((2))(model_mid)
        self.model = tf.keras.Model(inputs=input_model, outputs=model_mid)

        self.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

    def normalize_date(self, date_list):
        return_list = []
        for element in date_list:
            return_list.append(float(datetime.datetime.strptime(element, '%Y-%m-%d %H:%M:%S').strftime("%s")))
        return_list = normalize_list(return_list, max(return_list), min(return_list), 1.0, -1.0)
        return return_list

    def normalize(self, data, data_schema, path, target):

        return_dict = {}
        if type(data_schema) is dict:
            for element_key in data_schema.keys():
                local_element = data[element_key]
                return_dict[element_key] = self.normalize(local_element, data_schema[element_key], path + [element_key])
        elif type(data_schema) is list:
            for i, element_key in enumerate(data_schema):
                local_data = None
                local_data_path = ''
                for element_path in path:
                    local_data_path += '[' + element_path + ']'

                local_data = []
                for element in data:
                    local_data.append(element[i])
                if type(local_data) == list or type(local_data) == type(np.ndarray(shape=0)):
                    if element_key.type == 'int' and element_key.is_id == False:
                        return_dict[element_key.name] = normalize_list(local_data, max(local_data), min(local_data),
                                                                       1.0, -1.0)
                    elif element_key.type == 'date' and element_key.is_id == False:
                        return_dict[element_key.name] = self.normalize_date(local_data)
                    elif element_key.type == 'str' and element_key.is_id == False:
                        local_data = one_hot(local_data, element_key)
                        if len(local_data) > 0:
                            return_dict[element_key.name] = local_data
                    elif element_key.type == 'float' and element_key.is_id == False:
                        return_dict[element_key.name] = normalize_list(local_data, max(local_data), min(local_data),
                                                                       1.0, -1.0)
                    elif element_key.is_id == False:
                        exit(0)

        return return_dict

    def prepare_data(self, data, in_train=False):

        local_data_input = []
        for element in data:
            local_list = []
            for second_element in self.reg_input:

                if len(second_element.name) > 0:
                    local_element = element.source.get_by_name(second_element.name)
                    if local_element == None:
                        local_element = 0
                    if len(np.array(local_element).shape) == 0:
                     continue
                    else:
                     local_list.append(np.array(local_element).tolist())

            local_list = np.swapaxes(np.array(local_list),1,0)
            local_list = local_list.tolist()
            local_data_input+=local_list
        local_data_output = []
        for element in data:
            local_list = []
            for second_element in self.reg_output:
                if len(second_element.name) > 0:
                    local_list.append(element.source.get_by_name(second_element.name))
            local_list = np.swapaxes(np.array(local_list), 1, 0)
            local_list = local_list.tolist()
            local_data_output+=local_list

        normalized_data_input = local_data_input
        normalized_data_output = local_data_output

        #normalized_data_input = np.array(self.unlist(normalized_data_input))
        normalized_data_output = np.array(self.unlist(normalized_data_output))
        #print(normalized_data_input)

        #print(np.array(normalized_data_input).shape)
        norm_arr_x = normalized_data_input #np.array(normalized_data_input).tolist()

        norm_arr_y = normalized_data_output

        return norm_arr_x, norm_arr_y

    def train(self, images, force_train=False,only_fill=False):
        if images != None:
            self.local_bucket.append(images)
        if len(self.local_bucket) < 100:
            return
        if only_fill:
            return
        x_train, y_train = self.prepare_data(self.local_bucket, in_train=True)

        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(0)
        if Path('./checkpoints/' + ckpt_name).exists() and not force_train:
            self.model = tf.keras.models.load_model('./checkpoints/' + ckpt_name)
        else:
            #exit(0)
            self.model.fit(np.array(x_train), np.array(y_train), batch_size=32, epochs=1)#, validation_split=0.1)
            self.model.save('./checkpoints/' + ckpt_name)
        self.local_bucket = []

    def save(self):
        pass

    def predict(self, image):

        x_train, y_train = self.prepare_data([image], in_train=True)
        _ = self.model.predict(x_train)

        return abs(_[0])
