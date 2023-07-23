import copy
import csv
import datetime
import numpy as np
import tensorflow as tf
from utils.utils import normalize_list, one_hot, DataUnit, DataCollection, DataBundle,\
    DataInd,normalize_image,get_data_by_list,generate_path_list_from_dict,set_data_by_list,\
    REGRESSION, REGRESSION_CATEGORY, IMAGE, TIME_SERIES, try_convert_float,data_bundle_and_path
import pyarrow.parquet as pq
from  pandas import DataFrame
from NeuralNetworks.DenseScrable import DenseScrable
from NeuralNetworks.CellularAutomataAndData import CellularAutomataAndData
from NeuralNetworks.ImageAutoencoderDiscreteFunctions import ImageAutoencoderDiscreteFunctions
from NeuralNetworks.ImageAndData import ImageAndData


class Arbiter(object):

    def __init__(self, data_schema_input, data_schema_output, class_num, target_type, router_agent, skip_arbiter):
        self.data_schema_input = data_schema_input
        self.data_schema_output = data_schema_output
        self.router_agent = router_agent
        self.class_num = class_num
        self.tagrte_type = target_type
        self.bundle_bucket = {}
        self.registered_networks = {}
        self.input_arbiter_len = 0
        self.init_agents(router_agent)
        self.init_neural_network()
        self.skip_arbiter = skip_arbiter
        self.return_ids = []
        self.arbiter_router = {
            "": []
        }

    def add_bundle_bucket(self,key, input_dict):
        if type(input_dict) == type([]):
            self.bundle_bucket[key] = input_dict[0]
        else:
            self.bundle_bucket[key] = input_dict

    def register_neural_network(self, neural_network, input_shape, output_shape,keys_ins=None,keys_out=None,local_inputs=None,local_outpus=None):
        input_list = []
        output_list = []

        register_input = []
        register_output = []
        if keys_ins == None:
            keys_ins = ['base']
        if local_inputs == None:
            local_inputs = self.data_schema_input
        if local_outpus == None:
            local_outpus = self.data_schema_output

        if type(local_inputs) == type({}):
            return_template_input,return_path_list_input = data_bundle_and_path(local_inputs)
            return_template_output, return_path_list_output = data_bundle_and_path(local_outpus)
            for template_in,path_in in zip(return_template_input,return_path_list_input):
                for template_out, path_out in zip(return_template_output, return_path_list_output):
                     self.register_neural_network(neural_network,input_shape,output_shape,path_in,path_out,template_in,
                                                  template_out)
        else:
            for element in local_inputs:
                if not element.is_id:
                    input_list.append(element)
            for element in local_outpus:
                if not element.is_id:
                    output_list.append(element)

            for element in input_shape:
                is_shape_found = False
                for second_element in input_list:
                    if element.shape == second_element.shape:
                        element.name = second_element.name
                        register_input.append(element)
                        is_shape_found = True
                        break
                if is_shape_found:
                    input_list.remove(second_element)

            for element in output_shape:
                is_shape_found = False
                for second_element in output_list:
                    if element.shape == second_element.shape:
                        element.name = second_element.name
                        register_output.append(element)
                        is_shape_found = True
                        break
                if is_shape_found:
                    output_list.remove(second_element)

            if len(register_input) > 0 and len(register_output) > 0:
                self.registered_networks[neural_network] = {'neural_network': neural_network,
                                                            'input_path':keys_ins,
                                                            'input_list': register_input,
                                                            'output_path':keys_out,
                                                            'output_list': register_output}

    def init_neural_network(self):
        input_list = []
        output_len = 0
        for i in range(len(list( self.registered_networks.keys()))):
            for element in self.data_schema_output:
                if not element.is_id:
                    if element.shape == ():
                        input_list.append(1)
                    else:
                        input_list.append(element.shape)


        for element in self.data_schema_output:
            if not element.is_id:
                if element.shape == ():
                    output_len += 1
                elif type(element.shape) ==  type((1,2)):
                    output_len = element.shape

        self.input_arbiter_len = input_list
        if type(self.input_arbiter_len) == type((1, 2)):

            self.arbiter_neural_network_input = tf.keras.Input(shape=self.input_arbiter_len)
            layer_size = len(input_list)
            self.arbiter_neural_network = tf.keras.layers.Flatten()(self.arbiter_neural_network_input)
            self.arbiter_neural_network = tf.keras.layers.Dense(sum(input_list))(self.arbiter_neural_network_input)
            self.arbiter_neural_network = tf.keras.layers.Dense(sum(input_list))(self.arbiter_neural_network_input)
            self.arbiter_neural_network = tf.keras.layers.Dense(sum(input_list)/2)(self.arbiter_neural_network_input)
            tf.keras.layers.Conv2DTranspose(
                filters=30, kernel_size=[int(i/2) for i in input_list], strides=1, padding='same',
                activation='relu'),
            self.arbiter_neural_network = tf.keras.layers.Conv2DTranspose(sum(input_list))(self.arbiter_neural_network_input)

            self.arbiter_neural_network = tf.keras.layers.Dense(sum(input_list))(self.arbiter_neural_network_input)
            layer_size = len(input_list) / 2.0
            while layer_size > 1:
                self.arbiter_neural_network = tf.keras.layers.Dense(int(layer_size))( \
                    self.arbiter_neural_network)
                layer_size = layer_size / 2.0

            self.arbiter_neural_network = tf.keras.layers.Dense(output_len)( \
                self.arbiter_neural_network)
        else:
            self.arbiter_neural_network_input = tf.keras.Input(len(input_list))
            layer_size = len(input_list)
            self.arbiter_neural_network = tf.keras.layers.Dense(int(layer_size))( \
                self.arbiter_neural_network_input)
            layer_size = len(input_list) #/ len(list( self.registered_networks.keys()))

            self.arbiter_neural_network = tf.keras.layers.Dense(int(layer_size))( \
                    self.arbiter_neural_network)

        self.arbiter_neural_model = tf.keras.Model(inputs=self.arbiter_neural_network_input,
                                                   outputs=self.arbiter_neural_network)
        self.arbiter_neural_model.compile(optimizer="sgd", loss="mean_squared_error")

    def agents_schema_router(self):
        pass

    def normalize_date(self, date_list):
        return_list = []
        for element in date_list:
            return_list.append(float(datetime.datetime.strptime(element, '%Y-%m-%d %H:%M:%S').strftime("%s")))
        return_list = normalize_list(return_list, max(return_list), min(return_list), 1.0, -1.0)
        return return_list

    def normalize(self, data, data_schema, path, target):
        return_dict = {}
        return_dict_min = {}
        return_dict_max = {}

        if type(data_schema) is dict:
            for element_key in data_schema.keys():
                local_element = data[element_key]
                return_dict[element_key], l_min, l_max = self.normalize(local_element, data_schema[element_key],
                                                                        path + [element_key])
                return_dict_min[element_key.name] = l_min
                return_dict_max[element_key.name] = l_max
        elif type(data_schema) is list:
            for i, element_key in enumerate(data_schema):
                if element_key.is_id:
                    if element_key.name not in self.return_ids:
                        self.return_ids.append(element_key.name)
                local_data = None
                local_data_path = ''
                for element_path in path:
                    local_data_path += '[' + element_path + ']'

                local_data = []
                for element in data:
                    local_data.append(element.get_by_name(element_key.name))

                if  any(type(x) == type(None) for x in local_data):
                 for i in range(len(local_data)):
                    if local_data[i] == None:
                        local_data[i] = 0
                if type(local_data) == list or type(local_data) == type(np.ndarray(shape=0)):
                    if element_key.type == 'int' and element_key.is_id == False:

                        return_dict_min[element_key.name] = min(local_data)
                        return_dict_max[element_key.name] = max(local_data)

                        return_dict[element_key.name] = normalize_list(local_data, max(local_data), min(local_data),
                                                                       1.0, -1.0)

                    elif element_key.type == 'date' and element_key.is_id == False:
                        return_dict_min[element_key.name] = min(local_data)
                        return_dict_max[element_key.name] = max(local_data)
                        return_dict[element_key.name] = self.normalize_date(local_data)

                    elif element_key.type == 'str' and element_key.is_id == False:
                        local_data = one_hot(local_data, element_key)
                        if len(local_data) > 0:
                            return_dict_min[element_key.name] = min(local_data)
                            return_dict_max[element_key.name] = max(local_data)
                            return_dict[element_key.name] = local_data

                    elif element_key.type == 'float' and element_key.is_id == False:
                        return_dict_min[element_key.name] = min(local_data)
                        return_dict_max[element_key.name] = max(local_data)

                        return_dict[element_key.name] = normalize_list(local_data, max(local_data), min(local_data),
                                                                       1.0, -1.0)
                    elif element_key.type == 'bool' and element_key.is_id == False:
                        return_dict_min[element_key.name] = min(local_data)
                        return_dict_max[element_key.name] = max(local_data)
                        return_dict[element_key.name] = normalize_list(local_data, max(local_data), min(local_data),
                                                                       1.0, -1.0, element_key.type)
                    elif element_key.type == '2D_F' and element_key.is_id == False:

                        return_dict[element_key.name] = normalize_image(local_data)
                    elif element_key.is_id == False:
                        print('!!!!!!!!!!ERROR DATA NOMRALIZATION!!!!!!!!!!!!')
                        print(element_key.type)
                        exit(0)

        return return_dict, return_dict_min, return_dict_max

    def empty_bucket(self):
        self.bundle_bucket = {}


    def normalize_bundle_bucket(self, is_submit=False):
        local_key_list = list(self.bundle_bucket.keys())
        local_bundle_list = []
        for key in local_key_list:
             local_bundle_list.append(self.bundle_bucket[key])


        local_input_lista = generate_path_list_from_dict(self.data_schema_input, [], [], True)
        if local_input_lista == None:
                local_norm_arr = []
                for element in local_bundle_list:
                    local_norm_arr.append(element.source)

                norm_data = self.normalize_data_bundle(local_norm_arr)
                for i,key in enumerate(local_key_list):
                    set_data_by_list(self.bundle_bucket[key].source, [], norm_data,index=i)
        else:
            for element in local_input_lista:
                local_norm_arr = []
                for key in local_key_list:
                    local_norm_arr.append(get_data_by_list(self.bundle_bucket[key].source,element))

                norm_data = self.normalize_data_bundle(local_norm_arr)
                for i,key in enumerate(local_key_list):
                     set_data_by_list(self.bundle_bucket[key].source,element,norm_data,i)


    def normalize_data_bundle_dict(self,input_bucket,is_submit,keys,fill_dict):

        for element in list(input_bucket.keys()):
            if type(input_bucket[element]) == type({}):

                input_bucket[element] = self.normalize_data_bundle_dict(input_bucket[element],is_submit,keys+element,fill_dict)
            else:
                if str(keys+element) not in fill_dict.keys():
                    fill_dict[keys+element] = []
                fill_dict[keys+element].append(input_bucket[element])
        return input_bucket

    def normalize_data_bundle(self,input_bucket, is_submit=False):
        local_bucket = {}

        for key in list(input_bucket[0].get_dict()):
            local_bucket[key] = []
        source_ids = []
        target_ids = []

        for element in input_bucket:
            local_dict = element.get_dict()
            for k, v in local_dict.items():
                local_bucket[k].append(v)
            local_dict_ids = element.get_dict(include_only_id=True)
            for k, v in local_dict_ids.items():
                source_ids.append(v)

        max_len = len(source_ids)
        local_arr = []

        for element in input_bucket:
            local_arr.append(element)

        if not is_submit:
            local_arr, l_min, l_max = self.normalize(local_arr,input_bucket[0].data_schema,
                                                                         [], target='source')

            self.target_min = l_min
            self.target_max = l_max
        else:
            local_arr, _, _ = self.normalize(local_arr, input_bucket[0].data_schema, [],
                                             target='source')

        return local_arr

    def get_name_from_schema(self, schema, inputs):
        local_keys = []
        if type(schema) == dict:
            for element in schema.keys():
                local_keys += self.get_name_from_schema(schema[element], inputs)
        elif type(schema) == list:
            for element in schema:
                local_keys.append(element.name)

        else:
            raise Exception('NO foc king listr ')
        return local_keys

    def init_agents(self, agent_router):
        for agent_type in agent_router:
            agent_type_keys = [*agent_type.keys()]
            for input_list,output_list in zip(data_bundle_and_path(self.data_schema_input)[0],
                                              data_bundle_and_path(self.data_schema_output)[0]):
                exec('self.agent_local_' + agent_type_keys[0] + ' = ' + agent_type_keys[
                0]+'(input_list,output_list)')
                exec('self.agent_local_' + agent_type_keys[0] + '.register(self)')

    def save(self):
        local_agents = []
        for element in dir(self):
            if 'agent_' in element:
                local_agents.append(element)
        for element in local_agents:
            self.__getattribute__(element).save()

    def match_bundle_to_reg(self, local_bundle, model_io_reg):

        for bundle_i_element in model_io_reg:
            model_io_reg

    def train(self, force_train=False, train_arbiter=True):

        for key in list(self.bundle_bucket.keys()):
            for element in self.registered_networks.keys():

                local_bundle = DataBundle('000000',source=get_data_by_list(self.bundle_bucket[key].source,self.registered_networks[element]['input_path']),
                                          target=get_data_by_list(self.bundle_bucket[key].target,self.registered_networks[element]['output_path']))

                self.registered_networks[element]['neural_network'].train(local_bundle, force_train=force_train,only_fill=True)
        for element in self.registered_networks.keys():
            self.registered_networks[element]['neural_network'].train(None, force_train=force_train,
                                                                      only_fill=False)
        if train_arbiter and not self.skip_arbiter:
            local_y = []
            local_x = []
            for key in list(self.bundle_bucket.keys()):
                local_predictions = {}
                local_predictions_index = []
                x_element = []
                for element in self.registered_networks.keys():
                    local_predictions[element] = self.registered_networks[element]['neural_network'].predict(self.bundle_bucket[key])

                for element in self.registered_networks.keys():
                    local_predictions_index.append(local_predictions[element])

                for key in local_predictions.keys():
                    local_result = local_predictions[key]
                    if local_result is None:
                        local_result = 0
                    else:
                        local_result = local_result
                    x_element.append(local_result)
                local_x.append(x_element)

                local_y.append(self.bundle_bucket[key].target)
            self.train_arbiter(local_x, local_y)

    def train_arbiter(self, agent_results, target):
        x_fit = []
        y_fit = []
        class_num = self.class_num
        if self.tagrte_type == REGRESSION_CATEGORY:
            class_num = 100
        elif self.tagrte_type == REGRESSION:
            class_num = 1
        for agent_result, y in zip(agent_results, target):
            local_arr_x = []
            if type(agent_result) != list:
                x_fit.append(np.array(0))
            else:
                if self.tagrte_type == REGRESSION_CATEGORY:
                    local_arr_x += agent_result
                elif self.tagrte_type == REGRESSION:
                    local_arr_x += agent_result
                x_fit.append(local_arr_x)
            if self.tagrte_type == REGRESSION_CATEGORY:
                local_target = [0] * 100
                local_target[y - 1] = 1
                y_fit.append(local_target)
            elif self.tagrte_type == REGRESSION:

                y_fit.append(y / 100.0)
            else:
                y_fit.append(tf.one_hot(y, class_num))
        x_fit = np.array(x_fit)
        y_fit = np.array(y_fit)

        model_ = self.arbiter_neural_model.fit(x_fit, y_fit, epochs=200)

    def predict(self, image):
        local_agents = []
        local_predictions = {}
        local_predictions_index = []
        for element in dir(self):
            if 'agent_' in element:
                local_agents.append(element)
        for key in list(self.bundle_bucket.keys()):
            for element in self.registered_networks.keys():
                 local_bundle = DataBundle('000000', source=get_data_by_list(self.bundle_bucket[key].source,
                                                                             self.registered_networks[element][
                                                                                 'input_path']),
                                           target=get_data_by_list(self.bundle_bucket[key].target,
                                                                   self.registered_networks[element]['output_path']))
                 local_predictions[element] = self.registered_networks[element]['neural_network'].predict(local_bundle)

        for element in self.registered_networks.keys():
            local_predictions_index.append(local_predictions[element])
        local_arr_x = []
        local_x = []
        for key in local_predictions.keys():
            local_x.append(local_predictions[key])
        for x in local_x:
            if x is None:
                local_arr_x += [0]*int(len(self.input_arbiter_len)/len(list( self.registered_networks.keys())))
            else:
                if type(x) == type(np.zeros(1)):
                    x = x.tolist()
                local_arr_x += x

        if self.skip_arbiter:
            return np.array([local_arr_x]), None
        local_arr_x = np.squeeze(local_arr_x)
        result = self.arbiter_neural_model.predict(np.array([local_arr_x]))

        return result, local_predictions

    def evaluate(self, images):
        correct_count = 0
        wrong_count = 0
        div_arr = []
        preddicted_arr = []
        target_arr = []
        local_outputs = []
        for agent_type in self.router_agent:
            agent_type_keys = [*agent_type.keys()]

            for element_output in agent_type[agent_type_keys[0]]['outputs']:
                local_output = None
                for element_schema in self.data_schema_input['train']:
                    if element_schema.name == element_output['name']:
                        local_output = element_schema
                        break
                local_outputs.append(local_output)
        pred = self.predict(images)

        if self.tagrte_type == REGRESSION_CATEGORY:
            pred_indedx = np.argmax(pred, axis=1).tolist()[0]

            if pred_indedx == images['target']:
                correct_count += 1
            else:
                wrong_count += 1

        elif self.tagrte_type == REGRESSION:
            preddicted_arr.append(int(list(pred)[0][0] * 100))
            target_arr.append(images.get_by_name(local_outputs[0].name))
            div_arr.append(list(pred)[0][0] / images.get_by_name(local_outputs[0].name))
        elif self.tagrte_type == IMAGE:

            preddicted_arr.append(try_convert_float(pred[0][0][0]))
            target_arr.append(images.get_by_name(local_outputs[0].name))
        elif self.tagrte_type == TIME_SERIES:

            if type(pred[0][0]) != int and type(pred[0][0]) != np.int64:
                preddicted_arr += list(pred[0][0])

    def get_schema_names(self, schema):
        return_list = []
        if type(schema) is dict:
            for element_key in schema.keys():
                local_element = schema[element_key]
                return_list += self.get_schema_names(local_element)
        elif type(schema) is list:
            for element in schema:
                return_list.append(element.name)
        return return_list
    def get_data_ids(self, data,id_dict):
        if type(data) is dict:
            for element_key in data.keys():
                local_element = data[element_key]
                self.get_data_ids(local_element,id_dict)
        else:
            for element in data.data_collection:
                if element.is_id:
                    if id_dict[element.name] == None:
                          if type(element.data) == type(np.zeros((1,2))):
                              element.data = element.data.tolist()
                          id_dict[element.name]  = element.data
    def denormalize(self, data):
        if type(data) is not type([]):
            data = list(self.target_min.values())[0] + data * (
                    list(self.target_max.values())[0] - list(self.target_min.values())[0])
            return data

    def submit(self, file_dest=''):
        f = open(file_dest + 'submission.csv', 'w+')
        writer = csv.writer(f)
        local_arr = []
        local_dict ={}
        output_id_dict = {}
        local_arr.append('Id')
        #for element in self.return_ids:
        #    local_arr.append(element)

        if type(self.data_schema_output) is list:
            for element in self.data_schema_output:
                if element.is_id:
                    output_id_dict[element.name] = None
                else:
                  local_arr.append(element.name)

        else:
            local_arr = self.get_schema_names(self.data_schema_output)

        for element in self.get_schema_names(self.data_schema_output):
            local_dict[element] = []
        writer.writerow(local_arr)
        for key in list(self.bundle_bucket.keys()):

            results, _ = self.predict(self.bundle_bucket[key])
            results = np.squeeze(results)
            print('results',results)
            print('self.target_min',self.target_min)
            print('self.target_max', self.target_max)
            #exit(0)
            #results = self.denormalize(results)
            if type(results) == type(np.zeros((2))):
                results = results.tolist()
            local_arr = []
            final_ids = []
            try:
                local_id_dict = copy.deepcopy(output_id_dict)
                self.get_data_ids(self.bundle_bucket[key].source,local_id_dict)
                for element in self.data_schema_output:
                    if element.is_id:
                        final_ids.append(str(local_id_dict[element.name]))
                print(final_ids)
                local_arr.append('_'.join(final_ids).replace('.csv',''))
            except Exception as e:
                print(e)
                exit(0)
            if type(results) ==  type([]):
              for element in results:

                local_arr.append(element)


            else:
                local_arr.append(results)
            print('results', type(results))
            for element,arr_element in zip(self.get_schema_names(self.data_schema_output),local_arr):
                if element == 'Turn':
                    arr_element = int(arr_element)


                local_dict[element].append(arr_element)
            writer.writerow(local_arr)

