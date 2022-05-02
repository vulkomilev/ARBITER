import csv

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from utils.utils import DataCollection
from utils.utils import REGRESSION, REGRESSION_CATEGORY, IMAGE, TIME_SERIES, try_convert_float
from NeuralNetworks.LSTM import LSTM
from NeuralNetworks.MyResNet50 import MyResNet50
from NeuralNetworks.ConstructNetwork import ConstructNetwork

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tf.test.gpu_device_name():
#    print('GPU found')
# else:
#    print("No GPU found")
class Arbiter(object):

    def __init__(self, data_schema_input,data_schema_output, class_num, target_type, router_agent, skip_arbiter):
        self.data_schema_input = data_schema_input
        self.data_schema_output = data_schema_output
        self.router_agent = router_agent
        self.class_num = class_num
        self.tagrte_type = target_type
        self.init_agents(data_schema_input,data_schema_output, self.class_num, target_type, router_agent)
        # self.init_neural_network()
        self.skip_arbiter = skip_arbiter
        self.arbiter_router = {
            "": []
        }

    def init_neural_network(self):
        agent_size = 0
        for element in dir(self):
            if 'agent_' in element:
                if 'boost' in element:
                    agent_size += 100
                elif self.tagrte_type == REGRESSION_CATEGORY:
                    agent_size += 100
                elif self.tagrte_type == REGRESSION:
                    agent_size += 1
                else:
                    agent_size += self.class_num

        if self.tagrte_type == REGRESSION_CATEGORY:
            class_num = 100
            loss = 'categorical_crossentropy'
        elif self.tagrte_type == REGRESSION:
            class_num = 1
            loss = 'mean_squared_error'
        else:
            class_num = self.class_num
            loss = 'mean_squared_error'
        self.arbiter_neural_network_input = tf.keras.Input((agent_size))
        layer_size = agent_size
        self.arbiter_neural_network = tf.keras.layers.Dense(int(layer_size))( \
            self.arbiter_neural_network_input)
        layer_size = agent_size / 2.0
        while layer_size > 1:
            self.arbiter_neural_network = tf.keras.layers.Dense(int(layer_size))( \
                self.arbiter_neural_network)
            layer_size = layer_size / 2.0
        self.arbiter_neural_network = tf.keras.layers.Dense(class_num)( \
            self.arbiter_neural_network)
        print("arbiter_neural_network_input", self.arbiter_neural_network_input)
        self.arbiter_neural_model = tf.keras.Model(inputs=self.arbiter_neural_network_input,
                                                   outputs=self.arbiter_neural_network)
        self.arbiter_neural_model.compile(optimizer="sgd", loss=loss)

    def agents_schema_router(self):
        pass

    def init_agents(self, data_schema_input,data_schema_output, class_num, tagrte_type, agent_router):
        agent_id = 0
        for agent_type in agent_router:
            local_inputs = []
            local_outputs = []

            agent_type_keys = [*agent_type.keys()]
            for element_input in agent_type[agent_type_keys[0]]['inputs']:
                local_input = None
                for element_schema in data_schema_input:
                    if element_schema.name == element_input:
                        local_input = element_schema
                        break
                local_inputs.append(local_input)

            for element_output in agent_type[agent_type_keys[0]]['outputs']:
                local_output = None
                for element_schema in data_schema_output:
                    if element_schema.name == element_output['name']:
                        local_output = element_schema
                        break
                local_outputs.append(local_output)
            exec('self.agent_local_' + agent_type_keys[0] + ' = ' + agent_type_keys[
                0] + '(local_inputs,local_outputs,data_schema_input,data_schema_output,class_num)')
        # self.agent_boost = BoostClass(image_height=img_height, image_width=img_width, labels_size=class_num,tagrte_type=tagrte_type)

    def train(self, image_collection, train_target='', force_train=False, train_arbiter=True):
        local_agents = []
        for element in dir(self):
            if 'agent_' in element:
                local_agents.append(element)
        print(local_agents)
        for element in local_agents:
            self.__getattribute__(element).train(image_collection, force_train=force_train)

        if train_arbiter and not self.skip_arbiter:
            local_y = []
            local_x = []
            for image in image_collection:
                local_predictions = {}
                local_predictions_index = []
                x_element = []
                for element in local_agents:
                    local_predictions[element] = self.__getattribute__(element).predict(image)

                for element in local_agents:
                    local_predictions_index.append(local_predictions[element])

                for key in local_predictions.keys():
                    local_result = local_predictions[key]
                    if local_result is None:
                        local_result = 0
                    else:
                        local_result = local_result
                    x_element.append(local_result)
                local_x.append(x_element)

                local_y.append(image.get_by_name(train_target))
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
            print('type(agent_result)', type(agent_result))
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

        model_ = self.arbiter_neural_model.fit(x_fit, y_fit, epochs=200)  # x_fit,y_fit
        # print([y_fit[0]])
        # print(self.arbiter_neural_model.predict(np.array([x_fit[0]])))
        # plt.plot(list(model_.history.values())[0], 'k-o')
        # plt.show()
        # exit(0)

    def predict(self, image):
        local_agents = []
        local_predictions = {}
        local_predictions_index = []
        for element in dir(self):
            if 'agent_' in element:
                local_agents.append(element)
        for element in local_agents:
            local_predictions[element] = self.__getattribute__(element).predict(image)

        for element in local_agents:
            local_predictions_index.append(local_predictions[element])
        local_arr_x = []
        local_x = []
        for key in local_predictions.keys():
            local_x.append(local_predictions[key])
        for x in local_x:
            if x is None:
                local_x += [0]
            else:
                local_arr_x += [x]

        if self.skip_arbiter:
            return None,np.array([local_arr_x])
        local_arr_x = np.squeeze(local_arr_x)
        result = self.arbiter_neural_model.predict(np.array([local_arr_x]))
        # print('-------')
        # print(result)
        return result, local_predictions

    def evaluate(self, images):
        correct_count = 0
        wrong_count = 0
        print('-------------------')
        print(len(images))
        div_arr = []
        preddicted_arr = []
        target_arr = []
        local_outputs = []
        for agent_type in self.router_agent:
            agent_type_keys = [*agent_type.keys()]

            for element_output in agent_type[agent_type_keys[0]]['outputs']:
                local_output = None
                for element_schema in self.data_schema:
                    if element_schema.name == element_output['name']:
                        local_output = element_schema
                        break
                local_outputs.append(local_output)
        for image in images:
            pred = self.predict(image)

            if self.tagrte_type == REGRESSION_CATEGORY:
                pred_indedx = np.argmax(pred, axis=1).tolist()[0]

                print("pred_indedx", pred_indedx, '-', image['target'])
                if pred_indedx == image['target']:
                    correct_count += 1
                else:
                    wrong_count += 1
                print(correct_count, '/', (correct_count + wrong_count))
                print('', end='\r')
            elif self.tagrte_type == REGRESSION:
                preddicted_arr.append(int(list(pred)[0][0] * 100))
                target_arr.append(image.get_by_name(local_outputs[0].name))
                # print(int(pred.tolist()[0][0]*100),image['target'])
                # print(list(pred)[0][0],image.get_by_name(local_outputs[0].name),image.get_by_name('Open'))
                div_arr.append(list(pred)[0][0] / image.get_by_name(local_outputs[0].name))
            elif self.tagrte_type == IMAGE:
                if len(pred.shape) == 2:
                    continue
                preddicted_arr.append(try_convert_float(pred[0][0][0]))
                target_arr.append(image.get_by_name(local_outputs[0].name))
            elif self.tagrte_type == TIME_SERIES:

                if type(pred[0][0]) != int and type(pred[0][0]) != np.int64:
                    preddicted_arr += list(pred[0][0])

                # preddicted_arr.append(try_convert_float(pred[0][0]))
                target_arr.append(image.get_by_name(local_outputs[0].name))
        print("len(preddicted_arr)", len(preddicted_arr))
        print("len(target_arr)", len(target_arr))
        plt.plot(preddicted_arr[-100:])
        plt.plot(target_arr[:len(preddicted_arr)][-100:])
        plt.show()
        if len( preddicted_arr) > len(target_arr):
            preddicted_arr = preddicted_arr[:len(target_arr)]
        mse = tf.keras.losses.MeanSquaredError()(tf.cast(preddicted_arr, tf.float32),
                                                 tf.cast(target_arr[:len(preddicted_arr)], tf.float32))

        print("!!MSE!!!")
        print(mse)

    def submit(self, images):
        f = open('submission.csv', 'w')
        writer = csv.writer(f)
        local_arr = []
        for element in self.data_schema_output:
                local_arr.append(element.name)
        writer.writerow(local_arr)
        for image in images:
            pred_indedx, _ = self.predict(image)
            if type(_[0][0]) ==  DataCollection:
                _ = _[0][0]
                local_arr = []
                for element in self.data_schema_output:
                    local_arr.append(_.get_by_name(element.name))
                writer.writerow(local_arr)


        writer.writerow([])
        f.close()
