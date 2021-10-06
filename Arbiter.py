import csv

from NeuralNetwork import MyResNet50
from boost import BoostClass
import tensorflow as tf
import numpy as np


class Arbiter(object):

    def __init__(self, img_height, img_width, class_num):
        self.img_height = img_height
        self.img_width = img_width
        self.class_num = class_num
        self.init_agents(img_height, img_width, self.class_num)
        self.init_neural_network()

    def init_neural_network(self):
        agent_size = 0
        for element in dir(self):
            if 'agent_' in element:
                agent_size += 1
        self.arbiter_neural_network_input = tf.keras.Input((agent_size * self.class_num))
        layer_size = agent_size
        self.arbiter_neural_network = tf.keras.layers.Dense(int(layer_size) * self.class_num)( \
            self.arbiter_neural_network_input)
        layer_size = agent_size / 2.0
        while layer_size > 1:
            self.arbiter_neural_network = tf.keras.layers.Dense(int(layer_size) * self.class_num)( \
                self.arbiter_neural_network)
            layer_size = agent_size / 2.0
        self.arbiter_neural_network = tf.keras.layers.Dense(self.class_num)( \
            self.arbiter_neural_network)
        self.arbiter_neural_model = tf.keras.Model(inputs=self.arbiter_neural_network_input,
                                                   outputs=self.arbiter_neural_network)
        self.arbiter_neural_model.compile(optimizer="adam", loss='categorical_crossentropy')

    def init_agents(self, img_height, img_width, class_num):
        print("class_num", class_num)
        self.agent_local_resnet50 = MyResNet50(img_height, img_width, class_num)

        self.agent_boost = BoostClass(image_height=img_height, image_width=img_width, labels_size=class_num)

    def train(self, image_collection, force_train=False, train_arbiter=True):
        local_agents = []
        for element in dir(self):
            if 'agent_' in element:
                local_agents.append(element)

        for element in local_agents:
            self.__getattribute__(element).train(image_collection)

        if train_arbiter:
            local_y = []
            local_x = []
            for image in image_collection:
                local_x_row = []
                _, local_predict_dict = self.predict(image)
                for key in local_predict_dict.keys():
                    local_x_row += local_predict_dict[key].tolist()
                local_x.append(local_x_row)
                local_y.append(image['img_id'])
            self.train_arbiter(local_x, local_y)

    def train_arbiter(self, agent_results, target):
        x_fit = []
        y_fit = []
        for agent_result, y in zip(agent_results, target):
            local_arr_x = []
            for x in agent_result:
                local_arr_x += x
            x_fit.append(local_arr_x)
            y_fit.append(tf.one_hot(y, self.class_num))
        x_fit = np.array(x_fit)
        y_fit = np.array(y_fit)

        self.arbiter_neural_model.fit(x_fit, y_fit, epochs=80)

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
            local_arr_x += x.tolist()[0]
        local_arr_x = np.squeeze(local_arr_x)
        result = self.arbiter_neural_model.predict(np.array([local_arr_x]))
        return result, local_predictions

    def evaluate(self, images):
        correct_count = 0
        wrong_count = 0
        print('-------------------')
        for image in images:
            pred_indedx, _ = self.predict(image)
            pred_indedx = np.argmax(pred_indedx, axis=1).tolist()[0]

            print("pred_indedx", pred_indedx, '-', image['img_id'])
            if pred_indedx == image['img_id']:
                correct_count += 1
            else:
                wrong_count += 1
            print(correct_count, '/', (correct_count + wrong_count))
            print('', end='\r')

    def submit(self, images, images_train):
        f = open('submission.csv')
        writer = csv.writer(f)

        f.close()
        for image in images:
            submit_row = [image['img_name'], '']
            pred_indedx, _ = self.predict(image)
            pred_indedx = np.argmax(pred_indedx, axis=1).tolist()[0]
            for image_second in images_train:
                if pred_indedx == image_second['img_id']:
                    submit_row[1] += image_second['img_name'] + '\n'

            writer.writerow(submit_row)
        f.close()
