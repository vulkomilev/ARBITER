import tensorflow as tf
import xgboost as xgb
from utils.utils import REGRESSION, CATEGORY, TIME_SERIES, REGRESSION_CATEGORY
from tensorflow.keras.layers.experimental import preprocessing
from xgboost import XGBClassifier
import numpy as np
import cv2
from pathlib import Path


class BoostClass(object):

    def __init__(self, inputs, outputs, data_schema):
        self.local_input = inputs[0]
        self.local_output = outputs[0]
        if self.local_output.type == REGRESSION_CATEGORY or self.local_output.type == REGRESSION:
            num_classes = 100
        self.target_type = self.local_output.type
        if self.local_input.type == 'int':
            width_img = self.local_input.data
            height_img = 1
            depth_img = 1
        elif self.local_input.type == '2D_F':
            width_img = self.local_input.shape[0]
            height_img = self.local_input.shape[1]
            depth_img = self.local_input.shape[2]
        self.data_schema = data_schema
        self.local_inputs = inputs
        self.local_inputs_name = []
        self.local_outputs_name = []
        for element in inputs:
            self.local_inputs_name.append(element.name)
        self.local_outputs = outputs
        for element in outputs:
            self.local_outputs_name.append(element.name)
        self.classifier = self.init_network(labels_size=num_classes, image_height=height_img, image_width=width_img)

    def init_network(self, labels_size=10, image_height=256, image_width=256):

        if image_width is not None:
            network_input = tf.keras.Input(shape=(image_height, image_width, 3))
            network = preprocessing.Rescaling(1.0 / 255)(network_input)
            network = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(image_height, image_width, 3),
                                                       pooling='avg')(network)

            network = tf.keras.layers.Dense(labels_size, activation='sigmoid')(network)
            self.cnn_model = tf.keras.Model(inputs=network_input, outputs=network)
            self.cnn_model.layers[1].trainable = False
            self.cnn_model.compile(optimizer='adam', loss='categorical_crossentropy')
        return XGBClassifier(objective='multi:softprob', num_class=200, learning_rate=0.05)

    def contur_image(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.blur(grey_img, (1, 1), 0)
        contur_img = cv2.Sobel(src=blur_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        contur_img = np.float32(contur_img)
        contur_img = np.stack((contur_img,) * 3, axis=-1)
        return contur_img

    def prepare_data(self, images, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []
        if not in_train:
            return np.array(self.contur_image(images))
        for image in images:
            local_x = []
            local_y = []
            if 'cnn_model' in dir(self):
                local_x_train_arr.append(np.array(self.contur_image(image['img'])))
            else:
                for element in image.data_collection:
                    if element.name != self.local_output.name and element.name in self.local_inputs_name:
                        local_x.append(np.array(element.data))
                    elif element.name in self.local_outputs_name:
                        if self.local_output.type == CATEGORY:
                            local_y = tf.one_hot(element.data, self.labels_size)
                        elif self.local_output.type == REGRESSION_CATEGORY:
                            local_target = [0] * 100
                            local_target[int(element.data) - 1] = 1
                            local_y = local_target
                        elif self.local_output.type == REGRESSION:
                            local_target = [0] * 100
                            local_target[int(element.data) - 1] = 1
                            local_y = local_target
                if len(local_y) > 0 and len(local_x) > 0:
                    local_y_train_arr.append(local_y)
                    local_x_train_arr.append(local_x)
        if len(local_y_train_arr) == 0 or len(local_x_train_arr) == 0:
            return [], []
        return np.array(local_x_train_arr), np.expand_dims(np.array(local_y_train_arr), axis=2)

    def predict(self, image):

        if 'cnn_model' in dir(self):
            x = self.prepare_data(np.array(image['Image']))
            x_boost_train = np.expand_dims(self.cnn_model.predict(np.array([x]))[0], axis=1)
            x_boost_train = np.squeeze(x_boost_train)
        else:
            local_input_arr = []
            for element in image.data_collection:
                if element.name in self.local_inputs_name:
                    local_input_arr.append(element.data)
            x_boost_train = np.squeeze(local_input_arr)
        ypred = self.classifier.predict(
            np.expand_dims(x_boost_train, axis=0))  # xgb.DMatrix(np.expand_dims(x_boost_train, axis=0)))
        if self.local_output.type == REGRESSION:
            return ypred / 100.0
        elif self.local_output.type == REGRESSION_CATEGORY:
            return ypred

    def train(self, images, force_train=False):
        if Path('./checkpoints/' + 'cnn_boost_model').exists() and Path(
                "../checkpoints/boost-model.json").exists() and not force_train:
            self.classifier = xgb.Booster()
            self.classifier.load_model("./checkpoints/boost-model.json")
            self.model = tf.keras.models.load_model('./checkpoints/' + 'cnn_boost_model')
        else:

            x_train, y_train = self.prepare_data(images, in_train=True)
            if 'cnn_model' in dir(self):
                self.cnn_model.fit(x_train, y_train, epochs=1)
                self.cnn_model.save('./checkpoints/' + 'cnn_boost_model')
            x_boost_train = []
            y_boost_train = y_train[:]
            x_train_arr = []
            y_train_arr = []

            for image in images:

                if 'cnn_model' in dir(self):
                    x = self.prepare_data(np.array(image.get_by_name('Image')))
                    x_boost_train.append(np.expand_dims(self.cnn_model.predict(np.array([x]))[0], axis=1))
                else:

                    x_boost_train = x_train
            for x, y in zip(x_boost_train, y_boost_train):
                x_train_arr.append(x)
                y_train_arr.append(tf.argmax(y, axis=0))
            if len(x_train_arr) == 0 or len(y_train_arr) == 0:
                return
            self.classifier = self.classifier.fit(np.array(x_train_arr), np.array(y_train_arr), verbose=True)

            self.classifier.save_model("./checkpoints/boost-model.json")
        return
