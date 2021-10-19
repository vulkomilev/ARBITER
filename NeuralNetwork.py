import cv2
import tensorflow as tf
import numpy as np
from utils import REGRESSION, CATEGORY, TIME_SERIES
from tensorflow.keras.layers.experimental import preprocessing
import re
from pathlib import Path


class MyResNet50(object):

    def __init__(self, height_img, width_img, num_classes, target_type):
        self.model = None
        self.num_classes = num_classes
        self.target_type = target_type
        self.init_neural_network(height_img, width_img, num_classes, target_type)

    def init_neural_network(self, height_img, width_img, num_classes, tagrte_type):
        if tagrte_type == 'Regression':
            num_classes = 100
        input_model = tf.keras.Input(shape=(height_img, width_img, 3))
        model_mid = preprocessing.Rescaling(1.0 / 255)(input_model)
        model_mid = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(height_img, width_img, 3),
                                                     pooling='avg')(
            model_mid)
        model_mid = tf.keras.layers.Dense(num_classes, activation='sigmoid')(model_mid)
        self.model = tf.keras.Model(inputs=input_model, outputs=model_mid)
        self.model.layers[1].trainable = False
        self.model.compile(optimizer="adam", loss="categorical_crossentropy")

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
            local_x_train_arr.append(np.array(self.contur_image(image['img'])))
            if self.target_type == CATEGORY:
                local_y_train_arr.append(tf.one_hot(image['target'], self.num_classes))
            elif self.target_type == REGRESSION:
                local_target = [0] * 100
                local_target[int(image['target']) - 1] = 1
                local_y_train_arr.append(local_target)

        return np.array(local_x_train_arr), np.expand_dims(np.array(local_y_train_arr), axis=2)

    def train(self, images, force_train=False):
        x_train, y_train = self.prepare_data(images, in_train=True)

        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(0)
        if Path('./checkpoints/' + ckpt_name).exists() and not force_train:
            self.model = tf.keras.models.load_model('./checkpoints/' + ckpt_name)
        else:
            self.model.fit(x_train, y_train, epochs=1)
            self.model.save('./checkpoints/' + ckpt_name)

    def predict(self, image):
        x = self.prepare_data(np.array(image['img']))
        return self.model.predict(np.array([x]), batch_size=1)
