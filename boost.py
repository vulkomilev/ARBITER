import tensorflow as tf
import xgboost as xgb
from tensorflow.keras.layers.experimental import preprocessing
from xgboost import XGBClassifier
import numpy as np
import cv2
from pathlib import Path


class BoostClass(object):

    def __init__(self, image_height=256, image_width=256, labels_size=10):
        self.labels_size = labels_size
        self.classifier = self.init_network(labels_size=labels_size, image_height=image_height, image_width=image_width)

    def init_network(self, labels_size=10, image_height=256, image_width=256):

        network_input = tf.keras.Input(shape=(image_height, image_width, 3))
        network = preprocessing.Rescaling(1.0 / 255)(network_input)
        network = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(image_height, image_width, 3),
                                                   pooling='avg')(network)
        network = tf.keras.layers.Dense(labels_size, activation='sigmoid')(network)
        self.cnn_model = tf.keras.Model(inputs=network_input, outputs=network)
        self.cnn_model.layers[1].trainable = False
        self.cnn_model.compile(optimizer='adam', loss='categorical_crossentropy')
        return XGBClassifier(objective='multi:softprob', num_class=labels_size, learning_rate=0.05)

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
            local_y_train_arr.append(tf.one_hot(image['img_id'], self.labels_size))
        return np.array(local_x_train_arr), np.expand_dims(np.array(local_y_train_arr), axis=2)

    def predict(self, image):
        x = self.prepare_data(np.array(image['img']))
        x_boost_train = np.expand_dims(self.cnn_model.predict(np.array([x]))[0], axis=1)
        x_boost_train = np.squeeze(x_boost_train)
        ypred = self.classifier.predict(xgb.DMatrix(np.expand_dims(x_boost_train, axis=0)))
        return ypred

    def train(self, images, force_train=False):
        if Path('./checkpoints/' + 'cnn_boost_model').exists() and Path(
                "./checkpoints/boost-model.json").exists() and not force_train:
            self.classifier = xgb.Booster()
            self.classifier.load_model("./checkpoints/boost-model.json")
            self.model = tf.keras.models.load_model('./checkpoints/' + 'cnn_boost_model')
        else:

            x_train, y_train = self.prepare_data(images, in_train=True)
            self.cnn_model.fit(x_train, y_train, epochs=1)
            self.cnn_model.save('./checkpoints/' + 'cnn_boost_model')
            x_boost_train = []
            y_boost_train = y_train[:]

            for image in images:
                x = self.prepare_data(np.array(image['img']))
                x_boost_train.append(np.expand_dims(self.cnn_model.predict(np.array([x]))[0], axis=1))
            for x, y in zip(x_boost_train, y_boost_train):
                x = np.squeeze(x)
                x = np.expand_dims(x, axis=0)
                self.classifier = self.classifier.fit(x, np.expand_dims(tf.argmax(y, axis=0), axis=0), verbose=True)

            self.classifier.save_model("./checkpoints/boost-model.json")
        return
