from utils.Agent import *

import tensorflow as tf
import numpy as np
import random
from PIL import Image
import sys


# MEMORY_LIMIT = 3096 * 2.2
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# np.set_printoptions(threshold=sys.maxsize)
# if gpus:
#    try:
#        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])
#    except RuntimeError as e:
#        print(e)


class ConstrativeLearning(object):

    def __init__(self, inputs, outputs, data_schema):
        self.model = None
        self.init_neural_network(inputs, outputs, data_schema)

    def init_neural_network(self, inputs, outputs, data_schema):
        local_input = inputs[0]
        self.local_output = outputs[0]
        if local_input.type == '2D_F':
            image_width = local_input.shape[0]
            image_height = local_input.shape[1]
            image_depth = local_input.shape[2]
        self.cutout_height = 20
        self.cutout_width = 20
        # pls give money for something better than 3080
        if image_height > 32:
            image_height = 32
        if image_width > 32:
            image_width = 32
        self.image_depth = image_depth
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.N = 2
        self.s_arr = [[]] * self.N * 2
        self.tau = 0.1

        self.branch_A_input = tf.keras.Input(shape=(image_height, image_width, image_depth))

        self.branch_A = tf.keras.applications.ResNet50(include_top=False,
                                                       input_shape=(image_height, image_width, image_depth))(
            self.branch_A_input)

        self.branch_A = tf.keras.layers.Dense(128, name='my_dense_a')(self.branch_A)

        self.branch_A_model = tf.keras.Model(inputs=self.branch_A_input, outputs=self.branch_A)

        self.branch_A_model.compile(optimizer='Adam')

        self.branch_B_input = tf.keras.Input(shape=(image_height, image_width, image_depth))

        self.branch_B = tf.keras.applications.ResNet50(include_top=False,
                                                       input_shape=(image_height, image_width, image_depth))(
            self.branch_B_input)

        self.branch_B = tf.keras.layers.Dense(128, name='my_dense_b')(self.branch_B)

        self.branch_B_model = tf.keras.Model(inputs=self.branch_B_input, outputs=self.branch_B)

        self.branch_B_model.compile(optimizer='Adam')

        self.comm_model_input = tf.keras.Input(shape=(128))

        self.comm_model = tf.keras.layers.Dense(128)(self.comm_model_input)

        self.comm_model = tf.keras.Model(inputs=self.comm_model_input, outputs=self.comm_model)
        self.comm_model.compile(optimizer='Adam')

    def transformation(self, img, trans_type='crop'):
        img = img.copy()
        # pls give money for something better than 3080
        if img.shape[0] > 32 or img.shape[1] > 32:
            img = Image.fromarray(img)
            img = img.resize(size=(32, 32))
            img = np.asarray(img)
            img = img.tolist()
        if trans_type == 'crop':
            start_width = random.randint(0, len(img[0]) - self.cutout_width)
            start_height = random.randint(0, len(img) - self.cutout_height)

            for i in range(start_width, start_width + self.cutout_width):
                for j in range(start_height, start_height + self.cutout_height):
                    img[j][i] = [0] * self.image_depth
            return np.array(img)
        elif trans_type == 'color_dist':
            img = img[:][:][::-1]
            return np.array(img)

    def replacenan(self, t):
        t = tf.where(tf.math.is_inf(t), tf.ones_like(t), t)
        return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

    def s_sim(self, rep_1, rep_2):
        rep_1 = self.replacenan(rep_1)
        rep_2 = self.replacenan(rep_2)
        return tf.divide(tf.tensordot(tf.transpose(rep_1), rep_2, axes=0),
                         tf.tensordot(tf.norm(rep_1), tf.norm(rep_2), axes=0))

    def small_loss(self, i, j, s_arr):
        local_exp = tf.exp(tf.divide(s_arr, 0.1))
        local_size = [*local_exp.shape[:3]]
        local_exp = tf.linalg.set_diag(local_exp, tf.zeros(local_size), name=None)
        local_sum = tf.math.multiply_no_nan(tf.math.reduce_sum(local_exp, axis=0), 1)
        local_sum = self.replacenan(local_sum)
        return -tf.math.log(tf.divide(local_sum, tf.exp(tf.divide(s_arr, 0.1))))

    def big_loss(self, s_arr):
        local_sum = 0

        for k in range(self.N):
            local_sum += tf.add(self.small_loss(2 * k - 1, 2 * k, s_arr), self.small_loss(2 * k, 2 * k - 1, s_arr))
        local_sum = self.replacenan(local_sum)
        return tf.reduce_mean(local_sum / 2 * self.N)

    def train(self, images, force_train=False):
        trained_imgs = 0
        for img in images:

            img = img.get_by_name('Image')
            if img is None:
                continue
            trained_imgs += 1
            self.aug_1 = self.transformation(img)
            self.aug_2 = self.transformation(img, trans_type='color_dist')
            self.aug_1 = tf.image.per_image_standardization(self.aug_1)
            self.aug_2 = tf.image.per_image_standardization(self.aug_2)
            local_my_trainable_weights = []
            for element in (self.branch_A_model.trainable_variables + self.branch_B_model.trainable_variables):
                if 'my_dense_' in element.name:
                    local_my_trainable_weights.append(element)
            with tf.GradientTape() as tape:
                tape.watch(local_my_trainable_weights)
                z_arr = []
                self.rep_1 = self.branch_A_model(np.expand_dims(self.aug_1, axis=0))
                self.rep_2 = self.branch_B_model(np.expand_dims(self.aug_2, axis=0))

                self.rep_1 = tf.squeeze(self.rep_1)
                self.rep_2 = tf.squeeze(self.rep_2)

                z_arr.append(self.rep_1)
                z_arr.append(self.rep_2)
                self.N = len(z_arr)
                for j in range(self.N):
                    for i in range(self.N):
                        self.s_arr[j].append(self.s_sim(self.rep_1, self.rep_2))

                local_loss = self.big_loss(self.s_arr)
                local_loss = self.replacenan(local_loss)

            gradient_context = tape.gradient(local_loss, local_my_trainable_weights)

            self.optimizer.apply_gradients(zip(gradient_context, local_my_trainable_weights))
            if trained_imgs > 20:
                break
        trained_imgs = 0
        for img in images:
            for data in img.data_collection:
                if data.name == self.local_output.name:
                    local_y = data.data
                    break
            img = img.get_by_name('Image')
            if img is None:
                continue
            trained_imgs += 1
            self.aug_1 = self.transformation(img)
            self.aug_2 = self.transformation(img, trans_type='color_dist')
            self.aug_1 = tf.image.per_image_standardization(self.aug_1)
            self.aug_2 = tf.image.per_image_standardization(self.aug_2)

            z_arr = []
            self.rep_1 = self.branch_A_model(np.expand_dims(self.aug_1, axis=0))
            self.rep_2 = self.branch_B_model(np.expand_dims(self.aug_2, axis=0))

            self.rep_1 = tf.squeeze(self.rep_1)
            self.rep_2 = tf.squeeze(self.rep_2)
            if np.isnan(self.rep_1).any() or np.isnan(self.rep_2).any():
                continue
            z_arr.append(self.rep_1)
            z_arr.append(self.rep_2)

            self.comm_model.fit(np.array(z_arr), np.array(local_y))

    def predict(self, image):
        local_image = image.get_by_name('Image')
        if local_image is None:
            return [0]
        self.aug_1 = self.transformation(img)
        self.aug_2 = self.transformation(img, trans_type='color_dist')
        self.aug_1 = tf.image.per_image_standardization(self.aug_1)
        self.aug_2 = tf.image.per_image_standardization(self.aug_2)

        z_arr = []
        self.rep_1 = self.branch_A_model(np.expand_dims(self.aug_1, axis=0))
        self.rep_2 = self.branch_B_model(np.expand_dims(self.aug_2, axis=0))

        self.rep_1 = tf.squeeze(self.rep_1)
        self.rep_2 = tf.squeeze(self.rep_2)
        if np.isnan(self.rep_1).any() or np.isnan(self.rep_2).any():
            return None
        z_arr.append(self.rep_1)
        z_arr.append(self.rep_2)
        return self.comm_model.predict(np.array(z_arr))[0]
