import random
import string

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.Agent import *


# from google.colab.patches import cv2_imshow

class KerasOcr(Agent):

    def __init__(self, inputs, outputs, data_schema, class_num):
        self.model = None
        self.init_neural_network(inputs, outputs, data_schema, class_num)
        self.total_tested = 0
        self.good_tested = 0

        self.DEFAULT_ALPHABET = string.digits + string.ascii_lowercase
        self.alphabets = self.DEFAULT_ALPHABET
        self.blank_index = len(self.alphabets)

    def run_tflite_model(self, image_path, quantization):
        input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        input_data = cv2.resize(input_data, (200, 31))
        input_data = input_data[np.newaxis]
        input_data = np.expand_dims(input_data, 3)
        input_data = input_data.astype('float32') / 255
        path = f'ocr_{quantization}.tflite'
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        return output

    def init_neural_network(self, inputs, outputs, data_schema, class_num):
        local_input = inputs[0]
        self.local_output = outputs[0]
        class_num = 37
        self.local_output.type = CATEGORY
        if self.local_output.type == REGRESSION:
            self.num_classes = 1
            loss = 'mean_squared_error'
        elif self.local_output.type == REGRESSION_CATEGORY:
            self.num_classes = 100
            loss = 'categorical_crossentropy'
        elif self.local_output.type == CATEGORY or self.local_output.type == 'str':
            self.num_classes = class_num
            loss = 'categorical_crossentropy'
        self.num_classes = 37
        height_img = None
        width_img = None
        depth_img = None
        if local_input.type == 'int':
            width_img = local_input.data
            height_img = 1
            depth_img = 1
        elif local_input.type == '2D_F':
            width_img = local_input.shape[0]
            height_img = local_input.shape[1]
            depth_img = 3

        input_model = tf.keras.Input(shape=(75, 75, 3))
        # model_mid = preprocessing.Rescaling(1.0 / 255)(input_model)
        model_mid = tf.keras.applications.MobileNetV2(include_top=False,
                                                      weights='imagenet')(
            input_model)
        model_mid = tf.keras.layers.GlobalAveragePooling2D()(model_mid)
        model_mid = tf.keras.layers.Dense(100, activation='relu')(model_mid)
        model_mid = tf.keras.layers.Dropout(0.1)(model_mid)
        # model_mid = tf.keras.layers.Flatten()(model_mid)
        model_mid = tf.keras.layers.Dense(100, activation='relu')(model_mid)
        model_mid = tf.keras.layers.Dense(37, activation='softmax')(model_mid)
        self.model = tf.keras.Model(inputs=input_model, outputs=model_mid)

        self.model.layers[1].trainable = False
        self.model.compile(optimizer="sgd", loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])

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
        # np.pad(np.array(images), ((5,6),(5,6),(0, 0)), mode='constant',constant_values=255)
        local_y_labels = []
        if not in_train:
            images = np.array(images)
            print('images.shape', images.shape)
            return_image = np.repeat(images[:, :, np.newaxis], 3, axis=2)
            print('return_image.shape', return_image.shape)
            return return_image
            # return np.pad(np.array(images), ((5,6),(5,6),(0, 0)), mode='constant',constant_values=255)#(np.array(images),(75,75,3))
        for image in images:
            local_image = image.get_by_name('Image')
            if local_image is None:
                continue
            local_image = np.array(local_image)
            # print(sum(np.hstack(local_image)))
            # exit(0)

            mean = 0.0  # some constant
            std = 32.0  # some constant (standard deviation)
            noisy_img = local_image + np.random.normal(mean, std, local_image.shape)
            local_image = np.clip(noisy_img, 0, 255)

            local_image = np.pad(np.array(local_image), ((5, 6), (5, 6), (0, 0)), mode='constant', constant_values=255)

            local_image = np.roll(local_image, (random.randint(0, 30), random.randint(0, 30)), (0, 1))

            x_resize = random.randint(40, 75)
            y_resize = random.randint(40, 75)

            local_image = cv2.resize(local_image, dsize=(x_resize, y_resize), interpolation=cv2.INTER_CUBIC)
            x_resize_1 = int((75 - x_resize) / 2)
            x_resize_2 = (75 - x_resize) - x_resize_1

            y_resize_1 = int((75 - y_resize) / 2)
            y_resize_2 = (75 - y_resize) - y_resize_1

            local_image = np.pad(np.array(local_image), ((y_resize_1, y_resize_2), (x_resize_1, x_resize_2), (0, 0)),
                                 mode='constant', constant_values=255)

            # pyplot.imshow(local_image)
            # pyplot.show()
            local_x_train_arr.append(local_image)
            for data in image.data_collection:
                if data.name == self.local_output.name:
                    local_data = data.data
                    break
            if self.local_output.type == CATEGORY or 'str':
                # print('self.local_output.name',image.get_by_name(self.local_output.name))
                if image.get_by_name(self.local_output.name) not in local_y_labels:
                    local_y_labels.append(image.get_by_name(self.local_output.name))
                print("len(local_y_labels)", image.get_by_name(self.local_output.name))
                local_y_train_arr.append(tf.one_hot(image.get_by_name(self.local_output.name), self.num_classes))
            elif self.local_output.type == REGRESSION_CATEGORY:
                local_target = [0] * 100
                local_target[int(local_data) - 1] = 1
                local_y_train_arr.append(local_target)
            elif self.local_output.type == REGRESSION:
                local_target = [int(local_data) / 100.0]
                local_y_train_arr.append(local_target)
        # local_x_train_arr = local_x_train_arr[:10000]
        # local_y_train_arr = local_y_train_arr[:10000]
        return np.array(local_x_train_arr), np.array(local_y_train_arr)

    def train(self, images, force_train=False):
        pass

    def predict(self, image):

        filenamePNG = str(random.randint(1000000, 2000000)) + ".png"
        image_path = './data_sets/captcha/createCaptcha10.png'
        image_path = 'None'
        tflite_output = self.run_tflite_model(image_path, 'dr')
        for output in tflite_output:
            print(output)
        final_output = "".join(
            self.alphabets[index] for index in tflite_output[0] if index not in [self.blank_index, -1])
        print(final_output)
        plt.imshow(cv2.imread(image_path))
        plt.show()
        # Running Float16 Quantization
        tflite_output = self.run_tflite_model(image_path, 'float16')
        final_output = "".join(
            self.alphabets[index] for index in tflite_output[0] if index not in [self.blank_index, -1])
        print(final_output)
        plt.imshow(cv2.imread(image_path))
        plt.show()
        return None  # result


def preprocess_remove_gray(raw_img):
    """
    The preprocess function for removing the gray
    portion of the image.Please note that here we
    also normalize the values between 0.0 and 1.0
    for the neural network.

    Parameters
    ----------
    raw_img : array
        The array to be processed

    Returns
    ----------
    raw_img: array
        The processed array
    """

    for row in range(0, len(raw_img)):
        for column in range(0, len(raw_img[row])):
            if raw_img[row][column] < 128:
                raw_img[row][column] = 0
            else:
                raw_img[row][column] = 255  # 1.0

    return raw_img


def preprocess_remove_dots(raw_img):
    """
    The preprocess function for removing the dots
    from the image.This function is not used due
    to high time complexity and poor results.

    Parameters
    ----------
    raw_img : array
        The array to be processed

    Returns
    ----------
    raw_img: array
        The processed array
    """
    for i in range(0, len(raw_img)):
        for j in range(0, len(raw_img[i])):
            isAlone = True
            nStart = -1
            nEnd = 2
            mStart = -1
            mEnd = 2
            if i == 0:
                nStart = 0
            if i == len(raw_img):
                nEnd = 1
            if j == 0:
                Start = 0
            if j == len(raw_img[i]):
                mEnd = 1
            for n in range(nStart, nEnd):
                for m in range(mStart, mEnd):
                    if n == m:
                        continue
                    if raw_img[n][m] == 1.0:
                        isAlone = False
                        break
            if isAlone:
                raw_img[i][j] = 0

    return raw_img


def preprocess_image(raw_img):
    """
    Prepares the image for the finding of
    the contours.

    Parameters
    ----------
    raw_img : numpy array
        Array of the image

    Returns
    ----------
    raw_img: cv2Image
        The resized original image
    processed_res_img: 8-bit single-channel image
        The image for finding the contours
    """
    height, width, depth = raw_img.shape

    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    raw_img = cv2.resize(raw_img, (width * 13, height * 13))
    raw_img = cv2.copyMakeBorder(raw_img, 5, 5, 5, 5, cv2.BORDER_REPLICATE)

    processed_res_img = cv2.dilate(raw_img, np.ones((4, 4), np.uint8))
    processed_res_img = cv2.medianBlur(processed_res_img, 15)
    processed_res_img = cv2.erode(processed_res_img, np.ones((6, 2), np.uint8), iterations=2)

    ret1, processed_res_img = cv2.threshold(processed_res_img, thresh=128, maxval=255,
                                            type=cv2.THRESH_BINARY_INV)  # | cv2.THRESH_OTSU)

    return raw_img, processed_res_img


def preprocess_fix_dim(raw_img, height=75, width=75):
    """
    Makes the dimensions of the image
    compatible with the input of the
    neural network.

    Parameters
    ----------
    raw_img : numpy array
        Array of the image
    height : int
        wanted height
    width : int
        wanted width

    Returns
    ----------
    raw_img: array
        The resized original image

    """
    raw_img = cv2.resize(np.array(raw_img, np.uint8), (height, width))
    raw_img = raw_img.tolist()
    # for i in range(0, len(raw_img)):
    #    for j in range(len(raw_img[i]), height):
    #        raw_img[i].append(0)

    # for i in range(0, len(raw_img)):
    #    for j in range(0, len(raw_img[i])):
    #        num = raw_img[i][j]
    #        raw_img[i][j] = [num, num, num]
    return raw_img
