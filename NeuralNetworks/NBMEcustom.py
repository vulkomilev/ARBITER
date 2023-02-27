import random

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot

from utils.Agent import *


class NBMEcustom(Agent):

    def __init__(self, inputs, outputs, data_schema, class_num):
        self.model = None
        self.init_neural_network(inputs, outputs, data_schema, class_num)
        self.total_tested = 0
        self.good_tested = 0

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

        input_model = tf.keras.Input(shape=(331, 331, 3))
        # model_mid = preprocessing.Rescaling(1.0 / 255)(input_model)
        # ResNet152V2  70%
        model_mid = tf.keras.applications.NASNetLarge(include_top=False,
                                                      weights='imagenet')(
            input_model)

        model_mid = tf.keras.layers.GlobalAveragePooling2D()(model_mid)
        # model_mid = tf.keras.layers.Dense(100, activation='relu')(model_mid)
        # model_mid = tf.keras.layers.Dropout(0.1)(model_mid)
        # model_mid = tf.keras.layers.Flatten()(model_mid)
        # model_mid = tf.keras.layers.Dense(100, activation='relu')(model_mid)
        model_mid = tf.keras.layers.Dense(37, activation='softmax')(model_mid)
        self.model = tf.keras.Model(inputs=input_model, outputs=model_mid)

        self.model.layers[1].trainable = False
        self.model.compile(optimizer="adam", loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])

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
            local_image = images.get_by_name('Image')
            if local_image is None:
                return []
            local_image = np.array(local_image)
            # print(sum(np.hstack(local_image)))
            # exit(0)

            mean = 0.0  # some constant
            std = 32.0  # some constant (standard deviation)
            print("local_image.shape", local_image.shape)

            local_image = preprocess_fix_dim(local_image, 331,
                                             331)  # np.pad(np.array(local_image), ((5, 6), (5, 6), (0, 0)), mode='constant', constant_values=255)
            local_image = cv2.threshold(local_image, 84, 255, cv2.THRESH_BINARY)[1]
            return local_image
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
            print("local_image.shape", local_image.shape)

            local_image = preprocess_fix_dim(local_image, 331,
                                             331)  # local_image = np.pad(np.array(local_image), ((5, 6), (5, 6), (0, 0)), mode='constant', constant_values=255)
            local_image = cv2.threshold(local_image, 84, 255, cv2.THRESH_BINARY)[1]
            local_image = np.roll(local_image, (random.randint(0, 10), random.randint(0, 10)), (0, 1))
            '''
            noisy_img = local_image + np.random.normal(mean, std, local_image.shape)
            local_image = np.clip(noisy_img, 0, 255)




            x_resize = random.randint(40,75)
            y_resize = random.randint(40, 75)

            local_image = cv2.resize(local_image, dsize=(x_resize, y_resize), interpolation=cv2.INTER_CUBIC)
            x_resize_1 = int((75-x_resize)/2)
            x_resize_2 = (75-x_resize)-x_resize_1

            y_resize_1 = int((75-y_resize)/2)
            y_resize_2 = (75-y_resize)-y_resize_1

            local_image = np.pad(np.array(local_image), ( (y_resize_1, y_resize_2),(x_resize_1, x_resize_2), (0, 0)), mode='constant', constant_values=255)
            '''

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
        x_train, y_train = self.prepare_data(images, in_train=True)

        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(0)
        if Path('./checkpoints/' + ckpt_name).exists() and not force_train:
            self.model = tf.keras.models.load_model('./checkpoints/' + ckpt_name)
        else:
            print(self.model.summary())
            print(np.array(y_train).shape)
            # for element in x_train:
            #    plt.imshow(element)
            #    plt.show()
            print('FIT')
            self.model.fit(x_train, y_train, batch_size=32, epochs=30, validation_split=0.1)
            self.model.save('./checkpoints/' + ckpt_name)

    def predict(self, image):
        img = self.prepare_data(image, in_train=False)
        if len(img) == 0:
            return
        plt.imshow(img)
        plt.show()
        result = self.model.predict(np.array([img]), batch_size=1)
        print('-------------------------------')
        for local_result in result:
            # print('letter',image.get_by_name('letter'))
            key_arr = ['M', 'U', 'J', '6', 'C', 'V', 'E', 'F', 'N', 'L', 'A', 'G', 'P', '4', '7', 'Y', 'O', 'T',
                       '2', 'I', '5', 'K', '0', '8', 'W', '3', 'X', '1', 'H', 'S', '9', 'Q', 'Z', 'R', 'D', 'B']
            print(local_result)
            print('result', key_arr[np.argmax(local_result) - 1], np.argmax(local_result))
        return


        img, preprocessed_image = preprocess_image(img)

        image, contours = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image2, contours2 = cv2.findContours(preprocessed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        letter_image_regions = []
        letterImages = []
        letterImagesX = []
        letterImagesY = []
        if len(image) == 3:
            image = image2
        maxHeight = 0
        for c in image:

            (x, y, w, h) = cv2.boundingRect(c)
            if h > maxHeight:
                maxHeight = h
        print("len(image) ", len(image))
        xArray = []
        xOrder = []
        for c in image:

            (x, y, w, h) = cv2.boundingRect(c)

            # if x == 0 and y == 0:
            #    continue
            # if len(image) != 4:
            #    if w < 50 or maxHeight < 50:
            #        continue
            if len(image) == 5:

                print("x, y ", (x, y))
                letter_image_regions.append((x, y - (maxHeight - h), w, maxHeight))
                xArray.append(x)
            else:
                if w / h > 1.8:
                    third_width = int(w / 3)
                    xArray.append(x)
                    letter_image_regions.append((x, y, third_width, maxHeight))
                    letter_image_regions.append((x + third_width, y, third_width, maxHeight))
                    letter_image_regions.append(
                        (x + third_width + third_width, y, third_width + third_width, maxHeight))
                elif w / h > 1.25:
                    half_width = int(w / 2)
                    xArray.append(x)
                    letter_image_regions.append((x, y, half_width, maxHeight))
                    letter_image_regions.append((x + half_width, y, half_width, maxHeight))
                else:
                    xArray.append(x)
                    letter_image_regions.append((x, y, w, maxHeight))

        # if len(letter_image_regions) != 4:
        #    print('====== ERROR {} out of 4 letters found in file: {}'.format(len(letter_image_regions), filename))
        #    return '''
        #                <!doctype html>
        #                <title>ERROR</title>
        #                <h1>less than  4 letters found </h1>
        #                '''
        print(xArray)
        sortedList = xArray[:]
        sortedList.sort()
        for element in sortedList:
            xOrder.append(xArray.index(element))
        print('xOrder ', xOrder)
        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            print('y ', y)
            print('x ', x)
            print('h ', h)
            print('w ', w)
            letter_image = img[y - 2:y + h + 2, x - 2:x + w + 2]
            letterImages.append(letter_image)
            letterImagesX.append(x - 2)
            letterImagesY.append(y - 2)

        returnCapcha = ""

        oldLabels = []

        for i in range(0, 1):
            for imageLetter, imageY, imageX in zip(letterImages, letterImagesY, letterImagesX):
                if not len(oldLabels) == 0:
                    oldLabels = []

                letter_image = imageLetter.tolist()

                letter_image = preprocess_remove_gray(letter_image)
                if len(letter_image) == 0:
                    continue
                letter_image = preprocess_fix_dim(letter_image)
                letter_image = np.array(letter_image, dtype=np.uint8)

                # draw.text((x, y),"Sample Text",(r,g,b))
                im = Image.fromarray(letter_image)

                local_image = letter_image
                # pyplot.imshow(local_image)
                # pyplot.show()
                if local_image is None:
                    return [0]
                x = self.prepare_data(np.array(local_image))
                # pyplot.imshow(x)
                # pyplot.show()
                print('x.shape', x.shape)

                result = self.model.predict(np.array([x]), batch_size=1)

                print('-------------------------------')

                for local_result in result:
                    # print('letter',image.get_by_name('letter'))
                    key_arr = ['M', 'U', 'J', '6', 'C', 'V', 'E', 'F', 'N', 'L', 'A', 'G', 'P', '4', '7', 'Y', 'O', 'T',
                               '2', 'I', '5', 'K', '0', '8', 'W', '3', 'X', '1', 'H', 'S', '9', 'Q', 'Z', 'R', 'D', 'B']
                    print('result', key_arr[np.argmax(local_result) - 1], np.argmax(local_result))
                # im.save("./" + str(key_arr[np.argmax(local_result)-1])+'-'+str(random.randint(1000000, 2000000)) + ".jpeg")
                pyplot.imshow(np.array(x))
                pyplot.show()
                self.total_tested += 1
                # if int(image.get_by_name('letter'))  == np.argmax(result):
                #    self.good_tested+=1
                # print((self.good_tested/self.total_tested)*100)

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

    # for i in range(0, len(raw_img)):
    #    for j in range(len(raw_img[i]), height):
    #        raw_img[i].append(0)

    # for i in range(0, len(raw_img)):
    #    for j in range(0, len(raw_img[i])):
    #        num = raw_img[i][j]
    #        raw_img[i][j] = [num, num, num]
    return raw_img
