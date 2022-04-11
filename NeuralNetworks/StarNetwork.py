from utils.Agent import *


class StarNetwork(Agent):

    def __init__(self, inputs, outputs, data_schema):
        self.model = None
        self.init_neural_network(inputs, outputs, data_schema)

    def init_neural_network(self, inputs, outputs, data_schema):
        local_input = inputs[0]
        self.local_output = outputs[0]

        if self.local_output.type == REGRESSION:
            self.num_classes = 1
            loss = 'mean_squared_error'
        elif self.local_output.type == REGRESSION_CATEGORY:
            self.num_classes = 100
            loss = 'categorical_crossentropy'
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
            depth_img = local_input.shape[2]

        input_model_1 = tf.keras.Input(shape=(height_img, width_img, depth_img))
        model_mid_1 = tf.keras.layers.Dense(self.num_classes, activation='relu')(input_model_1)
        model_mid_1 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_1)
        model_mid_1 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_1)
        self.model_1 = tf.keras.Model(inputs=input_model_1, outputs=model_mid_1)
        self.model_1.compile(optimizer="adam", loss=loss)

        input_model_2 = tf.keras.Input(shape=(height_img, width_img, depth_img))
        model_mid_2 = tf.keras.layers.Dense(self.num_classes, activation='relu')(input_model_2)
        model_mid_2 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_2)
        model_mid_2 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_2)
        self.model_2 = tf.keras.Model(inputs=input_model_2, outputs=model_mid_2)
        self.model_2.compile(optimizer="adam", loss=loss)

        input_model_3 = tf.keras.Input(shape=(height_img, width_img, depth_img))
        model_mid_3 = tf.keras.layers.Dense(self.num_classes, activation='relu')(input_model_3)
        model_mid_3 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_3)
        model_mid_3 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_3)
        self.model_3 = tf.keras.Model(inputs=input_model_3, outputs=model_mid_3)
        self.model_3.compile(optimizer="adam", loss=loss)

        input_model_4 = tf.keras.Input(shape=(height_img, width_img, depth_img))
        model_mid_4 = tf.keras.layers.Dense(self.num_classes, activation='relu')(input_model_4)
        model_mid_4 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_4)
        model_mid_4 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_4)
        self.model_4 = tf.keras.Model(inputs=input_model_4, outputs=model_mid_4)
        self.model_4.compile(optimizer="adam", loss=loss)

        input_model_5 = tf.keras.Input(shape=(height_img, width_img, depth_img))
        model_mid_5 = tf.keras.layers.Dense(self.num_classes, activation='relu')(input_model_5)
        model_mid_5 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_5)
        model_mid_5 = tf.keras.layers.Dense(self.num_classes, activation='relu')(model_mid_5)
        self.model_5 = tf.keras.Model(inputs=input_model_5, outputs=model_mid_5)
        self.model_5.compile(optimizer="adam", loss=loss)

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
            local_image = image.get_by_name('Image')
            if local_image is None:
                continue
            local_x_train_arr.append(np.array(self.contur_image(local_image)))
            for data in image.data_collection:
                if data.name == self.local_output.name:
                    local_data = data.data
                    break
            if self.local_output.type == CATEGORY:
                local_y_train_arr.append(tf.one_hot(image['target'], self.num_classes))
            elif self.local_output.type == REGRESSION_CATEGORY:
                local_target = [0] * 100
                local_target[int(local_data) - 1] = 1
                local_y_train_arr.append(local_target)
            elif self.local_output.type == REGRESSION:
                local_target = [int(local_data) / 100.0]
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
        local_image = image.get_by_name('Image')
        if local_image is None:
            return [0]
        x = self.prepare_data(np.array(local_image))
        return self.model.predict(np.array([x]), batch_size=1)[0]
