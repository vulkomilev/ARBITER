from matplotlib import pyplot

from utils.Agent import *


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()

        # self.encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss = 'categorical_crossentropy')

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x_img, x_type):

        mean, logvar = tf.split(self.encoder(inputs=[x_img]), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class GraphSearch(Agent):

    def __init__(self, inputs, outputs, data_schema, class_num):
        self.model = None
        self.func_map = {}
        self.optimizer = tf.keras.optimizers.Adam(1e-6)
        self.init_neural_network(inputs, outputs, data_schema, latent_dim=(4), class_num=class_num)  # 8*4*3
        self.calc_map_plot_counter = 0
        self.confusion_matrix = {}

    def grap_conv(self, image, activation_matrix, addition_matrix):
        if len(np.intersect1d(image, activation_matrix)) == len(activation_matrix):
            return np.sum([image, addition_matrix])
        return image

    def generate_random_matrix(self, size):
        local_activation_matrix = np.random.randint(0, 2, size)
        local_sum_matrix = np.random.randint(0, 2, size)
        return {'local_activation_matrix': local_activation_matrix,
                'local_sum_matrix': local_sum_matrix}

    def grap_simulation(self, image, matrix_arr):
        for element in matrix_arr:
            image = self.grap_conv(image, element['local_activation_matrix'],
                                   element['local_sum_matrix'])
        return image

    def init_neural_network(self, inputs, outputs, data_schema, latent_dim, class_num):
        local_input = inputs[0]
        self.local_output = outputs[0]
        self.latent_dim = latent_dim
        if local_input.type == 'int':
            width_img = local_input.data
            height_img = 1
            depth_img = 1
        elif local_input.type == '2D_F':
            width_img = local_input.shape[0]
            height_img = local_input.shape[1]
            depth_img = 3

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(width_img, height_img, depth_img)),
            tf.keras.layers.Conv2D(filters=96, kernel_size=(6, height_img), strides=(6, height_img), activation='relu'),
            # tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(36, activation='tanh'),
            # tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(36, activation='tanh'),
            # tf.keras.layers.Dense(36),
            tf.keras.layers.Dense(latent_dim + latent_dim, activation='sigmoid'),
        ]
        )

        self.decoder = tf.keras.Sequential(
            [

                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=8 * 4 * 3, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(8, 4, 3)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(width_img, width_img), strides=(2, 2), padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=48, kernel_size=(width_img, width_img), strides=(1, 2), padding='same',
                    activation='relu'),

                tf.keras.layers.Reshape((width_img, height_img, depth_img), input_shape=(16,))

            ]
        )

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
        local_target_train_arr = []
        if not in_train:
            return np.array(self.contur_image(images))
        for image in images:

            local_image = image.get_by_name('Image')
            if local_image is None:
                continue
            if len(local_image.shape) == 2:
                local_image = np.expand_dims(local_image, axis=1)
                image.set_by_name('Image', local_image)
            local_x_train_arr.append(np.expand_dims(np.array(self.contur_image(local_image)), axis=0))
            for data in image.data_collection:
                if data.name == self.local_output.name:
                    local_data = data.data
                    break

            local_y_train_arr.append(local_data)
            target_type = 'None'
            if image.get_by_name('fill') == 1:
                target_type = 'fill'
            if image.get_by_name('rotate') == 1:
                target_type = 'rotate'
            if image.get_by_name('rotate') and image.get_by_name('fill'):
                target_type = 'fill+rotate'
            local_target_train_arr.append(target_type)
        return np.array(local_x_train_arr), np.array(local_x_train_arr), np.array(local_target_train_arr)

    def calc_map(self, local_map, is_plot_now=False):
        local_sum = {}
        for key in local_map.keys():
            local_sum[key] = [[0.0] * len(local_map[key][0])]
            for a, b in zip(local_map[key], local_sum[key]):
                local_sum[key] = [i + j for i, j in zip(a, b)]
            local_sum[key] = [i / len(local_map[key]) for i in local_sum[key]]
        self.calc_map_plot_counter += 1
        if self.calc_map_plot_counter % 8000 == 0 or is_plot_now:
            for key in local_sum:
                pyplot.plot(local_sum[key], label=key)
            pyplot.legend(loc='best')
            pyplot.show()

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.005 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x_img, x_type):
        mean, logvar = tf.split(self.encoder(inputs=[x_img]), num_or_size_splits=2, axis=1)
        return mean, logvar

    def encode_ord_dense(self, x_img, x_type):
        print(x_img.shape)
        result = self.encoder(inputs=x_img)
        return result

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compute_loss_encoder_ordinary_dense(self, model, x, y, func_name='', is_plot=False, is_plot_now=False):
        func_type = 0
        # func_map_decode = {'target_function_a': [0, 0, 0, 1]
        #    , 'target_function_b': [0, 0, 1, 0]
        #    , 'target_function_c': [0, 1, 0, 0]
        #    , 'target_function_d': [1, 0, 0, 0]}
        x_img = x
        # x_type = tf.convert_to_tensor([func_map_decode[func_name]])
        result = self.encode_ord_dense(x_img, func_name)
        # z = result
        # z = self.reparameterize(mean, logvar)
        if len(func_name) > 0:

            if is_plot:

                if func_name not in self.func_map.keys() or is_plot_now:
                    self.func_map[func_name] = []
                self.func_map[func_name].append(result.numpy().tolist()[0])

                self.calc_map(self.func_map, is_plot_now)
            z = [0] * 4

            if func_name == 'None':
                z[0] = 0
                z[1] = 0
                z[2] = 0
                z[3] = 1
            elif func_name == 'fill':
                z[0] = 1
                z[1] = 0
                z[2] = 0
                z[3] = 0
            elif func_name == 'rotate':
                z[0] = 0
                z[1] = 1
                z[2] = 0
                z[3] = 0
            elif func_name == 'fill+rotate':
                z[0] = 0
                z[1] = 0
                z[2] = 1
                z[3] = 0

        if z.index(1) not in self.confusion_matrix.keys():
            self.confusion_matrix[z.index(1)] = {'t': 0, 'f': 0}
        if is_plot_now:
            if z.index(1) == np.argmax(result.numpy()):
                self.confusion_matrix[z.index(1)]['t'] += 1
            else:
                self.confusion_matrix[z.index(1)]['f'] += 1

            # for i in range(0,4):
            #    print(i,self.confusion_matrix[i]['t'],self.confusion_matrix[i]['f'],
            #          round((self.confusion_matrix[i]['t']/(self.confusion_matrix[i]['t']+self.confusion_matrix[i]['f']+1))*100,2))
        z = tf.convert_to_tensor([z + z])
        # logpz = self.log_normal_pdf(z, 0., 0.)
        # logqz_x = self.log_normal_pdf(result,0., 0.)
        cce = tf.keras.losses.CategoricalCrossentropy()
        local_loss = cce(z, result)

        return local_loss

    def compute_loss_encoder(self, model, x, y, func_name='', is_plot=False):
        func_type = 0
        # func_map_decode = {'target_function_a': [0, 0, 0, 1]
        #    , 'target_function_b': [0, 0, 1, 0]
        #    , 'target_function_c': [0, 1, 0, 0]
        #    , 'target_function_d': [1, 0, 0, 0]}
        x_img = x
        # x_type = tf.convert_to_tensor([func_map_decode[func_name]])
        mean, logvar = self.encode(x_img, func_name)
        z = self.reparameterize(mean, logvar)
        if len(func_name) > 0:

            if is_plot:
                if func_name not in self.func_map.keys():
                    self.func_map[func_name] = []
                self.func_map[func_name].append(z.numpy().tolist()[0])
                self.calc_map(self.func_map)
            z = z.numpy().tolist()[0]

            if func_name == 'None':
                z[-4] = 0
                z[-3] = 0
                z[-2] = 0
                z[-1] = 1
            elif func_name == 'fill':
                z[-4] = 1
                z[-3] = 0
                z[-2] = 0
                z[-1] = 0
            elif func_name == 'rotate':
                z[-4] = 0
                z[-3] = 1
                z[-2] = 0
                z[-1] = 0
            elif func_name == 'fill+rotate':
                z[-4] = 1
                z[-3] = 1
                z[-2] = 1
                z[-1] = 0
        z = tf.convert_to_tensor([z])

        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpz - logqz_x)

    def compute_loss(self, model, x, y, func_name='', is_plot=False):

        func_type = 0
        # func_map_decode = {'target_function_a': [0, 0, 0, 1]
        #    , 'target_function_b': [0, 0, 1, 0]
        #    , 'target_function_c': [0, 1, 0, 0]
        #    , 'target_function_d': [1, 0, 0, 0]}
        x_img = x
        # x_type = tf.convert_to_tensor([func_map_decode[func_name]])
        mean, logvar = self.encode(x_img, func_name)
        z = self.reparameterize(mean, logvar)
        if len(func_name) > 0:

            if is_plot:
                if func_name not in self.func_map.keys():
                    self.func_map[func_name] = []
                self.func_map[func_name].append(z.numpy().tolist()[0])
                self.calc_map(self.func_map)

        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train(self, images, force_train=False):
        x_train, y_train, func_name = self.prepare_data(images, in_train=True)

        for x, y, target_type in zip(x_train, y_train, func_name):
            if target_type == 'fill+rotate':
                continue

            with tf.GradientTape(persistent=True) as tape:
                loss = self.compute_loss(self.model, x, y, target_type, is_plot=False)
                loss_enc = self.compute_loss_encoder_ordinary_dense(self.model, x, y, target_type, is_plot=False)
            # if tf.math.is_nan(loss):
            #     loss = tf.zeros(1)
            gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
            gradients_encoder = tape.gradient(loss_enc, self.encoder.trainable_variables)
            if None not in gradients_encoder:
                self.optimizer.apply_gradients(
                    zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
                self.optimizer.apply_gradients(zip(gradients_encoder, self.encoder.trainable_variables))

    def predict(self, image):
        x_train, y_train, func_name = self.prepare_data([image], in_train=True)
        print('func_name', func_name)
        self.compute_loss_encoder_ordinary_dense(self.model, x_train[0], y_train[0], func_name[0], is_plot=True,
                                                 is_plot_now=False)

        if x_train.shape[0] == 0:
            return None
        x_train = np.squeeze(x_train, axis=1)
        mean, logvar = self.encode(x_train, func_name)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        print(tf.keras.losses.MSE(x_logit, x_train))
        # return [x_logit]
        return None
