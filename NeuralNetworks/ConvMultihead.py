from utils.Agent import *
from matplotlib import pyplot as plt
from keras import backend as K

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


class ConvMultihead(Agent):

    def __init__(self, inputs, outputs, data_schema,class_num):
        self.model = None
        self.func_map = {}
        self.optimizer = tf.keras.optimizers.Adam(1e-6)
        self.init_neural_network(inputs, outputs, data_schema, latent_dim=(4))  # 8*4*3
        self.calc_map_plot_counter = 0
        self.test_counter = 0
        self.confusion_matrix = {}
        self.loss_arr =[]

    def init_neural_network(self, inputs, outputs, data_schema, latent_dim):
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
        print(width_img, height_img)

        inp1_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp2_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp3_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp4_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp5_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp6_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp7_in= tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp8_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp9_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp10_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp11_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp12_in = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))

        inp1_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp2_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp3_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp4_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp5_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp6_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp7_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp8_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp9_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp10_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp11_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        inp12_in_w = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))

        inp1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, height_img),
                                      strides=(width_img, height_img), activation='relu')(inp1_in)
        inp2 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, height_img),
                                      strides=(3, height_img), activation='relu')(inp2_in)
        inp3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(1, height_img), activation='relu')(inp3_in)
        inp4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(2, height_img), activation='relu')(inp4_in)
        inp5 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(3, height_img), activation='relu')(inp5_in)
        inp6 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(4, height_img), activation='relu')(inp6_in)
        inp7 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(5, height_img), activation='relu')(inp7_in)
        inp8 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(6, height_img), activation='relu')(inp8_in)
        inp9 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(7, height_img), activation='relu')(inp9_in)
        inp10 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(8, height_img), activation='relu')(inp10_in)
        inp11 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(9, height_img), activation='relu')(inp11_in)
        inp12 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, height_img),
                                      strides=(10, height_img), activation='relu')(inp12_in)

        inp1_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, height_img),
                                      strides=(width_img, height_img), activation='relu')(inp1_in_w)
        inp2_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 3),
                                      strides=(width_img, 3), activation='relu')(inp2_in_w)
        inp3_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                      strides=(width_img, 1), activation='relu')(inp3_in_w)
        inp4_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                      strides=(width_img, 2), activation='relu')(inp4_in_w)
        inp5_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                      strides=(width_img, 3), activation='relu')(inp5_in_w)
        inp6_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                      strides=(width_img, 4), activation='relu')(inp6_in_w)
        inp7_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                      strides=(width_img, 5), activation='relu')(inp7_in_w)
        inp8_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                      strides=(width_img, 6), activation='relu')(inp8_in_w)
        inp9_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                      strides=(width_img, 7), activation='relu')(inp9_in_w)
        inp10_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                       strides=(width_img, 8), activation='relu')(inp10_in_w)
        inp11_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                       strides=(width_img, 9), activation='relu')(inp11_in_w)
        inp12_w = tf.keras.layers.Conv2D(filters=96, kernel_size=(width_img, 1),
                                       strides=(width_img, 10), activation='relu')(inp12_in_w)

        inp1 = tf.keras.layers.Flatten()(inp1)
        inp2 = tf.keras.layers.Flatten()(inp2)
        inp3 = tf.keras.layers.Flatten()(inp3)
        inp4 = tf.keras.layers.Flatten()(inp4)
        inp5 = tf.keras.layers.Flatten()(inp5)
        inp6 = tf.keras.layers.Flatten()(inp6)
        inp7 = tf.keras.layers.Flatten()(inp7)
        inp8 = tf.keras.layers.Flatten()(inp8)
        inp9 = tf.keras.layers.Flatten()(inp9)
        inp10 = tf.keras.layers.Flatten()(inp10)
        inp11 = tf.keras.layers.Flatten()(inp11)
        inp12 = tf.keras.layers.Flatten()(inp12)

        inp1_w = tf.keras.layers.Flatten()(inp1_w)
        inp2_w = tf.keras.layers.Flatten()(inp2_w)
        inp3_w = tf.keras.layers.Flatten()(inp3_w)
        inp4_w = tf.keras.layers.Flatten()(inp4_w)
        inp5_w = tf.keras.layers.Flatten()(inp5_w)
        inp6_w = tf.keras.layers.Flatten()(inp6_w)
        inp7_w = tf.keras.layers.Flatten()(inp7_w)
        inp8_w = tf.keras.layers.Flatten()(inp8_w)
        inp9_w = tf.keras.layers.Flatten()(inp9_w)
        inp10_w = tf.keras.layers.Flatten()(inp10_w)
        inp11_w = tf.keras.layers.Flatten()(inp11_w)
        inp12_w = tf.keras.layers.Flatten()(inp12_w)

        x = tf.keras.layers.concatenate([inp1, inp2, inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,inp11,inp12,
                                         inp1_w, inp2_w, inp3_w,inp4_w,inp5_w,inp6_w,inp7_w,inp8_w,inp9_w,
                                         inp10_w,inp11_w,inp12_w])
        x = tf.keras.layers.Dense(384)(x)
        x = tf.keras.layers.Dense(192)(x)
        x =  tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Dense(36)(x)
            # tf.keras.layers.Dense(36),
        x = tf.keras.layers.Dense(latent_dim, activation='softmax')(x)

        self.encoder = tf.keras.Model(inputs = [inp1_in,inp2_in,inp3_in,inp4_in,inp5_in,inp6_in,inp7_in,inp8_in,inp9_in
            ,inp10_in,inp11_in,inp12_in,
                                                inp1_in_w, inp2_in_w, inp3_in_w, inp4_in_w, inp5_in_w, inp6_in_w, inp7_in_w, inp8_in_w,
                                                inp9_in_w
            , inp10_in_w, inp11_in_w, inp12_in_w
                                                ],outputs = x)

        '''
        self.decoder = tf.keras.Sequential(
            [

                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu),
                #tf.keras.layers.Dense(units=8* 4*3, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(8, 4, 3)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(width_img, width_img), strides=(2, 2), padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=48, kernel_size=(width_img, width_img), strides=(1, 2), padding='same',
                    activation='relu'),


                tf.keras.layers.Reshape((width_img, height_img,depth_img), input_shape=(16,))

            ]
        )
  '''

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
                plt.plot(local_sum[key], label=key)
            plt.legend(loc='best')
            plt.show()

    def running_mean(self,x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

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
        result = self.encoder(inputs=[x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img,
                                      x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img,x_img])
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
        if is_plot_now:
            print(result, z)
        if z.index(1) not in self.confusion_matrix.keys():
            self.confusion_matrix[z.index(1)] = {'t': 0, 'f': 0}
        print(np.argmax(result.numpy()), z.index(1))
        if is_plot_now:
            if z.index(1) == np.argmax(result.numpy()):
                self.confusion_matrix[z.index(1)]['t'] += 1
            else:
                self.confusion_matrix[z.index(1)]['f'] += 1
            print(' ', 't', 'f')
            for i in range(0, 4):
                print(i, self.confusion_matrix[i]['t'], self.confusion_matrix[i]['f'],
                      round((self.confusion_matrix[i]['t'] / (
                                  self.confusion_matrix[i]['t'] + self.confusion_matrix[i]['f'] + 1)) * 100, 2))
        z = tf.convert_to_tensor([z])
        # logpz = self.log_normal_pdf(z, 0., 0.)
        # logqz_x = self.log_normal_pdf(result,0., 0.)
        cce = tf.keras.losses.CategoricalCrossentropy()
        local_loss = cce(z, result)

        print(local_loss, z, result)
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
            with tf.GradientTape(persistent=True) as tape:
                # loss = self.compute_loss(self.model, x, y, target_type,is_plot=False)
                loss_enc = self.compute_loss_encoder_ordinary_dense(self.model, x, y, target_type, is_plot=False)
            # if tf.math.is_nan(loss):
            #     loss = tf.zeros(1)
            # gradients = tape.gradient(loss,self.encoder.trainable_variables + self.decoder.trainable_variables)
            print('loss_enc',loss_enc)
            self.loss_arr.append(loss_enc)
            gradients_encoder = tape.gradient(loss_enc, self.encoder.trainable_variables)
            if None not in gradients_encoder:
                # self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
                self.optimizer.apply_gradients(zip(gradients_encoder, self.encoder.trainable_variables))

    def predict(self, image):
        plt.plot(self.running_mean(self.loss_arr,100))
        plt.show()
        x_train, y_train, func_name = self.prepare_data([image], in_train=True)
        #print('func_name', func_name[0],func_name[0] != 'fill')
        if func_name[0] != 'None' or self.test_counter >4:
              return None
        self.test_counter +=1
        # self.save_layers()
        self.compute_loss_encoder_ordinary_dense(self.model, x_train[0], y_train[0], func_name[0], is_plot=False,
                                                 is_plot_now=True)

        if x_train.shape[0] == 0:
            return None
        x_train = np.squeeze(x_train, axis=1)
        #mean, logvar = self.encode_ord_dense(x_train[0], func_name)
        #z = self.reparameterize(mean, logvar)
        # x_logit = self.decode(z)
        # return [x_logit]
        return None


    # ---------------------------------------------------------------------------------------------------
    # Utility function for generating patterns for given layer starting from empty input image and then
    # applying Stochastic Gradient Ascent for maximizing the response of particular filter in given layer
    # ---------------------------------------------------------------------------------------------------

    def deprocess_image(self,x):

        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
        x += 0.5
        x = np.clip(x, 0, 1)
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x
    def generate_pattern(self,layer_name, filter_index, size=150):

        layer_output = self.encoder.get_layer(layer_name).output
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.GradientTape(loss, self.encoder.input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([self.encoder.input], [loss, grads])
        input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
        step = 1.
        for i in range(80):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

        img = input_img_data[0]
        return self.deprocess_image(img)

    # ------------------------------------------------------------------------------------------
    # Generating convolution layer filters for intermediate layers using above utility functions
    # ------------------------------------------------------------------------------------------
    def save_layers(self):
        layer_name = 'conv2d_4'
        size = 299
        margin = 5
        results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

        for i in range(8):
            for j in range(8):
                filter_img = [v for v in self.encoder.trainable_variables if 'kernel' in v.name ][0]

                horizontal_start = i * size + i * margin
                horizontal_end = horizontal_start + size
                vertical_start = j * size + j * margin
                vertical_end = vertical_start + size
                results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

        pyplot.figure(figsize=(20, 20))
        pyplot.savefig(results)
