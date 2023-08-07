# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np

from custom_layers.ConvSymb import ConvSymb
from utils.Agent import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.animation as animation
from multiprocessing import Pipe
import cv2
from utils.utils import DataUnit
import copy


import traceback
last_img = np.zeros((100, 1))
# last_img = np.zeros((12,8,3))
#tf.compat.v1.enable_eager_execution()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
print('gpus',gpus)
#exit(0)
def runGraph(pipe):
    global last_img
    # Parameters
    x_len = 200  # Number of points to display
    y_range = [10, 40]  # Range of possible Y values to display

    # Create figure for plotting
    fig = plt.figure()
    ax = plt.subplot(111)
    xs = list(range(0, 200))
    ys = [0] * x_len

    # Create a blank line. We will update the line in animate
    line = ax.plot(last_img)

    # This function is called periodically from FuncAnimation
    def animate(i, ys):

        # Update line with new Y values
        # print("RECV",pipe.recv())
        try:
            last_img.put(0, pipe.recv())
            # last_img(0)
        except Exception as e:
            pass
        # print(e)
        return last_img,

    # Set up plot to call animate() function periodically

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  fargs=(ys,),
                                  interval=50,
                                  blit=True)
    ani.save('continuousSineWave.mp4',
             writer='ffmpeg', fps=30)
    plt.show()


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


class ImageAutoencoderDiscreteFunctions(Agent):
    def __init__(self, input_list, output_list):
        self.model = None
        self.func_map = {}
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.reg_input = [
            DataUnit('str', (), None, 'Id', is_id=True),
            DataUnit('2D_F', (32, 32, 3), None, 'Image'),

        ]
        self.reg_output = [
            DataUnit('str', (), None, 'Id', is_id=True),
            DataUnit('2D_F', (32, 32, 3), None, 'Image')
        ]
        self.local_image_list = []
        self.is_init = True
        self.init_neural_network(latent_dim=(2000))  # 8*4*3
        self.is_init = False
        self.calc_map_plot_counter = 0
        self.confusion_matrix = {}
        self.func_arr = [[], []]
        conn1, conn2 = Pipe()
        self.conn_send = conn2
        # Initialize the Model Card Toolkit with a path to store generate assets
        model_card_output_path = './'
        mct = model_card_toolkit.ModelCardToolkit(model_card_output_path)

        # Initialize the model_card_toolkit.ModelCard, which can be freely populated
        model_card = mct.scaffold_assets()
        model_card.model_details.name = 'My Model'

        # Write the model card data to a proto file
        mct.update_model_card(model_card)

        # Return the model card document as an HTML page
        html = mct.export_format()

        # p = Process(target=runGraph, args=(conn1,))
        # p.start()

    def register(self, arbiter):
        arbiter.register_neural_network(self, self.reg_input, self.reg_output)

    def init_neural_network(self, latent_dim):
        self.latent_dim = latent_dim
        width_img = self.reg_input[1].shape[0]
        height_img = self.reg_input[1].shape[1]
        depth_img = 3

        @tf.function()
        def custom_func(inputs):

            values = [1.0]  # A list of values corresponding to the respective
            # coordinate in indices.

            shape = [1, 32, 32, 3]  # The shape of the corresponding dense tensor, same as `c`.

            i = 0

            def while_three(n, j, i, inputs):
                n += 1

                indices = [[0, i, j, 1]]
                delta = tf.SparseTensor(indices, values, shape)
                # print(delta)
                tf.sparse.to_dense(delta)
                inputs = inputs + tf.sparse.to_dense(delta)
                return n, j, i, inputs

            def while_two(j, i):
                n = 0
                j += 1

                tf.while_loop(cond=lambda n, j, i, inputs: tf.less(n, inputs.shape.as_list()[3]), body=while_three,
                              loop_vars=[n, j, i, inputs])
                return i, j

            @tf.function()
            def while_one(i):
                i += 1
                j = 0
                tf.while_loop(lambda j, i: tf.less(j, inputs.shape.as_list()[2] - 1), while_two, [j, i])
                tf.greater_equal(inputs[0][i][0][0], 0.5)
                tf.cond(tf.greater_equal(inputs[0][i][0][0], 0.5),lambda i=i ,j=j : tf.while_loop(lambda j, i: tf.less(j, inputs.shape.as_list()[2] - 1), while_two, [j, i]),lambda i=i  :[0,0])
                return [i]
            tf.while_loop(lambda i: tf.less(i, inputs.shape.as_list()[1] - 1), while_one, [i])

            return inputs

        @tf.function()
        def custom_func_compare_v1(inputs):

            values = [1.0]  # A list of values corresponding to the respective
            # coordinate in indices.

            shape = [1, 32, 32, 3]  # The shape of the corresponding dense tensor, same as `c`.

            i = 0


            result  =  tf.SparseTensor([[0, 0, 0, 1]], values, shape)
            result = tf.sparse.to_dense(result)
            result  = tf.cast(result,dtype=tf.float32)
            def while_three(n, j, i, inputs):
                n += 1

                indices = [[0, i, j, 1]]
                delta = tf.SparseTensor(indices, values, shape)
                # print(delta)
                tf.sparse.to_dense(delta)
                result =  tf.sparse.to_dense(delta)
                return n, j, i, inputs

            def while_two(j, i):
                n = 0
                j += 1

                tf.while_loop(cond=lambda n, j, i, inputs: tf.less(n, inputs.shape.as_list()[3]), body=while_three,
                              loop_vars=[n, j, i, inputs])
                return i, j

            @tf.function()
            def while_one(i):
                i += 1
                j = 0
                tf.while_loop(lambda j, i: tf.less(j, inputs.shape.as_list()[2] - 1), while_two, [j, i])
                tf.greater_equal(inputs[0][i][0][0], 0.5)
                tf.cond(tf.greater_equal(inputs[0][i][0][0], 0.5),lambda i=i ,j=j : tf.while_loop(lambda j, i: tf.less(j, 32 - 1), while_two, [j, i]),lambda i=i  :[0,0])
                return [i]
            tf.while_loop(lambda i: tf.less(i, 32 - 1), while_one, [i])

            return tf.math.squared_difference(result,inputs[:][:32][:][:])

        @tf.function()
        def custom_func_compare_v2(inputs):
            values = [1.0]  # A list of values corresponding to the respective
            # coordinate in indices.

            shape = [1, 32, 32, 3]  # The shape of the corresponding dense tensor, same as `c`.

            result = inputs[:][:32][:][:]
            result = tf.image.rot90(result)

            return tf.reduce_sum(tf.math.squared_difference(result, inputs[:][32:][:][:]), keepdims=True)
            #return inputs


        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(width_img, height_img, depth_img)),
            tf.keras.layers.Dense(3, activation='relu'),
            tf.keras.layers.Dense(3, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(32,32,3)),
            ConvSymb(kernel_size=1,
                     filters=3, rank=2, custom_function=custom_func, strides=(1, 1), activation='relu'),
            ConvSymb(kernel_size=1,
                     filters=3, rank=2, custom_function=custom_func, strides=(1, 1), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim, activation='gelu')
        ]
        )
        '''
        self.encoder_input_1 = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        self.encoder_1 = tf.keras.layers.Dense(3, activation='relu')(self.encoder_input_1)
        self.encoder_1 = tf.keras.layers.Dense(3, activation='relu')(self.encoder_1)
        self.encoder_1 = tf.keras.layers.Reshape(target_shape=(32,32,3))(self.encoder_1)
        #self.encoder_1 = ConvSymb(kernel_size=1,
      #              filters=1, rank=2, custom_function=custom_func_compare_v2, strides=(1, 1), activation='relu')(self.encoder_1)
        #self.encoder_1 = ConvSymb(kernel_size=1,
        #             filters=1, rank=2, custom_function=custom_func_compare_v2, strides=(1, 1), activation='relu')(self.encoder_1)
        self.encoder_1 = tf.keras.layers.Flatten()(self.encoder_1)

        self.encoder_input_2 = tf.keras.layers.Input(shape=(width_img, height_img, depth_img))
        self.encoder_2 = tf.keras.layers.Dense(3, activation='relu')(self.encoder_input_2)
        self.encoder_2 = tf.keras.layers.Dense(3, activation='relu')(self.encoder_2)
        self.encoder_2 = tf.keras.layers.Flatten()(self.encoder_2)

       # self.encoder = tf.keras.layers.Concatenate(axis=1)([self.encoder_1, self.encoder_2])
        self.encoder = tf.keras.layers.Flatten()(self.encoder_2)
        self.encoder = tf.keras.layers.Dense(latent_dim + latent_dim, activation='relu')(self.encoder)
        self.encoder = tf.keras.layers.Dense(latent_dim + latent_dim, activation='relu')(self.encoder)
        self.encoder = tf.keras.layers.Dense(latent_dim + latent_dim, activation='gelu')(self.encoder)
        self.encoder_model = tf.keras.Model(inputs=[self.encoder_input_2],outputs= self.encoder)#self.encoder_input_1,
        '''

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=16 * 16 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(16, 16, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=300, kernel_size=5, strides=1, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=2, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Reshape(target_shape=(32, 32, 3))
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
        images_collection = {}

        for image in images:

            local_image_input = image.source.get_by_name('Image')
            local_image_output = image.target.get_by_name('Image')
            local_id = str(image.source.get_by_name('Id'))
            if local_image_output == None:
                continue
            if local_id[:8] not in images_collection.keys():
                images_collection[local_id[:8]] = {'input': None, 'output': None}

            # for i in range(len(local_image_input)):
            plt.imshow(np.array(local_image_input) / 256.0)

            plt.imshow(np.array(local_image_output) / 256.0)

            if type(local_image_output) == type(None) or type(local_image_input) == type(None):
                continue
            # plt.imshow(local_image_output[2])
            # plt.show()
            local_image = copy.deepcopy(
                local_image_input)  # np.concatenate((local_image_input, local_image_output), axis=0)
            # plt.imshow(local_image_input)
            # plt.show()
            # plt.imshow(local_image_output)
            # plt.show()
            local_x_train_arr.append(
                np.array(np.resize(np.float32(local_image_input), (1, 32, 32, 3))))  # self.contur_image(local_image)

            local_y_train_arr.append(np.array(np.resize(np.float32(local_image_output), (1, 32, 32, 3))))

        return np.array(local_x_train_arr), np.array(local_y_train_arr), np.array(local_target_train_arr)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.005 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x_img):
        mean, logvar = tf.split(self.encoder(inputs=[np.array(x_img)]), num_or_size_splits=2, axis=1)#np.array(x_img),
        return mean, logvar

    def encode_ord_dense(self, x_img, x_type):
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

    def custom_function_1(self, mean):
        local_np = mean.numpy()
        new_np = [[]]

        for element in local_np[0]:
            if element >= 0.0:
                new_np[0].append(1.0)
            elif element < 0.0:
                new_np[0].append(-1.0)
        return tf.convert_to_tensor(new_np)

    def compute_loss(self, model, x, y, is_plot=False):
        mean, logvar = self.encode(x)
        #mean = self.custom_function_1(mean)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        self.calc_map_plot_counter += 1
        if self.calc_map_plot_counter % 100 == 0:
            plt.imshow(x[0])
            plt.show()
            plt.imshow(y[0])
            plt.show()
            plt.imshow(x_logit[0])
            plt.show()
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train(self, images, force_train=False, only_fill=True):
        global last_img
        if images != None:
            self.local_image_list.append(images)

        if only_fill:
            return
        x_train, y_train, func_name = self.prepare_data(self.local_image_list, in_train=True)
        print('prepare_data done',x_train,y_train)
        for x, y in zip(x_train, y_train):

            with tf.GradientTape(persistent=True) as tape:

                loss = self.compute_loss(self.model, x, y, is_plot=False)
                try:
                    self.conn_send.send(loss)
                except BrokenPipeError as e:

                    pass
                print('local_loss', loss)

            self.gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)

        if 'gradients' not in dir(self):
            return
        self.optimizer.apply_gradients(
            zip(self.gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))

        print('gradients applied')
        mean, logvar = self.encode(x_train[0])
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        image = x_logit.numpy()[0]
        image *= 255.0 / image.max()

        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(1)

        self.encoder.save('./checkpoints/' + ckpt_name + '_encoder')
        self.decoder.save('./checkpoints/' + ckpt_name + '_decoder')
        last_img = z

    def save(self):
        pass

    def predict(self, image):

        x_train, y_train, func_name = self.prepare_data([image], in_train=False)

        for x_element, y_element in zip(x_train, y_train):
            if x_train.shape[0] == 0:
                return None
            image_1 = x_element
            mean, logvar = self.encode(x_element)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z)
            # return [x_logit]
            image_1 = image_1.astype('int')
            y_image = y_element.astype('int')

            image = x_logit.numpy()[0]
            image = image * (255.0 / image.max())
            image_1 = image_1 * (1.0 / image_1.max())
            y_image = y_image * (1.0 / y_image.max())
            plt.imshow(image)
            plt.show()
            plt.imshow(x_element)
            plt.show()
            plt.imshow(y_element)
            plt.show()

        return None
