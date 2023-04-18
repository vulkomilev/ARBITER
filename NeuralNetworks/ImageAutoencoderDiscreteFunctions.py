import random
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

from utils.Agent import *
from custom_layers.ConvSymb import ConvSymb
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.animation as animation
from multiprocessing import Process,Pipe
import cv2
from utils.utils import normalize_list,one_hot,DataUnit
import copy
last_img = np.zeros((100,1))
#last_img = np.zeros((12,8,3))
def runGraph(pipe):
    global last_img
    # Parameters
    x_len = 200         # Number of points to display
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
        #print("RECV",pipe.recv())
        try:
         last_img.put(0,pipe.recv())
         #last_img(0)
        except Exception as  e:
            print(e)
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
    def __init__(self, ):
        self.model = None
        self.func_map = {}
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.reg_input = [
            DataUnit('str', (), None, 'Id', is_id=True),
            DataUnit('2D_F', (64,64), None, 'Image'),


            ]
        self.reg_output = [
            DataUnit('str', (), None, 'Id', is_id=True),
            DataUnit('2D_F', (64, 64), None, 'Image')
        ]
        self.local_image_list = []
        self.init_neural_network( latent_dim=(2000))  # 8*4*3
        self.calc_map_plot_counter = 0
        self.confusion_matrix = {}
        self.func_arr = [[],[]]
        conn1, conn2 = Pipe()
        self.conn_send = conn2
        #p = Process(target=runGraph, args=(conn1,))
        #p.start()

    def register(self,arbiter):
        arbiter.register_neural_network(self,self.reg_input,self.reg_output)

    def init_neural_network(self, latent_dim):
        self.latent_dim = latent_dim
        width_img = self.reg_input[1].shape[0]
        height_img = self.reg_input[1].shape[1]
        depth_img = 3
        def custom_func(inputs):

            inputs /= 2
            return inputs
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(width_img, height_img, depth_img)),
            ConvSymb(
                filters=3,rank=2, custom_function=custom_func, strides=(2, 2), activation='relu'),
            ConvSymb(
                filters=3, rank=2,custom_function=custom_func, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim, activation='gelu')
        ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=32 * 32 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(32, 32, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=300, kernel_size=5, strides=1, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=2, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Reshape(target_shape=( 64, 64,3))
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

            local_image = image.source.get_by_name('Image')
            local_id = str(image.source.get_by_name('Id'))

            if local_id[:8] not in images_collection.keys():
                images_collection[local_id[:8]] = {'input': None, 'output': None}
            if 'input' in local_id:
                images_collection[local_id[:8]]['input'] = local_image
            elif 'output' in local_id:
                images_collection[local_id[:8]]['output'] = local_image
        for local_key in images_collection.keys():
            local_image_input = images_collection[local_key]['input']
            if images_collection[local_key]['output'] is None or images_collection[local_key]['input'] is None:
                continue

            local_image_output = images_collection[local_key]['output']
            if type(local_image_output) == type(None) or type(local_image_input) == type(None):
                continue
            local_image = copy.deepcopy(local_image_input)   # np.concatenate((local_image_input, local_image_output), axis=0)

            local_x_train_arr.append(
         np.array(np.resize(np.float32(local_image),(1,64,64,3))))  # self.contur_image(local_image)

            local_y_train_arr.append( np.array(np.resize(np.float32(local_image_output), (1, 64, 64, 3))))

        return np.array(local_x_train_arr), np.array(local_y_train_arr), np.array(local_target_train_arr)



    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(     -.005 * ((sample - mean ) ** 2. * tf.exp(-logvar) + logvar + log2pi),  axis=raxis)

    #tf.reduce_sum(     -.005 * ((sample - mean ) ** 2. * tf.exp(-logvar) + logvar + log2pi),  axis=raxis)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x_img):
        mean, logvar = tf.split(self.encoder(inputs=np.array(x_img)), num_or_size_splits=2, axis=1)
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

    def custom_function_1(self,mean):
        local_np = mean.numpy()
        new_np = [[]]

        for element in local_np[0]:
           if element >= 0.0:
            new_np[0].append(1.0)
           elif element < 0.0:
            new_np[0].append(-1.0)
        #print(mean.shape)
        #print(np.array(new_np).shape)
        return tf.convert_to_tensor(new_np)

    def compute_loss(self,model, x,y,is_plot=False):
        mean, logvar = self.encode(x)
        mean = self.custom_function_1(mean)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x =  self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train(self, images, force_train=False):
        global  last_img
        print('loaded images',images)
        self.local_image_list.append(images)
        x_train, y_train, func_name = self.prepare_data(self.local_image_list, in_train=True)
        print('prepare_data',np.array(x_train).shape)

        for x, y in zip(x_train, y_train):
            with tf.GradientTape(persistent=True) as tape:

                loss = self.compute_loss(self.model, x, y, is_plot=False)
                try:
                 self.conn_send.send(loss)
                except BrokenPipeError as e:
                    print(e)
                    pass
                print('local_loss',loss)
                #loss_enc = self.compute_loss_encoder_ordinary_dense(self.model, x, y, target_type, is_plot=False)
            # if tf.math.is_nan(loss):
            #     loss = tf.zeros(1)
            self.gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
           #gradients_encoder = tape.gradient(loss_enc, self.encoder.trainable_variables)
        #loss_arr.append(loss.numpys().item())
        if 'gradients' not in dir(self):
            return
        self.optimizer.apply_gradients(
                    zip(self.gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
           # loss_enc_arr.append(loss_enc.numpy().item())
            #if None not in gradients_encoder:
            #    self.optimizer.apply_gradients(
            #        zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
                #self.optimizer.apply_gradients(zip(gradients_encoder, self.encoder.trainable_variables))
        print('gradients applied')
        mean, logvar = self.encode(x_train[0])
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        # return [x_logit]
        #image_1 = image_1.astype('float64')
        #image_1 *= 255.0 / image_1.max()
        image = x_logit.numpy()[0]
        image *= 255.0 / image.max()

        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(1)

        self.encoder.save('./checkpoints/' + ckpt_name+'_encoder')
        self.decoder.save('./checkpoints/' + ckpt_name+'_decoder')
        last_img =z
        #if self.calc_map_plot_counter %1 == 0 :
        #    pyplot.imshow(image)
        #    pyplot.show()
            #pyplot.imshow(image_1)
            #pyplot.show()
            #pyplot.imshow(x_train[0][0])
            #pyplot.show()
    def save(self):
        pass
    def predict(self, image):
        x_train, y_train, func_name = self.prepare_data(image, in_train=True)
        #self.compute_loss_encoder_ordinary_dense(self.model, x_train[0], y_train[0], func_name[0], is_plot=True,
        #                                         is_plot_now=False)
        #x_train = x_train[0]
        #y_image = y_train[0]
        for x_element ,y_element in zip(x_train,y_train):
            if x_train.shape[0] == 0:
                return None
            image_1 = x_element
            mean, logvar = self.encode(x_element)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z)
            # return [x_logit]
            image_1 = image_1.astype('int')
            y_image = y_element.astype('int')

            print('image_1.max()', image_1.max())
            image = x_logit.numpy()[0]
            image = image*(255.0 / image.max())
            image_1 = image_1*(1.0 / image_1.max())
            y_image = y_image*(1.0 / y_image.max())
            print('z',image.shape)
            print('z', image_1.shape)
            print('z', y_image.shape)
            plt.imshow(image)
            plt.show()
            plt.imshow(x_element)
            plt.show()
            plt.imshow(y_element)
            plt.show()


        return None
