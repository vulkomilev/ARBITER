import random

import numpy as np

from utils.Agent import *
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from multiprocessing import Process,Pipe
import copy
last_img = np.zeros((128,96,3))
#last_img = np.zeros((12,8,3))
def contur_image( img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.blur(grey_img, (1, 1), 0)
    contur_img = cv2.Sobel(src=blur_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)

    contur_img = np.float32(contur_img)

    contur_img = np.stack((contur_img,) * 3, axis=-1)
    return contur_img
def prepare_data( images, in_train=False):
    local_x_train_arr = []
    local_y_train_arr = []
    local_target_train_arr = []
    images_collection = {}
    if not in_train:
        return np.array(contur_image(images))
    for image in images:

        local_image = image.get_by_name('Image')
        local_id = image.get_by_name('Id')

        # plt.imshow(local_image)
        # plt.show()
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
        if random.choice([True, False]):
            local_target_train_arr.append('rot')
            # print(images_collection[local_key]['output'])

            # print(images_collection[local_key]['input'])
            images_collection[local_key]['output'] = np.rot90(images_collection[local_key]['output'])
        else:
            local_target_train_arr.append('none')
        local_image_output = images_collection[local_key]['output']
        if type(local_image_output) == type(None) or type(local_image_input) == type(None):
            continue
        local_image =local_image_input*255# np.concatenate((local_image_input, local_image_output), axis=0)

        if local_image is None:
            continue
        if len(local_image.shape) == 2:
            local_image = np.expand_dims(local_image, axis=1)
            image.set_by_name('Image', local_image)
        local_x_train_arr.append(
            np.expand_dims(np.array(np.float32(local_image)) , axis=0))  # self.contur_image(local_image)
        for data in image.data_collection:
            if data.name == 'Id':
                local_data = data.data
                break

        local_y_train_arr.append(local_image_output*255)

    return np.array(local_x_train_arr), np.array(local_x_train_arr), np.array(local_target_train_arr)


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
    line = ax.imshow(last_img)



    # This function is called periodically from FuncAnimation
    def animate(i, ys):

        # Read temperature (Celsius) from TMP102
        temp_c = np.random.random(1)*40

        # Add y to list
        ys.append(temp_c)

        # Limit y list to set number of items
        ys = ys[-x_len:]

        # Update line with new Y values
        #print(pipe.recv())
        line.set_array(pipe.recv())

        return line,


    # Set up plot to call animate() function periodically

    ani = animation.FuncAnimation(fig,
        animate,
        fargs=(ys,),
        interval=50,
        blit=True)
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

def animate(i, xs, ys):

    # Read temperature (Celsius) from TMP102
    temp_c = round(random.randint(0,10)/10, 2)

    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(temp_c)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

class FunctionalAutoencoder(Agent):
    def __init__(self, inputs, outputs,  data_schema_input, data_schema_output, class_num):
        self.model = None
        self.func_map = {}
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.init_neural_network(inputs, outputs, data_schema_input, latent_dim=(1), class_num=class_num)  # 8*4*3
        self.calc_map_plot_counter = 0
        self.confusion_matrix = {}
        self.func_arr = [[],[]]
        conn1, conn2 = Pipe()
        self.conn_send = conn2
        #p = Process(target=runGraph, args=(conn1,))
        #p.start()

    def init_neural_network(self, inputs, outputs, data_schema, latent_dim, class_num):
        local_input = inputs[0]
        self.local_output = outputs[0]
        self.latent_dim = latent_dim
        for element in data_schema:
            if element.name  == local_input:
                local_input = element
                break
        if local_input.type == 'int':
            width_img = local_input.data
            height_img = 1
            depth_img = 1
        elif local_input.type == '2D_F':
            width_img = local_input.shape[0]*2
            height_img = local_input.shape[1]
            depth_img = 3

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(width_img, height_img, depth_img)),
            tf.keras.layers.Conv2D(
                filters=3, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=3, kernel_size=3, strides=(2, 2), activation='relu'),
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
                tf.keras.layers.Dense(units=32 * 16 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(32, 16, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=4, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, strides=1, padding='same'),
            ]
        )

        '''
         self.decoder = tf.keras.Sequential(
                [
    
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=32 * 16 * 3, activation=tf.nn.relu),
                    tf.keras.layers.Reshape(target_shape=(32, 16, 3),name='r1'),
                    tf.keras.layers.Conv2DTranspose(
                        filters=3, kernel_size=(width_img, width_img), strides=(2, 2), padding='same',
                        activation='relu'),
                    tf.keras.layers.Conv2DTranspose(
                        filters=3, kernel_size=(width_img*4, height_img*4), strides=(2, 2), padding='same',
                        activation='relu'),
    
                    tf.keras.layers.Reshape((width_img, height_img, depth_img), input_shape=(16,),name='r2')
    
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
        images_collection = {}
        if not in_train:
            return np.array(self.contur_image(images))
        for image in images:

            local_image = image.get_by_name('Image')
            local_id = image.get_by_name('Id')

            #plt.imshow(local_image)
            #plt.show()
            if local_id[:8] not in images_collection.keys():
                images_collection[local_id[:8]] = {'input': None, 'output': None}
            if 'input' in local_id:
                images_collection[local_id[:8]]['input'] = local_image
            elif 'output' in local_id:
                images_collection[local_id[:8]]['output'] = local_image
        for local_key in images_collection.keys():
            local_image_input = images_collection[local_key]['input']
            if images_collection[local_key]['output'] is None or images_collection[local_key]['input']  is None:
                continue
            if random.choice([True,False]):
                local_target_train_arr.append('rot')
                #print(images_collection[local_key]['output'])

                #print(images_collection[local_key]['input'])
                images_collection[local_key]['output'] = np.rot90(images_collection[local_key]['output'])
            else:
                local_target_train_arr.append('none')
            local_image_output = images_collection[local_key]['output']
            if type(local_image_output) == type(None) or type(local_image_input) == type(None):
                continue
            local_image =np.concatenate((local_image_input,local_image_output), axis=0)

            if local_image is None:
                continue
            if len(local_image.shape) == 2:
                local_image = np.expand_dims(local_image, axis=1)
                image.set_by_name('Image', local_image)
            local_x_train_arr.append(np.expand_dims(np.array(np.float32(local_image))/255.0, axis=0))#self.contur_image(local_image)
            for data in image.data_collection:
                if data.name == self.local_output['name']:
                    local_data = data.data
                    break

            local_y_train_arr.append(local_data)

        return np.array(local_x_train_arr), np.array(local_x_train_arr), np.array(local_target_train_arr)

    def calc_map(self, local_map, is_plot_now=False):
        local_sum = {}
        for key in local_map.keys():
            local_sum[key] = [[0.0] * len(local_map[key][0])]
            for a, b in zip(local_map[key], local_sum[key]):
                local_sum[key] = [i + j for i, j in zip(a, b)]
            local_sum[key] = [i / len(local_map[key]) for i in local_sum[key]]
        self.calc_map_plot_counter += 1
        if self.calc_map_plot_counter % 80 == 0 or is_plot_now:
            for key in local_sum:
                pyplot.plot(local_sum[key], label=key)
            pyplot.legend(loc='best')
            pyplot.show()

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    #tf.reduce_sum(     -.005 * ((sample - mean ) ** 2. * tf.exp(-logvar) + logvar + log2pi),  axis=raxis)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x_img, x_type):
        mean, logvar = tf.split(self.encoder(inputs=[x_img]), num_or_size_splits=2, axis=1)
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
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        z_fix = np.zeros((96,1))
        #bottom left
        if func_name == 'rot':
            self.func_arr[0].append(z[0][0])
        else:
            self.func_arr[1].append(z[0][0])
        #print("self.func_arr[0]",np.mean(np.array(self.func_arr[0])))
        #print("self.func_arr[1]", np.mean(np.array(self.func_arr[1])))
        #print('z_fix[0][0]',z_fix[0][0])
        #print('z[0] loss', z[0][0]-z_fix[0][0])
        logpz = self.log_normal_pdf(z+z_fix, 0., 0.)
        logqz_x = self.log_normal_pdf(z+z_fix, mean, logvar)
        image_1 = x_img.astype('float64')
        image_1 *= 255.0 / image_1.max()

        image = x_logit.numpy()[0]
        image *= 255.0 / image.max()
        #self.last_img = image
        z = z.numpy()[0]
       # z += 3#z.min()
        #z *= 255.0 / z.max()
        #z = np.reshape(z,newshape=(12,8,1))
        #z = np.pad(z, [(0, 0), (0, 0),(0,2)], mode='constant')
        self.calc_map_plot_counter+=1
        #if self.calc_map_plot_counter %1 == 0 :
        # self.conn_send.send(image_1[0])#[z.tolist()] image
        #cv2.imshow('frame',  np.zeros(shape=(128,128,3), dtype = np.uint8 ) )#np.array(image, dtype = np.uint8 ) )
        #plt.imshow(data, interpolation='nearest')

        #plt.show()

        #pyplot.imshow(image_1[0])
        #pyplot.show()
        #pyplot.imshow(y[0])
        #pyplot.show()
        #pyplot.imshow(image_1[0])
        #pyplot.show()
        #exit(0)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)/100

    def save(self):
        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(1)
        self.encoder.save('./checkpoints/' + ckpt_name+'_encoder')
        self.decoder.save('./checkpoints/' + ckpt_name+'_decoder')

    def train(self, images, force_train=False):
        global  last_img
        x_train, y_train, func_name = self.prepare_data(images, in_train=True)

        loss_arr = []
        #rand_loc = random.randint(0,len(x_train)-101)
        #x_train = x_train[rand_loc:rand_loc+100]
        #y_train = y_train[rand_loc:rand_loc+100]
        #plt.imshow(x_train[0][0])
        #plt.show()
        #exit(0)
        image_1 = images[0].get_by_name('Image')
        loss_enc_arr = []
        loss = 0
        for x, y, target_type in zip(x_train, y_train, func_name):
            if target_type == 'fill+rotate':
                continue
            with tf.GradientTape(persistent=True) as tape:

                loss += self.compute_loss(self.model, x, y, target_type, is_plot=False)
                #loss_enc = self.compute_loss_encoder_ordinary_dense(self.model, x, y, target_type, is_plot=False)
            # if tf.math.is_nan(loss):
            #     loss = tf.zeros(1)
            #print(loss/100)
            gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
           #gradients_encoder = tape.gradient(loss_enc, self.encoder.trainable_variables)
        #loss_arr.append(loss.numpys().item())
        if 'gradients' not in dir(self):
            return
        self.optimizer.apply_gradients(
                    zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
           # loss_enc_arr.append(loss_enc.numpy().item())
            #if None not in gradients_encoder:
            #    self.optimizer.apply_gradients(
            #        zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
                #self.optimizer.apply_gradients(zip(gradients_encoder, self.encoder.trainable_variables))

        mean, logvar = self.encode(x_train[0], func_name)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        # return [x_logit]
        #image_1 = image_1.astype('float64')
        #image_1 *= 255.0 / image_1.max()
        image = x_logit.numpy()[0]
        image *= 255.0 / image.max()


        last_img =z
        #if self.calc_map_plot_counter %1 == 0 :
        #    pyplot.imshow(image)
        #    pyplot.show()
            #pyplot.imshow(image_1)
            #pyplot.show()
            #pyplot.imshow(x_train[0][0])
            #pyplot.show()

    def predict(self, image):

        self.encoder = tf.keras.models.load_model('./checkpoints/FunctionalAutoencoder_encoder')
        x_train, y_train, func_name = self.prepare_data(image, in_train=True)
        #self.compute_loss_encoder_ordinary_dense(self.model, x_train[0], y_train[0], func_name[0], is_plot=True,
        #                                         is_plot_now=False)

        if x_train.shape[0] == 0:
            return None
        for i in range(100):
          for element,func in zip(x_train,func_name):
             mean, logvar = self.encode(element, func_name)
             z = self.reparameterize(mean, logvar)
             print(func,z.numpy().tolist()[0][0])
        return None
