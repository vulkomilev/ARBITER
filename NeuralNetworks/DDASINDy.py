import matplotlib.pyplot as plt

from utils.Agent import *


class DDASINDy(Agent):

    def __init__(self, inputs, outputs, data_schema, class_num):
        self.model = None
        self.init_neural_network(inputs, outputs, data_schema, class_num)
        self.sigma_list = []
        self.r = 15
        self.m = 15
        self.sigma_list.append(self.f_1)
        self.sigma_list.append(self.f_2)
        self.sigma_list.append(self.f_3)
        self.fixed_xi_matrix = np.random.rand(self.m, self.r)

    def f_1(self, inp):
        # for i in range(len(inp)):
        #    inp[i] /=2
        return tf.math.divide(inp, tf.Variable(2.0))

    def f_2(self, inp):
        return tf.math.cos(inp)

    def f_3(self, inp):
        return tf.math.divide(inp, tf.Variable(2.0))

    def init_neural_network(self, inputs, outputs, data_schema, class_num):
        local_input = inputs[0]
        self.local_output = outputs[0]

        self.optimizer = tf.keras.optimizers.Adam(1e-6)

        # self.l_filter = tf.keras.Dense(61)

        self.f_encoder_input = tf.keras.Input(shape=(1, 128))
        self.f_encoder = tf.keras.layers.Dense(244)(self.f_encoder_input)
        self.f_encoder = tf.keras.layers.Dense(122)(self.f_encoder)
        self.f_encoder = tf.keras.layers.Dense(61)(self.f_encoder)
        self.f_encoder = tf.keras.layers.Dense(30)(self.f_encoder)
        self.f_encoder = tf.keras.layers.Dense(15)(self.f_encoder)

        self.f_encoder_model = tf.keras.Model(inputs=[self.f_encoder_input], outputs=[self.f_encoder])

        self.psi_decoder_input = tf.keras.Input(shape=(15))
        self.psi_decoder = tf.keras.layers.Dense(15)(self.psi_decoder_input)
        self.psi_decoder = tf.keras.layers.Dense(30)(self.psi_decoder)
        self.psi_decoder = tf.keras.layers.Dense(61)(self.psi_decoder)
        self.psi_decoder = tf.keras.layers.Dense(122)(self.psi_decoder)
        self.psi_decoder = tf.keras.layers.Dense(244)(self.psi_decoder)
        self.psi_decoder = tf.keras.layers.Dense(128)(self.psi_decoder)
        self.psi_decoder = tf.keras.layers.Reshape((1, 128))(self.psi_decoder)
        self.psi_decoder_model = tf.keras.Model(inputs=[self.psi_decoder_input], outputs=[self.psi_decoder])

    def state_rec_loss(self, image_org):
        x, y = self.prepare_data(image_org)
        print(x.shape)
        context_arr = []
        for element in x[0][0]:
            context_arr.append(self.f_encoder_model(np.array([[element]])))
        output_arr = []
        for element in context_arr:
            output_arr.append(self.psi_decoder_model(np.array(element[0])))
        output = tf.linalg.pinv(output_arr)
        local_res = tf.math.subtract(tf.Variable(x, dtype=tf.double), tf.cast(output, dtype=tf.double))
        return tf.norm(local_res, ord=2)

    def sindy_loss_z1(self, image_org):
        x, y = self.prepare_data(image_org)
        context_arr = []
        for element in x[0][0]:
            context_arr.append(self.f_encoder_model(np.array([[element]])))
        return tf.norm(tf.transpose(np.array(x[0][0]))[0] - np.array(tf.transpose(tf.squeeze(context_arr))[0]), ord=2)

    def sindy_loss_const(self, image_org):
        x, y = self.prepare_data(image_org)

        # print('x',np.array(x).shape)
        # exit(0)

        context_arr = tf.Variable([0] * 15, dtype=tf.double)
        for element in x[0][0]:
            for j in range(2, len(element)):

                sigma_matrix = []
                local_context_arr = self.f_encoder_model(
                    np.array([[tf.pad(element[:j], [[128 - j, 0]], mode='CONSTANT', name=None)]]))
                for element_second in local_context_arr:
                    f_sigma_list = []
                    for i in range(len(self.sigma_list)):
                        f_sigma_list.append(self.sigma_list[i](element_second))
                    sigma_matrix.append(f_sigma_list)
                sigma_matrix = tf.squeeze(tf.reduce_sum(sigma_matrix, axis=1))
                context_arr = tf.cast(context_arr, dtype=tf.double) + tf.cast(sigma_matrix, dtype=tf.double)
        return tf.norm(context_arr, ord=2)

    def sindy_loss_in_prime_z(self, image_org, image_dif):

        x, y = self.prepare_data(image_org)
        x_dif, y_dif = self.prepare_data(image_dif)
        context_arr = []
        for element in x[0][0]:
            context_arr.append(self.f_encoder_model(np.array([[element]])))
        sigma_matrix = []

        for element in context_arr:
            f_sigma_list = []
            for i in range(len(self.sigma_list)):
                f_sigma_list.append(self.sigma_list[i](element))
            sigma_matrix.append(f_sigma_list)
        sigma_matrix = tf.squeeze(tf.reduce_sum(sigma_matrix, axis=1))
        context_arr = tf.squeeze(context_arr)
        sindy_result = tf.matmul(sigma_matrix, self.fixed_xi_matrix)

        x_dif = tf.Variable(x_dif)
        dy_dx_arr = []
        for element in x_dif[0][0]:
            element = tf.Variable([[element]])
            with tf.GradientTape() as tape_s:
                context_diff = self.f_encoder_model(element)
            dy_dx = tf.squeeze(tape_s.jacobian(context_diff, element))
            dy_dx_arr.append(tf.matmul(dy_dx, tf.expand_dims(element[0][0], axis=1)))
        dy_dx_arr = tf.squeeze(dy_dx_arr)
        return tf.norm(tf.cast(sindy_result, dtype=tf.double) - tf.cast(dy_dx_arr, dtype=tf.double), 2)

    def sindy_loss_in_prime_h(self, image_org, image_dif):

        x, y = self.prepare_data(image_org)
        x_dif, y_dif = self.prepare_data(image_dif)
        context_arr = []
        for element in x[0][0]:
            context_arr.append(self.f_encoder_model(np.array([[element]])))
        sigma_matrix = []

        for element in context_arr:
            f_sigma_list = []
            for i in range(len(self.sigma_list)):
                f_sigma_list.append(self.sigma_list[i](element))
            sigma_matrix.append(f_sigma_list)
        sigma_matrix = tf.squeeze(tf.reduce_sum(sigma_matrix, axis=1))
        context_arr = tf.squeeze(context_arr)
        sindy_result = tf.matmul(sigma_matrix, self.fixed_xi_matrix)

        x_dif = tf.Variable(x_dif)
        dy_dx_arr = []
        for element in x_dif[0][0]:
            element_d = tf.Variable(element)
            element_enc = tf.Variable([[element]])
            local_context = tf.Variable(tf.squeeze(self.f_encoder_model(element_enc), axis=0))
            with tf.GradientTape() as tape_s:
                y_diff = tf.math.multiply(tf.cast(self.psi_decoder_model(local_context)[0][0], dtype=tf.double),
                                          tf.cast(element,
                                                  dtype=tf.double))  # tf.matmul(tf.cast(,tf.expand_dims(tf.cast(element,dtype=tf.double),axis=1))

            dy_dx = tf.squeeze(tape_s.jacobian(y_diff, local_context))

            dy_dx_arr.append(tf.linalg.matvec(tf.cast(dy_dx, dtype=tf.double), element, transpose_a=True))
        dy_dx_arr = tf.squeeze(dy_dx_arr)

        return tf.norm(tf.squeeze(x_dif) - tf.matmul(tf.cast(sindy_result, dtype=tf.double),
                                                     tf.cast(tf.reshape(dy_dx_arr, (15, 128)), dtype=tf.double)), 2)

    def sindy_loss_in_prime_hx(self, image_org, image_dif):

        image = self.prepare_data(image_org)
        context = self.f_encoder_model(image)
        sigma_matrix = []

        for j in range(len(context)):
            f_sigma_list = []
            for i in range(len(self.sigma_list)):
                f_sigma_list.append(self.sigma_list[i](context[j]))
            sigma_matrix.append(f_sigma_list)
        sindy_result = tf.matmul(sigma_matrix, self.fixed_xi_matrix)
        with tf.GradientTape(persistent=True) as tape:

            res = tf.linalg.pinv(self.psi_decoder_model(self.f_encoder_model(image)))

        dz_dh = tape.jacobian(self.psi_decoder_model, res)
        return tf.norm(image_dif - tf.matmul(dz_dh, sindy_result), '2')

    def show_result(self, image_org):

        x, y = self.prepare_data(image_org)
        context_arr = []
        result_arr = []
        for element in x[0][0]:
            local_context = self.f_encoder_model(np.array([[element]]))[0]
            context_arr.append(local_context)

            result_arr.append(self.psi_decoder_model(local_context))

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(tf.squeeze(context_arr))
        ax2.plot(tf.squeeze(result_arr))
        plt.show()

    def prepare_data(self, x, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []
        u, s, vh = np.linalg.svd(x)

        u_colls = []
        # print(tf.transpose(np.array(x)[:,element]).shape)
        for element in range(u.shape[1]):
            u_colls.append(u[:, element])
        return np.array([[u_colls]]), np.array(local_y_train_arr)

    def train(self, images, force_train=False):

        for j in range(1000):
            samples = np.arange(128)
            x_arr = []
            x_arr_diff = []
            n_dis_samples = 128
            for i in range(n_dis_samples):
                samples = np.arange(i, 128 + i)
                x = np.sin(2 * np.pi * samples)
                x_dif = np.sin(2 * np.pi * samples)
                x_arr.append(x)
                x_arr_diff.append(x_dif)
            # self.show_result(x_arr)
            # exit(0)
            with tf.GradientTape(persistent=True) as tape:
                total_loss = 1 * self.state_rec_loss(x_arr)  # + 0.2 * self.sindy_loss_in_prime_z(x_arr, x_arr_diff) + \
                # 0.2 * self.sindy_loss_in_prime_h(x_arr, x_arr_diff) + 0.2 * self.sindy_loss_z1(x_arr) + \
                # 0.2 * self.sindy_loss_const(x_arr)
            print('total_loss', total_loss)
            gradients_encoder = tape.gradient(total_loss,
                                              self.f_encoder_model.trainable_variables + self.psi_decoder_model.trainable_variables)

            if True:  # None not in gradients_encoder:
                # print('gradients_encoder', gradients_encoder)
                self.optimizer.apply_gradients(zip(gradients_encoder,
                                                   self.f_encoder_model.trainable_variables + self.psi_decoder_model.trainable_variables))
        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(0)
        ckpt_name_enc = ckpt_name + 'enc'
        ckpt_name_dec = ckpt_name + 'dec'
        self.f_encoder_model.save('./checkpoints/' + ckpt_name_enc)
        self.psi_decoder_model.save('./checkpoints/' + ckpt_name_dec)

    def predict(self, image):
        pass
