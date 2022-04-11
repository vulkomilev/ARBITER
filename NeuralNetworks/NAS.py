import random


class NAS(object):

    def __init__(self, arvg):

        self.start_dim = arvg.get("start_dim", (28, 28, 3))
        self.end_dim = arvg.get("end_dim", (28, 28, 3))
        self.last_dim = arvg.get("start_dim", (28, 28, 3))
        self.max_random_output = arvg.get("max_random_output", 24)
        self.min_random_output = arvg.get("min_random_output", 1)
        self.max_depth = arvg.get("max_depth", 12)
        self.min_depth = arvg.get("min_depth", 5)
        self.rand_layer = ["tf.keras.layers.Dense", "tf.keras.layers.Conv2D"]  # , "tf.keras.layers.AveragePooling2D",
        #  "tf.keras.layers.Flatten"]

    def init_neural_network(self):
        local_depth = random.randint(self.min_depth, self.max_depth)
        model_str = "model_input = tf.keras.Input(shape=self.start_dim)\n"
        is_first_layer = True
        while local_depth > 0:
            local_rand_layer = random.choice(self.rand_layer)
            local_output = random.randint(self.min_random_output, self.max_random_output)
            # model_str += local_rand_layer
            if is_first_layer:
                next_layer = 'model_input'
            else:
                next_layer = 'model_middle'
            if "Dense" in local_rand_layer:
                model_str += "model_middle = " + local_rand_layer + "(" + str(local_output) + ")(" + next_layer + ")\n"
                self.last_dim = local_output
            if "Conv" in local_rand_layer:
                model_str += "model_middle = " + local_rand_layer + "(" + str(
                    local_output) + ",kernel_size=(2,2)" + ")(" + next_layer + ")\n"
            local_depth -= 1
            is_first_layer = False
        model_str += "self.model = tf.keras.Model(inputs = model_input,outputs = model_middle)\n"
        model_str += "self.model.compile(optimizer='adam', loss='mse',metrics=[tf.keras.metrics.CategoricalAccuracy()])"
        print(model_str)
        exec(model_str)


for i in range(100):
    local_nas = NAS({})
    local_nas.init_neural_network()
