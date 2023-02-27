from utils.Agent import *

import json

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

last_img = np.zeros((128, 96, 3))

CHANNEL_N = 8  # Number of CA state channels
TARGET_PADDING = 0  # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5
DIVISION = 1  # 16
EXPERIMENT_TYPE = "Regenerating"  # @param ["Growing", "Persistent", "Regenerating"]Regenerating
EXPERIMENT_MAP = {"Growing": 0, "Persistent": 1, "Regenerating": 2}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]


class CAModel(tf.keras.Model):

    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        self.dmodel = tf.keras.Sequential([
            Conv2D(128, 1, activation=tf.nn.relu),
            Conv2D(self.channel_n, 1, activation=None,
                   kernel_initializer=tf.zeros_initializer),
        ], name='dmodel_seq')

        self(tf.zeros([1, 64, 64, channel_n]))  # dummy call to build the model

    @tf.function
    def perceive(self, x, angle=0.0):
        identify = np.float32(
            [0, 1, 0])

        identify = np.outer(identify, identify)
        dx = np.outer([1, 0, 1], [1, 0, 1]) / 8.0
        dy = dx.T
        c, s = tf.cos(angle), tf.sin(angle)
        kernel = tf.stack([identify, c * dx - s * dy, s * dx + c * dy], -1)[:, :, None, :]
        kernel = tf.repeat(kernel, self.channel_n, 2)

        y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
        y = tf.math.l2_normalize(y)
        return y

    def get_living_mask(self, x):
        alpha = x[:, :, :, 3:4]
        return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1

    @tf.function
    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        global global_pool

        pre_life_mask = self.get_living_mask(x)
        # print('x',x.shape)
        y = self.perceive(x, angle)

        dx = self.dmodel(y) * step_size
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        # removing the + sign lead to no generation
        x += dx * tf.cast(update_mask, tf.float32)

        # x += global_pool.read()*0.1
        # global_pool.write(x)
        post_life_mask = self.get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * tf.cast(life_mask, tf.float32)


@tf.function
def make_circle_masks(n, h, w):
    x = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = tf.cast(x * x + y * y < 1.0, tf.float32)
    return mask


class CellularAutomataAndData(Agent):
    def export_model(self, ca, base_fn):
        ca.save_weights(base_fn)

        cf = ca.call.get_concrete_function(
            x=tf.TensorSpec([None, None, None, CHANNEL_N]),

            fire_rate=tf.constant(0.5),
            angle=tf.constant(0.0),
            step_size=tf.constant(1.0))
        cf = convert_to_constants.convert_variables_to_constants_v2(cf)
        graph_def = cf.graph.as_graph_def()
        graph_json = MessageToDict(graph_def)
        graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
        model_json = {
            'format': 'graph-model',
            'modelTopology': graph_json,
            'weightsManifest': [],
        }
        with open(base_fn + '.json', 'w') as f:
            json.dump(model_json, f)

    def __init__(self, local_inputs, local_outputs, data_schema_input, data_schema_output, class_num):
        self.model = CAModel()
        local_input = None
        data_schema_input = data_schema_input['train']
        self.data_schema_output = data_schema_output

        for element in data_schema_input:
            if element.name == 'image_data':
                local_input = element
        self.init_neural_network(local_input, local_outputs, data_schema_input, class_num)
        self.total_tested = 0
        self.good_tested = 0
        lr = 2e-3
        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [2000], [lr, lr * 0.1])
        self.model.load_weights('./model/')
        self.trainer = tf.keras.optimizers.Adam(lr_sched)
        self.loss_log = []

    def init_neural_network(self, inputs, outputs, data_schema, class_num):
        pass

    def prepare_data(self, images, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []

        for image in images:

            local_y = None
            try:

                local_image = image['train'].get_by_name('image_data')
                if image['train'].get_by_name('cancer')[0] == 1:
                    continue

                for element in self.data_schema_output['train']:
                    if not element.is_id:
                        local_y = image['train'].get_by_name(element.name)
            except Exception as e:
                local_image = image['train']['image_data']
                for element in self.data_schema_output['train']:
                    if not element.is_id:
                        local_y = image['train'][element.name]
            if len(local_image) == 0:
                continue
            if local_image is None:
                continue
            local_image = np.array(local_image)

            local_image = np.pad(np.expand_dims(preprocess_fix_dim(local_image, 256, 256), axis=2), ((0, 0), (0, 0), (2,
                                                                                                                      0)))  # local_image = np.pad(np.array(local_image), ((5, 6), (5, 6), (0, 0)), mode='constant', constant_values=255)

            np.array(local_y_train_arr.append(local_y))
            np.array(local_x_train_arr.append(local_image))

        return np.array(local_x_train_arr), np.array(local_y_train_arr)

    def to_rgba(self, x):

        return x[..., :4]

    @tf.function
    def loss_f(self, x):
        return tf.reduce_mean(tf.square(self.to_rgba(x) - self.pad_target), [-2, -3, -1])

    def train_step(self, x):
        iter_n = tf.random.uniform([], 64, 96, tf.int32)
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = self.model(x)
            loss = tf.reduce_mean(self.loss_f(x))

        grads = g.gradient(loss, self.model.weights)

        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        self.trainer.apply_gradients(zip(grads, self.model.weights))
        return x, loss

    def train(self, images, force_train=False):

        x_train, y_train = self.prepare_data(images, in_train=True)
        y_train = np.expand_dims(x_train, 0)

        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(0)
        if Path('./checkpoints/' + ckpt_name).exists() and not force_train:
            self.model = tf.keras.models.load_model('./checkpoints/' + ckpt_name)
        else:

            for i in range(50):
                for j in range(len(x_train)):

                    if USE_PATTERN_POOL:
                        x0 = x_train[j]

                        x0 = np.float32(x0)

                        print("x0.shape", x0.shape)
                        x1 = [np.pad(x0, ((0, 0), (0, 0), (0, CHANNEL_N - 3)), mode='edge') / 255.0] * 1

                        x0 = tf.convert_to_tensor(x1)

                    else:
                        pass

                    x, loss = self.train_step(x0)

                    self.loss_log.append(loss.numpy())

                    self.export_model(self.model, './data_sets/rsna-breast-cancer-detection/model')
                    print('\r step: %d, log10(loss): %.3f' % (len(self.loss_log), np.log10(loss)), end='')

    def save(self):
        pass

    def predict(self, image):
        img, _ = self.prepare_data([image], in_train=False)
        if len(img) == 0:
            return 0
        img = [np.pad(img[0], ((0, 0), (0, 0), (0, CHANNEL_N - 3)), mode='edge') / 255.0] * 1

        result = self.model.predict(np.array(img), batch_size=1)

        return np.linalg.norm(np.array(result[0][:, :, :3]) - img[0][:, :, :3])


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
    raw_img = cv2.resize(np.array(raw_img, np.float), (height, width))

    return raw_img
