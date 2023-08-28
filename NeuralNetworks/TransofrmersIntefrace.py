from utils.Agent import *


class SwinTransformer(object):

    def __init__(self, inputs, outputs, data_schema):
        self.model = None
        self.EPOCHS = 1
        self.init_neural_network(inputs, outputs, data_schema)

    def init_neural_network(self, inputs, outputs, data_schema,
                            model_name='swin_tiny_224', num_classes=1000,
                            include_top=True, pretrained=True, use_tpu=False, cfgs=CFGS):
        local_input = inputs[0]
        self.local_output = outputs[0]

        if self.local_output.type == REGRESSION:
            self.num_classes = 1
            loss = 'mean_squared_error'
        elif self.local_output.type == REGRESSION_CATEGORY:
            self.num_classes = 100
            loss = 'categorical_crossentropy'
        cfg = cfgs[model_name]
        net = SwinTransformer(
            model_name=model_name, include_top=include_top, num_classes=num_classes, img_size=cfg['input_size'],
            window_size=cfg[
                'window_size'], embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads']
        )
        net(tf.keras.Input(shape=(cfg['input_size'][0], cfg['input_size'][1], 3)))
        if pretrained is True:
            url = f'https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/{model_name}.tgz'
            pretrained_ckpt = tf.keras.utils.get_file(
                model_name, url, untar=True)
        else:
            pretrained_ckpt = pretrained

        if pretrained_ckpt:
            if tf.io.gfile.isdir(pretrained_ckpt):
                pretrained_ckpt = f'{pretrained_ckpt}/{model_name}.ckpt'

            if use_tpu:
                load_locally = tf.saved_model.LoadOptions(
                    experimental_io_device='/job:localhost')
                net.load_weights(pretrained_ckpt, options=load_locally)
            else:
                net.load_weights(pretrained_ckpt)
        net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-8),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        self.model = net

    def prepare_data(self, images, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []

        for image in images:
            local_image = image.get_by_name('Image')
            if local_image is None:
                continue
            local_x_train_arr.append(
                np.array(cv2.resize(local_image, (224, 224), interpolation=cv2.INTER_AREA), dtype=np.float32))
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
        history = self.model.fit(x_train, y_train, epochs=self.EPOCHS,
                                 validation_data=(x_train, y_train))

    def predict(self, image):
        local_image = image.get_by_name('Image')
        if local_image is None:
            return [0]
        x, _ = self.prepare_data([image])
        return self.model.predict(np.array(x), batch_size=1)[0]

    def evaluate(self):
        pass
