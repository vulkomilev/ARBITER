from utils.Agent import *
import Transformer

class TransformerInterfaceText(object):

    def __init__(self, inputs, outputs, data_schema):
        self.model = None
        self.EPOCHS = 1
        self.init_neural_network(inputs, outputs, data_schema)

    def masked_loss(self,label, pred):
        mask = label != 0
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        loss = loss_object(label, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss

    def masked_accuracy(self,label, pred):
        pred = tf.argmax(pred, axis=2)
        label = tf.cast(label, pred.dtype)
        match = label == pred

        mask = label != 0

        match = match & mask

        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(match) / tf.reduce_sum(mask)

    def init_neural_network(self, inputs, outputs, data_schema,
                            model_name='swin_tiny_224', num_classes=1000,
                            include_top=True, pretrained=True, use_tpu=False, cfgs=CFGS):

        cfg = cfgs[model_name]
        num_layers = 4
        d_model = 128
        dff = 512
        num_heads = 8
        dropout_rate = 0.1
        token = tf.keras.preprocessing.text.Tokenizer(['21','31'])

        net = Transformer.Transformer(num_layers, d_model, num_heads, dff, token.num_words,
                          token.num_words, pe_input=2048, pe_target=2048, rate=0.1)
        net(tf.keras.Input(shape=(cfg['input_size'][0], cfg['input_size'][1], 3)))


        net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-8),
            loss=self.masked_loss,
            metrics=self.masked_accuracy
        )
        self.model = net

    def prepare_data(self, data_bundle, in_train=False):
        local_x_train_arr = []
        local_y_train_arr = []


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
