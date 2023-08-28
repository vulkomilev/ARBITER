import copy

import numpy as np

from utils.utils import normalize_list, one_hot, DataUnit
from utils.Agent import *
import datetime
#from  .Transformer import Transformer
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import json
import tensorflow_models as tfm
nlp = tfm.nlp

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
tf.config.run_functions_eagerly(True)

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)

batch_size = 32
glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=32)
tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    lower_case=True)

tokens = tokenizer(tf.constant(["Hello TensorFlow!"]))

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

encoder_config = tfm.nlp.encoders.EncoderConfig({
    'type': 'bert',
    'bert': config_dict
})


bert_encoder = tfm.nlp.encoders.build_encoder(encoder_config)

bert_classifier = tfm.nlp.models.BertClassifier(network=bert_encoder, num_classes=2)



class ExportModel(tf.Module):
  def __init__(self, input_processor, classifier):
    self.input_processor = input_processor
    self.classifier = classifier

  @tf.function(input_signature=[{
      'sentence1': tf.TensorSpec(shape=[None], dtype=tf.string),
      'sentence2': tf.TensorSpec(shape=[None], dtype=tf.string)}])
  def __call__(self, inputs):
    packed = self.input_processor(inputs)
    logits =  self.classifier(packed, training=False)
    result_cls_ids = tf.argmax(logits)
    return {
        'logits': logits,
        'class_id': result_cls_ids,
        'class': tf.gather(
            tf.constant(info.features['label'].names),
            result_cls_ids)
    }





class BertInputProcessor(tf.keras.layers.Layer):
    def __init__(self, tokenizer, packer):
        super().__init__()
        self.tokenizer = tokenizer
        self.packer = packer

    def call(self, inputs):

        tok1 = self.tokenizer(inputs['sentence1'])
        #tok2 = self.tokenizer(inputs['sentence2'])
        #tok3 = self.tokenizer(inputs['sentence3'])

        packed = self.packer([tok1])#, tok2,tok3])
        if 'label' in inputs:
            return packed, inputs['label']
        elif 'label' in list(inputs.keys()):
            return packed, inputs['label']
        else:
            return packed

max_seq_length = 128

packer = tfm.nlp.layers.BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict = tokenizer.get_special_tokens_dict())
sentences1 = ["hello tensorflow"]
tok1 = tokenizer(sentences1)

sentences2 = ["goodbye tensorflow"]
tok2 = tokenizer(sentences2)

packed = packer([tok1, tok2])
bert_inputs_processor = BertInputProcessor(tokenizer, packer)
#glue_train = glue['train'].map(bert_inputs_processor).prefetch(1)

#example_inputs, example_labels = next(iter(glue_train))
#bert_classifier(
 #   example_inputs, training=True).numpy()[:10]

checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
checkpoint.read(
    os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()

epochs = 1
batch_size = 32
eval_batch_size = 32

train_data_size = info.splits['train'].num_examples
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(0.1 * num_train_steps)
initial_learning_rate=2e-5

linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    end_learning_rate=0,
    decay_steps=num_train_steps)

warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
    warmup_learning_rate = 0,
    after_warmup_lr_sched = linear_decay,
    warmup_steps = warmup_steps
)

optimizer = tf.keras.optimizers.experimental.Adam(
    learning_rate = warmup_schedule)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#glue_validation = glue['validation'].map(bert_inputs_processor).prefetch(1)
#glue_test = glue['test'].map(bert_inputs_processor).prefetch(1)
bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

#bert_classifier.evaluate(glue_validation)
#glue_train_1 = glue_train
#bert_classifier.fit(
#    glue_train,
#    validation_data=(glue_validation),
#    batch_size=32,
#    epochs=epochs)
my_examples = {
        'sentence1':[
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.'],
        'sentence2':[
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.']
    }
#ex_packed = bert_inputs_processor(my_examples)
#my_logits = bert_classifier(ex_packed, training=False)

#result_cls_ids = tf.argmax(my_logits)

#tf.gather(tf.constant(info.features['label'].names), result_cls_ids)
export_model = ExportModel(bert_inputs_processor, bert_classifier)

import tempfile

export_dir = tempfile.mkdtemp(suffix='_saved_model')
#tf.saved_model.save(export_model, export_dir=export_dir,
#                    signatures={'serving_default': export_model.__call__})

#original_logits = export_model(my_examples)['logits']

#reloaded = tf.saved_model.load(export_dir)
#reloaded_logits = reloaded(my_examples)['logits']



class DenseAndTransformersClassification(Agent):

    def __init__(self,input_dict,output_dict):
        self.model = None

        self.total_tested = 0
        self.good_tested = 0
        self.local_bucket = []
        self.reg_input = []
        self.reg_output = []
        for element in input_dict:
         if element.is_id == False:
             self.reg_input.append(DataUnit(str(element.type), (), None, '', is_id=element.is_id))
        for element in output_dict:
         if element.is_id == False:
             self.reg_output.append(DataUnit(str(element.type), (), None, '', is_id=element.is_id))

        #self.init_neural_network()
    def register(self, arbiter):
        arbiter.register_neural_network(self, self.reg_input, self.reg_output)

    def unlist(self, element):
        local_list = []
        if type(element) == list:
            if type(element[0]) == list:
                for local_element in element:
                    local_list.append(self.unlist(local_element))
            else:
                local_list = element
        elif type(element) == dict:
            for key, value in element.items():
                local_list.append(self.unlist(value))
        return local_list

    def init_neural_network(self):

        input_model = tf.keras.Input(shape=(57))
        num_layers = 1
        d_model = 4
        dff = 512
        num_heads = 2
        dropout_rate = 0.1
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(['21','31'])
        #print(len(token.num_words)
        #exit(0)

        input_model = tf.keras.Input(shape=(len(self.reg_input),1))#,dtype='int')
        self.vectorize_layer = tf.keras.layers.TextVectorization(
           max_tokens=5000,
           output_mode='int',
           output_sequence_length=4)

        model_mid = self.vectorize_layer(input_model)
        net = Transformer(num_layers, d_model, num_heads, dff, len(self.tokenizer.num_words),
                          len(self.tokenizer.num_words), pe_input=2048, pe_target=2048, rate=0.1)(input_model)
        model_mid = tf.keras.layers.Dense((len(self.reg_input)))(net)
        model_mid = tf.keras.layers.Dense((len(self.reg_input)/2))(model_mid)
        model_mid = tf.keras.layers.Dense((len(self.reg_input)/4))(model_mid)
        model_mid = tf.keras.layers.Dense((len(self.reg_input)/8))(model_mid)
        model_mid = tf.keras.layers.Dense((len(self.reg_output)),activation='softmax')(model_mid)
        self.model = tf.keras.Model(inputs=input_model, outputs=model_mid)

        self.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), run_eagerly=True)

    def normalize_date(self, date_list):
        return_list = []
        for element in date_list:
            return_list.append(float(datetime.datetime.strptime(element, '%Y-%m-%d %H:%M:%S').strftime("%s")))
        return_list = normalize_list(return_list, max(return_list), min(return_list), 1.0, -1.0)
        return return_list

    def normalize(self, data, data_schema, path, target):

        return_dict = {}
        if type(data_schema) is dict:
            for element_key in data_schema.keys():
                local_element = data[element_key]
                return_dict[element_key] = self.normalize(local_element, data_schema[element_key], path + [element_key])
        elif type(data_schema) is list:
            for i, element_key in enumerate(data_schema):
                local_data = None
                local_data_path = ''
                for element_path in path:
                    local_data_path += '[' + element_path + ']'

                local_data = []
                for element in data:
                    local_data.append(element[i])
                if type(local_data) == list or type(local_data) == type(np.ndarray(shape=0)):
                    if element_key.type == 'int' and element_key.is_id == False:
                        return_dict[element_key.name] = normalize_list(local_data, max(local_data), min(local_data),
                                                                       1.0, -1.0)
                    elif element_key.type == 'date' and element_key.is_id == False:
                        return_dict[element_key.name] = self.normalize_date(local_data)
                    elif element_key.type == 'str' and element_key.is_id == False:
                        local_data = one_hot(local_data, element_key)
                        if len(local_data) > 0:
                            return_dict[element_key.name] = local_data
                    elif element_key.type == 'float' and element_key.is_id == False:
                        return_dict[element_key.name] = normalize_list(local_data, max(local_data), min(local_data),
                                                                       1.0, -1.0)
                    #elif element_key.is_id == False:
                    #    exit(0)

        return return_dict

    def prepare_data(self, data, in_train=False):

        local_data_input = []
        local_data_input_str = []
        local_data_output = []
        for element in data:
            local_list = []
            local_list_str = []
            for second_element in self.reg_input:

                if len(second_element.name) > 0:
                    local_element = element.source.get_by_name(second_element.name)
                    if local_element == None:
                        local_list.append('')

                    if type(local_element) == type([]):
                          local_list.append(element.source.get_by_name(second_element.name)[0])
                    else:
                            local_list.append(element.source.get_by_name(second_element.name))
            local_data_input.append(local_list)


        for element in data:
           local_list = []
           for second_element in self.reg_output:
                if len(second_element.name) > 0:
                    local_element = element.target.get_by_name(second_element.name)
                    if local_element == None:
                        local_element = 0
                    local_list.append(local_element)
           local_data_output.append(local_list)
        normalized_data_input = []
        normalized_data_output = []
        for x,y in zip(local_data_input,local_data_output):

           local_data = bert_inputs_processor({'sentence1': tf.convert_to_tensor([x[1]], dtype=tf.string),

                                            'label': np.array([1])})#y[0]
           #print(int(y[0]))
           #exit(0)
           normalized_data_input.append(local_data[0])
           normalized_data_output.append(local_data[1])
        #    local_data_output.append(local_list)
        #local_data_input = np.array(local_data_input)
        #local_data_input_str = local_data_input_str#np.array(local_data_input)#.tolist()
        #normalized_data_input = np.array(self.unlist(normalized_data_input))
        #normalized_data_output = np.array(self.unlist(normalized_data_output))
        norm_arr_x = normalized_data_input#np.array(normalized_data_input).tolist()
        norm_arr_y = normalized_data_output

        return norm_arr_x, norm_arr_y

    def train(self, images, force_train=False,only_fill=False):
        if images != None:
            self.local_bucket.append(images)
        if len(self.local_bucket) < 100:
            return
        if only_fill:
            return

        x_train, y_train = self.prepare_data(self.local_bucket, in_train=True)

        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(0)
        if Path('./checkpoints/' + ckpt_name).exists() and not force_train:
            self.model = tf.keras.models.load_model('./checkpoints/' + ckpt_name)
        else:

            #self.vectorize_layer.adapt(x_train[0])

            for x,y in zip(x_train,y_train):

                bert_classifier.fit(
                    x=x,
                    y=y,
                   # validation_data=(glue_validation),
                    batch_size=32,
                    epochs=epochs)
            #self.model.fit(x_train, y_train, batch_size=32, epochs=1)
            #self.model.save('./checkpoints/' + ckpt_name)
        self.local_bucket = []

    def save(self):
        pass

    def predict(self, image):
        print("predict")
        x_train, y_train = self.prepare_data([image], in_train=True)
        #print(x_train)
        _ = self.model.predict(x_train)
        #print(_)
        return abs(_[0])
