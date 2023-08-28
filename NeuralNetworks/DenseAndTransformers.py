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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



class DenseAndTransformers(Agent):

    def __init__(self,input_dict,output_dict):
        self.model = None

        self.total_tested = 0
        self.good_tested = 0
        self.local_bucket = []
        self.reg_input = []
        self.reg_output = []
        self.loss_history = []
        for element in input_dict:
         if element.is_id == False:
             self.reg_input.append(DataUnit(str(element.type), (), None, '', is_id=element.is_id))
        for element in output_dict:
         if element.is_id == False:
             self.reg_output.append(DataUnit(str(element.type), (), None, '', is_id=element.is_id))

        self.init_neural_network()
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

        input_model = tf.keras.Input(shape=(57) ,dtype=tf.string)
        num_layers = 1
        d_model = 4
        dff = 512
        num_heads = 2
        dropout_rate = 0.1
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(['21','31'])
        #print(len(token.num_words)
        #exit(0)
        embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 32
        vocab_size = 20000  # Only consider the top 20k words
        maxlen = 200
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=maxlen)


        input_model = layers.Input(shape=(1,),dtype=tf.string)
        x =  self.vectorize_layer(input_model)
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = embedding_layer(x)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(2, activation="relu")(x)
        self.model = tf.keras.Model(inputs=input_model, outputs=outputs)

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
        maxlen = 200
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
        adapt_list = []
        for x,y in zip(local_data_input,local_data_output):


           local_data  = [x[1] ,np.array(y)]#y[0]
           #print(int(y[0]))
           #exit(0)
           adapt_list.append([local_data[0]])
           #self.vectorize_layer.adapt([[local_data[0]]])
           normalized_data_input.append(local_data[0])
           normalized_data_output.append(local_data[1])
        #    local_data_output.append(local_list)
        #local_data_input = np.array(local_data_input)
        self.vectorize_layer.adapt(adapt_list)
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

            #for x,y in zip(x_train,y_train):

             loss = self.model.fit(
                    x=np.asarray(x_train),
                    y=np.asarray(y_train),
                   # validation_data=(glue_validation),
                    batch_size=32,
                    epochs=1)
             self.loss_history +=loss.history['loss']
            #self.model.fit(x_train, y_train, batch_size=32, epochs=1)
            #self.model.save('./checkpoints/' + ckpt_name)
        self.local_bucket = []



    def save(self):
        pass

    def predict(self, image):
        print("predict")

        x_train, y_train = self.prepare_data([image], in_train=True)
        _ = self.model.predict(x_train)
        print(_)
        return abs(_[0])
