import numpy as np

from utils.Agent import *
from utils.utils import REGRESSION,REGRESSION_CATEGORY,TIME_SERIES
from utils.utils import DataCollection



class LSTM(Agent):

    def __init__(self, inputs,outputs,data_schema_input,data_schema_output,class_num,time_steps = 2):
        self.model = None
        self.time_steps = time_steps
        self.input_names = []
        self.output_names = []
        self.input_ids = []
        self.data_schema_input = data_schema_input
        self.data_schema_output  = data_schema_output
        self.init_neural_network(inputs,outputs,data_schema_input,data_schema_output)



    def init_neural_network(self, inputs,outputs,data_schema_input,data_schema_output):
        self.local_inputs = inputs
        self.local_outputs = outputs
        for element in data_schema_input:
            if element.is_id:
             self.input_ids.append(element.name)
        for element in self.local_inputs :
            self.input_names.append(element.name)
        for element in self.local_outputs :
            self.output_names.append(element.name)
        self.local_output = outputs[0]


        if self.local_output.type == REGRESSION:
            self.num_classes = 1
            loss = 'mean_squared_error'
        elif self.local_output.type == REGRESSION_CATEGORY:
            self.num_classes = 100
            loss = 'categorical_crossentropy'
        elif self.local_output.type == TIME_SERIES:
            self.num_classes = 100
            loss = 'categorical_crossentropy'
        local_input_dim = 0

        for element in self.local_inputs:
            local_input_dim += element.shape[0]
        print('self.time_steps',self.time_steps)
        print('local_input_dim',local_input_dim)
        input_dim = ( self.time_steps ,local_input_dim)


        model_inputs = tf.keras.Input(shape=input_dim)
        model = tf.keras.layers.LSTM(input_dim[1],return_sequences=True)(model_inputs)
        model = tf.keras.layers.LSTM(input_dim[1],return_sequences=True)(model)
        model = tf.keras.layers.LSTM(input_dim[1],return_sequences=True)(model)
        model = tf.keras.layers.Dense(1)(model)
        self.model = tf.keras.Model(inputs = model_inputs,outputs = model)
        self.model.compile(optimizer='sgd',loss = loss)

    def prepare_data(self, series,data_compresion_rate =32, in_train=False):
       x = []
       x_ids = []
       y = []

       if 'local_x' not in dir(self):
           self.local_x = []
       if 'local_y' not in dir(self):
           self.local_y = []
       if 'local_ids' not in dir(self):
           self.local_ids = []
       if in_train:
           self.local_max_by_name = {}
           self.local_min_by_name = {}
           for i, element in enumerate(series):
               for input_name in self.input_names:
                   if input_name not in self.local_min_by_name:
                       self.local_max_by_name[input_name] = []
                       self.local_min_by_name[input_name] = []
                   self.local_min_by_name[input_name].append(element.get_by_name(input_name))
           for i, element in enumerate(series):
               for output_name in self.output_names:
                   if output_name not in self.local_min_by_name:
                       self.local_max_by_name[output_name] = []
                       self.local_min_by_name[output_name] = []
                   self.local_min_by_name[output_name].append(element.get_by_name(output_name))
           for name_key in self.local_min_by_name.keys():
                self.local_max_by_name[name_key] =  max(self.local_min_by_name[name_key])
           for name_key in self.local_min_by_name.keys():
               self.local_min_by_name[name_key] = min(self.local_min_by_name[name_key])

       for i,element in enumerate(series):
           local_data = []
           for input_name in self.input_names:
               local_result = 0
               try:
                local_result = (element.get_by_name(input_name)-self.local_min_by_name[input_name])/(self.local_max_by_name[input_name]-self.local_min_by_name[input_name])
               except ZeroDivisionError as e:
                   local_result = element.get_by_name(input_name)/1.0
               if math.isnan(local_result):
                      local_result = 0
               local_data.append(local_result)

           self.local_x.append(np.array(local_data))
           local_data = {}
           for input_ids in self.input_ids:

               local_data[input_ids] = element.get_by_name(input_ids)

           self.local_ids.append(local_data)
           local_data = []
           for output_name in self.output_names:
                  local_result = 0
                  local_val = element.get_by_name(output_name)
                  if local_val  == None:
                      local_val = 0
                  try:
                    local_result = (local_val-self.local_min_by_name[output_name])/(self.local_max_by_name[output_name]-self.local_min_by_name[output_name])
                  except ZeroDivisionError as e:
                    local_result = local_val/1.0
                  if math.isnan(local_result):
                      local_result = 0
                  local_data.append(local_result)
           self.local_y.append(np.array(local_data))
           if in_train:
               if  i% data_compresion_rate == 0 and i >= data_compresion_rate:
                   if len(self.local_x) != data_compresion_rate or len(self.local_y)   != data_compresion_rate:
                       self.local_x = []
                       self.local_y = []
                       self.local_ids = []
                       continue
                   x_ids.append(self.local_ids)
                   x.append(np.array(self.local_x))
                   y.append(np.array(self.local_y))
                   self.local_x = []
                   self.local_y = []
                   self.local_ids = []

           else:

               if len(self.local_x) == data_compresion_rate or len(self.local_y) == data_compresion_rate:
                   x_ids.append(self.local_ids)
                   x.append(np.array(self.local_x))
                   y.append(np.array(self.local_y))
                   self.local_x = []
                   self.local_y = []
                   self.local_ids = []
                   continue

               #self.local_x = []
               #self.local_y = []
                   #self.local_x = []
                   #self.local_y = []

       return np.array(x), np.array(y),x_ids

    def train(self, series, force_train=False):
        print('---train---')
        x_train, y_train ,x_id= self.prepare_data(series,data_compresion_rate=self.time_steps, in_train=True)

        ckpt_name = 'default_cpkt_name'
        re_result = re.search(r'.*\.(\w*)', str(self.__class__), re.S | re.U | re.I)
        if re_result:
            ckpt_name = re_result.group(0)
        if Path('./checkpoints/' + ckpt_name).exists() and not force_train:
            self.model = tf.keras.models.load_model('./checkpoints/' + ckpt_name)
        else:
            self.model.fit(x_train, y_train, epochs=10)
            self.model.save('./checkpoints/' + ckpt_name)

    def predict(self, image):
        output_name = self.output_names[0]
        x,_,x_ids = self.prepare_data([image],in_train = False,data_compresion_rate=self.time_steps)

        if len(x)  == 0:
            return 0#np.zeros(1)
        local_predict  = self.model.predict(x, batch_size=1)
        #for element in local_predict:
        #    print(element)
        result =  np.squeeze(self.model.predict(x, batch_size=1)[0])

        for i in range(len(result)):

                result[i] =(result[i]-10)*(self.local_max_by_name[output_name]-self.local_min_by_name[output_name])
                result[i] =  result[i]*4
        row = []

        for i,element_schema in enumerate(self.data_schema_output):
            if element_schema.is_id :#and element_schema.name in:
                row.append(x_ids[0][-1][element_schema.name])

        row.append(result[-1])
        return DataCollection(data_size=0, data_schema=self.data_schema_output, data=row)
        #return result
#10 BA
#10 Dev
#200 +
#
#Programming + Research (YOLO ,debug and other research)
