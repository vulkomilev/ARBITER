import concurrent.futures
import copy
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
THREAD_COUNT = 32
loaded_ids = []
loaded_ids_target = []
index_csv = {}

IMAGE_EXTS_LIST = ['jpg', 'png']
DATA_FORMATS_SPECIAL = [ '2D_F', '2D_INT']
DATA_FORMATS = ['float', 'int', 'str'] + DATA_FORMATS_SPECIAL
REGRESSION_CATEGORY = 'int'
REGRESSION = 'float'
CATEGORY = 'Category'
TIME_SERIES = 'TimeSeries'
IMAGE = '2D_F'
TARGET_TYPES = [REGRESSION,REGRESSION_CATEGORY, CATEGORY, TIME_SERIES,IMAGE]



class DataCollection(object):

    def __init__(self, data_size, data_schema, data):
        self.data_size = data_size
        self.data_collection = []
        self.data_schema = data_schema
        self.add_data(data)
        for element in data_schema:
            if element.type not in DATA_FORMATS:
                raise RuntimeError('Data type in schema is not supported')

    def get_shape_by_name(self, name):
        for element in self.data_collection:
            if element.name == name:
                return element.shape

    def get_by_name(self, name):
        for element in self.data_collection:
            if element.name == name:
                return element.data

    def set_by_name(self, name,val):
        for i,element in enumerate(self.data_collection):
            if element.name == name:
                self.data_collection[i].data = copy.deepcopy(val)
                return


    def add_data(self, data):
        if len(data) == 0:
            return

            #assert (len(data_element) == self.data_size)
        data_element_collection = []
        added_elements = []
        for schema_element, element in zip(self.data_schema, data):
                data_unit = DataUnit(type=schema_element.type, shape=schema_element.shape, name=schema_element.name,
                                     data=element)
                added_elements.append(schema_element.name)
                data_element_collection.append(data_unit)
        for element in self.data_schema:
            if element.name == 'Image':
                data_unit = DataUnit(type=element.type, shape=element.shape, name=element.name,
                                     data=None)
                data_element_collection.append(data_unit)


        self.data_collection = data_element_collection

    def remove(self, name):

        for element in self.data_collection:
            if element.name == name:
                self.remove(element)

def try_convert_float(f):
    try:
        print(f)
        return float(f)
    except Exception as e:
        return 0.0
class DataUnit(object):

    def __init__(self, type, shape, data, name=''):
        if type not in DATA_FORMATS:
            raise RuntimeError('Data type in data provided is not supported')
        if data is not None:
            if np.array(data).shape != shape and shape != () and \
                np.array([data]).shape != shape:

                raise RuntimeError('Data shape is different form the schema',np.array([data]).shape , shape)
        if data != None:
            if type == 'str':
                data = str(data)
            elif type == 'int':
                data = int(data)
            elif type == 'float':
                data = float(data)
        self.type = type
        self.shape = shape
        self.data = data
        self.name = name


def contur_image(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.blur(grey_img, (3, 3), 0)
    contur_img = cv2.Sobel(src=blur_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    return contur_img


def is_int(i):
    try:
        int(i)
        return True
    except Exception as e:
        return False


def re_size_image(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return img


# is this thread save
def image_json_loader_worker(args):
    global loaded_ids_target
    image_paths = args
    local_image_collection = {'train_inputs':[],'train_outputs':[],'test_inputs':[],'test_outputs':[]}
    for i, image_col in enumerate(image_paths):
        if i % 1000 == 0:
            print(i, '/', len(image_paths))
        try:
            image_path, image_name = image_col
        except Exception as e:
            continue
        with open(image_path + image_name,mode='r') as f:
               local_json = f.read()
               local_json = json.loads(local_json)
               for element in local_json['train']:

                   local_image_collection['train_inputs'].append(np.array(element['input']))
                   local_image_collection['train_outputs'].append(np.array(element['output']))
               for element in local_json['test']:
                   local_image_collection['test_inputs'].append(np.array(element['input']))
                   local_image_collection['test_outputs'].append(np.array(element['output']))

    return local_image_collection

def image_loader_worker(args):
    global loaded_ids_target
    image_paths, image_ids, no_ids, target_name = args
    local_image_collection = {}

    for i, image_col in enumerate(image_paths):
        if i % 1000 == 0:
            print(i, '/', len(image_paths))
        try:
            image_path, image_name = image_col
        except Exception as e:
            continue
        if image_path[-1] != '/':
            image_path  = image_path +'/'
        local_img = cv2.imread(image_path + image_name)



        if image_ids is None:

            local_image_collection[image_name[:-4]] = \
            {"img": local_img, "data": None}
        else:
            if image_name[:-4] not in image_ids.keys():
                continue
            local_img_id = image_name[:-4]

            if local_img_id not in loaded_ids_target:
                loaded_ids_target.append(local_img_id)

            img_shape = image_ids[local_img_id].get_shape_by_name('Image')
            if  img_shape != None and len(img_shape) > 0:
                local_img = cv2.resize(local_img, (img_shape[0], img_shape[1]))
                local_data = copy.deepcopy(image_ids[local_img_id])
            local_image_collection[image_name[:-4]] ={"img": local_img,
                     "img_name": image_name[:-4]}

    return local_image_collection


def split_list(target_list, count, restrict=False, size=1000):
    interval = int(len(target_list) / count)
    splited_list = []
    for i in range(count):
        splited_list.append(target_list[i * interval:(i + 1) * interval])
    splited_list[-1].append(target_list[(count + 1) * interval:])
    if restrict:
        for i in range(len(splited_list)):
            random_start = random.randint(0, len(splited_list[i]) - (size + 1))
            splited_list[i] = splited_list[i][random_start:random_start + size]
    return splited_list

def image_loader_json_images(path, restrict=False, size=1000):
    image_paths = image_list(path)
    image_ids = None

    local_image_collection = {'num_classes': 0, 'image_arr': []}
    if size < len(image_paths):
        image_paths_list = split_list(image_paths, THREAD_COUNT, restrict, size)
    else:
        image_paths_list = []
        for element in image_paths:
            image_paths_list.append([element])

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:

        futures = [executor.submit(image_json_loader_worker, args) for args in
                   zip(image_paths_list)]
        print("THREAD COUNT:", len(futures))
        results = [f.result() for f in futures]


    return results

def image_loader(path,train_name = 'train', restrict=False, size=1000, no_ids=False,
                 load_image=False,target_name=None, data_schema=None,split=False,split_coef=0.9,THREAD_COUNT = 32):
    THREAD_COUNT = THREAD_COUNT
    image_paths = image_list(path)
    image_ids = None
    if Path(path + train_name + '.csv').exists():
        image_ids, loaded_ids_size = load_id_from_csv(path + train_name + '.csv', data_schema)
    local_image_collection = {'num_classes': 0, 'image_arr': []}
    if size < len(image_paths):
        image_paths_list = split_list(image_paths, THREAD_COUNT, restrict, size)
    else:
        image_paths_list = []
        for element in image_paths:
            image_paths_list.append([element])
    image_ids_splited = [image_ids] * THREAD_COUNT
    no_ids = [no_ids] * THREAD_COUNT
    target_name = [target_name] * THREAD_COUNT
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:

        futures = [executor.submit(image_loader_worker, args) for args in
                   zip(image_paths_list, image_ids_splited, no_ids, target_name)]
        print("THREAD COUNT:", len(futures))
        local_dict = [f.result() for f in futures]
        for element in local_dict:
            results.update(element)

    for key,val in enumerate(results):
        image_ids[val].set_by_name('Image',results[val].get('img',None))
    local_image_collection['num_classes'] = len(loaded_ids_target)
    if split:
        test_arr = []
        train_arr = []
        train_cutoff = int(split_coef*len(local_image_collection['image_arr']))
        #FIX THIS
        if not load_image:
            key_list = list(image_ids.keys())
            random.shuffle(key_list)
            train_cutoff = int(split_coef * len(key_list))
            for i in range(0,train_cutoff):
                train_arr.append(image_ids[key_list[i]])
            for i in range(train_cutoff,len(key_list)):
                test_arr.append(image_ids[key_list[i]])
            random.shuffle(train_arr)
            random.shuffle(test_arr)
            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr},\
                   {'num_classes': local_image_collection['num_classes'], 'image_arr': test_arr}
        else:
            for i in range(0,train_cutoff):
                train_arr.append(local_image_collection['image_arr'][i])
            for i in range(train_cutoff,len(local_image_collection['image_arr'])):
                test_arr.append(local_image_collection['image_arr'][i])
            random.shuffle(train_arr)
            random.shuffle(test_arr)
            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr},\
                   {'num_classes': local_image_collection['num_classes'], 'image_arr': test_arr}
    if not load_image:
        return image_ids

    return local_image_collection


def worker_load_image_data_from_csv(args):
    # global loaded_ids
    list, schema,cut_size = args
    local_data_arr = {}
    if cut_size != -1:
        list = list[:cut_size]
    for row in list:
        if len(row) == 0:
            continue
        try:
            local_data_arr[str(row[0])] = DataCollection(data_size=len(row), data_schema=schema, data=row)
        except Exception as e:
            print(e)
            pass
    return local_data_arr


def load_id_from_csv(csv_path, data_schema, restrict=False,size=100):
    local_ids = {}
    data = pd.read_csv(csv_path, low_memory=False)
    csv_reader = data.values.tolist()
    if len(csv_reader) > THREAD_COUNT:
        id_list = split_list(target_list=csv_reader, count=THREAD_COUNT, restrict=restrict,size=size)
        data_schema = [data_schema] * THREAD_COUNT
        if restrict:
            size = [size] * THREAD_COUNT
        else:
            size = [len(id_list[0])]* THREAD_COUNT
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(worker_load_image_data_from_csv, args) for args in zip(id_list, data_schema,size)]
            print("THREAD COUNT:", len(futures))
            results = [f.result() for f in futures]
        for element in results:
            local_ids = {**local_ids, **element}
    else:
        local_ids = worker_load_image_data_from_csv((csv_reader, data_schema,size))
    loaded_ids_size = len(local_ids.keys())
    return local_ids, loaded_ids_size


def image_list(path):
    image_path_list = []
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file[-3:] in IMAGE_EXTS_LIST:
                image_path_list.append((root, file))
    return image_path_list[1:]
