import concurrent.futures
import copy
import json
import os
import random
import math
from pathlib import Path
from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd
import pydicom
from matplotlib import image
from pydicom.pixel_data_handlers.util import apply_voi_lut

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
THREAD_COUNT = 32
loaded_ids = []
loaded_ids_target = []
index_csv = {}

IMAGE_EXTS_LIST = ['jpg', 'png']
DATA_FORMATS_SPECIAL = ['2D_F', '2D_INT']
DATA_FORMATS = ['float', 'int', 'str', 'date','bool'] + DATA_FORMATS_SPECIAL
REGRESSION_CATEGORY = 'int'
REGRESSION = 'float'
CATEGORY = 'Category'
STRING = 'str'
TIME_SERIES = 'TimeSeries'
IMAGE = '2D_F'
TARGET_TYPES = [REGRESSION, REGRESSION_CATEGORY, CATEGORY, TIME_SERIES, IMAGE]
HEURISTICS = {}
local_lib = {}


class DataCollection(object):

    def __init__(self, data_size, data_schema, data):
        self.data_size = data_size
        self.data_collection = []
        self.data_schema = data_schema
        self.add_data(data)
        for element in data_schema:
            if element.type not in DATA_FORMATS:
                raise RuntimeError('Data type in schema is not supported')

    def get_names(self):
        return_list = []
        for element in self.data_collection:
            return_list.append(element.name)
        return return_list

    def get_shape_by_name(self, name):
        for element in self.data_collection:
            if element.name == name:
                return element.shape

    def get_by_name(self, name):
        for element in self.data_collection:
            if element.name == name:
                return element.data
        return None

    def set_by_name(self, name, val):
        for i, element in enumerate(self.data_collection):
            if element.name == name:
                self.data_collection[i].data = copy.deepcopy(val)
                return

    def get_dict(self,include_only_id=False):

        return_dict = {}

        for element in self.data_collection:
            if not include_only_id:
                if not element.is_id:
                 return_dict[element.name] = element.data
            else:
                if element.is_id:
                 return_dict[element.name] = element.data
        return return_dict

    def add_data(self, data):
        if type(data) == dict:
            data_element_collection = []
            added_elements = []
            for schema_element, element in zip(self.data_schema, data):
                data_unit = DataUnit(type_val=schema_element.type,is_id=schema_element.is_id, shape=schema_element.shape, name=schema_element.name,
                                     data=data[schema_element.name])
                added_elements.append(schema_element.name)
                data_element_collection.append(data_unit)
                self.data_collection = data_element_collection
        else:
            if len(data) == 0:
                return

                # assert (len(data_element) == self.data_size)
            data_element_collection = []
            added_elements = []
            for schema_element, element in zip(self.data_schema, data):
                data_unit = DataUnit(type_val=schema_element.type,is_id=schema_element.is_id, shape=schema_element.shape, name=schema_element.name,
                                     data=element)
                added_elements.append(schema_element.name)
                data_element_collection.append(data_unit)
            for element in self.data_schema:
                if element.name == 'Image':
                    data_unit = DataUnit(type_val=element.type,is_id=schema_element.is_id, shape=element.shape, name=element.name,
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


class DataInd(object):
    def __init__(self, timestamp, cat_subcat, model_sign):
        self.timestamp = timestamp
        self.cat_subcat = cat_subcat
        self.model_sign = model_sign


class DataBundle(object):
    def __init__(self, data_ind, source, target):
        self.source = source
        self.target = target
        self.data_ind = data_ind

class ModelIOReg(object):
    def __init__(self,data_list):
        self.data_list =data_list

class DataUnit(object):

    def __init__(self, type_val, shape, data, name='', is_id=False, break_seq=False, break_size=100):
        self.is_id = is_id
        if type_val not in DATA_FORMATS:
            raise RuntimeError('Data type in data provided is not supported')
        if data is not None:
            if np.array(data).shape != shape and shape != () and \
                    np.array([data]).shape != shape:
                raise RuntimeError('Data shape is different form the schema', np.array([data]).shape, shape)

        if type(data) != type(None):

            if type_val == 'str':
                data = str(data)
            elif type_val == 'int':
                # try:
                if np.isnan(data).any():
                    data = np.nan
                else:
                    data = np.array(data, dtype=np.int)
            # except:
            #    print(type(data))
            #    data = np.array(data, dtype=np.int)

            elif type_val == 'float':
                if np.isnan(data):
                    data = np.nan
                else:
                    data = np.array(data, dtype=np.float)
        self.type = type_val
        self.shape = shape
        self.data = data
        self.name = name
        self.break_seq = break_seq
        self.break_size = break_size


def one_hot(string_list, element_key):
    unique_indentifier = list(set(string_list))
    if len(unique_indentifier) > 250 and element_key.break_seq == False:
        return []
    # exit(0)
    if element_key.break_seq:
        return_arr = []
        local_alphabet = []
        for arr in string_list:
            for element in arr:
                if element not in local_alphabet:
                    local_alphabet.append(element)
        unique_indentifier = list(set(local_alphabet))
        for arr in string_list:
            local_arr = []
            for element_pos in range(len(arr)):
                if element_pos % element_key.break_size == 0:
                    local_arr.append([])
                local_entry = [0] * len(unique_indentifier)
                local_entry[unique_indentifier.index(arr[element_pos])] = 1
                local_arr[-1].append(local_entry)
            for i in range(element_key.break_size - len(local_arr[-1])):
                local_arr[-1].append([0] * len(unique_indentifier))
            return_arr.append(local_arr)
        return return_arr

    else:
        unique_str = len(unique_indentifier)
        one_hot_list = []
        for element in string_list:
            local_entry = [0] * unique_str
            local_entry[unique_indentifier.index(element)] = 1
            one_hot_list.append(local_entry)
        # print('len(one_hot_list)',len(one_hot_list))
        return one_hot_list


def contur_image(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.blur(grey_img, (3, 3), 0)
    contur_img = cv2.Sobel(src=blur_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    return contur_img

def find_image_by_name(img_name, path):
    global local_lib
    local_path = ''
    img_name = str(img_name)
    img_name = img_name.replace('^', '')
    # print(len(local_lib.keys()))
    if len(local_lib.keys()) == 0:
        for root, dirs, files in os.walk(path):
            for element in files:
                if element[:-4] not in list(local_lib.keys()):
                    local_lib[element[:-4]] = os.path.join(root, element)
    if img_name in list(local_lib.keys()):

        if len(local_lib[img_name]) > 0:
            # print(local_lib)
            ds =  image.imread(local_lib[img_name])

            pixel_array_numpy = ds
            return pixel_array_numpy

    if len(local_path) > 0:
        # if 'dcm' not in local_path:
        #    local_path = local_path+'.dcm'
        ds = image.imread(local_path)
        pixel_array_numpy = ds
        return pixel_array_numpy

def is_int(i):
    try:
        int(i)
        return True
    except Exception as e:
        return False


def is_float(i):
    try:
        float(i)
        return True
    except Exception as e:
        return False


def is_float_arr(i):
    try:
        for element in i:
            float(element)
        return True
    except Exception as e:
        return False


def normalize_list(local_list, max_val, min_val, target_max, target_min,type_in=None):
    print('type',type)
    if type_in == 'bool':
        for i in range(len(local_list)):
            local_list[i] = int(local_list[i])
            return local_list
    if (abs(min_val) + abs(max_val)) == 0:
        return local_list
    if len(local_list) == 0:
        return local_list
    if type(local_list[0]) == type([]):
        return local_list
    for i in range(len(local_list)):

        local_list[i] = ((local_list[i] + abs(min_val)) / ((abs(min_val) + abs(max_val)))) - abs(target_min) * 1
        if math.isnan(local_list[i]):
            local_list[i] = 0
    return local_list

def flatten_list(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_list(x)
        else:
            yield x


def re_size_image(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return img


# is this thread save
def image_json_loader_worker(args):
    global loaded_ids_target
    image_paths = args
    local_image_collection = {'train_inputs': [], 'train_outputs': [], 'test_inputs': [], 'test_outputs': []}
    for i, image_col in enumerate(image_paths):
        if i % 1000 == 0:
            print(i, '/', len(image_paths))
        try:
            image_path, image_name = image_col
        except Exception as e:
            continue
        with open(image_path + image_name, mode='r') as f:
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
            image_path = image_path + '/'
        local_img = cv2.imread(image_path + image_name)

        if image_ids is None:

            local_image_collection[image_name[:-4]] = \
                {"img": local_img, "data": None}
        else:
            if image_name[:-4] + '^' not in image_ids.keys():
                continue
            local_img_id = image_name[:-4] + '^'

            if local_img_id not in loaded_ids_target:
                loaded_ids_target.append(local_img_id)

            img_shape = image_ids[local_img_id].get_shape_by_name('Image')
            if img_shape != None and len(img_shape) > 0:
                local_img = cv2.resize(local_img, (img_shape[0], img_shape[1]))
                local_data = copy.deepcopy(image_ids[local_img_id])
            local_image_collection[image_name[:-4]] = {"img": local_img,
                                                       "img_name": image_name[:-4]}
    return local_image_collection


def split_list_second(target_dict, count, restrict=False, size=1000):
    interval = int(len(target_dict[list(target_dict.keys())[0]]) / count)
    splited_dict = []
    for local_key in target_dict.keys():
        splited_dict[local_key] = []
    for local_key in splited_dict.keys():
        for i in range(count):
            splited_dict[local_key].append(target_dict[local_key][i * interval:(i + 1) * interval])
    for local_key in splited_dict.keys():
        splited_list_rest = splited_dict[local_key][(count + 1) * interval:]
        for i, element in enumerate(splited_list_rest):
            splited_dict[local_key][i].append(element)

    if restrict:
        for local_key in splited_dict.keys():
            for i in range(len(splited_dict[local_key])):
                random_start = random.randint(0, len(splited_dict[local_key][i]) - (size + 1))
                splited_dict[local_key][i] = splited_dict[local_key][i][random_start:random_start + size]

    return splited_dict


def split_list(target_list, count, restrict=False, size=1000):
    interval = int(len(target_list) / count)
    splited_list = []
    for i in range(count):
        splited_list.append(target_list[i * interval:(i + 1) * interval])

    splited_list_rest = splited_list[(count + 1) * interval:]
    for i, element in enumerate(splited_list_rest):
        splited_list[i].append(element)

    if restrict:

        for i in range(len(splited_list)):
            random_start = random.randint(0, len(splited_list[i]) - (size + 1))
            splited_list[i] = splited_list[i][random_start:random_start + size]

    return splited_list


def split_dict(target_dict, count, restrict=False, size=1000):
    interval = int(len(target_dict[list(target_dict.keys())[0]]) / count)
    splited_dict = {}
    for local_key in target_dict.keys():
        splited_dict[local_key] = []
    for local_key in splited_dict.keys():
        for i in range(count):
            splited_dict[local_key].append(target_dict[local_key][i * interval:(i + 1) * interval])
    for local_key in splited_dict.keys():
        splited_list_rest = splited_dict[local_key][(count + 1) * interval:]
        for i, element in enumerate(splited_list_rest):
            splited_dict[local_key][i].append(element)

    if restrict:
        for local_key in splited_dict.keys():
            for i in range(len(splited_dict[local_key])):
                random_start = random.randint(0, len(splited_dict[local_key][i]) - (size + 1))
                splited_dict[local_key][i] = splited_dict[local_key][i][random_start:random_start + size]

    return splited_dict


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



def image_loader(path, train_name='train', restrict=False, size=1000, no_ids=False,
                 load_image=False, target_name=None, data_schema_input=None,data_schema_output=None, split=False, split_coef=0.9, THREAD_COUNT_V=32,
                 dir_tree=False):
    global THREAD_COUNT
    global train_df, valid_df
    THREAD_COUNT = THREAD_COUNT_V
    image_paths = image_list(path + 'train_images/')
    image_ids = None
    if Path(path + train_name + '.csv').exists():
        if not dir_tree:
            image_ids, loaded_ids_size = load_id_from_csv(path + train_name + '.csv',  data_schema_input,data_schema_output, restrict, size)
        else:
            image_ids, loaded_ids_size = load_id_from_dir_tree_csv(path, data_schema_input,data_schema_output, restrict, size)
    elif dir_tree:
        image_ids, loaded_ids_size = load_id_from_dir_tree_csv(path, data_schema_input,data_schema_output, restrict, size)
    print('csv load done')
    local_image_collection = {'num_classes': 0, 'image_arr': []}
    if size < len(image_paths):
        image_paths_list = split_list(image_paths, THREAD_COUNT, restrict, size)
    else:
        image_paths_list = []
        for element in image_paths:
            image_paths_list.append([element])

    if split:
        test_arr = []
        train_arr = []
        train_cutoff = int(split_coef * len(local_image_collection['image_arr']))
        # FIX THIS
        if not load_image:
            key_list = list(image_ids.keys())
            random.shuffle(key_list)
            train_cutoff = int(split_coef * len(key_list))
            for i in range(0, train_cutoff):
                train_arr.append(image_ids[key_list[i]])
            for i in range(train_cutoff, len(key_list)):
                test_arr.append(image_ids[key_list[i]])
            random.shuffle(train_arr)
            random.shuffle(test_arr)
            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}, \
                   {'num_classes': local_image_collection['num_classes'], 'image_arr': test_arr}
        else:
            for i in range(0, train_cutoff):
                train_arr.append(local_image_collection['image_arr'][i])
            for i in range(train_cutoff, len(local_image_collection['image_arr'])):
                test_arr.append(local_image_collection['image_arr'][i])
            random.shuffle(train_arr)
            random.shuffle(test_arr)
            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}, \
                   {'num_classes': local_image_collection['num_classes'], 'image_arr': test_arr}
    else:
        train_arr = []
        # FIX THIS

        train_cutoff = int(len(local_image_collection['image_arr']))
        if not load_image:
            key_list = list(image_ids)
            random.shuffle(key_list)
            train_cutoff = int(len(key_list))
            for i in range(0, train_cutoff):
                train_arr.append(image_ids[i])
            random.shuffle(train_arr)
            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}, {
                'num_classes': local_image_collection['num_classes'], 'image_arr': []}
        else:
            for i in range(0, train_cutoff):
                train_arr.append(local_image_collection['image_arr'][i])

            random.shuffle(train_arr)
            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}, {
                'num_classes': local_image_collection['num_classes'], 'image_arr': []}

def worker_load_image_data_from_dir_tree_csv(args):
    # global loaded_ids
    local_list, schema, cut_size, restrict, dir_path = args

    # print(type(schema))
    # print(type(cut_size))
    local_data_arr = {}
    # local_id_poss = []
    # for i,element in enumerate(schema):
    #    if element.is_id:
    #        local_id_poss.append(i)
    if restrict:
        if cut_size != -1:
            local_list = local_list[:cut_size]
    schema_transformed = {}
    for element in local_list:
        if os.path.exists(dir_path + element):

            for key in schema.keys():
                print(os.path.exists(dir_path + element))
                if os.path.isfile(dir_path + element):
                    if (key) not in schema_transformed.keys():
                        schema_transformed[key] = {}

                    for local_data_unit in schema[key]:
                        if local_data_unit.name not in schema_transformed[key].keys():
                            schema_transformed[key][local_data_unit.name] = []
                else:
                    for second_path in os.listdir(dir_path + element):
                        if (element + '/' + second_path) not in schema_transformed.keys():
                            schema_transformed[element + '/' + second_path] = {}
                        for key in schema.keys():
                            if key not in schema_transformed[element + '/' + second_path].keys():
                                schema_transformed[element + '/' + second_path][key] = {}
                            for local_data_unit in schema[key]:
                                if local_data_unit.name not in schema_transformed[element + '/' + second_path][
                                    key].keys():
                                    schema_transformed[element + '/' + second_path][key][local_data_unit.name] = []
    local_norm_list = []
    for key in schema.keys():

        if os.path.isfile(dir_path + key + '.csv'):

            df = pd.read_csv(dir_path + key + '.csv')
            local_dict = df.to_dict()
            for second_key in schema[key]:
                for element_second in local_dict[second_key.name].values():
                    schema_transformed[key][second_key.name].append(element_second)
        else:
            for element in local_list:
                for second_path in os.listdir(dir_path + element):

                    df = pd.read_csv(dir_path + element + '/' + second_path + '/' + key + '.csv')
                    local_dict = df.to_dict()
                    for second_key in schema[key]:
                        for element_second in local_dict[second_key.name].values():
                            schema_transformed[element + '/' + second_path][key][second_key.name].append(element_second)

    return schema_transformed
    '''
    for i in range(len(local_dict[list(local_dict.keys())[0]])):
        local_row = []
        for element in schema:
            if element.name in list(local_dict.keys()):
             local_row.append(local_dict[element.name][i])
            else:
                local_row.append(None)
        local_norm_list.append(local_row)
    for row in local_norm_list:
        if len(row) == 0:
            continue
        try:
            local_id_name = ''
            for element in local_id_poss:
                local_id_name += str(row[element]) + '^'
            local_data_arr[local_id_name] = DataCollection(data_size=len(row), data_schema=schema, data=row)
        except Exception as e:
            print(e)
            pass
    return local_data_arr
    '''


def worker_load_image_data_from_csv(args):
    # global loaded_ids
    local_dict, schema, cut_size, restrict = args
    data_schema_input, data_schema_output = schema
    local_data_arr_input = {}
    local_norm_list_input = []
    local_data_arr_output = {}
    local_norm_list_output = []
    local_id_poss = []
    for i, element in enumerate(data_schema_input):
        if element.is_id:
            local_id_poss.append(i)
    if restrict:
        for local_key in local_dict.keys():
            if cut_size != -1:
                local_dict[local_key] = local_dict[local_key][:cut_size]

    for i in range(len(local_dict[list(local_dict.keys())[0]])):
        local_row = []
        for element in data_schema_input:
            if element.name in list(local_dict.keys()):
                local_row.append(local_dict[element.name][i])
            elif element.type in DATA_FORMATS_SPECIAL:
                local_row.append(np.resize(find_image_by_name(local_row[local_id_poss[0]],
                                 './data_sets/abstraction-and-reasoning-challenge/image_train_small/'),element.shape))
            else:
                local_row.append(None)
        local_norm_list_input.append(local_row)
    local_count = 1
    for row in local_norm_list_input:
        if local_count % 10 == 0:
            print(local_count, '/', len(local_norm_list_input))
        local_count += 1
        if len(row) == 0:
            continue
        try:
            local_id_name = ''
            # print('local_id_poss',local_id_poss)
            for element in local_id_poss:
                local_id_name += str(row[element]) + '^'
            local_data_arr_input[local_id_name] = DataCollection(data_size=len(row), data_schema=data_schema_input, data=row)

        except Exception as e:
           pass
#////////////////////////////////
    local_id_poss = []
    for i, element in enumerate(data_schema_output):
        if element.is_id:
            local_id_poss.append(i)
    for i in range(len(local_dict[list(local_dict.keys())[0]])):
        local_row = []
        for element in data_schema_output:
            if element.name in list(local_dict.keys()):
                local_row.append(local_dict[element.name][i])
            else:
                local_row.append(None)
        local_norm_list_output.append(local_row)
    local_count = 1
    for row in local_norm_list_output:
        if local_count % 10 == 0:
            print(local_count, '/', len(local_norm_list_output))
        local_count += 1
        if len(row) == 0:
            continue
        try:
            local_id_name = ''
            # print('local_id_poss',local_id_poss)
            for element in local_id_poss:
                local_id_name += str(row[element]) + '^'
            local_data_arr_output[local_id_name] = DataCollection(data_size=len(row), data_schema=data_schema_output, data=row)

        except Exception as e:
            pass
    data_bundle_list  = []
    for input,target in zip(local_data_arr_input.values(),local_data_arr_output.values()):
        data_bundle_list.append(DataBundle(data_ind=DataInd(timestamp='00000', cat_subcat='00000', model_sign='00000'),source=input,target=target))
    return data_bundle_list


def load_id_from_dir_tree_csv(dir_path, data_schema, restrict=False, size=100):
    local_ids = {}
    data = os.listdir(dir_path)

    if len(data) >= THREAD_COUNT:
        id_list = []
        id_list_ordered = split_list(target_list=data, count=THREAD_COUNT, restrict=restrict, size=size)
        data_schema = [data_schema] * THREAD_COUNT

        for i in range(THREAD_COUNT):
            id_list.append(id_list_ordered[i])

        if restrict:
            size = [size] * THREAD_COUNT
            dir_path = [dir_path] * THREAD_COUNT
        else:
            size = [len(id_list[0])] * THREAD_COUNT
            dir_path = [dir_path] * THREAD_COUNT
        restrict = [restrict] * THREAD_COUNT
        print('id_list', type(id_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(worker_load_image_data_from_dir_tree_csv, args) for args in
                       zip(id_list, data_schema, size,
                           restrict, dir_path)]
            print("THREAD COUNT:", len(futures))
            results = [f.result() for f in futures]
        for element in results:
            local_ids = {**local_ids, **element}
    else:

        local_ids = worker_load_image_data_from_dir_tree_csv((data, data_schema, size, restrict, dir_path))
    loaded_ids_size = len(local_ids.keys())
    return local_ids, loaded_ids_size


def load_id_from_csv(csv_path,  data_schema_input=None,data_schema_output=None, restrict=False, size=100):
    local_ids = {}
    data = pd.read_csv(csv_path, low_memory=False)
    csv_reader = data.to_dict(orient='list')
    print('len(data) >= THREAD_COUNT', len(data) >= THREAD_COUNT)
    if len(csv_reader) > THREAD_COUNT:

        id_dict = split_dict(target_dict=csv_reader, count=THREAD_COUNT, restrict=restrict, size=size)
        data_schema = [(data_schema_input,data_schema_output)] * THREAD_COUNT
        id_list = []
        for i in range(THREAD_COUNT):
            local_dict = {}
            local_len = len(id_dict.keys()) / THREAD_COUNT
            for local_key in list(id_dict.keys()[local_len * i:local_len * (i + 1)]):
                local_dict[local_key] = id_dict[local_key][i]
            id_list.append(local_dict)
        if restrict:
            size = [size] * THREAD_COUNT
        else:
            size = [len(id_list[0])] * THREAD_COUNT
        print('id_list', type(id_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(worker_load_image_data_from_csv, args) for args in
                       zip(id_list, data_schema[csv_path[csv_path.rindex('/') + 1:csv_path.rindex('.')]], size,
                           restrict)]
            print("THREAD COUNT:", len(futures))
            results = [f.result() for f in futures]

    else:

        results = worker_load_image_data_from_csv(
            (csv_reader, (data_schema_input,data_schema_output), size, restrict))
    loaded_ids_size = len(local_ids)
    return results, loaded_ids_size


def image_list(path):
    image_path_list = []
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file[-3:] in IMAGE_EXTS_LIST:
                image_path_list.append((root, file))
    return image_path_list[1:]
