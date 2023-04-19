import concurrent.futures
import copy
import json
import os
import random
from pathlib import Path
from utils.utils import DataCollection, DataUnit,DataBundle,DataInd


import cv2
import numpy as np
import pandas as pd
import pyarrow

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
THREAD_COUNT = 32
loaded_ids = []
loaded_ids_target = []
index_csv = {}

IMAGE_EXTS_LIST = ['jpg', 'png', 'tiff']
DATA_FORMATS_SPECIAL = ['2D_F', '2D_INT']
DATA_FORMATS = ['float', 'int', 'str'] + DATA_FORMATS_SPECIAL
REGRESSION_CATEGORY = 'int'
REGRESSION = 'float'
CATEGORY = 'Category'
STRING = 'str'
TIME_SERIES = 'TimeSeries'
IMAGE = '2D_F'
TARGET_TYPES = [REGRESSION, REGRESSION_CATEGORY, CATEGORY, TIME_SERIES, IMAGE]
HEURISTICS = {}

train_df = None

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
            if image_name[:image_name.index('.')] + '^' not in image_ids.keys():
                continue
            local_img_id = image_name[:image_name.index('.')] + '^'

            if local_img_id not in loaded_ids_target:
                loaded_ids_target.append(local_img_id)

            img_shape = image_ids[local_img_id].get_shape_by_name('Image')
            if img_shape != None and len(img_shape) > 0:
                local_img = cv2.resize(local_img, (img_shape[0], img_shape[1]))
                local_data = copy.deepcopy(image_ids[local_img_id])
            local_image_collection[image_name[:image_name.index('.')]] = {"img": local_img,
                                                                          "img_name": image_name[
                                                                                      :image_name.index('.')]}

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


local_lib = {}





def image_loader(path, train_name='train', restrict=False, size=1000, no_ids=False,
                 load_image=False, target_name=None, data_schema_input=None,data_schema_output=None, split=False, split_coef=0.9, THREAD_COUNT_V=32,
                 dir_tree=False):
    global THREAD_COUNT
    global train_df, valid_df
    THREAD_COUNT = THREAD_COUNT_V
    image_paths = image_list(path + 'train_images/')
    image_ids = None
    if Path(path).exists():
        data = pd.read_parquet(path +'train_meta.parquet')
        print('loaded train_meta.parquet')
        data = data.iloc[0:size].to_dict(orient='list')
        #exit(0)
        if restrict:
            for element in list(data.keys()):
                data[element] = data[element][:size]
        if not dir_tree:
            image_ids, loaded_ids_size = load_id_from_parquet(path+train_name,  data_schema_input,data_schema_output, restrict, size,data)
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
            image_ids = np.array(image_ids)
            image_ids = image_ids.flatten()
            image_ids = image_ids.tolist()
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


def file_dict_by_id_value(dict_to_filter, filter_keys, value):
    return_dict = {}
    for local_key in list(dict_to_filter.keys()):
        return_dict[local_key] = []
    filter_indexies = []
    clean_dict = {}
    for local_key in list(dict_to_filter.keys()):
        clean_dict[local_key] = list(dict_to_filter[local_key].values())

    for i, element in enumerate(clean_dict[value]):
        if str(element) in filter_keys:
            filter_indexies.append(i)
    for local_key in list(dict_to_filter.keys()):
        for index_key in filter_indexies:
            return_dict[local_key].append(dict_to_filter[local_key][index_key])
    return return_dict




def worker_load_image_data_from_csv(args):
    # global loaded_ids
    parquet_dir, schema, cut_size, restrict,meta_data = args
    data_schema_input, data_schema_output = schema
    local_data_arr_input = {}
    local_norm_list_input = []
    local_data_arr_output = {}
    local_norm_list_output = []
    local_id_poss = []
    for i, element in enumerate(data_schema_input):
        if element.is_id:
            local_id_poss.append(i)
    print(parquet_dir)
    arrow_dataset = pyarrow.parquet.ParquetDataset(parquet_dir)
    arrow_table = arrow_dataset.read()
    pandas_df = arrow_table.to_pandas()
    data =pandas_df#pd.read_parquet(parquet_dir)
    data = data.iloc[0:cut_size].to_dict(orient='list')
    if restrict:
        for element in list(data.keys()):
            data[element] = data[element][:cut_size]

    for i in range(len(data[list(data.keys())[0]])):
        local_row = []
        for element in data_schema_input:
            if element.name in list(data.keys()):
                local_row.append(data[element.name][i])
            elif element.name in list(meta_data.keys()):
                local_row.append(meta_data[element.name][i])
            else:
                local_row.append(None)
        local_norm_list_input.append(local_row)
    data = meta_data
    for i in range(len(data[list(data.keys())[0]])):
        local_row = []
        for element in data_schema_output:
            if element.name in list(data.keys()):
                local_row.append(data[element.name][i])
            elif element.name in list(meta_data.keys()):
                local_row.append(meta_data[element.name][i])
                local_row.append(None)
        local_norm_list_output.append(local_row)

    local_count = 1
    for row in local_norm_list_input:
        if local_count % 10 == 0:
            print(local_count, '/', len(local_norm_list_input))
        local_count += 1
        if len(row) == 0:
            continue
        try:
            local_id_name = ''
            for element in local_id_poss:
                local_id_name += str(row[element]) + '^'
            local_data_arr_input[local_id_name] = DataCollection(data_size=len(row), data_schema=data_schema_input, data=row)

        except Exception as e:
           pass
    local_id_poss = []
    for i, element in enumerate(data_schema_output):
        if element.is_id:
            local_id_poss.append(i)

    local_count = 1
    for row in local_norm_list_output:
        if local_count % 10 == 0:
            print(local_count, '/', len(local_norm_list_output))
        local_count += 1
        if len(row) == 0:
            continue
        try:
            local_id_name = ''
            for element in local_id_poss:
                local_id_name += str(row[element]) + '^'
            local_data_arr_output[local_id_name] = DataCollection(data_size=len(row), data_schema=data_schema_output, data=row)

        except Exception as e:
            pass
    data_bundle_list  = []
    for input,target in zip(local_data_arr_input.values(),local_data_arr_output.values()):
        data_bundle_list.append(DataBundle(data_ind=DataInd(timestamp='00000', cat_subcat='00000', model_sign='00000'),source=input,target=target))
    return data_bundle_list


def load_id_from_dir_tree_csv(dir_path,  data_schema_input=None,data_schema_output=None, restrict=False, size=100):
    local_ids = {}
    data = os.listdir(dir_path + 'test_images' )# list(data_schema.keys())[0])
    print('len(data) >= THREAD_COUNT', len(data) >= THREAD_COUNT)
    if len(data) >= THREAD_COUNT:
        id_list = []
        id_list_ordered = split_list(target_list=data, count=THREAD_COUNT, restrict=restrict, size=size)
        data_schema = [(data_schema_input,data_schema_output)] * THREAD_COUNT

        for i in range(THREAD_COUNT):
            id_list.append(id_list_ordered[i])

        if restrict:
            size = [size] * THREAD_COUNT
            dir_path = [dir_path] * THREAD_COUNT
        else:
            size = [len(id_list[0])] * THREAD_COUNT
            dir_path = [dir_path] * THREAD_COUNT
        restrict = [restrict] * THREAD_COUNT

    else:

        local_ids = worker_load_image_data_from_dir_tree_csv((0, data, (data_schema_input,data_schema_output), size, restrict, dir_path))
    loaded_ids_size = len(local_ids.keys())
    return local_ids, loaded_ids_size


def load_id_from_parquet(parquet_dir,  data_schema_input=None,data_schema_output=None, restrict=False, size=100,meta_data=None):
    local_ids = {}
    file_paths = []
    for element in os.listdir(parquet_dir):
        file_paths.append(parquet_dir +'/'+element)

    file_paths = file_paths[:10]
    print('len(data) >= THREAD_COUNT', len(file_paths) >= THREAD_COUNT, len(file_paths) )
    if len(file_paths) > THREAD_COUNT:

        data_schema = [(data_schema_input,data_schema_output)] * THREAD_COUNT
        id_list = []
        for i in range(THREAD_COUNT):
            local_list = []
            local_len = int(len(file_paths) / THREAD_COUNT)
            for local_key in list(file_paths[local_len * i:local_len * (i + 1)]):
                local_list.append( local_key)
            id_list.append(local_list)
        if restrict:
            size = [size] * THREAD_COUNT
        else:
            size = [len(id_list[0])] * THREAD_COUNT
        restrict = [restrict]*THREAD_COUNT
        meta_data = [meta_data]*THREAD_COUNT
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(worker_load_image_data_from_csv, args) for args in
                       zip(id_list, data_schema, size,
                           restrict,meta_data)]
            print("THREAD COUNT:", len(futures))
            results = [f.result() for f in futures]

    else:
        id_list = []
        for local_key in list(file_paths):
            id_list.append(local_key)

        results = worker_load_image_data_from_csv((id_list, (data_schema_input,data_schema_output), size,
                           restrict,meta_data))
    loaded_ids_size = len(local_ids)
    return results, loaded_ids_size


def image_list(path):
    image_path_list = []
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file[file.index('.') + 1:] in IMAGE_EXTS_LIST:
                image_path_list.append((root, file))
    return image_path_list[1:]
