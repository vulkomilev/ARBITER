import pydicom
import concurrent.futures
import copy
import json
import os
import random
import math
from pathlib import Path
from collections.abc import Iterable
from utils.utils import DataCollection,DataUnit

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
THREAD_COUNT = 32
loaded_ids = []
loaded_ids_target = []
index_csv = {}

IMAGE_EXTS_LIST = ['jpg', 'png','tiff']
DATA_FORMATS_SPECIAL = [ '2D_F', '2D_INT']
DATA_FORMATS = ['float', 'int', 'str'] + DATA_FORMATS_SPECIAL
REGRESSION_CATEGORY = 'int'
REGRESSION = 'float'
CATEGORY = 'Category'
STRING = 'str'
TIME_SERIES = 'TimeSeries'
IMAGE = '2D_F'
TARGET_TYPES = [REGRESSION,REGRESSION_CATEGORY, CATEGORY, TIME_SERIES,IMAGE]
HEURISTICS = {}



cfg = {
    'train_df': '/tmp/data_sets/hubmap-organ-segmentation/train.csv',
    'train_img': '/tmp/data_sets/hubmap-organ-segmentation/train_images/',
    'train_annot': '/tmp/data_sets/hubmap-organ-segmentation/train_annotations/',
    'slices': 4, #Batch size = slices ** 2
    'epochs': 3,
    'img_shape': 2304,
    'lr': 1e-03
}

def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T
train_df = None
def display_img(idx: int):
    '''
    Displays image along with its mask
    '''

    row = train_df.loc[train_df['id'] == int(idx)]
    h, w = row['img_height'].iloc[-1], row['img_width'].iloc[-1]
    img = Image.open(row['paths'].iloc[-1])
    img = np.array(img)
    sample = row['rle'].iloc[0]
    mask = rle2mask(sample, shape=(int(h), int(w)))
    mask = np.expand_dims(mask,axis=2)
    mask = np.pad(mask, pad_width=((0,0),(0,0),(0,2)), mode='constant', constant_values=0)

    return img, mask

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
        ds = pydicom.read_file(
            image_path + image_name)

        local_img =ds.pixel_array#= cv2.imread(image_path + image_name)



        if image_ids is None:

            local_image_collection[image_name[:-4]] = \
            {"img": local_img, "data": None}
        else:
            if image_name[:image_name.index('.')]+'^' not in image_ids.keys():
                continue
            local_img_id = image_name[:image_name.index('.')]+'^'

            if local_img_id not in loaded_ids_target:
                loaded_ids_target.append(local_img_id)

            img_shape = image_ids[local_img_id].get_shape_by_name('Image')
            if  img_shape != None and len(img_shape) > 0:
                local_img = cv2.resize(local_img, (img_shape[0], img_shape[1]))
                local_data = copy.deepcopy(image_ids[local_img_id])
            local_image_collection[image_name[:image_name.index('.')]] ={"img": local_img,
                     "img_name": image_name[:image_name.index('.')]}

    return local_image_collection
def split_list_second(target_dict, count, restrict=False, size=1000):
    interval = int(len(target_dict[list(target_dict.keys())[0]]) / count)
    splited_dict = []
    for local_key  in target_dict.keys():
        splited_dict[local_key] = []
    for local_key in splited_dict.keys():
       for i in range(count):
        splited_dict[local_key].append(target_dict[local_key][i * interval:(i + 1) * interval])
    for local_key in splited_dict.keys():
        splited_list_rest = splited_dict[local_key][(count + 1) * interval:]
        for i,element in enumerate(splited_list_rest):
            splited_dict[local_key][i].append(element)

    if restrict:
        for local_key in splited_dict.keys():
            for i in range(len( splited_dict[local_key])):
                random_start = random.randint(0, len( splited_dict[local_key][i]) - (size + 1))
                splited_dict[local_key][i] =  splited_dict[local_key][i][random_start:random_start + size]

    return splited_dict

def split_list(target_list, count, restrict=False, size=1000):
    interval = int(len(target_list) / count)
    splited_list = []
    for i in range(count):
        splited_list.append(target_list[i * interval:(i + 1) * interval])

    splited_list_rest = splited_list[(count + 1) * interval:]
    for i,element in enumerate(splited_list_rest):
            splited_list[i].append(element)

    if restrict:

            for i in range(len(splited_list)):
                random_start = random.randint(0, len( splited_list[i]) - (size + 1))
                splited_list[i] =  splited_list[i][random_start:random_start + size]

    return splited_list

def split_dict(target_dict, count, restrict=False, size=1000):
    interval = int(len(target_dict[list(target_dict.keys())[0]]) / count)
    splited_dict = {}
    for local_key  in target_dict.keys():
        splited_dict[local_key] = []
    for local_key in splited_dict.keys():
       for i in range(count):
        splited_dict[local_key].append(target_dict[local_key][i * interval:(i + 1) * interval])
    for local_key in splited_dict.keys():
        splited_list_rest = splited_dict[local_key][(count + 1) * interval:]
        for i,element in enumerate(splited_list_rest):
            splited_dict[local_key][i].append(element)

    if restrict:
        for local_key in splited_dict.keys():
            for i in range(len( splited_dict[local_key])):
                random_start = random.randint(0, len( splited_dict[local_key][i]) - (size + 1))
                splited_dict[local_key][i] =  splited_dict[local_key][i][random_start:random_start + size]

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

def image_loader(path,train_name = 'train', restrict=False, size=1000, no_ids=False,
                 load_image=False,target_name=None, data_schema=None,split=False,split_coef=0.9,THREAD_COUNT_V = 32,
                 dir_tree = False):
    global THREAD_COUNT
    global train_df, valid_df
    THREAD_COUNT = THREAD_COUNT_V
    image_paths = image_list(path+'train_images/')
    image_ids = None
    if Path(path + train_name + '.csv').exists():
        if not dir_tree:
            image_ids, loaded_ids_size = load_id_from_csv(path + train_name + '.csv', data_schema,restrict, size)
        else:
            image_ids, loaded_ids_size = load_id_from_dir_tree_csv(path +'train_images', data_schema, restrict, size)
    elif   dir_tree:
            image_ids, loaded_ids_size = load_id_from_dir_tree_csv(path , data_schema, restrict, size)
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

    df = pd.read_csv(cfg['train_df'])
    df['paths'] = [f'/tmp/data_sets/hubmap-organ-segmentation/train_images/{str(i)}.tiff' for i in tqdm(df['id'])]
    train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=20)
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    train_df.head()

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:

        futures = [executor.submit(image_loader_worker, args) for args in
                   zip(image_paths_list, image_ids_splited, no_ids, target_name)]
        print("THREAD COUNT:1", len(futures))
        local_dict = [f.result() for f in futures]
        for element in local_dict:
            results.update(element)
    for key,val in enumerate(results):
        try:

            img, mask = display_img(val)
            image_ids[val+'^'].set_by_name('Image',cv2.resize(img, dsize=(900, 900), interpolation=cv2.INTER_CUBIC))
            image_ids[val + '^'].set_by_name('ImageMask', cv2.resize(mask, dsize=(900, 900), interpolation=cv2.INTER_CUBIC))
        except Exception as e:
            pass
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
    else:
        train_arr = []
        # FIX THIS

        train_cutoff = int(len(local_image_collection['image_arr']))
        if not load_image:
            key_list = list(image_ids.keys())
            random.shuffle(key_list)
            train_cutoff = int( len(key_list))
            for i in range(0, train_cutoff):
                train_arr.append(image_ids[key_list[i]])
            random.shuffle(train_arr)
            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}
        else:
            for i in range(0, train_cutoff):
                train_arr.append(local_image_collection['image_arr'][i])

            random.shuffle(train_arr)
            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}

def worker_load_image_data_from_dir_tree_csv(args):
    # global loaded_ids
    local_list, schema,cut_size,restrict,dir_path = args

    #print(type(schema))
    #print(type(cut_size))
    local_data_arr = {}
    #local_id_poss = []
    #for i,element in enumerate(schema):
    #    if element.is_id:
    #        local_id_poss.append(i)
    if restrict:
        if cut_size != -1:
            local_list = local_list[:cut_size]
    schema_transformed = {}
    for element in local_list:

        for key in schema.keys():
            if 'csv' not in element:
             for second_path in os.listdir(dir_path+'/'+element):
                if ( element+'/'+second_path) not in schema_transformed.keys():
                    schema_transformed[ element+'/'+second_path] = {}
                for key in schema.keys():
                    if key not in schema_transformed[ element+'/'+second_path].keys():
                        schema_transformed[ element+'/'+second_path][key] = {}
                    for local_data_unit in schema[key]:
                        if local_data_unit.name not in schema_transformed[ element+'/'+second_path][key].keys():
                            schema_transformed[ element+'/'+second_path][key][local_data_unit.name] = []
    local_norm_list = []
    for element in local_list:

        for key in schema.keys():
            for second_path in os.listdir(dir_path+'/'+element):

                df = pd.read_csv(dir_path+'/' +'train.csv')
                local_dict = df.to_dict()
                for second_key in schema[key]:
                    for element_second in local_dict[second_key.name].values():
                        schema_transformed[element+'/'+second_path][key][second_key.name].append(element_second)

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
    local_dict, schema,cut_size,restrict = args
    #print(type(schema))
    #print(type(cut_size))
    local_data_arr = {}
    local_id_poss = []
    for i,element in enumerate(schema):
        if element.is_id:
            local_id_poss.append(i)
    if restrict:
      for local_key in local_dict.keys():
        if cut_size != -1:
            local_dict[local_key] = local_dict[local_key][:cut_size]
    local_norm_list = []
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

def load_id_from_dir_tree_csv(dir_path, data_schema, restrict=False,size=100):
    local_ids = {}
    data = os.listdir(dir_path)

    if len(data) >= THREAD_COUNT:
        id_list = []
        id_list_ordered = split_list(target_list=data, count=THREAD_COUNT, restrict=restrict,size=size)
        data_schema = [data_schema] * THREAD_COUNT

        for i in range(THREAD_COUNT):
           id_list.append(id_list_ordered[i])

        if restrict:
            size = [size] * THREAD_COUNT
            dir_path = [dir_path] * THREAD_COUNT
        else:
            size = [len(id_list[0])]* THREAD_COUNT
            dir_path = [dir_path] * THREAD_COUNT
        restrict  =  [restrict] * THREAD_COUNT
        print('id_list', type(id_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(worker_load_image_data_from_dir_tree_csv, args) for args in zip(id_list, data_schema,size,
                                                                                              restrict,dir_path)]
            print("THREAD COUNT:", len(futures))
            results = [f.result() for f in futures]
        for element in results:
            local_ids = {**local_ids, **element}
    else:

        local_ids = worker_load_image_data_from_dir_tree_csv((data, data_schema,size,restrict,dir_path))
    loaded_ids_size = len(local_ids.keys())
    return local_ids, loaded_ids_size

def load_id_from_csv(csv_path, data_schema, restrict=False,size=100):
    local_ids = {}
    data = pd.read_csv(csv_path, low_memory=False)
    csv_reader = data.to_dict(orient='list')

    if len(csv_reader) > THREAD_COUNT:

        id_dict = split_dict(target_dict=csv_reader, count=THREAD_COUNT, restrict=restrict,size=size)
        data_schema = [data_schema] * THREAD_COUNT
        id_list = []
        for i in range(THREAD_COUNT):
           local_dict = {}
           for local_key in list(id_dict.keys()):
            local_dict[local_key] = id_dict[local_key][i]
           id_list.append(local_dict)
        if restrict:
            size = [size] * THREAD_COUNT
        else:
            size = [len(id_list[0])]* THREAD_COUNT
        print('id_list', type(id_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(worker_load_image_data_from_csv, args) for args in zip(id_list, data_schema,size,
                                                                                              restrict)]
            print("THREAD COUNT:", len(futures))
            results = [f.result() for f in futures]
        for element in results:
            local_ids = {**local_ids, **element}
    else:

        local_ids = worker_load_image_data_from_csv((csv_reader, data_schema,size,restrict))
    loaded_ids_size = len(local_ids.keys())
    return local_ids, loaded_ids_size


def image_list(path):
    image_path_list = []
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file[file.index('.')+1:] in IMAGE_EXTS_LIST:
                image_path_list.append((root, file))
    return image_path_list[1:]
