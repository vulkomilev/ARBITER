import copy
import os
import cv2
import concurrent.futures
import pandas as pd
import random
from pathlib import Path
import numpy as np

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
THREAD_COUNT = 32
loaded_ids = []
loaded_ids_target = []
index_csv = {}

IMAGE_EXTS_LIST = ['jpg', 'png']
DATA_FORMATS = ['float', 'int', 'str', '2D_F', '2D_INT']
REGRESSION = 'Regression'
CATEGORY = 'Category'
TIME_SERIES = 'TimeSeries'
TARGET_TYPES = [REGRESSION, CATEGORY, TIME_SERIES]


class DataCollection(object):

    def __init__(self, data_size, data_schema, data):
        self.data_size = data_size
        self.data_collection = []
        self.add_data(data)
        for element in data_schema:
            if element.type not in DATA_FORMATS:
                raise RuntimeError('Data type in schema is not supported')
        self.data_schema = data_schema

    def get_by_name(self, name):
        for element in self.data_collection:
            if element.name == name:
                return element.data

    def add_data(self, data):
        if len(data) == 0:
            return

        for data_element in data:
            assert (len(data_element) == self.data_size)
            data_element_collection = []
            for schema_element, element in zip(self.data_schema, data_element):
                data_unit = DataUnit(type=schema_element.type, shape=schema_element.shape, name=schema_element.name,
                                     data=element)
                data_element_collection.append(data_unit)

            self.data_collection = data_element_collection

    def remove(self, name):

        for element in self.data_collection:
            if element.name == name:
                self.remove(element)


class DataUnit(object):

    def __init__(self, type, shape, data, name=''):
        if type not in DATA_FORMATS:
            raise RuntimeError('Data type in data provided is not supported')
        if data is not None:
            if np.array(data).shape != shape:
                raise RuntimeError('Data shape is different form the schema')
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
def image_loader_worker(args):
    global loaded_ids_target
    image_paths, image_ids, no_ids, target_name = args
    local_image_collection = {'image_arr': []}

    for i, image_col in enumerate(image_paths):
        if i % 1000 == 0:
            print(i, '/', len(image_paths))
        try:
            image_path, image_name = image_col
        except Exception as e:
            continue
        local_img = cv2.imread(image_path + image_name)
        local_img = cv2.resize(local_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        if image_ids is None:

            local_image_collection['image_arr'].append(
                {"img": local_img, "data": None, "img_name": image_name[:-4]})
        else:
            if image_name[:-4] not in image_ids.keys():
                continue
            local_img_id = image_name[:-4]

            if local_img_id not in loaded_ids_target:
                loaded_ids_target.append(local_img_id)
            local_data = copy.deepcopy(image_ids[local_img_id])
            local_image_collection['image_arr'].append(
                {"img": local_img, "target": image_ids[local_img_id].get_by_name(target_name), "data": local_data,
                 "img_name": image_name[:-4]})

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


def image_loader(path, restrict=False, size=1000, no_ids=False, target_name=None, data_schema=None):
    image_paths = image_list(path)
    image_ids = None
    if Path(path + path[2:-1] + '.csv').exists():
        image_ids, loaded_ids_size = load_id_from_csv(path + path[2:-1] + '.csv', data_schema)
    local_image_collection = {'num_classes': 0, 'image_arr': []}
    if size < len(image_paths):
        image_paths_list = split_list(image_paths, THREAD_COUNT, restrict, size)
    else:
        image_paths_list = image_paths
    image_ids = [image_ids] * THREAD_COUNT
    no_ids = [no_ids] * THREAD_COUNT
    target_name = [target_name] * THREAD_COUNT
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:

        futures = [executor.submit(image_loader_worker, args) for args in
                   zip(image_paths_list, image_ids, no_ids, target_name)]
        print("THREAD COUNT:", len(futures))
        results = [f.result() for f in futures]

    for result in results:
        local_image_collection['image_arr'] += result['image_arr']
    local_image_collection['num_classes'] = len(loaded_ids_target)
    return local_image_collection


def worker_load_image_data_from_csv(args):
    # global loaded_ids
    list, schema = args
    local_data_arr = {}
    for row in list:
        if len(row) == 0:
            continue
        try:
            local_data_arr[row[0]] = DataCollection(data_size=len(row), data_schema=schema, data=[])
        except Exception as e:
            pass
        local_data_arr[row[0]].add_data([row])
    return local_data_arr


def load_id_from_csv(csv_path, data_schema):
    local_ids = {}
    data = pd.read_csv(csv_path, low_memory=False)
    csv_reader = data.values.tolist()
    id_list = split_list(target_list=csv_reader, count=THREAD_COUNT, restrict=False)
    data_schema = [data_schema] * THREAD_COUNT
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = [executor.submit(worker_load_image_data_from_csv, args) for args in zip(id_list, data_schema)]
        print("THREAD COUNT:", len(futures))
        results = [f.result() for f in futures]
    for element in results:
        local_ids = {**local_ids, **element}
    loaded_ids_size = len(local_ids.keys())
    return local_ids, loaded_ids_size


def image_list(path):
    image_path_list = []
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file[-3:] in IMAGE_EXTS_LIST:
                image_path_list.append((root, file))
    return image_path_list[1:]
