import concurrent.futures
import copy
import json
import math
import os
import random
from collections.abc import Iterable
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import image
from ..utils import DataUnit,DataBundle,DataCollection,DATA_FORMATS_SPECIAL,find_image_by_name,DataInd,IMAGE_EXTS_LIST


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






def image_loader(path, train_name='train', restrict=False, size=1000, no_ids=False,
                 load_image=False, target_name=None, data_schema_input=None, data_schema_output=None, split=False,
                 split_coef=0.9, THREAD_COUNT_V=32,
                 dir_tree=False):
    global THREAD_COUNT
    global train_df, valid_df
    global GLOBAL_DATA
    global GLOBAL_ID_DATA
    THREAD_COUNT = THREAD_COUNT_V
    GLOBAL_DATA = {}
    GLOBAL_ID_DATA = {}
    image_paths = image_list(path + 'train_images/')
    image_ids = None
    image_ids, loaded_ids_size = load_id_from_dir_tree_csv(path, data_schema_input, data_schema_output, restrict,
                                                               size)
    print('csv load done')
    local_image_collection = {'num_classes': 0, 'image_arr': []}
    if size < len(image_paths):
        image_paths_list = split_list(image_paths, THREAD_COUNT, restrict, size)
    else:
        image_paths_list = []
        for element in image_paths:
            image_paths_list.append([element])

    if split:
        test_arr = {}
        train_arr = {}
        train_cutoff = int(split_coef * len(local_image_collection['image_arr']))
        if not load_image:
            key_list = list(image_ids.keys())
            # random.shuffle(key_list)
            train_cutoff = int(split_coef * len(key_list))
            for i in range(0, train_cutoff):
                train_arr[key_list[i]] = image_ids[key_list[i]]
            for i in range(train_cutoff, len(key_list)):
                test_arr[key_list[i]] = image_ids[key_list[i]]

            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}, \
                   {'num_classes': local_image_collection['num_classes'], 'image_arr': test_arr}
        else:
            for i in range(0, train_cutoff):
                train_arr = local_image_collection['image_arr'][i]
            for i in range(train_cutoff, len(local_image_collection['image_arr'])):
                test_arr = local_image_collection['image_arr'][i]

            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}, \
                   {'num_classes': local_image_collection['num_classes'], 'image_arr': test_arr}
    else:
        train_arr = {}

        train_cutoff = int(len(local_image_collection['image_arr']))
        if not load_image:
            if type(image_ids) == type({}):
                key_list = list(image_ids.keys())
                # random.shuffle(key_list)
                train_cutoff = int(len(key_list))
                for i in range(0, train_cutoff):
                    train_arr[key_list[i]] = (image_ids[key_list[i]])
                return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}, {
                    'num_classes': local_image_collection['num_classes'], 'image_arr': []}
            elif type(image_ids) == type([]):
                # random.shuffle(image_ids)
                train_cutoff = int(len(image_ids))
                for i in range(0, train_cutoff):
                    train_arr[i] = (image_ids[i])
                return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}, {
                    'num_classes': local_image_collection['num_classes'], 'image_arr': []}
        else:
            key_list = list(image_ids.keys())
            # random.shuffle(key_list)
            for i in range(0, train_cutoff):
                train_arr[image_ids[i]] = (image_ids[key_list[i]])

            return {'num_classes': local_image_collection['num_classes'], 'image_arr': train_arr}, {
                'num_classes': local_image_collection['num_classes'], 'image_arr': []}


def generate_path_list_from_dict(input_dict, key_list, return_list, is_first=True):
    if type(input_dict) == type({}):
        if not is_first:
            for key in list(input_dict.keys()):
                generate_path_list_from_dict(input_dict[key], key_list + [key], return_list, is_first=False)
        else:
            local_return_list = []
            for key in list(input_dict.keys()):
                generate_path_list_from_dict(input_dict[key], key_list + [key], local_return_list, is_first=False)
            return local_return_list
    else:
        return_list.append(key_list)


def get_data_by_list(input_dict, key_list):
    if type(input_dict) != type({}):
        return input_dict
    if len(key_list) > 0:
        return get_data_by_list(input_dict[key_list[0]], key_list[1:])
    return input_dict


def data_bundle_and_path(input_dict):
    local_path_list = generate_path_list_from_dict(input_dict, [], [], True)
    return_path_list = []
    return_template = []
    if local_path_list == None:
        return_template = input_dict
        return [return_template], [[]]
    for element in local_path_list:
        return_template.append(get_data_by_list(input_dict, element))
        return_path_list.append(element)
    return return_template, return_path_list


def set_data_by_list(input_dict, key_list, data, index=-1):
    if len(key_list) > 1:
        get_data_by_list(input_dict[key_list[0]], key_list[1:])
    else:
        if len(key_list) == 0:
            input_dict.set_by_dict(data, index)
        else:
            input_dict[key_list[0]].set_by_dict(data, index)


def add_dict_path_recs(target_dict, target_list, data, file_name):
    if len(target_list) > 1:
        add_dict_path_recs(target_dict[target_list[0]], target_list[1:], data, file_name)
        return
    if type(target_dict[target_list[0]]) == type({}):
        target_dict[target_list[0]] = []

    if target_list[0] == 'filename':
        target_dict[target_list[0]] += [file_name] * add_dict_path_recs.local_data_len
    else:
        target_dict[target_list[0]] += data
        add_dict_path_recs.local_data_len = len(data)


def create_dict_path_recs(target_dict, target_list):
    if len(target_list) == 0:
        return
    if target_list[0] not in list(target_dict.keys()):
        if len(target_list) <= 1:
            target_dict[target_list[0]] = []
            return
        else:
            target_dict[target_list[0]] = {}
            create_dict_path_recs(target_dict, target_list[1:])
    if type(target_dict[target_list[0]]) == type([]):
        if len(target_dict[target_list[0]]) > 0:
            return
        target_dict[target_list[0]] = {}
    create_dict_path_recs(target_dict[target_list[0]], target_list[1:])


def worker_load_image_data_from_dir_tree_csv(args):
    global GLOBAL_DATA
    local_list, schema, cut_size, restrict, dir_path = args
    data_schema_input, data_schema_output = schema

    local_data_arr = {}

    if restrict:
        if cut_size != -1:
            local_list = local_list[:cut_size]
    schema_transformed = {}
    for element in list( data_schema_input.keys()):
        element_tree = element
        print('==============')
        create_dict_path_recs(GLOBAL_DATA, [element_tree])
        if os.path.exists(dir_path + element+'.csv'):
            if not restrict:
                df = pd.read_csv(dir_path + element+'.csv', low_memory=True)
            else:
                df = pd.read_csv(dir_path + element+'.csv', low_memory=True,nrows=cut_size)
            local_dict = df.to_dict()
            local_keys =  list(local_dict.keys())
            for element in local_keys:
                create_dict_path_recs(GLOBAL_DATA, [element_tree] + [element])
            for element in local_keys:
                try:

                    if element not in list(local_dict.keys()):
                        local_data_arr = []
                    else:
                        local_data_arr = list(local_dict[element].values())
                    add_dict_path_recs.local_data_len = len(local_dict[list(local_dict.keys())[0]])
                    add_dict_path_recs(GLOBAL_DATA, [element_tree] + [element], local_data_arr, element_tree[-1])
                except Exception as e:
                    print('!!!!!!!!')
                    print(e)
                    exit(0)
    return

def match_schema_name(name,names_list):

        if name in names_list:
            return True
        for element in names_list:
            if name in element:
                return True
        return False

def get_arr_by_name(name,target_dict):
    return_arr = []
    for key in list(target_dict.keys()):
        if name in key:
            return_arr.append(target_dict[key])
    return return_arr

def map_schema_data_rec(data, schema, path):
    return_dict = {}
    if type(schema) == type({}):
        for key in list(schema.keys()):
            return_dict[key] = map_schema_data_rec(data[key], schema[key], path)
        return return_dict
    else:
        row = []
        for element in schema:
            if match_schema_name(element.name , list(data.keys())):
                if element.type == 'arr':
                    row.append(get_arr_by_name(element.name,data))
                else:
                    row.append(data[element.name])
            elif element.type in ['2D_F', '2D_INT']:
                row.append(
                    np.resize(find_image_by_name(data[element.name],
                                                 path), element.shape))
            else:
                row.append(None)
        return DataCollection(data_size=len(row), data_schema=schema, data=row)


def worker_load_image_data_from_csv_tree(args):
    local_dict, schema, cut_size, restrict, path = args
    data_schema_input, data_schema_output = schema
    input = map_schema_data_rec(local_dict, data_schema_input, path)
    target = map_schema_data_rec(local_dict, data_schema_output, path)
    return DataBundle(data_ind=DataInd(timestamp='00000', cat_subcat='00000', model_sign='00000'), source=input,
                      target=target)


def worker_load_image_data_from_csv(args):
    local_dict, schema, cut_size, restrict, path = args
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
            elif element.load_name != None and element.load_name in list(local_dict.keys()):
                local_row.append(local_dict[element.load_name][i])
            elif element.type in DATA_FORMATS_SPECIAL:
                local_row.append(find_image_by_name(local_dict[list(local_dict.keys())[local_id_poss[0] - 1]][i],
                                                    path))


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
            for element in local_id_poss:
                local_id_name += str(row[element]) + '^'
            local_data_arr_input[local_id_name] = DataCollection(data_size=len(row), data_schema=data_schema_input,
                                                                 data=row)

        except Exception as e:
            pass
    local_id_poss = []
    for i, element in enumerate(data_schema_output):
        if element.is_id:
            local_id_poss.append(i)
    for i in range(len(local_dict[list(local_dict.keys())[0]])):
        local_row = []
        for element in data_schema_output:
            if element.name in list(local_dict.keys()):
                local_row.append(local_dict[element.name][i])
            elif element.load_name != None and element.load_name in list(local_dict.keys()):
                local_row.append(local_dict[element.load_name][i])
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
            for element in local_id_poss:
                local_id_name += str(row[element]) + '^'
            local_data_arr_output[local_id_name] = DataCollection(data_size=len(row), data_schema=data_schema_output,
                                                                  data=row)

        except Exception as e:
            pass
    data_bundle_list = []
    for input, target in zip(local_data_arr_input.values(), local_data_arr_output.values()):
        data_bundle_list.append(
            DataBundle(data_ind=DataInd(timestamp='00000', cat_subcat='00000', model_sign='00000'), source=input,
                       target=target))
    return data_bundle_list


def list_files(startpath, dir_path):
    return_list = []
    for root, dirs, files in os.walk(startpath):

        for f in files:
            return_list.append('{}/{}'.format(root.replace(dir_path, ''), f))
    return return_list


def add_recursive(data, path, dict_add):
    if path[0] not in list(dict_add.keys()):

        if len(path) > 1:
            dict_add[path[0]] = {}
        else:
            dict_add[path[0]] = None

    if len(path) > 1:
        add_recursive(data, path[1:], dict_add[path[0]])
    else:
        dict_add[path[0]] = data


def fill_global_id_data(args):
    local_ids, local_data, array_ids, average_len, j, back_keys = args

    for i in range(len(array_ids[j * average_len:j * average_len + average_len])):

        print(i, '/', len(array_ids))
        for local_key in list(local_data.keys()):

            for element in back_keys:
                add_recursive(local_data[local_key][i], [array_ids[average_len * j + i]] + [element] + [local_key],
                              GLOBAL_ID_DATA)
    pass


def recursiv_match(input_dict, data_schema, key=''):
    global GLOBAL_ID_DATA

    if type(data_schema) == type({}):
        for second_key in list(data_schema.keys()):
            if second_key in list(input_dict.keys()):
                recursiv_match(input_dict[second_key], data_schema[second_key], key=key + [second_key])

    if type(data_schema) == type([]):
        array_id = []
        local_dict = {}
        for local_key in list(data_schema):
            if local_key.is_id:
                if len(array_id) == 0 and len(input_dict)>0:
                    array_id = copy.deepcopy(input_dict[local_key.name])
                else:
                  if len(input_dict)>0:
                    for i in range(len(input_dict[local_key.name])):
                        array_id[i] = str(array_id[i]) + '-' + str(input_dict[local_key.name][i])

        id_list_ordered = split_list(target_list=array_id, count=THREAD_COUNT, restrict=False, size=100)
        input_dict_splited = [input_dict] * THREAD_COUNT
        array_id_splited = [array_id] * THREAD_COUNT
        average_len = [len(id_list_ordered[0])] * THREAD_COUNT
        back_keys = [GLOBAL_DATA.keys()] * THREAD_COUNT
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(fill_global_id_data, args) for args in
                       zip(id_list_ordered, input_dict_splited, array_id_splited, average_len,
                           range(len(id_list_ordered)), back_keys)]
            results = [f.result() for f in futures]
        # exit(0)


def load_id_from_dir_tree_csv(dir_path, data_schema_input, data_schema_output, restrict=False, size=100):
    local_ids = {}
    data = list_files(dir_path, dir_path)

    if len(data) >= THREAD_COUNT:
        id_list = []
        id_list_ordered = split_list(target_list=data, count=THREAD_COUNT, restrict=restrict, size=size)
        data_schema = [(data_schema_input, data_schema_output)] * THREAD_COUNT

        for i in range(THREAD_COUNT):
            id_list.append(id_list_ordered[i])

        if restrict:
            size = [size] * THREAD_COUNT
            dir_path = [dir_path] * THREAD_COUNT
        else:
            size = [len(id_list[0])] * THREAD_COUNT
            dir_path = [dir_path] * THREAD_COUNT
        restrict = [restrict] * THREAD_COUNT
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(worker_load_image_data_from_dir_tree_csv, args) for args in
                       zip(id_list, data_schema, size,
                           restrict, dir_path)]
            print("THREAD COUNT:", len(futures))
            results = [f.result() for f in futures]
        recursiv_match(GLOBAL_DATA, data_schema_input, [])
        return_list = {}
        for key in list(GLOBAL_ID_DATA.keys()):
            return_list[key] = worker_load_image_data_from_csv_tree(
                (GLOBAL_ID_DATA[key], (data_schema_input, data_schema_output), 100, False, ''))
        return return_list, len(return_list)




    else:

        local_ids = worker_load_image_data_from_dir_tree_csv(
            (data, (data_schema_input, data_schema_output), size, restrict, dir_path))
    loaded_ids_size = len(local_ids.keys())
    return local_ids, loaded_ids_size



def image_list(path):
    image_path_list = []
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file[-3:] in IMAGE_EXTS_LIST:
                image_path_list.append((root, file))
    return image_path_list[1:]


def specific_submit(self, file_dest=''):
        import csv
        import numpy
        f = open(file_dest + 'submission.csv', 'a+')
        writer = csv.writer(f)
        local_arr = []
        local_dict ={}
        output_id_dict = {}
        local_arr.append('Id')
        #for element in self.return_ids:
        #    local_arr.append(element)
        is_dsm = False
        is_2A3 = False
        if type(self.data_schema_output) is list:
            for element in self.data_schema_output:
                if element.is_id:
                    output_id_dict[element.name] = None
                else:
                  local_arr.append(element.name)

        else:
            local_arr,local_ids = self.get_schema_names(self.data_schema_output)
        for element in local_ids:
            output_id_dict[element] = None
        for element in local_arr:
            local_dict[element] = []
        writer.writerow(["id","reactivity_DMS_MaP","reactivity_2A3_MaP"])
        local_id_list = {}
        #for key in list(self.bundle_bucket.keys()):
        #         local_id_list[key] = self.bundle_bucket[key].source['train_data'].get_by_name('experiment_type')

        results, _ = self.predict()
        results = np.squeeze(results)

        #results = self.denormalize(results)
        if type(results) == type(np.zeros((2))):
                results = results.tolist()


        #if type(results) == type({}):
        #    results = results[list(results.keys())[0]]
        seq_id = 0
        for key in list(results.keys()):
            print(key)
            local_arr = []
            final_ids = []
            is_2A3 = True
            #if local_id_list[key] == '2A3_MaP':
            #    is_2A3 = True
            #elif local_id_list[key] == 'DMS_MaP':
            #    is_dsm = True
            try:
                    local_id_dict = copy.deepcopy(output_id_dict)
                    self.get_data_ids(self.bundle_bucket[key].source,local_id_dict)
                    for element in local_ids:
                            final_ids.append(str(local_id_dict[element]))
                    local_arr.append(seq_id)
                    seq_id += 1
            except IOError as e:
                    print(e)
                    exit(0)
            if type(results) ==  type([]):
                for element in results[key]:

                    local_arr.append(round(element,3))


            else:
                       if is_dsm:
                           local_arr.append(0)
                           local_arr.append(round(numpy.mean(results[key][0]),3))
                       elif is_2A3:
                           local_arr.append(round(numpy.mean(results[key][0]),3))
                           local_arr.append(0)

            for element,arr_element in zip(self.get_schema_names(self.data_schema_output),local_arr):
                    if element == 'Turn':
                        arr_element = int(arr_element)


                    #local_dict[element].append(arr_element)
            if str(key) not in self.submited_ids:
                    writer.writerow(local_arr)
            self.submited_ids.append(str(key))