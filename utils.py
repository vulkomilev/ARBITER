import os
import cv2
import concurrent.futures
import pandas as pd
import random
from pathlib import Path

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
THREAD_COUNT = 32
loaded_ids = []
loaded_ids_target = []
index_csv = {}


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
    image_paths, image_ids = args
    local_image_collection = {'class_num': [], 'image_arr': []}

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
                {"img": local_img, "img_name": image_name[:-4]})
        else:
            if image_name[:-4] not in image_ids.keys():
                continue
            local_img_id = image_ids[image_name[:-4]]

            if local_img_id not in loaded_ids_target:
                loaded_ids_target.append(local_img_id)
            local_image_collection['image_arr'].append(
                {"img": local_img, "img_id": local_img_id, "img_name": image_name[:-4]})

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


def image_loader(path, restrict=False, size=1000):
    image_paths = image_list(path)
    image_ids = None
    if Path(path + 'train.csv').exists():
        image_ids, loaded_ids_size = load_id_from_csv(path + 'train.csv')
    local_image_collection = {'num_classes': 0, 'image_arr': []}
    image_paths_list = split_list(image_paths, THREAD_COUNT, restrict, size)
    image_ids = [image_ids] * THREAD_COUNT
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:

        futures = [executor.submit(image_loader_worker, args) for args in zip(image_paths_list, image_ids)]
        print("THREAD COUNT:", len(futures))
        results = [f.result() for f in futures]

    for result in results:
        local_image_collection['image_arr'] += result['image_arr']
    local_image_collection['num_classes'] = len(loaded_ids_target)
    return local_image_collection


def worker_load_id_from_csv(args):
    # global loaded_ids
    list = args
    local_ids = {}
    list = list[0]
    for row in list:
        if len(row) != 2:
            continue
        if is_int(row[1]):
            local_ids[row[0]] = int(row[1])
    return local_ids


def load_id_from_csv(csv_path):
    local_ids = {}
    data = pd.read_csv(csv_path, low_memory=False)
    csv_reader = data.values.tolist()
    id_list = split_list(target_list=csv_reader, count=32, restrict=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = [executor.submit(worker_load_id_from_csv, args) for args in zip(id_list)]
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
            image_path_list.append((root + '/', file))
    return image_path_list[1:]
