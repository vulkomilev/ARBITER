import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from main import runner

dataset_path = '/kaggle/input/playground-series-s3e9/'
runner(dataset_path, train_name='test_images', restrict=False,
       size=10, target_name='letter', no_ids=False,
       submit_file='test',
       train_file='train',
       utils_name='specific_loaders.playgroundSeriesUtils',
       split=False, THREAD_COUNT=32, dir_tree=False)
