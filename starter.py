import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from main import runner

dataset_path = '/kaggle/input/rsna-breast-cancer-detection'
runner(dataset_path, train_name='train', restrict=False,
       size=10, target_name='letter', no_ids=False,
       submit_file='test',
       train_file='train',
       split=True, THREAD_COUNT=32, dir_tree=True
       , utils_name='specific_loaders.hubmapOrganSegmentationUtils')
