import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from main import runner

dataset_path = "./data_sets/icecube-neutrinos-in-deep-ice/"#'/kaggle/input/icecube-neutrinos-in-deep-ice/'
runner(dataset_path, train_name='test_images', restrict=True,
       size=200000, target_name='letter', no_ids=False,
       submit_file='test',
       train_file='train',
       utils_name='specific_loaders.icecubeNeutrinos',
       split=False, THREAD_COUNT=2, dir_tree=False)
