import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from main import runner
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
#dataset_path = "./data_sets/tlvmc-parkinsons-freezing-gait-prediction/train/"
#dataset_path = "./data_sets/abstraction-and-reasoning-challenge/image_train_small_second/"
dataset_path = "./data_sets/icr-identify-age-related-conditions/"
runner(dataset_path, train_name='test_images', restrict=False,
       size=3, target_name='letter', no_ids=False,
       submit_file='test',
       train_file='train',
       #utils_name='specific_loaders.icecubeNeutrinos',
       split=False, THREAD_COUNT=32, dir_tree=False)
