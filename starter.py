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
#dataset_path = "./data_sets/abstraction-and-reasoning-challenge/training/"
dataset_path = "./data_sets/commonlit-evaluate-student-summaries/"
runner(dataset_path, train_name='test_images', restrict=False,
       size=3, target_name='letter', no_ids=False,
       submit_file='prompts_test',
       train_file='train',
       utils_name='specific_loaders.commonLitLoader',
       split=False, THREAD_COUNT=4, dir_tree=True)
