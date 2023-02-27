import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import pyplot

from utils.utils import REGRESSION, REGRESSION_CATEGORY, IMAGE, TIME_SERIES,CATEGORY

from utils.Agent import *


class EvolutionProgrammingAutoEncoderAssist():

    def __init__(self, inputs, outputs, data_schema, class_num):
        self.model = None
        self.init_neural_network(inputs, outputs, data_schema, class_num)
        self.total_tested = 0
        self.good_tested = 0

    def init_neural_network(self, inputs, outputs, data_schema, class_num):
        pass

    def prepare_data(self, images, in_train=False):
        pass

    def predict(self, image):
        pass