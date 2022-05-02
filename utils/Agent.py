import abc
import numpy as np
import tensorflow as tf
import re
from pathlib import Path
import math


class Agent(object):

    @abc.abstractmethod
    def train(self, images, force_train=False):
        pass

    @abc.abstractmethod
    def predict(self, image):
        pass
