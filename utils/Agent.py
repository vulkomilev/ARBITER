import abc


class Agent(object):

    @abc.abstractmethod
    def train(self, images, force_train=False):
        pass

    @abc.abstractmethod
    def predict(self, image):
        pass
