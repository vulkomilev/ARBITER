import cv2
import numpy as np
from matplotlib import pyplot


class TextLocator(object):

    def __init__(self, inputs, outputs, data_schema):
        pass

    def train(self, images, force_train=False):
        pass

    def predict(self, image):
        local_image = image.get_by_name('Image')
        if local_image is None:
            return [0]
        # Get input size
        dst = cv2.GaussianBlur(local_image, (25, 25), cv2.BORDER_DEFAULT)

        '''
        height, width = local_image.shape[:2]

        # Desired "pixelated" size
        w, h = (28, 28)

        # Resize input to "pixelated" size
        temp = cv2.resize(local_image, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = 1.0  # Simple contrast control
        beta = 0  # Simple brightness control
        # Initialize output image
        dst = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        for y in range(dst.shape[0]):
            for x in range(dst.shape[1]):
                for c in range(dst.shape[2]):
                     dst[y, x, c] = np.clip(alpha * dst[y, x, c] + beta, 0, 255)
        '''
        ret, dst = cv2.threshold(dst, 254, 255, cv2.THRESH_BINARY)

        # display input and output image
        pyplot.imshow(np.hstack((local_image, dst)))
        pyplot.show()
