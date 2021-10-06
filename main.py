from utils import image_loader
from Arbiter import Arbiter

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

print('Loading images ...')
image_collection_train = image_loader('./landmark-retrieval-2021/', restrict=True, size=100)
image_collection_test = image_loader('./landmark-retrieval-2021/', restrict=True, size=0)
print('Images loaded!!!')
arbiter = Arbiter(IMAGE_WIDTH, IMAGE_HEIGHT, image_collection_train['num_classes'])
arbiter.train(image_collection_train['image_arr'], force_train=False, train_arbiter=True)
arbiter.submit(image_collection_test['image_arr'], image_collection_train['image_arr'])
