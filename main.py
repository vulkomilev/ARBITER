from utils import image_loader
from utils import DataUnit
from Arbiter import Arbiter

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

print('Loading images ...')

# TODO:add different types of target like linear ,classifier ,timeseries etc

data_schema = [DataUnit('str', (), None, 'Id'),
               DataUnit('int', (), None, 'Subject Focus'),
               DataUnit('int', (), None, 'Eyes'),
               DataUnit('int', (), None, 'Face'),
               DataUnit('int', (), None, 'Near'),
               DataUnit('int', (), None, 'Action'),
               DataUnit('int', (), None, 'Accessory'),
               DataUnit('int', (), None, 'Group'),
               DataUnit('int', (), None, 'Collage'),
               DataUnit('int', (), None, 'Human'),
               DataUnit('int', (), None, 'Occlusion'),
               DataUnit('int', (), None, 'Info'),
               DataUnit('int', (), None, 'Blur'),
               DataUnit('int', (), None, 'Pawpularity')]
target_type = 'Regression'
image_collection_train = image_loader('./train/', restrict=True, size=100, target_name='target_name', no_ids=True,
                                      data_schema=data_schema)
print('Images loaded!!!')
arbiter = Arbiter(IMAGE_WIDTH, IMAGE_HEIGHT, image_collection_train['num_classes'], target_type)
arbiter.train(image_collection_train['image_arr'], force_train=False, train_arbiter=True)
arbiter.evaluate(image_collection_train['image_arr'])
