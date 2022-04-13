from arbiter import Arbiter
from utils.utils import DataUnit
from utils.utils import REGRESSION, TIME_SERIES
from utils.utils import image_loader

print('Loading images ...')

# TODO:add different types of target like linear ,classifier ,timeseries etc


'''

data_schema = [  DataUnit('str', (), None, 'image_name'),
                 DataUnit('int', (), None, 'bar'),
                 DataUnit('int', (), None, 'buble'),
                 DataUnit('int', (), None, 'fill'),
                 DataUnit('int', (), None, 'rotate'),
                 DataUnit('2D_F',(64,64), None, 'Image')]

data_schema = [DataUnit('str', (), None, 'timestamp'),
               DataUnit('int', (1,), None, 'Asset_ID'),
               DataUnit('int', (1,), None, 'Count'),
               DataUnit('float', (1,), None, 'Open'),
               DataUnit('float', (1,), None, 'High'),
               DataUnit('float', (1,), None, 'Low'),
               DataUnit('float', (1,), None, 'Close'),
               DataUnit('float', (1,), None, 'Volume'),
               DataUnit('float', (1,), None, 'VWAP')]

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
               DataUnit('float', (), None, 'Pawpularity'),
               DataUnit('2D_F',(64,64,3), None, 'Image')]
agent_router = [{'SwinTransformer':{'inputs':['Image'],
                                   'outputs':[{'name':'Pawpularity',
                                               'type':REGRESSION}]}}]
agent_router = [{'LSTM':{'inputs':['Count','Open','High','Low','Close','Volume'],
                               'outputs':[{'name':'VWAP','type':REGRESSION}]}}]
                               
agent_router = [{'FunctionalAutoencoder':{'inputs':['Image'],
                               'outputs':[{'name':'Image','type':REGRESSION}]}}]
agent_router = [{'ConvMultihead':{'inputs':['Image'],
                               'outputs':[{'name':'Image','type':REGRESSION}]}}]

data_schema = [  DataUnit('str', (), None, 'image_name'),
                 DataUnit('int', (), None, 'bar'),
                 DataUnit('int', (), None, 'buble'),
                 DataUnit('int', (), None, 'fill'),
                 DataUnit('int', (), None, 'rotate'),
                 DataUnit('2D_F',(64,64), None, 'Image')]

agent_router = [{'ConvMultihead':{'inputs':['Image'],
                               'outputs':[{'name':'Image','type':REGRESSION}]}}]
                               
data_schema = [  DataUnit('str', (), None, 'Image')]

agent_router = [{'MarkSpaces':{'inputs':['Image'],
                               'outputs':[{'name':'Image','type':CATEGORY}]}}]
                               '''
# data_schema = [
#                    DataUnit('str', (), None, 'name'),
#                 DataUnit('int', (), None, 'letter'),
#    DataUnit('2D_F', (64,64), None, 'Image')]

# agent_router = [{'MyResNet50':{'inputs':['Image'],
#                               'outputs':[{'name':'letter','type':CATEGORY}]}}]


# KerasOcrFineTune
# MyResNet50

'''
data_schema = [
                    DataUnit('int', (), None, 'id'),
                 DataUnit('int', (), None, 'case_num'),
                 DataUnit('int', (), None, 'pn_num'),
                 DataUnit('int', (), None, 'feature_num'),
                 DataUnit('str', (), None, 'annotation'),
                 DataUnit('str', (), None, 'location'),
]
agent_router = [{'DDASINDy':{'inputs':['annotation'],
                               'outputs':[{'name':'location','type':CATEGORY}]}}]
'''
'''
data_schema = [DataUnit('str', (), None, 'timestamp'),
               DataUnit('int', (1,), None, 'Asset_ID'),
               DataUnit('int', (1,), None, 'Count'),
               DataUnit('float', (1,), None, 'Open'),
               DataUnit('float', (1,), None, 'High'),
               DataUnit('float', (1,), None, 'Low'),
               DataUnit('float', (1,), None, 'Close'),
               DataUnit('float', (1,), None, 'Volume'),
               DataUnit('float', (1,), None, 'VWAP')]
agent_router = [{'LSTM':{'inputs':['Count','Open','High','Low','Close','Volume'],
                               'outputs':[{'name':'VWAP','type':REGRESSION}]}}]
'''

'''
data_schema = [  DataUnit('str', (), None, 'image_name'),
                 DataUnit('int', (), None, 'bar'),
                 DataUnit('int', (), None, 'buble'),
                 DataUnit('int', (), None, 'fill'),
                 DataUnit('int', (), None, 'rotate'),
                 DataUnit('2D_F',(64,64), None, 'Image')]
agent_router = [{'FunctionalAutoencoder':{'inputs':['Image'],
                               'outputs':[{'name':'Image','type':REGRESSION}]}}]
'''
'''
data_schema = [  DataUnit('str', (), None, 'image_name'),
                 DataUnit('int', (), None, 'bar'),
                 DataUnit('int', (), None, 'buble'),
                 DataUnit('int', (), None, 'fill'),
                 DataUnit('int', (), None, 'rotate'),
                 DataUnit('2D_F',(64,64), None, 'Image')]
agent_router = [{'GraphSearch':{'inputs':['Image'],
                               'outputs':[{'name':'Image','type':REGRESSION}]}}]
'''
data_schema = [DataUnit('str', (), None, 'timestamp'),
               DataUnit('int', (1,), None, 'Asset_ID'),
               DataUnit('int', (1,), None, 'Count'),
               DataUnit('float', (1,), None, 'Open'),
               DataUnit('float', (1,), None, 'High'),
               DataUnit('float', (1,), None, 'Low'),
               DataUnit('float', (1,), None, 'Close'),
               DataUnit('float', (1,), None, 'Volume'),
               DataUnit('float', (1,), None, 'VWAP')]
agent_router = [{'LSTM': {'inputs': ['Count', 'Open', 'High', 'Low', 'Close', 'Volume'],
                          'outputs': [{'name': 'VWAP', 'type': REGRESSION}]}}]
target_type = TIME_SERIES
# ./data_sets/solvedCaptchas/
#./data_sets/g-research-crypto-forecasting/
# MAKE A ARCH SEARCH OR SOMETHING OTHER SEARCH BASED ON GENETIC ALGORITHM SO THE PC WILL EXPLORE WHILE YOU ARE GONE
def runner(dataset_path, train_name='train', restrict=True, \
                                                                 size=10, target_name='letter', no_ids=False,
                                                                 data_schema=data_schema, split=True,THREAD_COUNT = 32):
    image_collection_train, image_collection_test = image_loader(dataset_path
                                                                 , train_name='train', restrict=True, \
                                                                 size=800, target_name='letter', no_ids=False,
                                                                 data_schema=data_schema, split=True,THREAD_COUNT_V = THREAD_COUNT)

    arbiter = Arbiter(data_schema=data_schema, target_type=target_type, class_num=image_collection_train['num_classes'],
                      router_agent=agent_router, skip_arbiter=True)
    for i in range(10):
        arbiter.train(image_collection_train['image_arr'], train_target='letter', force_train=True, train_arbiter=False)
    arbiter.evaluate(image_collection_train['image_arr'])
