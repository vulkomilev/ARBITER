import importlib

from arbiter import Arbiter
from utils.utils import CATEGORY
from utils.utils import DataUnit,IMAGE

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
'''
data_schema_input = [

                 DataUnit('str', (), None, 'name',is_id=True),
DataUnit('int', (), None, 'number'),
    DataUnit('2D_F', (64,64), None, 'Image')]
data_schema_output = [

                 DataUnit('str', (), None, 'name',is_id=True),
DataUnit('int', (), None, 'number'),
    DataUnit('2D_F', (64,64), None, 'Image')]
agent_router = [{'ConstructNetwork':{'inputs':['Image'],
                               'outputs':[{'name':'letter','type':CATEGORY}]}}]
'''
'''
data_schema = [DataUnit('str', (), None, 'timestamp',is_id=True),
               DataUnit('int', (1,), None, 'Asset_ID',is_id=True),
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
data_schema_input = [DataUnit('str', (), None, 'timestamp',is_id=True),
               DataUnit('int', (1,), None, 'Asset_ID',is_id=True),
               DataUnit('int', (1,), None, 'Count'),
               DataUnit('float', (1,), None, 'Open'),
               DataUnit('float', (1,), None, 'High'),
               DataUnit('float', (1,), None, 'Low'),
               DataUnit('float', (1,), None, 'Close'),
               DataUnit('float', (1,), None, 'Volume'),
               DataUnit('float', (1,), None, 'VWAP'),
               DataUnit('float', (1,), None, 'Target'),
               DataUnit('int', (1,), None, 'group_num',is_id=True),
               DataUnit('int', (1,), None, 'row_id',is_id=True)]

data_schema_output = [DataUnit('int', (), None, 'group_num',is_id=True),
               DataUnit('int', (1,), None, 'row_id',is_id=True),
               DataUnit('float', (1,), None, 'Target')]
agent_router = [{'LSTM':{'inputs':['Count','Open','High','Low','Close','Volume'],
                               'outputs':[{'name':'Target','type':REGRESSION}]}}]


data_schema_input = {
                 'device_imu':[DataUnit('str', (), None, 'MessageType',is_id=False),
                               DataUnit('int', (), None, 'utcTimeMillis',is_id=False),
                               DataUnit('float', (), None, 'MeasurementX', is_id=False),
                               DataUnit('float', (), None, 'MeasurementY', is_id=False),
                               DataUnit('float', (), None, 'MeasurementZ', is_id=False),
                               DataUnit('float', (), None, 'BiasX', is_id=False),
                               DataUnit('float', (), None, 'BiasY', is_id=False),
                               DataUnit('float', (), None, 'BiasZ', is_id=False)
                               ],
                 'device_gnss':[
                               DataUnit('str', (), None, 'MessageType',is_id=False),
                               DataUnit('int', (), None, 'utcTimeMillis',is_id=False),
                               DataUnit('int', (), None, 'TimeNanos',is_id=False),
                               DataUnit('str', (), None, 'LeapSecond',is_id=False),
                               DataUnit('int', (), None, 'FullBiasNanos',is_id=False),
                               DataUnit('float', (), None, 'BiasNanos', is_id=False),
                               DataUnit('float', (), None, 'BiasUncertaintyNanos', is_id=False),
                               DataUnit('float', (), None, 'DriftNanosPerSecond', is_id=False),
                               DataUnit('float', (), None, 'DriftUncertaintyNanosPerSecond', is_id=False),
                               DataUnit('int', (), None, 'HardwareClockDiscontinuityCount', is_id=False),
                               DataUnit('int', (), None, 'Svid', is_id=True),
                               DataUnit('int', (), None, 'TimeOffsetNanos', is_id=False),
                               DataUnit('int', (), None, 'State', is_id=False),
                               DataUnit('int', (), None, 'ReceivedSvTimeNanos', is_id=False),
                               DataUnit('int', (), None, 'ReceivedSvTimeUncertaintyNanos', is_id=False),
                               DataUnit('float', (), None, 'Cn0DbHz', is_id=False),
                               DataUnit('float', (), None, 'PseudorangeRateMetersPerSecond', is_id=False),
                               DataUnit('float', (), None, 'PseudorangeRateUncertaintyMetersPerSecond', is_id=False),
                               DataUnit('int', (), None, 'AccumulatedDeltaRangeState', is_id=False),
                               DataUnit('float', (), None, 'AccumulatedDeltaRangeMeters', is_id=False),
                               DataUnit('float', (), None, 'AccumulatedDeltaRangeUncertaintyMeters', is_id=False),
                               DataUnit('int', (), None, 'CarrierFrequencyHz', is_id=False),
                               DataUnit('int', (), None, 'MultipathIndicator', is_id=False),
                               DataUnit('int', (), None, 'ConstellationType', is_id=False),
                               DataUnit('str', (), None, 'CodeType', is_id=False),
                               DataUnit('int', (), None, 'ChipsetElapsedRealtimeNanos', is_id=False),
                               DataUnit('float', (), None, 'ArrivalTimeNanosSinceGpsEpoch', is_id=False),
                               DataUnit('float', (), None, 'RawPseudorangeMeters', is_id=False),
                               DataUnit('float', (), None, 'RawPseudorangeUncertaintyMeters', is_id=False),
                               DataUnit('str', (), None, 'SignalType', is_id=False),
                               DataUnit('float', (), None, 'RawPseudorangeUncertaintyMeters', is_id=False),
                               DataUnit('float', (), None, 'SvPositionXEcefMeters', is_id=False),
                               DataUnit('float', (), None, 'SvPositionYEcefMeters', is_id=False),
                               DataUnit('float', (), None, 'SvPositionZEcefMeters', is_id=False),
                               DataUnit('float', (), None, 'SvElevationDegrees', is_id=False),
                               DataUnit('float', (), None, 'SvAzimuthDegrees', is_id=False),
                               DataUnit('float', (), None, 'SvVelocityXEcefMetersPerSecond', is_id=False),
                               DataUnit('float', (), None, 'SvVelocityYEcefMetersPerSecond', is_id=False),
                               DataUnit('float', (), None, 'SvVelocityZEcefMetersPerSecond', is_id=False),
                               DataUnit('float', (), None, 'SvClockBiasMeters', is_id=False),
                               DataUnit('float', (), None, 'IsrbMeters', is_id=False),
                               DataUnit('float', (), None, 'IonosphericDelayMeters', is_id=False),
                               DataUnit('float', (), None, 'TroposphericDelayMeters', is_id=False),
                               DataUnit('float', (), None, 'WlsPositionXEcefMeters', is_id=False),
                               DataUnit('float', (), None, 'WlsPositionYEcefMeters', is_id=False),
                               DataUnit('float', (), None, 'WlsPositionZEcefMeters', is_id=False),
                 ]
              }
data_schema_output = [

                 DataUnit('str', (), None, 'name',is_id=True),
DataUnit('int', (), None, 'number'),
    DataUnit('2D_F', (64,64), None, 'Image')]
#tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
agent_router = [{'TransformerExpSmartphoneD2022':{'inputs':['MessageType','utcTimeMillis','TimeNanos','LeapSecond',
                                                            'FullBiasNanos','BiasNanos','BiasUncertaintyNanos',
                                                            'DriftNanosPerSecond','DriftUncertaintyNanosPerSecond',
                                                            'HardwareClockDiscontinuityCount','Svid','TimeOffsetNanos',
                                                            'State','ReceivedSvTimeNanos',
                                                            'ReceivedSvTimeUncertaintyNanos','Cn0DbHz',
                                                            'PseudorangeRateMetersPerSecond',
                                                            'PseudorangeRateUncertaintyMetersPerSecond',
                                                            'AccumulatedDeltaRangeState','AccumulatedDeltaRangeMeters',
                                                            'AccumulatedDeltaRangeUncertaintyMeters',
                                                            'CarrierFrequencyHz','MultipathIndicator',
                                                            'ConstellationType','CodeType',
                                                            'ChipsetElapsedRealtimeNanos',
                                                            'ArrivalTimeNanosSinceGpsEpoch','RawPseudorangeMeters',
                                                            'RawPseudorangeUncertaintyMeters','SignalType',
                                                            'RawPseudorangeUncertaintyMeters','SvPositionXEcefMeters',
                                                            'SvPositionYEcefMeters','SvPositionZEcefMeters',
                                                            'SvElevationDegrees','SvAzimuthDegrees',
                                                            'SvVelocityXEcefMetersPerSecond',
                                                            'SvVelocityYEcefMetersPerSecond',
                                                            'SvVelocityZEcefMetersPerSecond','SvClockBiasMeters',
                                                            'IsrbMeters','IonosphericDelayMeters',
                                                            'TroposphericDelayMeters','WlsPositionXEcefMeters',
                                                            'WlsPositionYEcefMeters','WlsPositionZEcefMeters'],
                               'outputs':[
{'name':'tripId','type':CATEGORY},
                                          {'name':'UnixTimeMillis','type':REGRESSION},
                                          {'name':'LatitudeDegrees','type':REGRESSION},
                                          {'name':'LongitudeDegrees','type':REGRESSION}]}}]
                        
data_schema_input = [
                     DataUnit('2D_F', (64,64), None, 'Image'),
                     DataUnit('str', (), None, 'Id',is_id=True)]

data_schema_output = [
                   DataUnit('str', (), None, 'Id',is_id=True),
                   DataUnit('2D_F', (64,64), None, 'Image')]
#tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
agent_router = [{'FunctionalAutoencoder':{'inputs':['Image','Id'],
                               'outputs':[{'name':'Image','type':IMAGE}]}}]
                               
data_schema_input = [
                     DataUnit('2D_F', (900,900), None, 'Image'),
                     DataUnit('2D_F', (900, 900), None, 'ImageMask'),
                     DataUnit('str', (), None, 'id',is_id=True),
                     DataUnit('str', (), None, 'organ',is_id=False),
                     DataUnit('str', (), None, 'data_source',is_id=False),
                     DataUnit('int', (), None, 'img_height',is_id=False),
                     DataUnit('int', (), None, 'img_width',is_id=False),
                     DataUnit('float', (), None, 'pixel_size',is_id=False),
                     DataUnit('int', (), None, 'tissue_thickness',is_id=False),
                     DataUnit('str', (), None, 'rle',is_id=False),
                     DataUnit('int', (), None, 'age',is_id=False),
                     DataUnit('str', (), None, 'sex',is_id=False)
]

data_schema_output = [
                   DataUnit('str', (), None, 'id',is_id=True) ,
                   DataUnit('str', (), None, 'rle',is_id=False)]
agent_router = [{'MyResNet50':{'inputs':['Image','Id'],
                               'outputs':[{'name':'rle','type':STRING}]}}]
                               
data_schema_input = [
                     DataUnit('2D_F', (64,64), None, 'Image'),
                     DataUnit('str', (), None, 'Id',is_id=True)]

data_schema_output = [
                   DataUnit('str', (), None, 'Id',is_id=True),
                   DataUnit('2D_F', (64,64), None, 'Image')]
#tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
agent_router = [{'NeuralCellularAutomata_moded':{'inputs':['Image','Id'],
                               'outputs':[{'name':'Image','type':IMAGE}]}}]
'''
"""
data_schema_input = {
    'games':[
        DataUnit('int', (), None, 'game_id', is_id=True),
        DataUnit('str', (), None, 'first', is_id=False),
        DataUnit('str', (), None, 'time_control_name', is_id=False),
        DataUnit('str', (), None, 'game_end_reason', is_id=False),
        DataUnit('int', (), None, 'winner', is_id=False),
        DataUnit('date', (), None, 'created_at', is_id=False),
        DataUnit('str', (), None, 'lexicon', is_id=False),
        DataUnit('int', (), None, 'initial_time_seconds', is_id=False),
        DataUnit('int', (), None, 'increment_seconds', is_id=False),
        DataUnit('str', (), None, 'rating_mode', is_id=False),
        DataUnit('int', (), None, 'max_overtime_minutes', is_id=False),
        DataUnit('float', (), None, 'game_duration_seconds', is_id=False)
    ],
    'train': [
        DataUnit('int', (), None, 'game_id', is_id=True),
        DataUnit('str', (), None, 'nickname', is_id=False),
        DataUnit('int', (), None, 'score', is_id=False),
        DataUnit('int', (), None, 'rating', is_id=False)],
    'turns': [
        DataUnit('int', (), None, 'game_id', is_id=True),
        DataUnit('int', (), None, 'turn_number', is_id=False),
        DataUnit('str', (), None, 'nickname', is_id=False),
        DataUnit('str', (), None, 'rack', is_id=False),
        DataUnit('str', (), None, 'location', is_id=False),
        DataUnit('str', (), None, 'move', is_id=False),
        DataUnit('int', (), None, 'points', is_id=False),
        DataUnit('int', (), None, 'score', is_id=False),
        DataUnit('str', (), None, 'turn_type', is_id=False)]
data_schema_input = {
    'train': [
        DataUnit('2D_F', (900,900), None, 'Image'),
        DataUnit('int', (), None, 'site_id', is_id=True),
        DataUnit('int', (), None, 'patient_id', is_id=True,break_seq=True),
        DataUnit('int', (), None, 'image_id', is_id=True),
        DataUnit('str', (), None, 'laterality', is_id=False),
        DataUnit('str', (), None, 'view', is_id=False),
        DataUnit('int', (), None, 'age', is_id=False),
        DataUnit('int', (), None, 'cancer', is_id=False),
        DataUnit('int', (), None, 'biopsy', is_id=False),
        DataUnit('int', (), None, 'invasive', is_id=False),
        DataUnit('int', (), None, 'BIRADS', is_id=False)]
}

data_schema_output = {
                'train':[
                   DataUnit('int', (), None, 'seq_id',is_id=True),
                   DataUnit('float', (), None, 'tm',is_id=False)]}

#tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
agent_router = [{'DenseScrable':{'inputs':['seq_id','protein_sequence',
                                                      'pH',
                                                      'data_source','tm'],
                               'outputs':[{'name':'seq_id','type':'int'},
                                          {'name':'tm','type':'float'}]}}]


data_schema_input = [
                     DataUnit('2D_F', (32,32,3), None, 'Image'),
                     DataUnit('str', (), None, 'Id',is_id=True)]

data_schema_output = [
                   DataUnit('str', (), None, 'Id',is_id=True),
                   DataUnit('2D_F', (32,32,3), None, 'Image')]
#tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
agent_router = [{'ImageAutoencoderDiscreteFunctions':{'inputs':['Image','Id'],
                               'outputs':[{'name':'Image','type':IMAGE}]}}]
data_schema_input = {
    'train': [
        DataUnit('2D_F', (900,900), None, 'image_data'),
        DataUnit('int', (), None, 'site_id', is_id=True),
        DataUnit('int', (), None, 'patient_id', is_id=True,break_seq=True),
        DataUnit('int', (), None, 'image_id', is_id=True),
        DataUnit('str', (), None, 'laterality', is_id=False),
        DataUnit('str', (), None, 'view', is_id=False),
        DataUnit('int', (), None, 'age', is_id=False),
        DataUnit('int', (), None, 'cancer', is_id=False),
        DataUnit('int', (), None, 'biopsy', is_id=False),
        DataUnit('int', (), None, 'invasive', is_id=False),
        DataUnit('int', (), None, 'BIRADS', is_id=False)]
}

data_schema_output = {
                'train':[
                   DataUnit('int', (), None, 'site_id',is_id=True),
                   DataUnit('int', (), None, 'cancer',is_id=False)]}
#tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
agent_router = [{'ImageAndData':{'inputs':['Image','Id','site_id','patient_id', 'image_id', 'laterality', 'view',
                                           'age', 'cancer', 'biopsy', 'invasive', 'BIRADS'],
                               'outputs':[{'name':'cancer','type':REGRESSION}]}}]
    data_schema_input = [
        DataUnit('int', (), None, 'id', is_id=True),
        DataUnit('int', (), None, 'CementComponent', is_id=False),
        DataUnit('float', (), None, 'BlastFurnaceSlag', is_id=False),
        DataUnit('float', (), None, 'FlyAshComponent', is_id=False),
        DataUnit('int', (), None, 'WaterComponent', is_id=False),
        DataUnit('float', (), None, 'SuperplasticizerComponent', is_id=False),
        DataUnit('float', (), None, 'CoarseAggregateComponent', is_id=False),
        DataUnit('float', (), None, 'FineAggregateComponent', is_id=False),
        DataUnit('int', (), None, 'AgeInDays', is_id=False)]

    data_schema_output = [
        DataUnit('int', (), None, 'id', is_id=True),
        DataUnit('float', (), None, 'Strength')]
        
        
    data_schema_input = [
        DataUnit('str', (), None, 'Id', is_id=True),
        DataUnit('2D_F', (1,64, 64,3), None, 'Image')
        ]

    data_schema_output = [
        DataUnit('str', (), None, 'Id', is_id=True),
        DataUnit('2D_F', (64, 64), None, 'Image')]
                                                                       

data_schema_input = [
    DataUnit('int', (), None, 'batch_id', is_id=True),
    DataUnit('int', (), None, 'event_id', is_id=True),
    DataUnit('int', (), None, 'first_pulse_index', is_id=False),
    DataUnit('int', (), None, 'last_pulse_index', is_id=False),
    DataUnit('float', (), None, 'azimuth', is_id=False),
    DataUnit('float', (), None, 'zenith', is_id=False),
    DataUnit('int', (), None, 'sensor_id', is_id=True),
    DataUnit('int', (), None, 'time', is_id=False),
    DataUnit('float', (), None, 'charge', is_id=False),
    DataUnit('bool', (), None, 'auxiliary', is_id=False),
]

data_schema_output = [
    DataUnit('int', (), None, 'event_id', is_id=True),
    DataUnit('float', (), None, 'azimuth', is_id=False),
    DataUnit('float', (), None, 'zenith', is_id=False)]

agent_router = [{'DenseScrable': {'inputs': ['batch_id', 'event_id', 'first_pulse_index', 'last_pulse_index', 'azimuth',
                                             'zenith', 'sensor_id', 'time', 'charge', 'auxiliary'],
                                  'outputs': [{'name': 'event_id', 'type': 'int'},
                                              {'name': 'azimuth', 'type': 'float'},
                                              {'name': 'zenith', 'type': 'float'}

                                              ]}}]
data_schema_input =     {
        'defog':[

            DataUnit('str', (), None, 'filename', is_id=True),
            DataUnit('int', (), None, 'Time',is_id=True),
            DataUnit('float', (), None, 'AccV'),
            DataUnit('float', (), None, 'AccML'),
            DataUnit('float', (), None, 'AccAP'),
            DataUnit('float', (), None, 'StartHesitation'),
            DataUnit('float', (), None, 'Turn'),
            DataUnit('float', (), None, 'Walking'),
            DataUnit('bool', (), None, 'Valid'),
            DataUnit('bool', (), None, 'Task')],
        'notype': [

            DataUnit('str', (), None, 'filename', is_id=True),
            DataUnit('int', (), None, 'Time',is_id=True),
            DataUnit('float', (), None, 'AccV'),
            DataUnit('float', (), None, 'AccML'),
            DataUnit('float', (), None, 'AccAP'),
            DataUnit('float', (), None, 'Event'),
            DataUnit('bool', (), None, 'Valid'),
            DataUnit('bool', (), None, 'Task')],
        'tdcsfog': [

            DataUnit('str', (), None, 'filename', is_id=True),
            DataUnit('int', (), None, 'Time',is_id=True),
            DataUnit('float', (), None, 'AccV'),
            DataUnit('float', (), None, 'AccML'),
            DataUnit('float', (), None, 'AccAP'),
            DataUnit('float', (), None, 'StartHesitation'),
            DataUnit('float', (), None, 'Turn'),
            DataUnit('float', (), None, 'Walking')]

    }

data_schema_output = [
    DataUnit('str', (), None, 'filename', is_id=True),
    DataUnit('int', (), None, 'Time', is_id=True),
    DataUnit('float', (), None, 'StartHesitation'),
    DataUnit('float', (), None, 'Turn'),]

agent_router = [{'DenseScrable': ''}]


data_schema_input = [
                     DataUnit('str', (), None, 'Id',is_id=True),
                    DataUnit('float', (), None, 'AB',),
                    DataUnit('float', (), None, 'AF',),
                    DataUnit('float', (), None, 'AH',),
                    DataUnit('float', (), None, 'AM',),
                    DataUnit('float', (), None, 'AR',),
                    DataUnit('float', (), None, 'AX',),
                    DataUnit('float', (), None, 'AY',),
                    DataUnit('float', (), None, 'AZ',),
                    DataUnit('float', (), None, 'BC',),
                    DataUnit('float', (), None, 'BD' ,),
                    DataUnit('float', (), None, 'BN',),
                    DataUnit('float', (), None, 'BP',),
                    DataUnit('float', (), None, 'BQ',),
                    DataUnit('float', (), None, 'BR',),
                    DataUnit('float', (), None, 'BZ',),
                    DataUnit('float', (), None, 'CB',),
                    DataUnit('float', (), None, 'CC',),
                    DataUnit('float', (), None, 'CD' ,),
                    DataUnit('float', (), None, 'CF',),
                    DataUnit('float', (), None, 'CH',),
                    DataUnit('float', (), None, 'CL',),
                    DataUnit('float', (), None, 'CR',),
                    DataUnit('float', (), None, 'CS',),
                    DataUnit('float', (), None, 'CU',),
                    DataUnit('float', (), None, 'CW' ,),
                    DataUnit('float', (), None, 'DA',),
                    DataUnit('float', (), None, 'DE',),
                    DataUnit('float', (), None, 'DF',),
                    DataUnit('float', (), None, 'DH',),
                    DataUnit('float', (), None, 'DI',),
                    DataUnit('float', (), None, 'DL',),
                    DataUnit('float', (), None, 'DN',),
                    DataUnit('float', (), None, 'DU',),
                    DataUnit('float', (), None, 'DV',),
                    DataUnit('float', (), None, 'DY',),
                    DataUnit('float', (), None, 'EB',),
                    DataUnit('float', (), None, 'EE',),
                    DataUnit('float', (), None, 'EG',),
                    DataUnit('float', (), None, 'EH',),
                    DataUnit('str', (), None, 'EJ',),
                    DataUnit('float', (), None, 'EL',),
                    DataUnit('float', (), None, 'EP',),
                    DataUnit('float', (), None, 'EU',),
                    DataUnit('float', (), None, 'FC',),
                    DataUnit('float', (), None, 'FD' ,),
                    DataUnit('float', (), None, 'FE',),
                    DataUnit('float', (), None, 'FI',),
                    DataUnit('float', (), None, 'FL',),
                    DataUnit('float', (), None, 'FR',),
                    DataUnit('float', (), None, 'FS',),
                    DataUnit('float', (), None, 'GB',),
                    DataUnit('float', (), None, 'GE',),
                    DataUnit('float', (), None, 'GF',),
                    DataUnit('float', (), None, 'GH',),
                    DataUnit('float', (), None, 'GI',),
                    DataUnit('float', (), None, 'GL',),
                   ]

data_schema_output = [
                DataUnit('str', (), None, 'Id',is_id=True),
                DataUnit('int', (), None, 'class_0',load_name='DL'),
                DataUnit('int', (), None, 'class_1',load_name='FR')]
#tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
agent_router = [{'DenseScrable':{'inputs':['Image','Id'],
                               'outputs':[{'name':'Image','type':IMAGE}]}}]
                                                """

target_type = CATEGORY


data_schema_input = [
                     DataUnit('str', (), None, 'Id',is_id=True),
                    DataUnit('float', (), None, 'AB',),
                    DataUnit('float', (), None, 'AF',),
                    DataUnit('float', (), None, 'AH',),
                    DataUnit('float', (), None, 'AM',),
                    DataUnit('float', (), None, 'AR',),
                    DataUnit('float', (), None, 'AX',),
                    DataUnit('float', (), None, 'AY',),
                    DataUnit('float', (), None, 'AZ',),
                    DataUnit('float', (), None, 'BC',),
                    DataUnit('float', (), None, 'BD' ,),
                    DataUnit('float', (), None, 'BN',),
                    DataUnit('float', (), None, 'BP',),
                    DataUnit('float', (), None, 'BQ',),
                    DataUnit('float', (), None, 'BR',),
                    DataUnit('float', (), None, 'BZ',),
                    DataUnit('float', (), None, 'CB',),
                    DataUnit('float', (), None, 'CC',),
                    DataUnit('float', (), None, 'CD' ,),
                    DataUnit('float', (), None, 'CF',),
                    DataUnit('float', (), None, 'CH',),
                    DataUnit('float', (), None, 'CL',),
                    DataUnit('float', (), None, 'CR',),
                    DataUnit('float', (), None, 'CS',),
                    DataUnit('float', (), None, 'CU',),
                    DataUnit('float', (), None, 'CW' ,),
                    DataUnit('float', (), None, 'DA',),
                    DataUnit('float', (), None, 'DE',),
                    DataUnit('float', (), None, 'DF',),
                    DataUnit('float', (), None, 'DH',),
                    DataUnit('float', (), None, 'DI',),
                    DataUnit('float', (), None, 'DL',),
                    DataUnit('float', (), None, 'DN',),
                    DataUnit('float', (), None, 'DU',),
                    DataUnit('float', (), None, 'DV',),
                    DataUnit('float', (), None, 'DY',),
                    DataUnit('float', (), None, 'EB',),
                    DataUnit('float', (), None, 'EE',),
                    DataUnit('float', (), None, 'EG',),
                    DataUnit('float', (), None, 'EH',),
                    DataUnit('str', (), None, 'EJ',),
                    DataUnit('float', (), None, 'EL',),
                    DataUnit('float', (), None, 'EP',),
                    DataUnit('float', (), None, 'EU',),
                    DataUnit('float', (), None, 'FC',),
                    DataUnit('float', (), None, 'FD' ,),
                    DataUnit('float', (), None, 'FE',),
                    DataUnit('float', (), None, 'FI',),
                    DataUnit('float', (), None, 'FL',),
                    DataUnit('float', (), None, 'FR',),
                    DataUnit('float', (), None, 'FS',),
                    DataUnit('float', (), None, 'GB',),
                    DataUnit('float', (), None, 'GE',),
                    DataUnit('float', (), None, 'GF',),
                    DataUnit('float', (), None, 'GH',),
                    DataUnit('float', (), None, 'GI',),
                    DataUnit('float', (), None, 'GL',),
                   ]

data_schema_output = [
                DataUnit('str', (), None, 'Id',is_id=True),
                DataUnit('int', (), None, 'DL',load_name='DL'),
                DataUnit('int', (), None, 'FR',load_name='FR')]
#tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
agent_router = [{'DenseScrable':{'inputs':['Image','Id'],
                               'outputs':[{'name':'Image','type':IMAGE}]}}]
# MAKE A ARCH SEARCH OR SOMETHING OTHER SEARCH BASED ON GENETIC ALGORITHM SO THE PC WILL EXPLORE WHILE YOU ARE GONE
def runner(dataset_path, train_name='train', restrict=True, \
           size=10, target_name='letter', no_ids=False,
           data_schema_input=data_schema_input,
           data_schema_output=data_schema_output,
           submit_file='test',
           train_file='train',
           split=True, THREAD_COUNT=32, dir_tree=True,
           utils_name='utils'):
    exec('from utils.' + utils_name + ' import image_loader')
    print('data_schema_input 1',data_schema_input)
    image_loader = importlib.import_module('utils.' + utils_name, package='.').image_loader
    print(image_loader)
    image_collection_train, image_collection_test = image_loader(dataset_path
                                                                 , train_name=train_file, restrict=restrict, \
                                                                 size=size, target_name='letter', no_ids=False,
                                                                 data_schema_input=data_schema_input,
                                                                 data_schema_output=data_schema_output,
                                                                 split=split, THREAD_COUNT_V=THREAD_COUNT,
                                                                 dir_tree=dir_tree)
    print('data_bundle_list',len(image_collection_train['image_arr']))
    print('DATA COLLECTED')
    arbiter = Arbiter(data_schema_input=data_schema_input,
                      data_schema_output=data_schema_output, target_type=target_type,
                      class_num=image_collection_train['num_classes'],
                      router_agent=agent_router, skip_arbiter=True)
    # print(type(image_collection_train['image_arr']))
    # print(image_collection_train['image_arr'][0])
    # exit(0)

    if type(image_collection_train['image_arr']) == type([]):
        for i, element in zip(range(len(image_collection_train['image_arr'])),image_collection_train['image_arr']):
            arbiter.add_bundle_bucket(i,element)
    else:
     for key in list(image_collection_train['image_arr'].keys()):
        arbiter.add_bundle_bucket(key, image_collection_train['image_arr'][key])

    # print(image_collection_train)
    arbiter.normalize_bundle_bucket()
    for i in range(2000):
        arbiter.train(force_train=True, train_arbiter=False)
    arbiter.save()
    arbiter.empty_bucket()

    image_collection_train, image_collection_test = image_loader(dataset_path
                                                                 , train_name=submit_file, restrict=restrict, \
                                                                 size=size, target_name='letter', no_ids=False,
                                                                 data_schema_input=data_schema_input,
                                                                 data_schema_output=data_schema_output,
                                                                 split=split, THREAD_COUNT_V=THREAD_COUNT,
                                                                 dir_tree=dir_tree)
    for key in list(image_collection_train['image_arr'].keys())[:100]:
        arbiter.add_bundle_bucket(key, image_collection_train['image_arr'][key])
    arbiter.normalize_bundle_bucket(is_submit=True)
    arbiter.submit('/kaggle/working/')
