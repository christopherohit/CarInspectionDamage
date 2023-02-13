from tensorflow.keras.models import load_model
from PIL import ImageFile
from RestructModelings import model_zoo
from RestructModelings.engine import DefaultPredictor
from RestructModelings.config import get_cfg
import tensorflow as tf
import torch

def FullConfigToSet():
    """
    > It sets the `LOAD_TRUNCATED_IMAGES` flag to `True` and then sets the visible devices to an empty
    list
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'

def loadModelSimple(whatClassify, version, type: str):
    """
    > This function loads a model of a specified type, and returns the model
    
    :param whatClassify: What are you classifying?
    :param version: the version of the model you want to load
    :param type: The type of model you want to load
    :type type: string
    :param metadata: This is the metadata of the model. It's a dictionary that contains the following
    keys:
    :return: The model is being returned.
    """
    announError = 'VersionError: Version which you load not compatible'
    allofTypeModel = ['yolo', 'tensorflow', 'pytorch']
    if type in allofTypeModel:
        if str(type).lower() == 'tensorflow':
            try:
                model = load_model(f'models/{str(type).lower()}/Classify{whatClassify}{version}.h5')
            except:
                raise Exception(announError)
        elif str(type).lower() == 'pytorch':
            try:
                if version == 'final':
                    model = f'models/{str(type).lower()}/{whatClassify}/model_final.pth'
                else:    
                    model = f"models/{str(type).lower()}/{whatClassify}/model_00{version}.pth" 
            except:
                raise Exception(announError)
        else:
            try:
                model = torch.hub.load('RestructYolo', 'custom', f'models/{str(type).lower()}/yolov{version}.pt', source= 'local')
                model.classes = [2, 7]
            except:
                raise Exception(announError)
        return model
    else:
        raise Exception('TypeError: Not found any type you request')

def configModelCarObject():
    """
    It loads the model from the model zoo, sets the threshold for the model to be confident enough to
    make a prediction, sets the number of classes to be predicted, and sets the device to be used for
    prediction
    :return: The predictor is being returned.
    """
    configCarParts = get_cfg()
    configCarParts.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    configCarParts.DATALOADER.NUM_WORKERS = 4
    configCarParts.MODEL.WEIGHTS =(loadModelSimple(whatClassify= 'carpath', version= 1,
                                                type = 'pytorch'))  # Let training initialize from model zoo
    configCarParts.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    configCarParts.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    configCarParts.MODEL.ROI_HEADS.NUM_CLASSES = 19
    configCarParts['MODEL']['DEVICE'] = 'cuda:0'
    predictor = DefaultPredictor(configCarParts)
    return predictor

def configModelSegmentation():
    """
    It loads the model from the model zoo, sets the number of workers to 4, sets the batch size to 128,
    sets the number of classes to 7, sets the score threshold to 0.7, and sets the device to cuda:0
    :return: The predictor is being returned.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml'))
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGES = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.RETINANET.NUM_CLASSES = 7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = (loadModelSimple(whatClassify= 'damage', version= 'final', type = 'pytorch'))
    cfg['MODEL']['DEVICE'] = 'cuda:0'
    predictor = DefaultPredictor(cfg)
    return predictor

def configModelCarPart():
    """
    It loads the model from the model zoo, sets the number of workers to 4, sets the score threshold to
    0.5, sets the batch size to 64, sets the number of classes to 19, and sets the device to cuda:0
    :return: The predictor is being returned.
    """
    configCarParts = get_cfg()
    configCarParts.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    configCarParts.DATALOADER.NUM_WORKERS = 4
    configCarParts.MODEL.WEIGHTS =("/home/ai_car/dev/models/pytorch/carpath/model_final.pth")  # Let training initialize from model zoo
    configCarParts.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    configCarParts.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    configCarParts.MODEL.ROI_HEADS.NUM_CLASSES = 19
    configCarParts['MODEL']['DEVICE'] = 'cuda:0'
    predictor = DefaultPredictor(configCarParts)
    return predictor