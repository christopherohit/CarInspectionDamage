# coding=utf-8
from RestructModelings import model_zoo
from RestructModelings.engine import DefaultPredictor
from RestructModelings.config import get_cfg
from RestructModelings.data.catalog import Metadata
from utils.util import base64_to_pil
from flask_restful import Resource
from src.utilsV3.config import loadModelSimple
from PIL import ImageFile
from src.utilsV3.visualDamage import visualOnlyMaskDamage
from src.utilsV3.utils import requestParser
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
List variable and datatype of variable

listPart : list
listLocation : list
imgCV : numpy.ndarray
arrayPartOnImage : list numpy.ndarray
out : numpy.ndarray
predictor : RestructModelings.engine.defaults.DefaultPredictor
cfg : RestructModelings..config.config.CfgNode
masks : torch.Tensor
v : RestructModelings.utils.visualizer.Visualizer
MetaData_PhysicalDamage : RestructModelings.data.catalog.Metadata
'''

parser = requestParser()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml'))
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGES = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
cfg.MODEL.RETINANET.NUM_CLASSES = 7
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = (loadModelSimple(whatClassify= 'damage', version= '59999', type = 'pytorch'))
cfg['MODEL']['DEVICE'] = 'cuda:0'
predictor = DefaultPredictor(cfg)
MetaData_PhysicalDamage = Metadata()
MetaData_PhysicalDamage.set(thing_classes = ['Móp lõm', 'Trầy sơn', 'Rách', 'Mất bộ phận', 'Thủng', 'Bể đèn', 'Vỡ kính'])


def predictSegmentationV3(ArrayImagePart):
    arrayDamageByPart = []
    for i in range(len(ArrayImagePart)):
        outputs = predictor(ArrayImagePart[i])
        if len(outputs["instances"].to('cpu').pred_classes.numpy().tolist()) == 0:
            VisualizeData = ['None find Damage']
        else:
            VisualizeData = visualOnlyMaskDamage(image= ArrayImagePart[i], damageLocation= outputs,
                                                 metaData= MetaData_PhysicalDamage, get_image= True)
        arrayDamageByPart.append(VisualizeData)
    return arrayDamageByPart

    
def ConvertCoordinateToArray(imgCV, lisLocationAttempt):
    arrayPart = []
    for i in range(len(lisLocationAttempt)):
        PartCrop = imgCV[int(lisLocationAttempt[i][1]):int(lisLocationAttempt[i][3]),
                         int(lisLocationAttempt[i][0]):int(lisLocationAttempt[i][2])]
        arrayPart.append(PartCrop)
    return arrayPart

class DetectDamageByPart(Resource):
    def post(self):
        args = parser.parse_args()
        if args['data'] != None:
            base64 = args['data']
            if args['listPart'] != None:
                listPart = args['listPart']
                listLocation = args['listlocationPart']
                imgCV = base64_to_pil(img_base64= base64, type= 1 , ext= False)
                arrayPartOnImage = ConvertCoordinateToArray(imgCV= imgCV, lisLocationAttempt= listLocation)
                arrayDamageByPart = predictSegmentationV3(arrayPartOnImage)
                


    def get():
        return DetectDamageByPart.post()
