import time
from RestructModelings.data.catalog import Metadata
from src.utilsV3.utils import cropImage, cropImageByList, toListLocatetion, requestParser, concatenateImage
from src.utilsV3.visualDamage import visualOnlyMaskDamage
from src.trainCarObject import ObjectDetect_Car
from src.mainCodeCarPart import CarPartObjectBoundingBox
from src.utilsV3.config import loadModelSimple, configModelCarPart, configModelSegmentation
from flask_restful import Resource
from PIL import ImageFile
from flask import jsonify
from utils.util import base64_to_pil, getPoint
from src.utilsV3.toOutputs import toBase64HardScores

ImageFile.LOAD_TRUNCATED_IMAGES = True
modelDetectCar = loadModelSimple(whatClassify= None, version= '5x', type = 'yolo')
parser = requestParser()

modelDetectCarPart = configModelCarPart()
modelSegmentation = configModelSegmentation()
MetaData_PhysicalDamage = Metadata()
MetaData_PhysicalDamage.set(thing_classes = ['Móp lõm', 'Trầy sơn', 'Rách', 'Mất bộ phận', 'Thủng', 'Bể đèn', 'Vỡ kính'])

def predict(img):
    """
    It takes an image as input and returns a segmentation mask
    
    :param img: The image to be segmented
    :return: The modelSegmentation function is being returned.
    """
    return modelSegmentation(img)



def processObject(image):
    start = time.time()
    dictDetectCar = ObjectDetect_Car(VariableImage= image, model= modelDetectCar)
    array = []
    if dictDetectCar['rc'] == '00t00':
        cropCar = cropImage(xmin= dictDetectCar['xminLocation'], ymin= dictDetectCar['yminLocation'],
                            xmax= dictDetectCar['xmaxLocation'], ymax= dictDetectCar['ymaxLocation'],
                            imageOriginal= image)
        outCarPart = CarPartObjectBoundingBox(ImageArray= cropCar, start= start)
        if outCarPart['rc'] == '00p001':
            return outCarPart
        else:
            xmin, xmax, ymin, ymax = getPoint(outCarPart)
            cropPart = cropImageByList(imageOriginal= cropCar,
                       xmin= toListLocatetion(data= outCarPart['CoordinateXY'], attemp=0),
                       xmax= toListLocatetion(data= outCarPart['CoordinateXY'], attemp=2),
                       ymin= toListLocatetion(data= outCarPart['CoordinateXY'], attemp=1),
                       ymax= toListLocatetion(data= outCarPart['CoordinateXY'], attemp=3))
            for i in range(len(cropPart)):
                out = predict(cropPart[i])
                if len(out['instances'].to('cpu').pred_classes.numpy().tolist()) == 0:
                    array.append('None detect')
                else:
                    toarray = visualOnlyMaskDamage(image= cropPart[i], damageLocation= out,
                                                   metaData= MetaData_PhysicalDamage, get_image= True)
                    array.append(toarray)
            for i in range(len(array)):
                if array[i] != 'None detect':
                    imageMainObject = concatenateImage(cropImage= array[i], originalImage=cropCar,
                                             xmin=xmin[i], ymin=ymin[i], delay= 0)
                else:
                    pass
            Lastresult = concatenateImage(cropImage= imageMainObject, originalImage= image,
                                          xmin= dictDetectCar['xminLocation'],
                                          ymin= dictDetectCar['yminLocation'],
                                          delay= 0)
            imageBase64 = toBase64HardScores(Lastresult)
            data = {
                    'Base64Image' : imageBase64,
                    'locationxmin' : xmin,
                    'locationxmax' : xmax,
                    'locationymin' : ymin,
                    'locationymax' : ymax
                   }
            return data
    else:
        return dictDetectCar

class VersionIsNew(Resource):
    def post(self):
        args = parser.parse_args()
        if args['data'] != None:
            base64 = args['data']
            imgCV = base64_to_pil(img_base64= base64, type= 1, ext= False)
            result = processObject(image= imgCV)
            return jsonify(result)
        else:
            raise Exception('Value Error: Not found any image in param')
    def get():
        return VersionIsNew.post()