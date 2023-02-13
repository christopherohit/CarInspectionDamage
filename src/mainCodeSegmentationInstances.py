# coding=utf-8
from src.utilsV3.toOutputs import toBase64HardScores
from src.utilsV3.utils import concatenateImage, requestParser
from RestructModelings.data.catalog import Metadata
from utils.util import base64_to_pil
from flask import jsonify
import time
from src.utilsV3.config import configModelSegmentation
from src.utilsV3.visualDamage import visualBothDamage
from flask_restful import Resource
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


ParamParser = requestParser()

predictor = configModelSegmentation()
MetaData_PhysicalDamage = Metadata()
MetaData_PhysicalDamage.set(thing_classes = ['Móp lõm', 'Trầy sơn', 'Rách', 'Mất bộ phận', 'Thủng', 'Bể đèn', 'Vỡ kính'])

#region Function process private PhysicalDamage
def OriginalImg(outputs, out, t0, ImageOriginal = None, xminLocation = None, yminLocation = None):
    """
    It takes the output of the model, the original image, the time it took to process the image, and the
    location of the crop image, and returns the scores, the image in base64, and the time it took to
    process the image
    
    :param outputs: The output of the model
    :param out: the output of the model
    :param t0: the time when the image was received
    :param ImageOriginal: The original image that is sent to the server
    :param xminLocation: xmin of the bounding box
    :param yminLocation: ymin of the bounding box
    :return: The scores, the image in base64, and the time it took to process the image.
    """
    scores = outputs["instances"].to("cpu").scores if outputs["instances"].to("cpu").has("scores") else None
    # scores_pre = scores.numpy()[0].split(' ')
    # print(scores.numpy().tolist())
    # output = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    output = out.get_image()
    if xminLocation != None:
        ImageConcatenate = concatenateImage(cropImage= output, originalImage= ImageOriginal,
                                            xmin= xminLocation, ymin= yminLocation, delay= 0)
        ImageBase64 = toBase64HardScores(ImageConcatenate)
        TimeProcess = time.time() - t0
    else:
        ImageBase64 = toBase64HardScores(output)
        TimeProcess = time.time() - t0
    return scores, ImageBase64, TimeProcess

def PredictSegmentationInstances(ImageArray, predictor, start, xminLocation = None, yminLocation = None, ImageOriginal = None):
    """
    It takes an image, runs it through the model, and returns a dictionary containing the predicted
    class, the image, and the time it took to process the image
    
    :param ImageArray: The image you want to predict
    :param predictor: the predictor object that is used to run inference on the image
    :param start: the time when the function is called
    :param xminLocation: The x-coordinate of the top left corner of the image
    :param yminLocation: The y-coordinate of the top left corner of the image
    :param ImageOriginal: The original image that the user uploaded
    :return: a dictionary with the following keys:
        rc: return code
        msg: message
        Class: list of class names
        Base64Image: base64 encoded image
        TimeProcess: time taken to process the image
    """
    outputs = predictor(ImageArray)
    if len(outputs["instances"].to('cpu').pred_classes.numpy().tolist()) == 0:
        if xminLocation == None:
            ImageArray = toBase64HardScores(imageArray= ImageArray)
            data = {'rc': '00d01',
                    'msg': "Don't detect any Damage In Image\nPlease zoom or move to Damage position.", 
                    'Base64Image': ImageArray}
        else:
            ImageArray = toBase64HardScores(imageArray= ImageOriginal)
            data = {'rc': '00d01',
                    'msg': "Don't detect any Damage In Image\nPlease zoom or move to Damage position.", 
                    'Base64Image': ImageArray}
    else:
        if xminLocation == None:
            VisualizeData = visualBothDamage(damageLocation= outputs , image= ImageArray, metaData= MetaData_PhysicalDamage)
            Scores, ImgBase64, TimeProcess = OriginalImg(outputs= outputs, out= VisualizeData,
                                                         t0= start)
            data = {'rc': '00p00', 'msg':'Đã xác định được tổn thất',
                    'Scores' : Scores.numpy().tolist(), 'Base64Image' : ImgBase64,
                    'TimeProcess' : TimeProcess,
'Class': outputs["instances"].to('cpu').pred_classes.numpy().tolist()}
        else:
            VisualizeData = visualBothDamage(damageLocation= outputs, image= ImageArray, metaData= MetaData_PhysicalDamage)
            Scores, ImgBase64, TimeProcess = OriginalImg(outputs= outputs, out= VisualizeData, t0= start,
                                                         ImageOriginal= ImageOriginal, xminLocation= xminLocation,
                                                         yminLocation= yminLocation)
            data = {'rc': '00p02', 'msg':'Đã xác định được tổn thất',
                    'Scores' : Scores.numpy().tolist(), 'Base64Image' : ImgBase64,
                    'TimeProcess' : TimeProcess,
'Class': outputs["instances"].to('cpu').pred_classes.numpy().tolist()}
    return data
#endregion

# The post() method takes in a base64 image, converts it to a PIL image, crops it to the bounding box
# coordinates, and then passes it to the PredictSegmentationInstances function
class Physical_Segmentations(Resource):
    def post(self):
        """
        The function takes in a base64 image, converts it to a PIL image, crops it to the bounding box
        coordinates, and then passes it to the PredictSegmentationInstances function
        :return: The return is a json object with the following keys:
            - rc: return code, 01 is success, 02 is failure
            - msg: message, if rc is 01, msg is "success", if rc is 02, msg is "failure"
            - Base64Image: the base64 string of the image
            - Base64ImageWithBox: the base64
        """
        args = ParamParser.parse_args()
        if args['data'] != None:
            start = time.time()
            base64 = args['data']
            if args['xmaxLocation'] != None:
                xminLocation = int(float(args['xminLocation']))
                xmaxLocation = int(float(args['xmaxLocation']))
                yminLocation = int(float(args['yminLocation']))
                ymaxLocation = int(float(args['ymaxLocation']))
                # sizeObject = int(float(args['SizeObjectImage']))
                # if sizeObject >= 90:
                #     imgCV  = base64_to_pil(base64, type= 1)
                #     FinalResult = PredictSegmentationInstances(ImageArray= imgCV, predictor= predictor, start = start, )
                #     return jsonify(FinalResult)
                # else:
                imgCV  = base64_to_pil(img_base64= base64, type= 1, ext= False)
                ImageCrop = imgCV[int(yminLocation):int(ymaxLocation), int(xminLocation):int(xmaxLocation)]
                FinalResult = PredictSegmentationInstances(ImageArray= ImageCrop, predictor= predictor, start= start,
                                                           xminLocation= xminLocation, yminLocation= yminLocation,
                                                           ImageOriginal=imgCV)
                return jsonify(FinalResult)
            else:
                imgCV  = base64_to_pil(base64, type= 1, ext= False)
                FinalResult = PredictSegmentationInstances(ImageArray= imgCV, predictor= predictor, start = start )
                return jsonify(FinalResult)
        else:
            data = {'rc' : '00x0001',
                    'msg' : 'Not found Image'}
            return jsonify(data)
    def get():
        """
        :return: The post() method of the Physical_Segmentations class.
        """
        return Physical_Segmentations.post()


