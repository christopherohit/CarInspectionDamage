# coding=utf-8
from RestructModelings.data.catalog import Metadata
from src.utilsV3.visualDamage import visualBoxCarPart
from src.utilsV3.toOutputs import toBase64HardScores
from src.utilsV3.utils import cropImage, requestParser, concatenateImage
from utils.util import base64_to_pil
from flask import jsonify
import time
from flask_restful import Resource
from PIL import ImageFile
from src.utilsV3.config import configModelCarPart

ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = requestParser()
predictor = configModelCarPart()
# metaDataCarPart = Metadata()
# metaDataCarPart.set(thing_classes = ['Cản sau', 'Kính sau', 'Cửa trái sau', 'Đèn sau trái', 'Cửa sau phải', 'Đèn sau phải',
#                                      'Cản trước', 'Kính trước', 'Cửa trái trước', 'Đèn trước trái', 'Cửa trước phải',
#                                      'Đèn trước phải', 'Mui xe', 'Gương trái', 'Gương phải', 'Cửa sau', 'Trucks', 'Bánh xe'])

def OriginalImage(out, t0, ImageOriginal = None, xminLocation = None, yminLocation = None, fromCarDetect = False):
    """
    It takes the output of the model, the original image, and the time it took to process the image, and
    returns the scores, the image, and the time it took to process the image
    :param outputs: the output of the model
    :param out: the image that was processed
    :param t0: the time when the image was uploaded
    :return: the scores, the image as a base64 string, and the time it took to process the image.
    """
    #scores = outputs["instances"].to("cpu").scores if outputs["instances"].to("cpu").has("scores") else None
    # scores_pre = scores.numpy()[0].split(' ')
    # print(scores.numpy().tolist())
    # output = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    output = out.get_image()
    if xminLocation != None:
        ImageConcatenate = concatenateImage(cropImage= output, originalImage= ImageOriginal,
                                            xmin= xminLocation, ymin= yminLocation, delay= 2)
        # cv2.imwrite(os.path.join(app.config["IMAGE_ALIGN"], filename), output)
        ImageBase64 = toBase64HardScores(ImageConcatenate)
        TimeProcess = time.time() - t0
    else:
        ImageBase64 = toBase64HardScores(output)
        TimeProcess = time.time() - t0
    return ImageBase64, TimeProcess

def CarPartObjectBoundingBox(ImageArray, start, xminLocation = None, yminLocation = None, ImageOriginal = None, fromCarDetect = False):
    """
    It takes an image, runs it through a model, and returns a dictionary containing the predicted
    classes and their corresponding scores
    
    :param ImageArray: The image you want to predict
    :param predictor: the predictor object that we created earlier
    :param start: the time when the function is called
    :param xminLocation: The x-coordinate of the top left corner of the bounding box
    :param yminLocation: The y-coordinate of the top-left corner of the bounding box
    :param ImageOriginal: The original image that the user uploaded
    """
    outputs = predictor(ImageArray)
    if fromCarDetect == True:
        if len(outputs["instances"].to('cpu').pred_classes.numpy().tolist()) == 0:
            return None
        else:
            data = {'numObject' : len(outputs["instances"].to('cpu').pred_classes.numpy().tolist()),
                    'rc' : '00p00', # Thông qua bằng đường phụ tránh bị reject
                    'msg' : "Warning: We don't detect any car but we recognize some part of car"}
            return data
    else:
        if len(outputs["instances"].to('cpu').pred_classes.numpy().tolist()) == 0:
            if yminLocation == None:
                ImageArray = toBase64HardScores(ImageArray)
                data = {'rc': '00p01',
                        'msg': "Don't detect any Part of Cars\nPlease zoom out or move camera to get a general view.", 
                        'Base64Image': ImageArray}
            else:
                ImageOriginal = toBase64HardScores(ImageOriginal)
                data = {'rc': '00p01',
                        'msg': "Don't detect any Part of Cars\nPlease zoom or move camera to get a general view", 
                        'Base64Image': ImageOriginal}
        else:
            if xminLocation == None:
                VisualizeData = visualBoxCarPart(image = ImageArray, partLocation= outputs, metaData= None)
                ImgBase64, TimeProcess = OriginalImage(out= VisualizeData,
                                                       t0= start)
                # ImgBase64 = ImgBase64.decode('utf8')
                data = {'rc': '00p00', 'msg':'Đã xác định được bộ phân của xe',
                        'PartsCar' : outputs["instances"].to('cpu').pred_classes.numpy().tolist(), 
                        'Base64Image' : ImgBase64,
                        'CoordinateXY' : outputs["instances"].to('cpu').pred_boxes.tensor.numpy().tolist(),
                        'TimeProcess' : TimeProcess}
            else:
                VisualizeData = visualBoxCarPart(image = ImageArray, partLocation= outputs, metaData= None)
                ImgBase64, TimeProcess = OriginalImage(out= VisualizeData, t0= start,
                                                       ImageOriginal= ImageOriginal, xminLocation= xminLocation,
                                                       yminLocation= yminLocation)
                # ImgBase64 = ImgBase64.decode('utf8')
                data = {'rc': '00p02', 'msg':'Đã xác định được bộ phân của xe',
                        'PartsCar' : outputs["instances"].to('cpu').pred_classes.numpy().tolist(), 
                        'Base64Image' : ImgBase64,
                        'CoordinateXY' : outputs["instances"].to('cpu').pred_boxes.tensor.numpy().tolist(),
                        'TimeProcess' : TimeProcess}
        return data

class CarPartDetect(Resource):
    def post(self):
        args = parser.parse_args()
        if args['data'] != None:
            start = time.time()
            base64 = args['data']
            if args['xmaxLocation'] != None:
                xminLocation = int(float(args['xminLocation']))
                xmaxLocation = int(float(args['xmaxLocation']))
                yminLocation = int(float(args['yminLocation']))
                ymaxLocation = int(float(args['ymaxLocation']))
                imgCV = base64_to_pil(img_base64= base64, type= 1, ext= False)
                ImageCrop = cropImage(xmin= xminLocation, ymin= yminLocation,
                                      xmax= xmaxLocation, ymax= ymaxLocation)
                finalResultCarPart = CarPartObjectBoundingBox(ImageArray= ImageCrop, start= start,
                                                        xminLocation= xminLocation, yminLocation= yminLocation,
                                                        ImageOriginal=imgCV)
                return jsonify(finalResultCarPart)
            else:
                imgCV = base64_to_pil(base64, type= 1 , ext= True)
                FinalResult = CarPartObjectBoundingBox(ImageArray= imgCV, start= start)
                return jsonify(FinalResult)
        else:
            data = {'rc' : '00x0001',
                    'msg' : 'Not found Image'}
            return jsonify(data)
    def get():
        return CarPartDetect.post()