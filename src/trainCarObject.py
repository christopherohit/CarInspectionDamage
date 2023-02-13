# coding=utf-8
from PIL import ImageFile
import time
from RestructModelings.data.catalog import Metadata
from flask import jsonify
from utils.util import base64_to_pil
from src.utilsV3.config import loadModelSimple, configModelCarObject
from src.utilsV3.utils import CalculateAreaByBoxBaseCoordinateDataFrame, requestParser
from src.mainCodeCarPart import CarPartObjectBoundingBox
from src.utilsV3.toOutputs import toBase64HardScores
from flask_restful import Resource
ImageFile.LOAD_TRUNCATED_IMAGES = True
metaDataCarPart = Metadata()
metaDataCarPart.set(thing_classes = ['Cản sau', 'Kính sau', 'Cửa trái sau', 'Đèn sau trái', 'Cửa sau phải', 'Đèn sau phải',
                                     'Cản trước', 'Kính trước', 'Cửa trái trước', 'Đèn trước trái', 'Cửa trước phải',
                                     'Đèn trước phải', 'Mui xe', 'Gương trái', 'Gương phải', 'Cửa sau', 'Trucks', 'Bánh xe'])
model = loadModelSimple(whatClassify= None, version= '5x', type = 'yolo')
paramParser = requestParser()
predictor = configModelCarObject()

def ObjectDetect_Car(VariableImage, model):
    """
    The function takes an image as input, and returns a dictionary containing the number of cars in the
    image, the size of the largest car, the accuracy of the detection, and the image itself
    :param VariableImage: Image to detect
    """
    result = model(VariableImage, size = 704)
    PandasToCalculateObject = result.pandas().xyxy[0]
    ImageToArray = result.render()[0]
    SizeImage = ImageToArray.shape
    TotalArea, Accuracy = CalculateAreaByBoxBaseCoordinateDataFrame(pandas_dataframe= PandasToCalculateObject, sizeImage= SizeImage)
    NumObject = len(TotalArea)
    lastImage = toBase64HardScores(imageArray= ImageToArray)
    if NumObject == 0:
        start = time.time()
        GetCarPart = CarPartObjectBoundingBox(ImageArray= ImageToArray, start= start, fromCarDetect= True) 
        if GetCarPart == None:
            data = {'rc': '00t01', 'message': "Don't Detect any car in image" ,
                    'Base64Image' : lastImage}
        else:
            data = GetCarPart
            data['Base64Image'] = lastImage
    else:
        MaxArea = max(TotalArea)
        PlaceInhold = TotalArea.index(MaxArea)
        lastImage = toBase64HardScores(imageArray= ImageToArray)
        data = {'rc': '00t00', 'Car in Image' : NumObject, 'Size Object Image' : TotalArea[PlaceInhold], 'Accuracy' : Accuracy[PlaceInhold] , 'Base64Image' : lastImage,
                'xminLocation' : PandasToCalculateObject['xmin'][PlaceInhold], 'xmaxLocation' : PandasToCalculateObject['xmax'][PlaceInhold],
                'yminLocation' : PandasToCalculateObject['ymin'][PlaceInhold], 'ymaxLocation' : PandasToCalculateObject['ymax'][PlaceInhold]}
    return data
    
# The class Calculate_AreaDetect is a subclass of the Resource class. It has two methods: post and
# get. The post method is called when a POST request is made to the server. The get method is called
# when a GET request is made to the server
# It takes a base64 string, converts it to a PIL image, then passes it to the ObjectDetect_Car
# function
class Calculate_AreaDetect(Resource):
    
    def post(seft):
        """
        :param seft: The first parameter is the class itself
        :return: The result is a list of dictionaries. Each dictionary contains the following keys:
            - 'box': the bounding box of the object
            - 'label': the label of the object
            - 'confidence': the confidence of the object
            - 'color': the color of the object
            - 'type': the type of the object
            - 'id': the
        """
        args = paramParser.parse_args()
        if args['data'] != None:
            base64 = args['data']
            imgCV = base64_to_pil(base64, type= 1 , ext= False)
            result = ObjectDetect_Car(imgCV, model= model)
            return jsonify(result)

    def get():
        """
        It takes the output of the function post() in the Calculate_AreaDetect.py file and returns it
        :return: The function post() is being returned.
        """
        return Calculate_AreaDetect.post()
