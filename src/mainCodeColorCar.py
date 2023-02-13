# coding=utf-8
from utils.util import base64_to_pil
from flask import jsonify
from flask_restful import Resource
import time
from src.utilsV3.utils import processImageSimple, requestParser
from src.utilsV3.config import FullConfigToSet, loadModelSimple

FullConfigToSet()
parser = requestParser()
model = loadModelSimple(whatClassify= 'Color'.capitalize(), version= None, type = 'tensorflow')
ColorCar = ['Beige', 'Black', 'Blue', 'Brown', 'Gold', 'Green', 'Grey',
            'Orange', 'Pink', 'Purple', 'Red', 'Silver', 'Tan', 'White', 'Yellow']

def preprocessColorImage(image):
    """
    > It takes an image, resizes it to 299x299, and returns the image
    
    :param image: The image to be processed
    :return: The image is being returned as a numpy array.
    """
    start = time.time()
    x = processImageSimple(image= image, size= (299,299))
    return x , start


def GetResultColor(x, timeProcess):
    """
    It takes an image, resizes it, converts it to a numpy array, and then feeds it to the model
    
    :param x: The input image
    :param timeProcess: The time the program starts running
    """
    output = model.predict(x)
    ClassColor = output[0].tolist().index(max(output[0]))
    data = {'rc' : '00c00', 'msg' : 'Đã xác định được màu sắc của xe',
            'ColorClass' : ColorCar[ClassColor],
            'Time Process': (time.time() - timeProcess)}
    return data


# It takes a base64 encoded image, converts it to a PIL image, preprocesses it, and returns the result
# of the preprocessed image
class ColorClassify(Resource):

    def post(self):
        args = parser.parse_args()
        if args['data'] != None:
            base64 = args['data']
            imgPIL  = base64_to_pil(base64,  type= 2)
            ColorIs, time = preprocessColorImage(imgPIL)
            ColorIs = GetResultColor(ColorIs, time)
            return jsonify(ColorIs)
        else:
            data = {'rc' : '00x0001',
                    'msg' : 'Not found Image'}
            return jsonify(data)
            
    def get():
        """
        It returns the output of the post() function in the ColorClassify class
        :return: The return value is the result of the post() method.
        """
        return ColorClassify.post()