from utils.util import base64_to_pil
from flask import jsonify
from flask_restful import Resource
import time
from src.utilsV3.utils import processImageSimple, requestParser
from src.utilsV3.toOutputs import toBase64Simple
from src.utilsV3.config import FullConfigToSet, loadModelSimple
FullConfigToSet()

MetaData_Corner = ['Right', 'Behind', 'Right Behind', 'Left Behind', 'Left', 'Front', 'Right Front', 'Left Front']
ParamParser = requestParser()
model = loadModelSimple(whatClassify= 'Corner'.capitalize(), version='New',
                        type= 'tensorflow')

def preprocessCornerImage(image):
    """
    > It takes an image, resizes it to 224x224, and returns the image as a numpy array
    
    :param image: The image to be processed
    :return: The image is being returned as a numpy array.
    """
    start = time.time()
    x = processImageSimple(image= image, size= (224, 224))
    return x , start

def GetCornerCar(x ,start):
    """
    It takes in an image, and then returns a dictionary with the
    class of the image and how long it took to classify the image
    
    :param x: The input image
    :param start: the time when the function was called
    """
    output = model.predict(x)
    ClassCorner = output[0].tolist().index(max(output[0]))
    data = {'rc': '00v00', 'msg':'Đã xác định được vị trí góc xe',
            'Corner-Class' : MetaData_Corner[ClassCorner] , 'TimeProcess' : (time.time() - start)}
    return  data



# The class is called ClassCorner, and it inherits from the Resource class. 
# The post method is called when a POST request is made to the server. 
# The get method is called when a GET request is made to the server. 
# The parser.parse_args() method parses the request and returns the data. 
# The jsonify method returns the data in JSON format. 
# The base64_to_pil method converts the base64 image to a PIL image. 
# The Classify_Corner method classifies the image and returns the result. 
# The ModelClassifyCorner is the model that is used to classify the image. 
# The get method simply calls the post method. 
# The get method is used when the user makes a GET request to the server. 
# The post method is used when the user makes a POST request to the
class ClassCorner(Resource):
    def post(self):
        """
        It takes a PIL image, converts it to a numpy array, and then converts that array into a base64
        string
        :return: The return is a json object with the following keys:
            - rc: return code
            - msg: return message
            - Base64Image: the base64 encoded image
            - CornerIs: the cornerIs array
            - time: the time it took to process the image
        """
        args = ParamParser.parse_args()
        if args['data'] != None:
            base64image = args['data']
            imgPIL  = base64_to_pil(base64image,  type= 2)
            imgPIL = imgPIL.convert('RGB')
            CornerIs, time = preprocessCornerImage(imgPIL)
            CornerIs = GetCornerCar(CornerIs, time)
            base64toOutputs = toBase64Simple(imageArray= imgPIL)
            CornerIs['Base64Image'] = base64toOutputs
            return jsonify(CornerIs)
        else:
            data = {'rc' : '00x0001',
                    'msg' : 'Not found Image'}
            return jsonify(data)

    def get():
        """
        > The function `get()` returns the value returned by the function `post()` in the class
        `ClassCorner`
        :return: The function get() is being returned.
        """
        return ClassCorner.post()
