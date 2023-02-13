from utils.util import base64_to_pil, estimate_blur
from flask import jsonify
from flask_restful import Resource
from PIL import ImageFile
from src.utilsV3.toOutputs import toBase64HardScores
from src.utilsV3.utils import requestParser


ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = requestParser()

def CheckBlurImage(imgArray, type = 'web'):
    """
    It takes an image array, converts it to base64, and then returns a dictionary with the score, the
    blurriness, and the base64 image
    
    :param imgArray: The image array that you want to check for blurriness
    :return: The data is being returned as a dictionary.
    """
    score, blurry = estimate_blur(imgArray, 100)
    base64toOutputs = toBase64HardScores(imageArray= imgArray)
    if type == 'web':
        data = {
            'Score' : score,
            'scoreBlur' : blurry,
            'Base64Image' : base64toOutputs
        }   
        return data
    if type == 'app':
        return blurry

class CheckBlur(Resource):
    def post(self):
        """
        It takes the image as a base64 string and returns the blurriness of the image.
        :return: The data is being returned as a json object.
        """
        args = parser.parse_args()
        if args['data'] != None:
            base64 = args['data']
            imgCV = base64_to_pil(img_base64= base64, type= 1, ext= False)
            data = CheckBlurImage(imgCV)
            return jsonify(data)    
    def get():
        """
        > The function `get()` returns the result of the function `post()` in the class `CheckBlur`
        :return: The post method is being returned.
        """
        return CheckBlur.post()