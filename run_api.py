# coding=utf-8
import os
# Main API Library
from flask import Flask
from flask_restful import Api
from src.mainBlurDetect import CheckBlur

# Main Config From model
from src.trainCarObject import Calculate_AreaDetect
from src.mainCodeCorner import ClassCorner
# from src.mainCodeColorCar import ColorClassify
from src.mainCodeSegmentationInstances import Physical_Segmentations
from src.mainCodeCarPart import CarPartDetect
from src.mainCodeMakeModel import Make_Model
from src.mainCodePriceEstimated import AIPredictCost
from src.utilsV3.utils import requestParser
from version3function.process import VersionIsNew

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
api = Api(app)


parser = requestParser()
api.add_resource(Calculate_AreaDetect, '/DetectCar')
api.add_resource(ClassCorner, '/ClassCorner')
# api.add_resource(ColorClassify, '/ColorCar')
api.add_resource(Physical_Segmentations, '/InstanceSegmentations')
api.add_resource(CarPartDetect, '/CarPartDetect')
api.add_resource(CheckBlur, '/CheckBlur')
api.add_resource(Make_Model,'/Make_Model')
api.add_resource(AIPredictCost, '/AIEstimate')
api.add_resource(VersionIsNew, '/VersionNew')

if __name__ == "__main__":
    port = os.environ['PORT']
    if (port == None):
        port = 4000
    app.run(debug=False, port=port)
