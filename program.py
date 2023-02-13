from src.mainBlurDetect import CheckBlurImage
# from src.mainCodeColorCar import *
from src.mainCodeCorner import *
from src.mainCodeSegmentationInstances import *
from utils.util import *
from src.trainCarObject import ObjectDetect_Car
import time
from src.utilsV3.utils import *
from src.utilsV3.config import *
from src.utilsV3.toOutputs import *

predictor = configModelSegmentation()
objectDetect = configModelCarObject()
if __name__ == "__main__":
    start = time.time()
    print('Check Physical Damage On Car')
    img = base64_to_pil(img_base64= str(input("Please paste Image by Base64 code: ")),
                        type= 1, ext= True)
    isCar, isPart = ObjectDetect_Car(VariableImage= img , model= objectDetect)
    if len(isCar) <= 3:
        print("Don't Detect any car in image")
    elif len(isCar) == 9:
        imgCrop = cropImage(xmin= isCar['xminLocation'], xmax= isCar['xmaxLocation'],
                            ymin= isCar['yminLocation'], ymax= isCar['ymaxLocation'],
                            imageOriginal= img)
        isBlur = CheckBlurImage(imgArray= imgCrop, type= 'app')
        if isBlur:
            print('This Image too Blur')
        else:
            FinalResult = PredictSegmentationInstances(ImageArray= imgCrop, start= start,
                                                       xminLocation= isCar['xminLocation'],
                                                       yminLocation= isCar['ymaxLocation'],
                                                       ImageOriginal= img, predictor= predictor)
            if len(FinalResult) == 3:
                print('Not Found any Damage')
            elif len(FinalResult) == 5:
                print(FinalResult['Base64Image'])
    else:
        isBlur = CheckBlurImage(imgArray= img, type= 'app')
        if isBlur:
            print('This Image too Blur')
        else:
            FinalResult = PredictSegmentationInstances(ImageArray= img, predictor= predictor,
                                                       start= start)