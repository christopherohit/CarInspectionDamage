'''
This file has functional code to process or navigate flow code
It helpfull to clean code and make clear
I divide it to more part and this file has response to process
a general function which was excute many time by many API in project 
'''
from utils.util import convert, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from flask_restful import reqparse

#region Config parameter API
def requestParser():
    """
    It takes a list of arguments and returns a dictionary of those arguments
    :return: The parser is being returned.
    """
    parser = reqparse.RequestParser()
    parser.add_argument('data')
    parser.add_argument('xminLocation')
    parser.add_argument('yminLocation')
    parser.add_argument('ymaxLocation')
    parser.add_argument('xmaxLocation')
    parser.add_argument('SizeObjectImage')
    parser.add_argument('listpart')
    parser.add_argument('listlocationPart')
    return parser

#endregion

#region Calculate Mathematics in Image
def CalculateAreaByBoxBaseCoordinateDataFrame(pandas_dataframe, sizeImage):
    """
    This function takes in a pandas dataframe and the size of the image as a tuple, and returns the area
    of the object in the image as a percentage of the total area of the image, and the confidence of the
    object
    
    :param pandas_dataframe: The dataframe that contains the bounding box coordinates
    :param sizeImage: The size of the image in pixels
    :return: the area of the object in the image and the confidence of the object.
    """
    totalArea_Image = []
    confidence = []
    areaImage = sizeImage[0] * sizeImage[1]
    if len(pandas_dataframe['class']) == 0:
        pass
    else:
        for i in range(len(pandas_dataframe['xmin'])):
            x_objectLenght = pandas_dataframe['xmax'][i] - pandas_dataframe['xmin'][i]
            y_objectLenght = pandas_dataframe['ymax'][i] - pandas_dataframe['ymin'][i]
            area = round(x_objectLenght * y_objectLenght)
            perImage = (area/areaImage)
            totalArea_Image.append(perImage)
            confidence.append(pandas_dataframe['confidence'][i])
    return totalArea_Image, confidence

def CalculateByAllCoordinateXY(xmin, ymin, xmax, ymax, shapeImage):
    """
    It takes the coordinates of the bounding box and the shape of the image and returns the area of the
    bounding box and the percentage of the bounding box in the image
    
    :param xmin: The x-coordinate of the top left corner of the bounding box
    :param ymin: The y coordinate of the top left corner of the bounding box
    :param xmax: The x-coordinate of the top-right corner of the bounding box
    :param ymax: The y coordinate of the bottom of the bounding box
    :param shapeImage: The shape of the image
    :return: The area of the object and the percentage of the object in the image.
    """
    areaImage = shapeImage[0] * shapeImage[1] 
    x_objectLenght = xmax - xmin
    y_objectLenght = ymax - ymin
    area = x_objectLenght * y_objectLenght
    perImage = (area/areaImage)
    return area , perImage
#endregion

#region Image Processing
def cropImage(xmin, ymin, xmax, ymax, imageOriginal):
    """
    It takes in the coordinates of the bounding box, and the original image, and returns the cropped
    image
    
    :param xmin: The x-coordinate of the upper left corner of the bounding box
    :param ymin: The y-coordinate of the top left corner of the bounding box
    :param xmax: The x coordinate of the right side of the bounding box
    :param ymax: ymin + height
    :param imageOriginal: The original image
    :return: The cropped image.
    """
    imageCrop = imageOriginal[int(ymin):int(ymax),
                              int(xmin):int(xmax)]
    return imageCrop

def cropImageByList(xmin, ymin, xmax, ymax, imageOriginal):
    """
    It takes in a list of xmin, ymin, xmax, ymax values and an image, and returns a list of cropped
    images
    
    :param xmin: The x-coordinate of the upper left corner of the bounding box
    :param ymin: The top coordinate of the bounding box
    :param xmax: The x-coordinate of the right side of the bounding box
    :param ymax: The y-coordinate of the bottom edge of the bounding box, normalized so that 1 is the
    maximum value of the image frame
    :param imageOriginal: the original image
    :return: A list of cropped images
    """
    listPart = []
    for i in range(len(xmin)):
        imageCrop = imageOriginal[int(ymin[i]):int(ymax[i]),
                            int(xmin[i]):int(xmax[i])]
        listPart.append(imageCrop)
    return listPart

def toListLocatetion(data, attemp):
    """
    It takes a list of dictionaries and a key, and returns a list of the values for that key in each
    dictionary
    
    :param data: the data you want to extract from
    :param attemp: the column name of the dataframe
    :return: A list of the values of the key "attemp" in the dictionary "data"
    """
    listObject = [i[attemp] for i in data]
    return listObject

def processImageSimple(image, size : tuple):
    """
    It takes an image, converts it to RGB, resizes it to the specified size, converts it to an array,
    adds a dimension to the array, and preprocesses the array
    
    :param image: The image to be processed
    :param size: The size of the image to be resized to
    :type size: tuple
    :return: The image is being returned as a numpy array.
    """
    img = convert(image, mode = 'RGB')
    img = img.resize(size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x, mode = 'tf')
    return x

def concatenateImage(cropImage, originalImage, xmin, ymin, delay : int = 0):
    """
    It takes a cropped image, an original image, the xmin and ymin coordinates of the cropped image, and
    a delay parameter, and it returns the original image with the cropped image pasted on top of it
    
    :param cropImage: the image you want to crop
    :param originalImage: the original image that you want to crop from
    :param xmin: The x-coordinate of the upper left corner of the bounding box
    :param ymin: The top coordinate of the bounding box
    :param delay: the number of pixels to shift the crop image from the original image, defaults to 0
    :type delay: int (optional)
    :return: The original image with the crop image added to it.
    """
    positionToSet = (int(ymin) + delay, int(xmin) + delay)
    try:
        originalImage[positionToSet[0]:(positionToSet[0] + cropImage.shape[0]),
                      positionToSet[1]:(positionToSet[1] + cropImage.shape[1])] = cropImage
        return originalImage
    except:
        raise Exception('ValueError: delay parameter too big it out range of original image')
#endregion