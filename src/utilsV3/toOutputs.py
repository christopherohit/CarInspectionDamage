from io import BytesIO
import base64
from PIL import Image
import cv2

def toBase64Simple(imageArray):
    """
    It takes an image array, converts it to a JPEG, then converts it to a base64 string
    
    :param imageArray: The image array that you want to convert to base64
    :return: The base64 string of the image
    """
    if 'PIL' not in str(type(imageArray)):
        buffered = BytesIO()
        imageBase64 = Image.fromarray(imageArray)
        imageBase64.save(buffered, format= 'JPEG')
        base64toOut = base64.b64decode(buffered.getvalue()).decode('utf-8')
    else:
        buffered = BytesIO()
        imageArray.save(buffered, format= 'JPEG')
        im_byte = buffered.getvalue()
        base64toOut = base64.b64encode(im_byte).decode('utf8')
    return base64toOut

def toBase64HardScores(imageArray):
    """
    It takes an image array, converts it to a jpg, converts the jpg to bytes, converts the bytes to
    base64, and then converts the base64 to a string.
    
    :param imageArray: The image array that you want to convert to base64
    :return: A base64 encoded string of the image
    """
    _, buffer = cv2.imencode('.jpg', imageArray)
    im_bytes = buffer.tobytes()
    jpg_as_text = base64.b64encode(im_bytes)
    imgArray = jpg_as_text.decode('utf8')
    return imgArray