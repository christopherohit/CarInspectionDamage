import random
from time import time
import numpy as np
from requests import post
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from io import BytesIO
import base64
import os
import pandas as pd
import time
from utils.util import base64_to_pil, convert, img_to_array
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

classes = ['AM General Hummer SUV 2000','Acura RL Sedan 2012','Acura TL Sedan 2012','Acura TL Type-S 2008','Acura TSX Sedan 2012','Acura Integra Type R 2001',
'Acura ZDX Hatchback 2012','Aston Martin V8 Vantage Convertible 2012','Aston Martin V8 Vantage Coupe 2012','Aston Martin Virage Convertible 2012','Aston Martin Virage Coupe 2012',
 'Audi RS 4 Convertible 2008','Audi A5 Coupe 2012','Audi TTS Coupe 2012','Audi R8 Coupe 2012','Audi V8 Sedan 1994','Audi 100 Sedan 1994','Audi 100 Wagon 1994',
 'Audi TT Hatchback 2011','Audi S6 Sedan 2011','Audi S5 Convertible 2012','Audi S5 Coupe 2012','Audi S4 Sedan 2012','Audi S4 Sedan 2007','Audi TT RS Coupe 2012',
 'BMW ActiveHybrid 5 Sedan 2012','BMW 1 Series Convertible 2012','BMW 1 Series Coupe 2012','BMW 3 Series Sedan 2012','BMW 3 Series Wagon 2012','BMW 6 Series Convertible 2007',
 'BMW X5 SUV 2007','BMW X6 SUV 2012','BMW M3 Coupe 2012','BMW M5 Sedan 2010','BMW M6 Convertible 2010','BMW X3 SUV 2012','BMW Z4 Convertible 2012','Bentley Continental Supersports Conv. Convertible 2012',
 'Bentley Arnage Sedan 2009','Bentley Mulsanne Sedan 2011','Bentley Continental GT Coupe 2012','Bentley Continental GT Coupe 2007','Bentley Continental Flying Spur Sedan 2007',
'Bugatti Veyron 16.4 Convertible 2009','Bugatti Veyron 16.4 Coupe 2009','Buick Regal GS 2012','Buick Rainier SUV 2007','Buick Verano Sedan 2012','Buick Enclave SUV 2012',
 'Cadillac CTS-V Sedan 2012','Cadillac SRX SUV 2012','Cadillac Escalade EXT Crew Cab 2007','Chevrolet Silverado 1500 Hybrid Crew Cab 2012','Chevrolet Corvette Convertible 2012',
 'Chevrolet Corvette ZR1 2012','Chevrolet Corvette Ron Fellows Edition Z06 2007','Chevrolet Traverse SUV 2012','Chevrolet Camaro Convertible 2012','Chevrolet HHR SS 2010',
 'Chevrolet Impala Sedan 2007','Chevrolet Tahoe Hybrid SUV 2012','Chevrolet Sonic Sedan 2012','Chevrolet Express Cargo Van 2007','Chevrolet Avalanche Crew Cab 2012',
 'Chevrolet Cobalt SS 2010','Chevrolet Malibu Hybrid Sedan 2010','Chevrolet TrailBlazer SS 2009','Chevrolet Silverado 2500HD Regular Cab 2012','Chevrolet Silverado 1500 Classic Extended Cab 2007',
 'Chevrolet Express Van 2007','Chevrolet Monte Carlo Coupe 2007','Chevrolet Malibu Sedan 2007','Chevrolet Silverado 1500 Extended Cab 2012','Chevrolet Silverado 1500 Regular Cab 2012',
 'Chrysler Aspen SUV 2009','Chrysler Sebring Convertible 2010','Chrysler Town and Country Minivan 2012','Chrysler 300 SRT-8 2010','Chrysler Crossfire Convertible 2008',
 'Chrysler PT Cruiser Convertible 2008','Daewoo Nubira Wagon 2002','Dodge Caliber Wagon 2012','Dodge Caliber Wagon 2007','Dodge Caravan Minivan 1997',
 'Dodge Ram Pickup 3500 Crew Cab 2010','Dodge Ram Pickup 3500 Quad Cab 2009','Dodge Sprinter Cargo Van 2009','Dodge Journey SUV 2012','Dodge Dakota Crew Cab 2010',
 'Dodge Dakota Club Cab 2007','Dodge Magnum Wagon 2008','Dodge Challenger SRT8 2011','Dodge Durango SUV 2012','Dodge Durango SUV 2007','Dodge Charger Sedan 2012',
 'Dodge Charger SRT-8 2009','Eagle Talon Hatchback 1998','FIAT 500 Abarth 2012','FIAT 500 Convertible 2012','Ferrari FF Coupe 2012','Ferrari California Convertible 2012',
 'Ferrari 458 Italia Convertible 2012','Ferrari 458 Italia Coupe 2012','Fisker Karma Sedan 2012','Ford F-450 Super Duty Crew Cab 2012','Ford Mustang Convertible 2007',
 'Ford Freestar Minivan 2007','Ford Expedition EL SUV 2009','Ford Edge SUV 2012','Ford Ranger SuperCab 2011','Ford GT Coupe 2006','Ford F-150 Regular Cab 2012',
'Ford F-150 Regular Cab 2007','Ford Focus Sedan 2007','Ford E-Series Wagon Van 2012','Ford Fiesta Sedan 2012','GMC Terrain SUV 2012','GMC Savana Van 2012',
 'GMC Yukon Hybrid SUV 2012','GMC Acadia SUV 2012','GMC Canyon Extended Cab 2012','Geo Metro Convertible 1993','HUMMER H3T Crew Cab 2010','HUMMER H2 SUT Crew Cab 2009',
 'Honda Odyssey Minivan 2012','Honda Odyssey Minivan 2007','Honda Accord Coupe 2012','Honda Accord Sedan 2012','Hyundai Veloster Hatchback 2012','Hyundai Santa Fe SUV 2012',
 'Hyundai Tucson SUV 2012','Hyundai Veracruz SUV 2012','Hyundai Sonata Hybrid Sedan 2012','Hyundai Elantra Sedan 2007','Hyundai Accent Sedan 2012','Hyundai Genesis Sedan 2012',
 'Hyundai Sonata Sedan 2012','Hyundai Elantra Touring Hatchback 2012', 'Hyundai Azera Sedan 2012', 'Infiniti G Coupe IPL 2012','Infiniti QX56 SUV 2011',
 'Isuzu Ascender SUV 2008','Jaguar XK XKR 2012','Jeep Patriot SUV 2012','Jeep Wrangler SUV 2012','Jeep Liberty SUV 2012','Jeep Grand Cherokee SUV 2012',
'Jeep Compass SUV 2012','Lamborghini Reventon Coupe 2008','Lamborghini Aventador Coupe 2012','Lamborghini Gallardo LP 570-4 Superleggera 2012','Lamborghini Diablo Coupe 2001',
 'Land Rover Range Rover SUV 2012','Land Rover LR2 SUV 2012','Lincoln Town Car Sedan 2011','MINI Cooper Roadster Convertible 2012','Maybach Landaulet Convertible 2012',
 'Mazda Tribute SUV 2011','McLaren MP4-12C Coupe 2012','Mercedes-Benz 300-Class Convertible 1993','Mercedes-Benz C-Class Sedan 2012','Mercedes-Benz SL-Class Coupe 2009',
 'Mercedes-Benz E-Class Sedan 2012','Mercedes-Benz S-Class Sedan 2012','Mercedes-Benz Sprinter Van 2012','Mitsubishi Lancer Sedan 2012','Nissan Leaf Hatchback 2012',
 'Nissan NV Passenger Van 2012','Nissan Juke Hatchback 2012','Nissan 240SX Coupe 1998','Plymouth Neon Coupe 1999','Porsche Panamera Sedan 2012','Ram C/V Cargo Van Minivan 2012',
 'Rolls-Royce Phantom Drophead Coupe Convertible 2012','Rolls-Royce Ghost Sedan 2012','Rolls-Royce Phantom Sedan 2012','Scion xD Hatchback 2012','Spyker C8 Convertible 2009',
 'Spyker C8 Coupe 2009','Suzuki Aerio Sedan 2007','Suzuki Kizashi Sedan 2012','Suzuki SX4 Hatchback 2012','Suzuki SX4 Sedan 2012','Tesla Model S Sedan 2012',
 'Toyota Sequoia SUV 2012','Toyota Camry Sedan 2012','Toyota Corolla Sedan 2012','Toyota 4Runner SUV 2012','Volkswagen Golf Hatchback 2012',
 'Volkswagen Golf Hatchback 1991','Volkswagen Beetle Hatchback 2012','Volvo C30 Hatchback 2012','Volvo 240 Sedan 1993','Volvo XC90 SUV 2007','smart fortwo Convertible 2012']


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('data')

json_file = open("models/final_all_transfer_learning_VGG16.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("models/final_all_transfer_learning_VGG16.h5")

def preprocessImage(image):
    """
    It takes an image, resizes it to 299x299, converts it to an array, adds a dimension to the array,
    and preprocesses it for InceptionV3
    
    :param image: The image to be preprocessed
    :return: The preprocessed image and the time at which the image was preprocessed.
    """
    start = time.time()
    img = convert(image, mode= 'RGB')
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    test_datagen = ImageDataGenerator(rescale=1./255)
    x = test_datagen.standardize(x)
    
    return x , start

def GetMakeModel(x ,start):
    """
    :param x: The input image
    :param start: The time when the request was received
    :return: a dictionary with the following keys:
        rc: return code, XX for success
        msg: message to be displayed
        Corner-Class: the class of the corner car
        TimeProcess: the time it took to process the image
    """
    preds = loaded_model.predict(x)
    class_id = np.argmax(preds)+1
    name = classes[class_id-1].split(' ')

    Model = ' '.join(name[1:-1])
    data = { "Year": name[-1], "Model":Model,"Brand": name[0], 'TimeProcess' : (time.time() - start)}

    return  data

class Make_Model(Resource):
    def post(self):
        """
        The function takes in a list of values, converts it to a dataframe, preprocesses it, and then
        predicts the price of the car
        :return: The return is a json file with the score and the time process
        """
        args = parser.parse_args()
        if args['data'] != None:
            # start = time.time()
            base64image = args['data']
            img = base64_to_pil(base64image, type = 2)
            img = img.convert('RGB')
            ModelIs, time = preprocessImage(img)
            ModelIs = GetMakeModel(ModelIs, time)
            buffered = BytesIO()
            img.save(buffered, format='JPEG')
            lastImage = base64.b64encode(buffered.getvalue()).decode('utf-8')
            ModelIs['Base64Image'] = lastImage           
            return jsonify(ModelIs)
    def get():
        """
        > The function `get()` returns the result of the function `post()` in the class
        `Physical_Segmentations`
        :return: The post method is being returned.
        """
        return Make_Model.post()
