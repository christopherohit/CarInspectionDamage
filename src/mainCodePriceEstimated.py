import random
from time import time
import numpy as np
from requests import post
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import os
import joblib
import pickle
import pandas as pd
import time


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('model_code')
parser.add_argument('brand_code')
parser.add_argument('manufactured_year')
unpickleFile = open('models/RanForest_models_08_09.pkl', 'rb')
ML_model = pickle.load(unpickleFile)


class AIPredictCost(Resource):
    def post(self):
        """
        The function takes in a list of values, converts it to a dataframe, preprocesses it, and then
        predicts the price of the car
        :return: The return is a json file with the score and the time process
        """
        # Parsing the arguments that are being passed to the function.
        args = parser.parse_args()
        start = time.time()
        input = []
        newlist = []
        if args['manufactured_year'] != None:
            year = args['manufactured_year']
            newlist.append(year)
        if args['brand_code'] != None:
            brand_code = args['brand_code']
            newlist.append(brand_code)
        if args['model_code'] != None:
            model_code = args['model_code']  
            newlist.append(model_code)
        if(len(newlist) == 3):
            input.append(newlist)
            output = ML_model.predict(input)
            end = time.time() - start
            data_out = {'AI_model':'RandomForest_model','manufactured_year':int(year),'brand_code':brand_code,'model_code':model_code
                    ,'price':float(output[0]),'time_process': float(end)}
                
            return jsonify(data_out)
        else: return("Input sai hoac khong ton tai trong database. Moi nhap lai")
    def get():
        """
        > The function `get()` returns the result of the function `post()` in the class
        `Physical_Segmentations`
        :return: The post method is being returned.
        """
        return AIPredictCost.post()


