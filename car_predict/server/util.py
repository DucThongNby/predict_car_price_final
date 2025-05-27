import pickle
import joblib
import json
import numpy as np
import pandas as pd

__fueltype = None
__aspiration = None
__doornumber = None
__carbody = None
__drivewheel = None
__enginelocation = None
__enginetype = None
__cylindernumber = None
__fuelsystem = None
__brand = None
__modelc = None

__columnsc = None
__model = None

def get_predict_price_car(symboling, fueltype, aspiration,
       doornumber, carbody, drivewheel,
       enginelocation, wheelbase, carlength, carwidth , carheight, curbweight
       , enginetype,cylindernumber, enginesize, fuelsystem
       ,boreratio, stroke, compressionratio , horsepower, 
       peakrpm, citympg, highwaympg,brand,  model , 
       power_to_weight_ratio, wheelbase_squared,
       carlength_squared, carwidth_squared,
       carheight_squared, curbweight_squared,
       enginesize_squared, boreratio_squared,
       stroke_squared, 
       compressionratio_squared, 
       horsepower_squared, peakrpm_squared, 
       citympg_squared, highwaympg_squared, 
       log_enginesize ):
   # print(wheelbase.type())
   
   
   categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                          'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model']
   
   categorical_input = pd.DataFrame([{
        'fueltype': fueltype,
        'aspiration': aspiration,
        'doornumber': doornumber,
        'carbody': carbody,
        'drivewheel': drivewheel,
        'enginelocation': enginelocation,
        'enginetype': enginetype,
        'cylindernumber': cylindernumber,
        'fuelsystem': fuelsystem,
        'brand': brand,
        'model': model
    }])
   
   import pickle

    # Load saved label encoders
   with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
   for column in categorical_columns:
        le = label_encoders[column]
        categorical_input[column] = le.transform(categorical_input[column])

   
   fueltype = categorical_input['fueltype'].iloc[0]
   aspiration = categorical_input['aspiration'].iloc[0]
   doornumber = categorical_input['doornumber'].iloc[0]
   carbody = categorical_input['carbody'].iloc[0]
   drivewheel = categorical_input['drivewheel'].iloc[0]
   enginelocation = categorical_input['enginelocation'].iloc[0]
   enginetype = categorical_input['enginetype'].iloc[0]
   cylindernumber = categorical_input['cylindernumber'].iloc[0]
   fuelsystem = categorical_input['fuelsystem'].iloc[0]
   brand = categorical_input['brand'].iloc[0]
   model = categorical_input['model'].iloc[0]

   
   '''
   try:
        loc_index = __fueltype.index(fueltype.lower())
    except:
        loc_index = -1
    print(loc_index)
    fueltype = loc_index
    
    try:
        loc_index1 = __aspiration.index(aspiration.lower())
    except:
        loc_index1 = -1
    aspiration = loc_index1
    
    try:
        loc_index2 = __doornumber.index(doornumber.lower())
    except:
        loc_index2 = -1
    doornumber = loc_index2
    
    try:
        loc_index3 = __carbody.index(carbody.lower())
    except:
        loc_index3 = -1
    carbody = loc_index3
    
    try:
        loc_index4 = __drivewheel.index(drivewheel.lower())
    except:
        loc_index4 = -1
    drivewheel = loc_index4
    
    try:
        loc_index5 = __enginelocation.index(enginelocation.lower())
    except:
        loc_index5 = -1
    enginelocation = loc_index5
    
    try:
        loc_index6 = __enginetype.index(enginetype.lower())
    except:
        loc_index6 = -1
    enginetype = loc_index6
    
    try:
        loc_index7 = __cylindernumber.index(cylindernumber.lower())
    except:
        loc_index7 = -1
    cylindernumber = loc_index7
    
    try:
        loc_index8 = __fuelsystem.index(fuelsystem.lower())
    except:
        loc_index8 = -1
    fuelsystem = loc_index8
    
    try:
        loc_index9 = __brand.index(brand.lower())
    except:
        loc_index9 = -1
    brand = loc_index9
    
    try:
        loc_index10 = __modelc.index(model.lower())
    except:
        loc_index10 = -1
    model = loc_index10
'''

        
    # new_sample = [[0.41, 0.43, 3, 153, 3, 0, 1, 1, 1]]
   new_sample = [[symboling, fueltype, aspiration,
           doornumber, carbody, drivewheel,
           enginelocation, wheelbase, carlength, carwidth , carheight, curbweight
           , enginetype,cylindernumber, enginesize, fuelsystem
           ,boreratio, stroke, compressionratio , horsepower, 
           peakrpm, citympg, highwaympg,brand,  model , 
           power_to_weight_ratio, wheelbase_squared,
           carlength_squared, carwidth_squared,
           carheight_squared, curbweight_squared,
           enginesize_squared, boreratio_squared,
           stroke_squared, 
           compressionratio_squared, 
           horsepower_squared, peakrpm_squared, 
           citympg_squared, highwaympg_squared, 
           log_enginesize]]
    
   x = pd.DataFrame(new_sample)
   x.columns = __columnsc
   
   predict = __model.predict(x)[0]
    
   return predict

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __fueltype 
    global __aspiration 
    global __doornumber 
    global __carbody 
    global __drivewheel 
    global __enginelocation 
    global __enginetype 
    global __cylindernumber 
    global __fuelsystem 
    global __brand 
    global __modelc 

    global __columnsc 
 

    with open("./artifacts/fueltype_value.json", "r") as f:
        __fueltype = json.load(f)['fueltype_value']
    with open("./artifacts/aspiration_value.json", "r") as f:
        __aspiration = json.load(f)['aspiration_value']
    with open("./artifacts/doornumber_value.json", "r") as f:
        __doornumber = json.load(f)['doornumber_value']
    with open("./artifacts/carbody_value.json", "r") as f:
        __carbody = json.load(f)['carbody_value']
    with open("./artifacts/drivewheel_value.json", "r") as f:
        __drivewheel = json.load(f)['drivewheel_value']
    with open("./artifacts/enginelocation_value.json", "r") as f:
        __enginelocation = json.load(f)['enginelocation_value']
    with open("./artifacts/enginetype_value.json", "r") as f:
        __enginetype = json.load(f)['enginetype_value']
    with open("./artifacts/cylindernumber_value.json", "r") as f:
        __cylindernumber = json.load(f)['cylindernumber_value']
    with open("./artifacts/fuelsystem_value.json", "r") as f:
        __fuelsystem = json.load(f)['fuelsystem_value']
    with open("./artifacts/brand_value.json", "r") as f:
        __brand = json.load(f)['brand_value']
    with open("./artifacts/modelc_value.json", "r") as f:
        __modelc = json.load(f)['modelc_value']    
        
    with open("./artifacts/columnsc_value.json", "r") as f:
        __columnsc = json.load(f)['columnsc_value']   

    global __model
    
    with open('./artifacts/model_car.sav', 'rb') as f:
        __model = joblib.load(f)
    print("loading saved artifacts...done")

def get_fueltype_names():
    return __fueltype

def get_aspiration_names():
    return __aspiration

def get_doornumber_names():
    return __doornumber

def get_carbody_names():
    return __carbody

def get_drivewheel_names():
    return __drivewheel

def get_enginelocation_names():
    return __enginelocation

def get_enginetype_names():
    return __enginetype

def get_cylindernumber_names():
    return __cylindernumber

def get_fuelsystem_names():
    return __fuelsystem

def get_brand_names():
    return __brand

def get_modelc_names():
    return __modelc

def get_columnsc():
    return __columnsc




if __name__ == '__main__':
    load_saved_artifacts()
   