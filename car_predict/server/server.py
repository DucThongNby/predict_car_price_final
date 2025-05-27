from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import util

app = Flask(__name__)

@app.route('/get_full_columns', methods=['GET'])
def get_full_columns():
    response = jsonify({
        'fueltype': util.get_fueltype_names(),
        'aspiration': util.get_aspiration_names(),
        'doornumber': util.get_doornumber_names(),
        'carbody': util.get_carbody_names(),
        'drivewheel': util.get_drivewheel_names(),
        'enginelocation': util.get_enginelocation_names(),
        'enginetype': util.get_enginetype_names(),
        'cylindernumber': util.get_cylindernumber_names(),
        'fuelsystem': util.get_fuelsystem_names(),
        'brand': util.get_brand_names(),
        'model': util.get_modelc_names(),   # model car
        'columns': util.get_columnsc()      # tất cả cột đặc trưng
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_price_car', methods=['GET', 'POST'])
def predict_car_price():
    
    symboling = int(request.form['symboling']) 
    wheelbase = float(request.form['wheelbase'])
    carlength = float(request.form['carlength'])
    carwidth = float(request.form['carwidth'])
    carheight = float(request.form['carheight'])
    curbweight = float(request.form['curbweight'])
    enginesize = float(request.form['enginesize'])
    boreratio = float(request.form['boreratio'])
    stroke = float(request.form['stroke'])
    compressionratio = float(request.form['compressionratio'])
    horsepower = float(request.form['horsepower'])
    peakrpm = float(request.form['peakrpm'])
    citympg = float(request.form['citympg'])
    highwaympg = float(request.form['highwaympg'])
    
    '''
    symboling =  3
    wheelbase = 88.6
    carlength = 168.8
    carwidth = 64.1
    carheight = 48.8
    curbweight = 2548
    enginesize = 130
    boreratio = 3.47
    stroke = 2.68
    compressionratio = 9
    horsepower = 111
    peakrpm = 5000
    citympg = 21
    highwaympg = 27
    '''
    
    
    
    
    fueltype = request.form['fueltype']
    aspiration = request.form['aspiration']
    doornumber = request.form['doornumber']
    carbody = request.form['carbody']
    drivewheel = request.form['drivewheel']
    enginelocation = request.form['enginelocation']
    enginetype = request.form['enginetype']
    cylindernumber = request.form['cylindernumber']
    fuelsystem = request.form['fuelsystem']
    brand = request.form['brand']
    model = request.form['model']
    
    '''
    fueltype = 'gas'
    aspiration = 'std'
    doornumber = 'two'
    carbody = 'convertible'
    drivewheel = 'rwd'
    enginelocation = 'front'
    enginetype = 'dohc'
    cylindernumber = 'four'
    fuelsystem = 'mpfi'
    brand = 'alfa-romero'
    model = 'giulia'
    '''
    
    
    
    

    power_to_weight_ratio = horsepower / curbweight

    wheelbase_squared = wheelbase ** 2
    carlength_squared = carlength ** 2
    carwidth_squared = carwidth ** 2
    carheight_squared = carheight ** 2
    curbweight_squared = curbweight ** 2
    enginesize_squared = enginesize ** 2
    boreratio_squared = boreratio ** 2
    stroke_squared = stroke ** 2
    compressionratio_squared = compressionratio ** 2
    horsepower_squared = horsepower ** 2
    peakrpm_squared = peakrpm ** 2
    citympg_squared = citympg ** 2
    highwaympg_squared = highwaympg ** 2
    
    log_enginesize = np.log(enginesize + 1)
    
    
    numerical_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                     'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                     'peakrpm', 'citympg', 'highwaympg']
    # Tạo DataFrame từ đầu vào
    numerical_input = pd.DataFrame([{
        'wheelbase': wheelbase,
        'carlength': carlength,
        'carwidth': carwidth,
        'carheight': carheight,
        'curbweight': curbweight,
        'enginesize': enginesize,
        'boreratio': boreratio,
        'stroke': stroke,
        'compressionratio': compressionratio,
        'horsepower': horsepower,
        'peakrpm': peakrpm,
        'citympg': citympg,
        'highwaympg': highwaympg
    }])
    numerical_input.head()
    
    import joblib

    # Load lại scaler đã lưu
    scaler = joblib.load('scaler.sav')
    
    scaled_array = scaler.transform(numerical_input)
    scaled_df = pd.DataFrame(scaled_array, columns=numerical_input.columns)

    ordered_columns = ["symboling", "fueltype", "aspiration", "doornumber", "carbody",
                   "drivewheel", "enginelocation", "wheelbase", "carlength", "carwidth",
                   "carheight", "curbweight", "enginetype", "cylindernumber", "enginesize",
                   "fuelsystem", "boreratio", "stroke", "compressionratio", "horsepower",
                   "peakrpm", "citympg", "highwaympg", "brand", "model",
                   "power_to_weight_ratio", "wheelbase_squared", "carlength_squared", "carwidth_squared",
                   "carheight_squared", "curbweight_squared", "enginesize_squared", "boreratio_squared",
                   "stroke_squared", "compressionratio_squared", "horsepower_squared", "peakrpm_squared",
                   "citympg_squared", "highwaympg_squared", "log_enginesize"]
    #print(scaled_df['wheelbase'].iloc[0].type())
    wheelbase = scaled_df['wheelbase'].iloc[0]
    carlength = scaled_df['carlength'].iloc[0]
    carwidth = scaled_df['carwidth'].iloc[0]
    carheight = scaled_df['carheight'].iloc[0]
    curbweight = scaled_df['curbweight'].iloc[0]
    enginesize = scaled_df['enginesize'].iloc[0]
    boreratio = scaled_df['boreratio'].iloc[0]
    stroke = scaled_df['stroke'].iloc[0]
    compressionratio = scaled_df['compressionratio'].iloc[0]
    horsepower = scaled_df['horsepower'].iloc[0]
    peakrpm = scaled_df['peakrpm'].iloc[0]
    citympg = scaled_df['citympg'].iloc[0]
    highwaympg = scaled_df['highwaympg'].iloc[0]
    
    response = jsonify({
        'predict_price': util.get_predict_price_car(symboling, fueltype, aspiration, doornumber, carbody, drivewheel,
        enginelocation,wheelbase , carlength, carwidth
        ,carheight ,curbweight ,
        enginetype, cylindernumber,enginesize, fuelsystem,
        boreratio,stroke ,compressionratio,
        horsepower,peakrpm ,
        citympg, highwaympg,
        brand, model,
        power_to_weight_ratio, wheelbase_squared, carlength_squared,
        carwidth_squared, carheight_squared, curbweight_squared,
        enginesize_squared, boreratio_squared, stroke_squared,
        compressionratio_squared, horsepower_squared, peakrpm_squared,
        citympg_squared, highwaympg_squared, log_enginesize)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response



if __name__ == "__main__":
    print("Starting Python Flask Server For Prediction...")
    util.load_saved_artifacts()
    app.run()
