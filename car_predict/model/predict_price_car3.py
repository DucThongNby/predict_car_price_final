# learn about data
'''
CarID: Identification Number for Each Car
SafetyRating: Car's Safety Rating
CarName: Name of the Car Model
FuelType: Type of Fuel Used (Gasoline, Diesel, Electric, etc.)
Aspiration: Type of Aspiration (Standard or Turbocharged)
NumDoors: Number of Doors on the Car
BodyStyle: Style of the Car's Body (Sedan, Coupe, SUV, etc.)
DriveWheelType: Type of Drive Wheels (Front, Rear, All)
EngineLocation: Location of the Car's Engine (Front or Rear)
Wheelbase: Length of the Car's Wheelbase
CarLength: Overall Length of the Car
CarWidth: Width of the Car
CarHeight: Height of the Car
CurbWeight: Weight of the Car without Passengers or Cargo
EngineType: Type of Engine (Gas, Diesel, Electric, etc.)
NumCylinders: Number of Cylinders in the Engine
EngineSize: Size of the Car's Engine
FuelSystem: Type of Fuel Delivery System
BoreRatio: Bore-to-Stroke Ratio of the Engine
Stroke: Stroke Length of the Engine
CompressionRatio: Compression Ratio of the Engine
Horsepower: Car's Engine Horsepower
PeakRPM: Engine's Peak RPM (Revolutions Per Minute)
CityMPG: Miles Per Gallon (MPG) in City Driving
HighwayMPG: MPG on the Highway
CarPrice: Price of the Car

'''
# Life Cycle of  Machine Learning Project
'''
Understanding the Problem Statement
Data Checks to Perform
Exploratory Data Analysis
Data Pre-Processing
Model Training
Choose Best Model
'''


#1 Import module
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

import warnings
warnings.filterwarnings('ignore')
import pickle
import joblib
df = pd.read_csv('C:/SONA/car_predict/car_predict/CarPrice_Assignment.csv')

#2 data checks to perform
'''
Check Missing values
Check Duplicates
Check data type
Check the number of unique values of each column
Check statistics of the dataset
Check various categories present in the different categorical columns
'''

# check Missing value -> 0
df.isnull().sum()

# Check Duplication -> 0
df.duplicated().sum()

#Check datatype
df.dtypes

# Check the number of unique values of each column
df.nunique()

#Check statistics of data set
df.describe()

categorical_columns = ['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype',
    'cylindernumber',
    'fuelsystem'
]

for col in categorical_columns:
    
    print(f"Category in {col} is : {df[col].unique()}")

''' 
Explotary data analysis
'''
# Distribution of Numerical Features
numerical_features = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                      'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                      'peakrpm', 'citympg', 'highwaympg', 'price']

plt.figure(figsize=(12, 8))
for feature in numerical_features:
    plt.subplot(3, 5, numerical_features.index(feature) + 1)
    sns.histplot(data=df[feature], bins=20, kde=True)
    plt.title(feature)
plt.tight_layout()
plt.show()

# Price Analysis
plt.figure(figsize=(8, 6))
sns.histplot(data=df['price'], bins=20, kde=True)
plt.title('Distribution of Price')
plt.show()

# doan define nay ben tren lam roi
# Define the list of categorical columns to analyze
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
axes = axes.ravel()  # Flatten the 2D array of axes

# Loop through each categorical column
for i, column in enumerate(categorical_columns):
    sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
    for container in axes[i].containers:
        axes[i].bar_label(container, color='black', size=10)
    axes[i].set_title(f'Count Plot of {column.capitalize()}')
    axes[i].set_xlabel(column.capitalize())
    axes[i].set_ylabel('Count')

# Adjust layout and show plots
plt.tight_layout()
plt.show()

n = 20  # Number of top car models to plot
top_car_models = df['CarName'].value_counts().head(n)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_car_models.values, y=top_car_models.index)
plt.title(f'Top {n} Car Models by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Car Model')
plt.tight_layout()
plt.show()


# Calculate average price for each car model
avg_prices_by_car = df.groupby('CarName')['price'].mean().sort_values(ascending=False)

# Plot top N car models by average price
n = 20  # Number of top car models to plot
top_car_models = avg_prices_by_car.head(n)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_car_models.values, y=top_car_models.index)
plt.title(f'Top {n} Car Models by Average Price')
plt.xlabel('Average Price')
plt.ylabel('Car Model')
plt.tight_layout()
plt.show()

# Categorical Feature vs. Price
plt.figure(figsize=(12, 8))
for feature in categorical_columns:
    plt.subplot(3, 3, categorical_columns.index(feature) + 1)
    sns.boxplot(data=df, x=feature, y='price')
    plt.title(f'{feature} vs. Price')
plt.tight_layout()
plt.show()


# Correlation Analysis
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


''' Analysis Results
We have DataSet > Car Price 👍
The Shape DataSet = (Rows = 205, columns = 26) 👊

No null value 🚫📛

No Duplicated value 🚫🔁

'''


''' Data Pre-Processing'''


# Extract brand and model from CarName
df['brand'] = df['CarName'].apply(lambda x: x.split(' ')[0])
df['model'] = df['CarName'].apply(lambda x: ' '.join(x.split(' ')[1:]))

# Define categorical and numerical columns
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model']
numerical_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                     'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                     'peakrpm', 'citympg', 'highwaympg']

df2 = df.copy()
# Encoding categorical variables
label_encoder = LabelEncoder()
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save encoder for this column

# Save all encoders into a pickle file
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)



# Lưu label_encoder ra file
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))
joblib.dump(label_encoder, 'label_encoder.sav' )

# Feature engineering
df['power_to_weight_ratio'] = df['horsepower'] / df['curbweight']
for column in numerical_columns:
    df[f'{column}_squared'] = df[column] ** 2
df['log_enginesize'] = np.log(df['enginesize'] + 1)

# Feature scaling
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


'''
Train the model
'''

# Splitting the dataset
X = df.drop(['price', 'CarName', 'car_ID'], axis=1)  # Include the engineered features and CarName
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')


pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
pred_df


'''
new_sample = [[67, 0, 0, 0, 0, 3, 2 , 0, 1.0227,0.0772503 ,0.0898123 , 0.276967, 0.278074, 3, 2, 0.170739, 3, 0.371023, 1.22937, 2.99254, -0.814171, -1.94427, 0.88574,1.20076, 11, 116, 0.0266667, 11004, 30625, 4369.21, 2959.36, 7290000,  17956,   11.7649, 13.2496, 484, 5148, 17640000, 961, 1521,  4.90527  ]]
df_newSample = pd.DataFrame(new_sample)
df_newSample.columns = X_train.columns
pred = model.predict(df_newSample)[0]
print(pred)

new_sample = X_test.iloc[[0]].copy()  # Lấy dòng đầu tiên, giữ định dạng DataFrame
pred = model.predict(new_sample)[0]
print(pred)

'''


'''
# Encoding categorical variables

for column in categorical_columns:
    df2[column] = label_encoder.fit_transform(df2[column])

X2 = df2.drop(['price', 'CarName','car_ID'], axis=1) 
# Include the engineered features and CarName


new_sample2 = X2.iloc[[0]].copy()  # Lấy dòng đầu tiên, giữ định dạng DataFrame
# Feature engineering
new_sample2['power_to_weight_ratio'] = new_sample2['horsepower'] / X2['curbweight']
for column in numerical_columns:
    new_sample2[f'{column}_squared'] = new_sample2[column] ** 2
new_sample2['log_enginesize'] = np.log(new_sample2['enginesize'] + 1)

# Feature scaling
new_sample2[numerical_columns] = scaler.transform(new_sample2[numerical_columns])

pred2 = model.predict(new_sample2)[0]
print(pred2)
'''
import pickle
import joblib
pickle.dump(model, open('model_car.pkl', 'wb'))
joblib.dump(model, open('model_car.sav', 'wb'))
# Lưu scaler ra file
pickle.dump(scaler, open('scaler.pkl', 'wb'))
joblib.dump(scaler, 'scaler.sav' )


df3 = pd.read_csv('C:/SONA/car_predict/car_predict/CarPrice_Assignment.csv')
# Extract brand and model from CarName
df3['brand'] = df3['CarName'].apply(lambda x: x.split(' ')[0])
df3['model'] = df3['CarName'].apply(lambda x: ' '.join(x.split(' ')[1:]))

# Define categorical and numerical columns
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model']

import json 
fueltypej = {'fueltype_value': list(df3.fueltype.unique())}
with open ('fueltype_value.json', 'w') as f:
   f.write(json.dumps(fueltypej))
   
aspirationj = {'aspiration_value': list(df3.aspiration.unique())}
with open ('aspiration_value.json', 'w') as f:
   f.write(json.dumps(aspirationj))
   
doornumberj = {'doornumber_value': list(df3.doornumber.unique())}
with open ('doornumber_value.json', 'w') as f:
   f.write(json.dumps(doornumberj))
   
carbodyj = {'carbody_value': list(df3.carbody.unique())}
with open ('carbody_value.json', 'w') as f:
   f.write(json.dumps(carbodyj))
   
drivewheelj = {'drivewheel_value': list(df3.drivewheel.unique())}
with open ('drivewheel_value.json', 'w') as f:
   f.write(json.dumps(drivewheelj))
   
enginelocationj = {'enginelocation_value': list(df3.enginelocation.unique())}
with open ('enginelocation_value.json', 'w') as f:
   f.write(json.dumps(enginelocationj))
   
enginetypej = {'enginetype_value': list(df3.enginetype.unique())}
with open ('enginetype_value.json', 'w') as f:
   f.write(json.dumps(enginetypej))
   
cylindernumberj = {'cylindernumber_value': list(df3.cylindernumber.unique())}
with open ('cylindernumber_value.json', 'w') as f:
   f.write(json.dumps(cylindernumberj))
   
fuelsystemj = {'fuelsystem_value': list(df3.fuelsystem.unique())}
with open ('fuelsystem_value.json', 'w') as f:
   f.write(json.dumps(fuelsystemj))
   
brandj = {'brand_value': list(df3.brand.unique())}
with open ('brand_value.json', 'w') as f:
   f.write(json.dumps(brandj))
   
modelcj = {'modelc_value': list(df3.model.unique())}
with open ('modelc_value.json', 'w') as f:
   f.write(json.dumps(modelcj))
   
columnsc = {'columnsc_value':list(X.columns)}
with open ('columnsc_value.json', 'w') as f:
   f.write(json.dumps(columnsc))
'''   
print(df3.columns.tolist())
print(new_sample2.columns.tolist())
'''

'''
new_sample2 = X2.iloc[[0]].copy()  # Lấy dòng đầu tiên, giữ định dạng DataFrame
# Feature engineering
new_sample2['power_to_weight_ratio'] = new_sample2['horsepower'] / new_sample2['curbweight']
for column in numerical_columns:
    new_sample2[f'{column}_squared'] = new_sample2[column] ** 2
new_sample2['log_enginesize'] = np.log(new_sample2['enginesize'] + 1)
   '''
df4 = pd.read_csv('C:/SONA/car_predict/car_predict/CarPrice_Assignment.csv')
# Extract brand and model from CarName
df4['brand'] = df4['CarName'].apply(lambda x: x.split(' ')[0])
df4['model'] = df4['CarName'].apply(lambda x: ' '.join(x.split(' ')[1:]))   
X4 = df4.drop(['price', 'CarName', 'car_ID'], axis=1)  # Include the engineered features and CarName
X4[numerical_columns] = scaler.transform(X4[numerical_columns])

   
'''
'fueltype', 'aspiration', 'doornumber', 
'carbody', 'drivewheel',
'enginelocation', 'enginetype', 
'cylindernumber', 'fuelsystem', 'brand',
 'model']

 '''





