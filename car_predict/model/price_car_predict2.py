import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/SONA/car_predict/car_predict/car_price_prediction.csv")
df.head()

print(f'Data has {df.shape[0]} rows , {df.shape[1]} columns.')

#first we can drop ID col
df.drop(['ID'], axis=1,inplace=True)
print(f'Data after dropping col "ID" has  {df.shape[1]} columns.')

# check for data type
df.dtypes

# "Levy" col should contain int or float values not object

print("Unique values in Levy\n",df['Levy'].unique())

print(df.iloc[2]['Levy'])

'''
"Levy" has '-' values
so before converting we should dealing with non numerical data
'''
# Replace non-numeric values with NaN
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')

df['Levy']


'''
Cylinders" col should contain int values not float
'''

df['Cylinders'] = df['Cylinders'].astype(int)
df['Cylinders']


'''
"Engine volume" col should contain float values not object
'''
df['Engine volume'].unique()

# ho bo nhung cai Turbo di luon
df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce') #non-numeric will be replaced with NaN
df['Engine volume']


#"Mileage" col should contain int values not object

df['Mileage'].unique()
# Remove non-numeric characters and convert 'Mileage' to integers
df['Mileage'] = df['Mileage'].str.replace(' km', '').str.replace(',', '')
df['Mileage']=df['Mileage'] .astype(int)
df['Mileage']

'''
Checking for duplicated Values
'''
num_duplicates = df.duplicated().sum()

print("Number of duplicated values:", num_duplicates)


df.drop_duplicates(inplace=True)
print(f'Data after dropping duplicated values has {df.shape[0]} rows ') #rows-=313


'''
Checking for NULL Values


'''

sns.heatmap(df.isnull(),cmap='cividis')
#Note:All NULL values be in "Levy" col and "Engine volume" col
num_nulls_Levy=df["Levy"].isnull().sum()
print("Number of Null values in 'Levy' col is          :", num_nulls_Levy)

num_nulls_Engine=df["Engine volume"].isnull().sum()
print("Number of Null values in 'Engine volume' col is :", num_nulls_Engine)

#  replace Null values in "Levy" with mean of "Levy" col
df['Levy'].fillna( df['Levy'].mean(), inplace=True)
df['Levy'] = df['Levy'].astype(int)# convert from float to int

#  replace Null values in "Engine volume" with mean of "Engine volume" col
df['Engine volume'].fillna( df['Engine volume'].mean(), inplace=True)

df.isnull().sum() #Done✅✅✅

num_nulls=df.isnull().sum().sum()
print("✅✅✅NOW Number of Null values in Data:", num_nulls)

df_setect_out=df[['Levy','Engine volume','Mileage','Cylinders','Airbags']]
df_setect_out

outlier_cols = []

for column in df_setect_out.columns:
    # Calculate the IQR (Interquartile Range)
    Q1 = df_setect_out[column].quantile(0.25)
    Q3 = df_setect_out[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Identify outliers based on the IQR
    outliers = (df_setect_out[column] < Q1 - 1.5 * IQR) | (df_setect_out[column] > Q3 + 1.5 * IQR)
    
    # Check if there are any outliers in the column
    if any(outliers):
        outlier_cols.append(column)

# Print columns with outliers
print("Columns with outliers:", outlier_cols)

data = {
    'Levy': [100, 150, 200, 250, 300, 5000],
    'Engine volume': [1.5, 2.0, 2.5, 3.0, 4.0, 10.0],
    'Mileage': [5000, 10000, 15000, 20000, 25000, 500000],
    'Cylinders': [4, 6, 8, 12, 16, 32]
}

df_select_out = pd.DataFrame(data)

# Function to replace outliers in a column with a specific value
def replace_outliers(column, replace_value):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    column.loc[(column < lower_bound) | (column > upper_bound)] = replace_value

for col in ['Levy', 'Engine volume', 'Mileage', 'Cylinders']:
    replace_outliers(df_select_out[col], replace_value=df_select_out[col].median())


# Getting avg of Prices by Year
average_prices = df.groupby('Prod. year')['Price'].mean().reset_index()
average_prices[:10]

# Mapping categorical values to numerical values
door_mapping = {'02-Mar': 2, '04-May': 4, '>5': 5}
df['Doors'] = df['Doors'].map({'02-Mar': 2, '04-May': 4, '>5': 5})

# Calculate correlation coefficient
corr = df['Doors'].corr(df['Price'])
print("Correlation between Doors and Prices is: {:.3f}".format(corr))

#"-0.033" refers to a weak negative correlation between the number of 
#doors and prices.

# Converting all categorical columns to Numerical
from sklearn.preprocessing import OrdinalEncoder

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
ordinal_encoder = OrdinalEncoder()
encoded_data = ordinal_encoder.fit_transform(df[categorical_columns])
df[categorical_columns] = encoded_data.astype(int)
df


X=df.drop(['Price','Color','Gear box type','Doors','Wheel','Fuel type','Drive wheels','Manufacturer'],axis=1)
X


from sklearn.preprocessing import StandardScaler
y = df['Price'].values.reshape(-1, 1)

scaler = StandardScaler()
y_standardized = scaler.fit_transform(y)

df['Price'] = y_standardized

df['Price']


'''
Replace Outliers in "Price" col
'''

Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_price = df['Price'].median()
df['Price'] = df['Price'].apply(lambda x: median_price if x < lower_bound or x > upper_bound else x)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y_standardized,test_size=.2,random_state=42)

#Normalizing the Data

from sklearn.preprocessing import StandardScaler

scl = StandardScaler()

X_train_scl = scl.fit_transform(X_train)
X_test_scl = scl.fit_transform(X_test)

# print("Normalized training data:", X_train_scl)
# print("Normalized test data:", X_test_scl)

from sklearn.linear_model import LinearRegression , Lasso , Ridge

models = [
    ['Linear Regression', LinearRegression()],
    ['Ridge Regression', Ridge()],
    ['Lasso Regression', Lasso()],
]

from sklearn.metrics import mean_absolute_error, mean_squared_error ,median_absolute_error,r2_score

predictions = []

for name, model in models:
    model.fit(X_train_scl, y_train)
    y_pred = model.predict(X_test_scl)
    predictions.append((name, y_pred))
    

for name, y_pred in predictions:
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)


    print('Model:', name)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Squared Error: {mae:.2f}')
    print(f'Median Absolute Error: {medae:.2f}')

   

    print('--------------------------\n')








