import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#dumpy
#label encoder
#original encoder

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

df_data = pd.read_csv('C:/SONA/car_predict/car_predict/car_price_prediction.csv')

df_data.info()

# =====Build model Machine Learning=====
# Step 1: Tách biến độc lập và biến phụ thuộc (biến mục tiêu ('left'))


#Step 2: Chia dữ liệu thành tập train và tập test

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 

#encoder = OrdinalEncoder()

'''
Miêu tả dữ liệu

1.   ID	: Mã định danh duy nhất cho mỗi xe trong tập dữ liệu
2.   Price : Giá bán của xe
3. Levy : Thuế hoặc phí bổ sung được áp dụng cho xe
4. Manufacturer : Hãng sản xuất oto
5. Model : Tên cụ thể của kiểu xe
6. Prod. year : Năm sản xuất xe
7. Category : Phân loại xe( SUV, Sedan,...)
8. Leather interior: Nội thất bằng da, cho biết ghế có được bọc da hay không?
9. Fuel type : Loại nhiên liệu, xăng, dầu hybrid
10. Engine volume : Kích thước động cơ, thường được đo bằng lít
11. Mileage : Tổng quãng đường xe đã đi
12. Cylinders : Số xi- lanh của động cơ
13. Gear box type : Loại hộp số, số động hay số sàn
14. Drive wheels : Bánh xe dẫn động( loại dẫn động)- dẫn động cầu trước, cầu sau, dẫn động 4 bánh
15. Doors: Số cửa, chia thành 2-3 , 4-5 , > 5 cửa
16. Wheel: Vị trí vô lăng , trái hoặc phải
17. Color : Màu ngoại thất của xe
18. Airbags : Số lượng túi khí

'''
df_data.duplicated().sum()
print(df_data.isna().sum())
df_data.head()
df_data.select_dtypes(include='object').head()
df_data.select_dtypes(include='object').tail()


df_data['Levy'].isna().sum()

# xac dinh nhung gia tri object bi loi, hoa ra la toan dau "-"
#df_data['Levy']=pd.to_numeric(df_data['Levy'])
invalid_values = df_data[~df_data['Levy'].apply(lambda x: pd.to_numeric(x, errors='coerce')).notna()]
print(invalid_values['Levy'])

#df_data = pd.read_csv('C:/SONA/car_predict/car_predict/car_price_prediction.csv')

# đổi giá trị thuế từ object sang float
df_data['Levy']=pd.to_numeric(df_data['Levy'],errors='coerce')
df_data['Levy'].head()
df_data['Levy'].isna().sum()

print(df_data['Mileage'])
# bo km khoi quang duong
df_data['Mileage'] = df_data['Mileage'].str.replace('km', '', regex=False).str.strip()
df_data['Mileage'].head()

# chuyển dữ liệu object của quãng đường sang dạng int
df_data['Mileage'] = (df_data['Mileage'].astype(int))
df_data['Mileage'].head()

# bỏ chữ Turbo khỏi dung tích động cơ
df_data['Engine volume'] = df_data['Engine volume'].str.replace('Turbo', '', regex=False).str.strip()
df_data['Engine volume'].tail()

# chuyển dung tích động cơ từ object về dạng số
df_data['Engine volume'] = pd.to_numeric(df_data['Engine volume'], errors='coerce')
df_data['Engine volume'].head()

df_data['Doors']

'''
**Sửa lỗi định dang excel trong cột Doors**

**Thay thế tất cả các giá trị ngày tháng theo 
định dạng Excel (ví dụ: "04-May") bằng các danh 
mục thứ bậc đúng (như "2-3", "4-5", ">5").**
'''

df_data['Doors'].unique()

df_data['Doors'] = df_data['Doors'].replace({
    '04-May': '4-5',
    '02-Mar': '2-3',
})
df_data['Doors']

df_data.dtypes

# Loai bo trung lap

df_data.duplicated().sum()
df_data.drop_duplicates(inplace=True)


df_data.drop(columns='ID',inplace=True)  
'''
# Không quan trọng trong quá trình phân tích dữ 
liệu khai phá ( Exploratory Data Analysis) EDA

'''
df_data.dtypes

numeric_cols = df_data.select_dtypes(include=['int64', 'float64']).columns
numeric_cols

'''
Phân tích outliers (giá trị ngoại lệ) trong các cột 
số bằng phương pháp IQR (Interquartile Range)
'''

total_rows = len(df_data) #tổng số dòng trong DataFrame
print(total_rows)

col=1
for col in numeric_cols: #Duyệt qua từng cột số (numeric_cols)
    Q1 = df_data[col].quantile(0.25) # Phân vị thứ 25 (quartile 1)
    Q3 = df_data[col].quantile(0.75) # Phân vị thứ 75 (quartile 3)
    IQR = Q3 - Q1 # Độ rộng khoảng trung vị
    #Xác định ngưỡng để coi là "ngoại lệ"
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df_data[(df_data[col] < lower) | (df_data[col] > upper)]
    count = len(outliers)
    percent = (count / total_rows) * 100

    print(f"{col}: {count} rows ({percent:.2f}%)")




''' 
xu ly outliners voi Price
'''
Q1 = df_data['Price'].quantile(0.25)
Q3 = df_data['Price'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_price = df_data['Price'].median()
df_data['Price'] = df_data['Price'].apply(lambda x: median_price if x < lower_bound or x > upper_bound else x)


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Prod. year', y='Price', data=df_data)
plt.title('Giá theo năm sản xuất')
plt.xlabel('Năm sản xuất')
plt.ylabel('Giá')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df_data['Price'], bins=30, kde=True, color='skyblue')
plt.title('Phân bố Giá sau khi loại bỏ Outliers')
plt.xlabel('Giá')
plt.ylabel('Số lượng')
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df_data['Price'], color='lightgreen')
plt.title('Boxplot Giá sau khi loại bỏ Outliers')
plt.xlabel('Giá')
plt.show()
'''
:Dù có outliers, nhưng giá trị vẫn có vẻ hợp lý 
( dung tích động cơ cao, số lượng xi-lanh hiếm...), 
nên KHÔNG xóa, vẫn giữ lại trong dữ liệu.
'''

df_data.isna().sum()
#Levy co 5709 gia trj NaN
'''
Thay thế các giá trị NaN (thiếu dữ liệu) 
trong cột Levy bằng giá trị trung vị (median)
'''
df_data['Levy']=df_data['Levy'].fillna(df_data['Levy'].median())
df_data.isna().sum()
# khong con cot nao co NaN nua

# Biểu đồ nhiệt giữa những đặc tính só
plt.figure(figsize=(10,6))
sns.heatmap(df_data[numeric_cols].corr(), annot=True, cmap='Blues')
plt.title('Biểu đồ nhiệt giữa những đặc tính số')
plt.show()

categorical_columns = df_data.select_dtypes(include='object').columns.tolist()
#categorical_cols
print(categorical_columns)

df_data_categorical = df_data.copy()

df_data['Manufacturer'].unique()
df_data['Manufacturer'].nunique()

df_data['Model'].unique()
df_data['Model'].nunique() #1590

'''
for model in df_data['Model'].unique():
    print(model)
'''
# Lưu các giá trị unique vào một mảng
#unique_models = df_data['Model'].unique().tolist()

# Tính số lượng phần tử 1/10 đầu tiên (làm tròn xuống)
#n = len(unique_models) // 10

# In ra 1/10 đầu danh sách
#print(f"In {n} giá trị đầu tiên trong tổng số {len(unique_models)} giá trị khác nhau:\n")
#for i in range(500,950):
#    print(f"{i}. {unique_models[i]}")
    


df_data['Category'].unique()
df_data['Category'].nunique()

df_data['Leather interior'].unique()
df_data['Leather interior'].nunique()

df_data['Fuel type'].unique()
df_data['Fuel type'].nunique()

df_data['Gear box type'].unique()
df_data['Gear box type'].nunique()

df_data['Drive wheels'].unique()
df_data['Drive wheels'].nunique()

df_data['Doors'].unique()
df_data['Doors'].nunique()

df_data['Wheel'].unique()
df_data['Wheel'].nunique()



df_data['Color'].unique()
df_data['Color'].nunique()

#df_data2 = df_data.copy()
#print(df_data2.head())

df_data2  = df_data.copy()

'''
chua can drop Model, vi phan tiep theo, hoj X.drop het catgories
'''
#df_data.drop(columns='Model',inplace=True)
#df_data.drop(columns='Index', inplace = True)  
#print(df_data.columns)

ordinal_encoder = OrdinalEncoder()
encoded_data = ordinal_encoder.fit_transform(df_data2[categorical_columns])
df_data2[categorical_columns] = encoded_data.astype(int)
df_data2


y = df_data2[['Price']] # bien phu thuoc
#X = df_data.drop('Price', axis=1) # bien doc lap
#df_data2.drop(columns = 'Model')
#print(df_data2)

X = df_data2.drop(columns = 'Price' )
X2 = X.copy()
y2 = y.copy()
'''
X2 = X.copy()
X2= X2.drop(columns='Levy')
X2= X2.drop(columns='Prod. year')
X2= X2.drop(columns='Mileage')
''' 


print(X2.head())
X2=X2.drop(['Manufacturer','Fuel type','Gear box type','Drive wheels','Doors','Wheel' ,'Color'],axis=1)
X2


y3 = df_data2['Price'].values.reshape(-1, 1)

scaler = StandardScaler()
y_standardized = scaler.fit_transform(y3)

df_data2['Price'] = y_standardized
df_data2['Price']



cols_to_standardize = ['Levy', 'Model', 'Prod. year', 'Category', 
                       'Leather interior', 'Engine volume', 
                       'Mileage', 'Cylinders', 'Airbags']

X4 = df_data2[cols_to_standardize].values
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X4)
df_data2[cols_to_standardize] = X_standardized


y5 = df_data2[['Price']] # bien phu thuoc
X5 = df_data2.drop(columns = 'Price' )
X5=X5.drop(['Manufacturer','Fuel type','Gear box type','Drive wheels','Doors','Wheel' ,'Color'],axis=1)
print(X5.columns)
'''
Index(['Levy', 'Model', 'Prod. year', 'Category', 'Leather interior',
       'Engine volume', 'Mileage', 'Cylinders', 'Airbags'],
'''

X_train5, X_test5, y_train5, y_test5 = train_test_split(X5,y5,test_size=.2,random_state=42)

from sklearn.linear_model import LinearRegression , Lasso , Ridge

models = [
    ['Linear Regression', LinearRegression()],
    ['Ridge Regression', Ridge()],
    ['Lasso Regression', Lasso()],
]



from sklearn.metrics import mean_absolute_error, mean_squared_error ,median_absolute_error,r2_score

predictions = []

for name, model in models:
    model.fit(X_train5, y_train5)
    y_pred5 = model.predict(X_test5)
    predictions.append((name, y_pred5))
    

for name, y_pred5 in predictions:
    mse = mean_squared_error(y_test5, y_pred5)
    mae = mean_absolute_error(y_test5, y_pred5)
    medae = median_absolute_error(y_test5, y_pred5)

    print('Model:', name)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Squared Error: {mae:.2f}')
    print(f'Median Absolute Error: {medae:.2f}')

   

    print('--------------------------\n')

#X3 = df_data2['Levy', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']
X3 = X2
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X3)
df_data2['levy'] = X_standardized
X3 = X_standardized.values
X3

y4 = df_data2['Price']
X4 = df_data2['Levy', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']





y_temp = df_data2['Price'].values.reshape(-1, 1)
y2['Price'] = y_temp.astype(int)

y_standardized = y2.copy()
scaler = StandardScaler()
y_temp2 = scaler.fit_transform(y2)
y_standardized = y_temp2.astype(float)

from sklearn.preprocessing import StandardScaler

scl = StandardScaler()

X2 = scl.fit_transform(X2)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2,test_size=.2,random_state=42)





encoder = OrdinalEncoder()


encoder3.fit(X_train[categorical_columns])
'''
minh se dung scaler nhu nguoi ta
'''
from sklearn.preprocessing import StandardScaler

scl = StandardScaler()

X_train_scl = scl.fit_transform(X_train2)
X_test_scl = scl.fit_transform(X_test2)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 

'''
X_train2 = X_train.copy()
X_test2 = X_test.copy()
y_train2 = y_train.copy()
y_test2 = y_test.copy()
'''

#encoder2 = OrdinalEncoder()

#encoder2.fit(X_train2[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']])

#X_train2[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']] = encoder2.transform(X_train[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']])
                         
#X_test2[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']] = encoder2.transform(X_test[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']])


if 'Model' in categorical_columns:
    categorical_columns.remove('Model')


# 2. Fit encoder trên tập huấn luyện
encoder3 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder3.fit(X_train[categorical_columns])

# 3. Transform cả train và test
X_train[categorical_columns] = encoder3.transform(X_train[categorical_columns])
X_test[categorical_columns] = encoder3.transform(X_test[categorical_columns])


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Danh gia mo hinh
mse = mean_squared_error(y_test, predictions)
r2_square = r2_score(y_test,predictions)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df_data['Price'], bins=20, kde=True)
plt.title('Phân bố giá')
plt.xlabel('Giá')
plt.ylabel('Số lượng')
plt.show()
print(df_data['Price'].head())



#Replace Outliners in "Price" col
Q1 = df_data['Price'].quantile(0.25)
Q3 = df_data['Price'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_price = df_data['Price'].median()
df_data['Price'] = df_data['Price'].apply(lambda x: median_price if x < lower_bound or x > upper_bound else x)

# xem lai outliners
Q1 = df_data['Price'].quantile(0.25)
Q3 = df_data['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_data[(df_data['Price'] < lower_bound) | (df_data['Price'] > upper_bound)]
print(outliers)
# ke doan outliners nay di
