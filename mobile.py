
import pandas as pd
print('imported')


data = pd.read_csv("mobile.data")


data.rename(columns={'Operating system':'Operating_system'},inplace = True)
data.rename(columns={'Battery capacity (mAh)':'Batterycapacity'},inplace = True)
data.rename(columns = {"Internal storage (GB)":"Internalstorage","Rear camera":"Rear_camera","Front camera":"Front_camera","RAM (MB)":"RAM"},inplace = True)

data.drop(columns = ['GPS','Number of SIMs','Operating_system','Name','Screen size (inches)','4G/ LTE','3G','Number of SIMs','Bluetooth','Wi-Fi','Unnamed: 0','Model','Resolution x','Resolution y'],inplace = True)  #9 left

data.drop(columns=['Touchscreen','Front_camera','Rear_camera'],inplace=True)

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for i in data.columns:
  if data[i].dtype == 'object':
    data[i] = enc.fit_transform(data[i])


x = data.iloc[:,:5]

y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)

x_train_std = scaler.transform(x_train)

x_test_std = scaler.transform(x_test)

x_test_std.std()

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


models = [LinearRegression(), Ridge(), Lasso(), SVR(kernel = 'linear'), RandomForestRegressor(), DecisionTreeRegressor(), KNeighborsRegressor()]

def compare_models():
  for model in models:
    model.fit(x_train,y_train)
    Y_predict = model.predict(x_test)
    accuracy = r2_score(y_test,Y_predict)*100
    print("For Model:",model,"=",accuracy,'\n')





model=RandomForestRegressor().fit(x_train,y_train)
Y_predict = model.predict(x_test)
accuracy = r2_score(y_test,Y_predict)*100

compare_models()

import pickle
pickle.dump(model, open('mobile.pkl', 'wb'))

print("done")