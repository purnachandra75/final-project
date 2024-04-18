
import pandas as pd
import numpy as np
import random
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

cropdf = pd.read_csv("C:/Users/DELL/Downloads/newdataset.csv")
cropdf['Season'] = cropdf['Season'].replace('Rabi', 'rabi')
cropdf['Season'].unique()

cropdf = cropdf.dropna(subset=['humudity'])
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
cropdf['Season'] = label_encoder.fit_transform(cropdf['Season'])

X = cropdf.drop('crop', axis=1)
y = cropdf['crop']
crop=['Maize(Rabi)', 'Rice', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee',
       'Moong(Green Gram)', 'Urad', 'Groundnut', 'Sugarcane', 'Wheat',
       'Rapeseed & Mustard', 'Arhar/Tur', 'Gram', 'Jowar', 'Onion',
       'Potato', 'Chillies', 'Sunflower', 'Bajra',
       'Peas & beans (Pulses)', 'Barley', 'Coriander', 'Cowpea(lobia)',
       'Garlic', 'Linseed', 'Masoor', 'Ragi', 'Safflower', 'Turmeric']

Y=[]
for i in y:
    Y.append(crop.index(i))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,shuffle = True, random_state = 0)

clf=XGBClassifier(n_estimators=33,random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#yeild_prediction 
'''print(X,y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle = True, random_state = 0)
print(y_train,y_test)

#random forest...
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn import metrics
acc1=[]
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(X_train,y_train)

predicted_values = RF.predict(X_test)

x = metrics.accuracy_score(y_test, predicted_values)
acc1.append(x)
#model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(y_test,predicted_values))
Training_sets_score=str(RF.score(X_train, y_train)*100)
Test_set_score=str(RF.score(X_test, y_test)*100)

print('Training set score: {:.4f}'.format(RF.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(RF.score(X_test, y_test)))'''

train = pd.read_csv('C:/Users/DELL/Downloads/crop_production.csv')
train.loc[train['Crop']=='Arecanut', 'Crop'] = 0
train.loc[train['Crop']=='Other Kharif pulses', 'Crop'] =1
train.loc[train['Crop']=='Rice', 'Crop'] =2
train.loc[train['Crop']=='Banana', 'Crop'] = 3
train.loc[train['Crop']=='Cashewnut', 'Crop'] =4
train.loc[train['Crop']=='Coconut ', 'Crop'] = 5
train.loc[train['Crop']=='Dry ginger', 'Crop'] =6
train.loc[train['Crop']=='Sugarcane', 'Crop'] = 7
train.loc[train['Crop']=='Sweet potato', 'Crop'] =8
train.loc[train['Crop']=='Tapioca', 'Crop'] = 9
train.loc[train['Crop']=='Black pepper', 'Crop'] =10
train.loc[train['Crop']=='Dry chillies', 'Crop'] =11
train.loc[train['Crop']=='other oilseeds', 'Crop'] =12
train.loc[train['Crop']=='Turmeric', 'Crop'] =13
train.loc[train['Crop']=='Maize', 'Crop'] =14
train.loc[train['Crop']=='Moong(Green Gram)', 'Crop'] =15

train.loc[train['Crop']=='Urad', 'Crop'] =16
train.loc[train['Crop']=='Arhar/Tur', 'Crop'] =17

train.loc[train['Crop']=='Groundnut', 'Crop'] =18
train.loc[train['Crop']=='Sunflower', 'Crop'] =19
train.loc[train['Crop']=='Bajra', 'Crop'] =20
train.loc[train['Crop']=='Castor seed', 'Crop'] =21
train.loc[train['Crop']=='Cotton(lint)', 'Crop'] =22
train.loc[train['Crop']=='Horse-gram', 'Crop'] =23
train.loc[train['Crop']=='Jowar', 'Crop'] =24
train.loc[train['Crop']=='Korra', 'Crop'] =25
train.loc[train['Crop']=='Ragi', 'Crop'] =26
train.loc[train['Crop']=='Tobacco', 'Crop'] =27
train.loc[train['Crop']=='Gram', 'Crop'] =28
train.loc[train['Crop']=='Wheat', 'Crop'] =29
train.loc[train['Crop']=='Masoor', 'Crop'] =30
train.loc[train['Crop']=='Sesamum', 'Crop'] =31
train.loc[train['Crop']=='Linseed', 'Crop'] =32
train.loc[train['Crop']=='Safflower', 'Crop'] =33
train.loc[train['Crop']=='Onion', 'Crop'] =34
train.loc[train['Crop']=='other misc. pulses', 'Crop'] =35
train.loc[train['Crop']=='Samai', 'Crop'] =36
train.loc[train['Crop']=='Small millets', 'Crop'] =37
train.loc[train['Crop']=='Coriander', 'Crop'] =38
train.loc[train['Crop']=='Potato', 'Crop'] =39
train.loc[train['Crop']=='Other  Rabi pulses', 'Crop'] =40
train.loc[train['Crop']=='Soyabean', 'Crop'] =41
train.loc[train['Crop']=='Beans & Mutter(Vegetable)', 'Crop'] =42
train.loc[train['Crop']=='Bhindi', 'Crop'] =43
train.loc[train['Crop']=='Brinjal', 'Crop'] =44
train.loc[train['Crop']=='Citrus Fruit', 'Crop'] =45
train.loc[train['Crop']=='Cucumber', 'Crop'] =46
train.loc[train['Crop']=='Grapes', 'Crop'] =47
train.loc[train['Crop']=='Mango', 'Crop'] =48
train.loc[train['Crop']=='Orange', 'Crop'] =49
train.loc[train['Crop']=='other fibres', 'Crop'] =50
train.loc[train['Crop']=='Other Fresh Fruits', 'Crop'] =51
train.loc[train['Crop']=='Other Vegetables', 'Crop'] =52
train.loc[train['Crop']=='Papaya', 'Crop'] =53
train.loc[train['Crop']=='Pome Fruit', 'Crop'] =54
train.loc[train['Crop']=='Tomato', 'Crop'] =55
train.loc[train['Crop']=='Rapeseed &Mustard', 'Crop'] =56
train.loc[train['Crop']=='Mesta', 'Crop'] =57
train.loc[train['Crop']=='Cowpea(Lobia)', 'Crop'] =58
train.loc[train['Crop']=='Lemon', 'Crop'] =59
train.loc[train['Crop']=='Pome Granet', 'Crop'] =60
train.loc[train['Crop']=='Sapota', 'Crop'] =61
train.loc[train['Crop']=='Cabbage', 'Crop'] =62
train.loc[train['Crop']=='Peas  (vegetable)', 'Crop'] =63
train.loc[train['Crop']=='Niger seed', 'Crop'] =64
train.loc[train['Crop']=='Bottle Gourd', 'Crop'] =65
train.loc[train['Crop']=='Sannhamp', 'Crop'] =66
train.loc[train['Crop']=='Varagu', 'Crop'] =67
train.loc[train['Crop']=='Garlic', 'Crop'] =68
train.loc[train['Crop']=='Ginger', 'Crop'] =69
train.loc[train['Crop']=='Oilseeds total', 'Crop'] =70
train.loc[train['Crop']=='Pulses total', 'Crop'] =71
train.loc[train['Crop']=='Jute', 'Crop'] =72
train.loc[train['Crop']=='Peas & beans (Pulses)', 'Crop'] =73
train.loc[train['Crop']=='Blackgram', 'Crop'] =74
train.loc[train['Crop']=='Paddy', 'Crop'] =75
train.loc[train['Crop']=='Pineapple', 'Crop'] =76
train.loc[train['Crop']=='Barley', 'Crop'] =77
train.loc[train['Crop']=='Khesari', 'Crop'] =78
train.loc[train['Crop']=='Guar seed', 'Crop'] =79
train.loc[train['Crop']=='Moth', 'Crop'] =80
train.loc[train['Crop']=='Other Cereals & Millets', 'Crop'] =81
train.loc[train['Crop']=='Cond-spcs other', 'Crop'] =82
train.loc[train['Crop']=='Turnip', 'Crop'] =83
train.loc[train['Crop']=='Carrot', 'Crop'] =84
train.loc[train['Crop']=='Redish', 'Crop'] =85
train.loc[train['Crop']=='Arcanut (Processed)', 'Crop'] =87
train.loc[train['Crop']=='Atcanut (Raw)', 'Crop'] =88
train.loc[train['Crop']=='Cashewnut Processed', 'Crop'] =86
train.loc[train['Crop']=='Cashewnut Raw', 'Crop'] =89
train.loc[train['Crop']=='Cardamom', 'Crop'] =90
train.loc[train['Crop']=='Rubber', 'Crop'] =91
train.loc[train['Crop']=='Bitter Gourd', 'Crop'] =92
train.loc[train['Crop']=='Drum Stick', 'Crop'] =93
train.loc[train['Crop']=='Jack Fruit', 'Crop'] =94
train.loc[train['Crop']=='Snak Guard', 'Crop'] =95
train.loc[train['Crop']=='Pump Kin', 'Crop'] =96
train.loc[train['Crop']=='Tea', 'Crop'] =97
train.loc[train['Crop']=='Coffee', 'Crop'] =98
train.loc[train['Crop']=='Cauliflower', 'Crop'] =99
train.loc[train['Crop']=='Other Citrus Fruit', 'Crop'] =100
train.loc[train['Crop']=='Water Melon', 'Crop'] =101
train.loc[train['Crop']=='Total foodgrain', 'Crop'] =102
train.loc[train['Crop']=='Kapas', 'Crop'] =103
train.loc[train['Crop']=='Colocosia', 'Crop'] =104
train.loc[train['Crop']=='Lentil', 'Crop'] =105
train.loc[train['Crop']=='Bean', 'Crop'] =106
train.loc[train['Crop']=='Jobster', 'Crop'] =107
train.loc[train['Crop']=='Perilla', 'Crop'] =108
train.loc[train['Crop']=='Rajmash Kholar', 'Crop'] =109
train.loc[train['Crop']=='Ricebean (nagadal)', 'Crop'] =110
train.loc[train['Crop']=='Ash Gourd', 'Crop'] =111
train.loc[train['Crop']=='Beet Root', 'Crop'] =112
train.loc[train['Crop']=='Lab-Lab', 'Crop'] =113
train.loc[train['Crop']=='Ribed Guard', 'Crop'] =114
train.loc[train['Crop']=='Yam', 'Crop'] =115
train.loc[train['Crop']=='Apple', 'Crop'] =116
train.loc[train['Crop']=='Peach', 'Crop'] =117
train.loc[train['Crop']=='Pear', 'Crop'] =118
train.loc[train['Crop']=='Plums', 'Crop'] =119
train.loc[train['Crop']=='Litchi', 'Crop'] =120
train.loc[train['Crop']=='Ber', 'Crop'] =121
train.loc[train['Crop']=='Jute & mesta', 'Crop'] =122
train.loc[train['Crop']=='Other Dry Fruit', 'Crop'] =123

train.loc[train['State_Name']=='Andaman and Nicobar Islands', 'State_Name'] = 0
train.loc[train['State_Name']=='Andhra Pradesh', 'State_Name'] =1
train.loc[train['State_Name']=='Assam', 'State_Name'] =3
train.loc[train['State_Name']=='Goa', 'State_Name'] =4
train.loc[train['State_Name']=='Karnataka', 'State_Name'] =5
train.loc[train['State_Name']=='Kerala', 'State_Name'] =6
train.loc[train['State_Name']=='Meghalaya', 'State_Name'] =7
train.loc[train['State_Name']=='Puducherry', 'State_Name'] =8
train.loc[train['State_Name']=='Tamil Nadu', 'State_Name'] =9
train.loc[train['State_Name']=='West Bengal', 'State_Name'] =10
train.loc[train['State_Name']=='Bihar', 'State_Name'] =11
train.loc[train['State_Name']=='Chhattisgarh', 'State_Name'] =12
train.loc[train['State_Name']=='Dadra and Nagar Haveli', 'State_Name'] =13
train.loc[train['State_Name']=='Gujarat', 'State_Name'] =14
train.loc[train['State_Name']=='Haryana', 'State_Name'] =15
train.loc[train['State_Name']=='Madhya Pradesh', 'State_Name'] =16
train.loc[train['State_Name']=='Maharashtra', 'State_Name'] =17
train.loc[train['State_Name']=='Manipur', 'State_Name'] =18
train.loc[train['State_Name']=='Rajasthan', 'State_Name'] =19
train.loc[train['State_Name']=='Telangana ', 'State_Name'] =20
train.loc[train['State_Name']=='Uttar Pradesh', 'State_Name'] =21
train.loc[train['State_Name']=='Arunachal Pradesh', 'State_Name'] =22
train.loc[train['State_Name']=='Himachal Pradesh', 'State_Name'] =23
train.loc[train['State_Name']=='Jammu and Kashmir ', 'State_Name'] =24
train.loc[train['State_Name']=='Nagaland', 'State_Name'] =25
train.loc[train['State_Name']=='Odisha', 'State_Name'] =26
train.loc[train['State_Name']=='Uttarakhand', 'State_Name'] =27
train.loc[train['State_Name']=='Mizoram', 'State_Name'] =28
train.loc[train['State_Name']=='Punjab', 'State_Name'] =29
train.loc[train['State_Name']=='Tripura', 'State_Name'] =30
train.loc[train['State_Name']=='Chandigarh', 'State_Name'] =31
train.loc[train['State_Name']=='Jharkhand', 'State_Name'] =32
train.loc[train['State_Name']=='Sikkim', 'State_Name'] =33          

train.loc[train['Season']=='Kharif     ', 'Season'] =1
train.loc[train['Season']=='Whole Year ', 'Season'] =2
train.loc[train['Season']=='Autumn     ', 'Season'] =3
train.loc[train['Season']=='Rabi       ', 'Season'] =4
train.loc[train['Season']=='Summer     ', 'Season'] =5
train.loc[train['Season']=='Winter     ', 'Season'] =6

p=train[["Crop","Area","State_Name","Season","Crop_Year"]]
q=train["Production"]

from sklearn.model_selection import train_test_split

p_train,p_test,q_train,q_test=train_test_split(p,q,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=16, random_state=0)
from sklearn.impute import SimpleImputer

# Impute NaN values with the mean
imputer = SimpleImputer(strategy='mean')
q_imputed = imputer.fit_transform(q.values.reshape(-1, 1))

# Convert back to DataFrame
q_imputed = pd.DataFrame(q_imputed, columns=['Production'])

# Now you can fit your model using y_imputed
regressor.fit(p, q_imputed.values.ravel())

def predicton(list):
    newdata=clf.predict([list])
    newdata
    print(newdata)
    l=newdata[0]
    print(l)
    return crop[l]
def yeildpreduction(l1):
    predictionss = regressor.predict([l1])
    print("Predictions:", predictionss)
    pred = format(int(predictionss[0]))
    print("Formatted Prediction:", pred)
    print("Formatted yield:", int(pred)/l1[2])
    return(pred)
    




    