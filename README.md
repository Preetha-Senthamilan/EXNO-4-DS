# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/1df99540-aa7c-4787-b193-60569e955c48)


data.isnull().sum()

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/eebba32d-ad52-41a6-9f44-1c34242f0f3b)

missing=data[data.isnull().any(axis=1)]
missing

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/d5699191-3ff5-4812-a2d4-2d4e3c54fd18)

data2=data.dropna(axis=0)
data2

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/417470e4-66d9-474f-83f4-dd3c6cadd717)

sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/a94a8874-49d2-4866-871c-a434b053e5c7)

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/64bc7d78-84da-4e38-8d0b-53508d24c678)

data2

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/cda10959-dbf2-40c1-8885-fbff9f801149)

new_data=pd.get_dummies(data2, drop_first=True)
new_data

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/54d24c45-bbfc-488b-b793-e36d89f0145f)

columns_list=list(new_data.columns)
print(columns_list)

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/314b36f9-e9d0-4653-9db8-a221346baf5c)


features=list(set(columns_list)-set(['SalStat']))
print(features)

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/94b89286-fa5b-4691-8b13-f283bb8c5303)

y=new_data['SalStat'].values
print(y)

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/4641736f-2e69-4849-bfa2-d5374e12e1d1)

x=new_data[features].values
print(x)

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/2d642735-9038-449a-9f8b-6246d9333e1c)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)


![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/d9ff18c5-ebb5-4f5e-bddc-d04b7bba6cee)

prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/b245869c-47a9-4c15-ae99-614a7d015f95)

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/386a2099-fb03-44a0-96b1-1a0443b241cb)


print("Misclassified Samples : %d" % (test_y !=prediction).sum())

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/72a92aca-f5b7-4031-843c-8cbb2fb1aaa7)

data.shape

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/532dda39-5f8f-42ba-8a37-bbefcc916b1b)

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)


![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/0833bbbc-ce34-43c9-9006-5be365b8364c)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/57523bf1-b45b-44d0-b764-589634ed2f93)

tips.time.unique()

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/6083fc78-a4dd-4298-9736-5761b012d6d8)


contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)


![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/4dba6fe9-8e06-45f6-96b6-6eed26d73e83)

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

![image](https://github.com/Preetha-Senthamilan/EXNO-4-DS/assets/119390282/50965f0c-6cc3-4afe-8fbd-f60247f56f12)









     
# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
