import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import pickle

df = pd.read_csv('diabetes.csv')
# delete the Pregnancies col to make it for both female and male
df = df.drop('Pregnancies', axis=1)
# separating the data (x , y)
x = df.drop(columns= 'Outcome' , axis=1)
y = df['Outcome']

#standrize the data

scaler = StandardScaler()
scaler.fit(x)

standrized_data = scaler.transform(x)
x = standrized_data

#  train test split

x_train, x_test, y_train, y_test = train_test_split(x,y , random_state=0 ,stratify= y, test_size=0.2)

# train the model

classifier = svm.SVC(kernel='linear')

classifier.fit(x_train,y_train)

# check the accuracy

x_test_prediction = classifier.predict(x_test)
training_data_accuracy = accuracy_score(x_test_prediction , y_test)

print('the accuracy score of the data is :' , training_data_accuracy)

# making a predictive system

input = (85,78,0,0,31.2,0.382,42)

input_as_numpy_array = np.asarray(input)

input_as_numpy_array_reshape = input_as_numpy_array.reshape(1 , -1)

std_data = scaler.transform(input_as_numpy_array_reshape)

prediction = classifier.predict(std_data)
if prediction[0] == 0:
    print(' you dont have a diabetec')
else :
    print('you have a diabetec')


pickle.dump(classifier,open('classifier.pkl', 'wb'))