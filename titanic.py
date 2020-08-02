# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv("/kaggle/input/titanic/train.csv")
print(train.count())

def clean_data(data):
    data.Age = data.Age.fillna(data.Age.dropna().median())
    
    data.loc[data.Sex == "male", "Sex"] = 0
    data.loc[data.Sex == "female", "Sex"] = 1
    
    data.Embarked = data.Embarked.fillna("S")
    data.loc[data.Embarked == "S", "Embarked"] = 0
    data.loc[data.Embarked == "Q", "Embarked"] = 1
    data.loc[data.Embarked == "C", "Embarked"] = 2
    
clean_data(train)

features_array = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

features = np.asarray(train[features_array].values).astype(np.float32)
target = np.asarray(train.Survived.values).astype(np.float32)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(64, input_dim = features.shape[1], activation = "relu"))
model.add(Dense(32, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss = "binary_crossentropy", optimizer = "adam")

model.fit(features, target, epochs = 600, batch_size = 1, verbose = 1)

test = pd.read_csv("/kaggle/input/titanic/test.csv")
clean_data(test)

arr = model.predict(np.asarray(test[features_array]).astype(np.float32))

test_predictions = []
for x in arr:
    if x[0] > 0.5:
        test_predictions.append(1)
    else:
        test_predictions.append(0)

        
submission = pd.DataFrame({
    'PassengerId': np.arange(start=892, stop=892+418, step=1),
    'Survived': test_predictions
})

submission.to_csv('submission-manual-cleansing.csv', index=False)
