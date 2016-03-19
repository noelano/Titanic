""" First test ML model
Author : Noel
Date : 19th March 2016

"""

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Load training data
df = pd.read_csv(r'..\Data\train2.csv')
df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

train = df.values

# Grab the mean fare value for later use
mean_fare = df['Fare'].mean()

# Load test data
df = pd.read_csv(r'..\Data\test2.csv')
ids = df['PassengerId'].values

df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

# Have one record with null fare
# We'll replace with the mean value
df.loc[df['Fare'].isnull(), 'Fare'] = mean_fare

test = df.values

# Build model
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train[0::,1::], train[0::,0] )

output = forest.predict(test).astype(int)

# Save predictions to csv
predictions_file = open("Forest1_Output.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
