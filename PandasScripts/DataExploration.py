""" Some initial exploratory analysis of the input data
Author : Noel
Date : 15th March 2016

"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'..\Data\train.csv')

print (df.head(5))
print(df.describe())

# Summaries of distinct features
print(df['Survived'].value_counts())
print(df['Pclass'].value_counts())
print(df['Embarked'].value_counts())

# Histograms of some data
plt.figure()
df[['Age', 'Fare', 'Pclass', 'Survived']].hist()
plt.show()

# Compare age with survival and class with survival

plt.scatter(df['Fare'], df['Age'], c=df['Survived'])
plt.legend()
plt.show()
