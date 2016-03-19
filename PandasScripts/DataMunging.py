""" Clean the data and add some derived features
Author : Noel
Date : 19th March 2016

"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'..\Data\train.csv')

# Numerical gender identifier:
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Fill missing ages
# We'll do it by gender and class and take the median of each group

df['FilledAges'] = df['Age']
median_ages = np.zeros((2, 3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'FilledAges'] = median_ages[i, j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

# Add new feature for family size
df['FamilySize'] = df['SibSp'] + df['Parch']

# Save changes to file
df.to_csv(r'..\Data\train2.csv', index=False, header=True)