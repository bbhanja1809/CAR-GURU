import pandas as pd
import numpy as np

df = pd.read_csv('dataset/car_new.csv')
print(df.head)
df['mileage'] = df['mileage'].replace(' kmpl', '', regex=True).astype(float)
df['max_power'] = df['max_power'].replace(' bhp', '', regex=True).astype(float)
df['engine'] = df['engine'].replace(' CC', '', regex=True).astype(float)
df['seats'] = df['seats'].astype(int)
df.dropna()
df = df.drop('seller_type',axis = 1)
df = df.drop('torque',axis = 1)
df = df.drop('owner',axis = 1)
df = df.drop('engine',axis = 1)
df = df.drop('max_power',axis = 1)
df = df.drop('year',axis = 1)
df = df.iloc[:400]
df.to_csv("car_final.csv",index=False)