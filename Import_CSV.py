'''
import csv
with open('C:/Users/rayru/OneDrive/Columbia University MS/EECS E6893 Topics in information processing_big data/Final Project/train.csv/train_sample.csv', newline='') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in readCSV:
        print(row)
        i += 1
        if i == 3:
            break
'''

import csv,sys,os,glob
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

df = pd.read_csv('C:/Users/rayru/OneDrive/Columbia University MS/EECS E6893 Topics in information processing_big data/Final Project/train.csv/train.csv')
print(len(df))

fre = df.song_id.value_counts()

df1 = pd.read_csv('C:/Users/rayru/OneDrive/Columbia University MS/EECS E6893 Topics in information processing_big data/Final Project/train.csv/frequency.csv')
#print(df1)

dataset = pd.merge(df,df1, how="left",on="song_id")
print(dataset)
dataset.to_csv('Visual.csv')

"""
train_data = np.array(df)   #np.ndarray()
train_x_list = train_data.tolist()    #list
print(train_x_list[0][5])

SongID = []
for i in range(0,len(df)):
    SongID.append(train_x_list[i][0])
#print(SongID)
#print(SongID.value_counts())
"""

"""
SongID = []
for i in range(0,len(df):
    SongID = SongID + train_x_list[i][0]
    #print(SongID)
"""
