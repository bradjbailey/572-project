import os  # file paths
import pandas as pd  # read_csv
import torch
from torch.utils.data import DataLoader, Dataset

df = pd.read_csv('data/all-data.csv', delimiter=',', encoding='latin-1')

df = df.rename(columns={'neutral':'sentiment',
                        'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .':'message'})

print(df[0:5])

print(df.shape)

sentiment  = {'positive': 0,'neutral': 1,'negative':2}

df.sentiment = [sentiment[item] for item in df.sentiment]

print(df.sentiment[0:5])
print(df.message[0:5])
print(df["sentiment"])
