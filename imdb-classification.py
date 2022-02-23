import spacy
import pandas as pd
import torch
import torchtext
from torchtext.datasets import IMDB

train_iter = IMDB(split='train')

for label, line in train_iter:
    print(label, line)
