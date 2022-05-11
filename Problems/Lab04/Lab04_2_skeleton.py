import torch
import pandas as pd
import torch.nn as nn

def main():
    num_epochs = 500
    filename = "Iris.csv"
    info = {"Iris-setosa":[1,0,0], "Iris-versicolor":[0,1,0], "Iris-virginica":[0,0,1]}

    df = pd.read_csv(filename)
    x_train = torch.tensor(df.loc[:, ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values.tolist(), dtype=torch.float32)
    y_train = torch.tensor([info[i] for i in df['Species'].values.tolist()], dtype=torch.float32)

    'Write your code here'

    for epoch in range(num_epochs):
        'Write your code here'

    return model