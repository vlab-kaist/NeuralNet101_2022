import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

filename = "Breast_cancer_train.csv"
nb_epochs = 10000

def get_prediction(x_predict):
    return prediction

if __name__=="__main__":
    x_test = torch.tensor([[17.99,10.38,122.8,1001,0.1184],[13.54,14.36,87.46,566.3,0.09779]], dtype=torch.float32)
    print(get_prediction(x_test))