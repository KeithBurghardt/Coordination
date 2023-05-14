import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle


class TestData(Dataset):
    def __init__(self):
        self.x = np.arange(13477).reshape(-1,1)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x, y):
        print(x.shape)
        return x



if __name__ == '__main__':
    loader = DataLoader(TestData(), batch_size=32)

    device = 'cuda:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    model = TestModel()
    model = torch.nn.DataParallel(model)
    model.to(device)
    print(f"{torch.cuda.device_count()} GPUs!")

    for idx, batch in enumerate(loader):
        print('-' * 50)
        batch = batch.to(device)
        x = model(batch, None)