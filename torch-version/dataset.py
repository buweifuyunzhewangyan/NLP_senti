import os

import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd

# 重新定义collate_fn
def collate_fn(batch):
    """
    :param batch: (一个__getitem__[tokens, label], 一个__getitem__[tokens, label],..., batch_size个)
    :return:
    """
    content, label = list(zip(*batch))
    from lib import ws, max_len
    content = [ws.transform(i, max_len=max_len) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content, label


class sentiDataSet(Dataset):
    '''data_path:数据所在路径'''
    def __init__(self,data_path):
        self.data_path = data_path
        self.txts = []
        self.labels = []
        data = pd.read_csv(data_path, sep='\t')
        self.txts = data['text_a'].tolist()
        self.labels = data['label'].tolist()


    def __getitem__(self, index):
        tokens =self.txts[index]
        labels =self.labels[index]

        return tokens,labels