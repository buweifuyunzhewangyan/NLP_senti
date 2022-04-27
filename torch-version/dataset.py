import os
import re

import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from models import lib


# 分词
def tokenlize(content):
    content = re.sub(r"<.*?>", " ", content)
    filters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?',
               '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    content = re.sub("|".join(filters), " ", content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens

# 重新定义collate_fn
def collate_fn(batch):
    """
    :param batch: (一个__getitem__[tokens, label], 一个__getitem__[tokens, label],..., batch_size个)
    :return:
    """
    content, label = list(zip(*batch))
    ws, max_len = lib.ws,lib.max_len

    content = [ws.to_indices(i) for i in content]
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
        tokens = tokenlize(tokens)
        labels =self.labels[index]

        return tokens,labels

    def __len__(self):
        return len(self.labels)

def getDataLoader(data_path):
    return DataLoader(sentiDataSet(data_path),shuffle=True, batch_size=2, collate_fn=collate_fn)