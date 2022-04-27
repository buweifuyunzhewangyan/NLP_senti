import os
import re
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from models import lib
from models.utils import CharTokenizer


tokenizer = CharTokenizer(lib.ws, 'ch', './models/punctuations')

# 分词
def tokenlize(content):
    tokens = tokenizer.encode(content)
    return tokens

# 重新定义collate_fn
def collate_fn(batch):
    """
    :param batch: (一个__getitem__[tokens, label], 一个__getitem__[tokens, label],..., batch_size个)
    :return:
    """
    content, label = list(zip(*batch))
    ws, max_len = lib.ws,lib.max_len

    content2 = []
    for i in content:
        if len(i)<lib.max_len:
            while len(i)<lib.max_len:
                i.append(0)
        else:
            i = i[0:lib.max_len]
        content2.append(i)

    content = torch.LongTensor(content2)
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

def getDataLoader(data_path,batch_size):
    return DataLoader(sentiDataSet(data_path),shuffle=True, batch_size=batch_size, collate_fn=collate_fn)