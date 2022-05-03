import json
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import lib
from models.utils import CharTokenizer
from models.lstm_model import biLSTM

'''数据定义部分--start'''
tokenizer = CharTokenizer(lib.ws, 'ch', './models/punctuations')
def tokenlize(content):
    tokens = tokenizer.encode(content)
    return tokens

# 重新定义collate_fn
def collate_fn(batch):
    """
    :param batch: (一个__getitem__[tokens, label], 一个__getitem__[tokens, label],..., batch_size个)
    :return:
    """
    id,content, content_len = list(zip(*batch))
    ws, max_len = lib.ws,lib.max_len

    content2 = []
    for i in content:
        if len(i)<max_len:
            while len(i)<max_len:
                i.append(0)
        else:
            i = i[0:max_len]
        content2.append(i)

    content = torch.LongTensor(content2)
    content_len = torch.LongTensor(content_len)
    return id,content, content_len

class interpreDataSet(Dataset):
    '''data_path:数据所在路径'''
    def __init__(self,data_path):
        self.data_path = data_path
        self.ids = []
        self.context = []
        self.sent_token = []
        with open(data_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                line_split = json.loads(line)
                self.ids.append(line_split['id'])
                self.context.append(line_split['context'])
                self.sent_token.append(line_split['sent_token'])


    def __getitem__(self, index):
        id = self.ids[index]
        tokens =self.context[index]
        tokens = tokenlize(tokens)
        #返回tokens以及tokens的长度
        return id,tokens,len(tokens)

    def __len__(self):
        return len(self.context)

def getInterDataLoader(data_path,batch_size):
    return DataLoader(interpreDataSet(data_path),shuffle=False, batch_size=batch_size,collate_fn=collate_fn)

'''数据定义部分--end'''


def init_lstm_var():
    vocab = lib.ws
    padding_idx = vocab.token_to_idx.get('[PAD]', 0)
    # 定义自己的模型
    model = biLSTM(
        num_embeddings=len(vocab),
        embedding_dim=lib.embedding_dim,
        hidden_size=lib.hidden_size,
        num_layer=lib.num_layer,
        bidirectional=lib.bidirectional,
        dropout=lib.drop_out,
        num_classes=lib.num_classes
    )

    # Reads data and generates mini-batches.
    test_data_path = '../data-part-1/senti_ch_part1.txt'
    test_loader = getInterDataLoader(test_data_path,batch_size=1)

    return model, tokenizer, test_loader


if __name__=='__main__':

    '''lstm模型部分'''
    model, tokenizer, dataloader = init_lstm_var()

    # 加载模型
    PATH = './model/model_9.pth'
    state = torch.load(PATH)
    model.load_state_dict(state['model_state'])
    # model.train()

    '''结果文件'''
    result_txt = './out_put/result_lstm.txt'
    if os.path.exists(result_txt):
        os.remove(result_txt)

    for step, d in tqdm(enumerate(dataloader)):
        #id：每一句对应的索引；text：中文编码为数字的数组；seq_len：数组长度
        id,text,seq_len = d
        fwd_args = [text, seq_len]
        embedded,output = model.forward_interpet(*fwd_args)

        pred_label = torch.argmax(output, axis=-1).tolist()[0]
        predicted_class_prob = output[0][pred_label]
        predicted_class_prob.backward()
        inter_score = embedded.grad
        inter_score = inter_score.sum(-1)

        #处理得分，得到最终结果
        inter_score = inter_score[0][0:seq_len]
        inter_score = torch.abs(inter_score).tolist()
        sum_score = sum(inter_score)
        inter_score = [i/sum_score for i in inter_score]

        #输出结果
        char_list = []
        for i,score in enumerate(inter_score):
            if float(score)>=0.05:
                char_list.append(i)
        char_list = str(char_list)
        with open(result_txt, 'a') as f:
            f.write(str(id[0]) + "\t")
            f.write(str(pred_label) + "\t")
            f.write(char_list[1:-1] + "\n")






