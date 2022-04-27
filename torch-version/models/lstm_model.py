import torch
import torch.nn as nn
import lib
import torch.nn.functional as F


'''双向LSTM网络'''
class biLSTM(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,hidden_size,num_layer,bidirectional,dropout,num_classes):
        super().__init__()
        self.embedding = nn.Embedding(
            # 字典的长度
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layer,
            batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size*2,num_classes)

    def forward(self,input):
        '''前向传播'''
        x = self.embedding(input)
        x, (h_n, c_n) = self.lstm(x)
        output_fw = h_n[-2, :, :]  # 正向最后一次输出
        # output = x[:,0,hidden_size:]   反向，等同下方
        output_bw = h_n[-1, :, :]  # 反向最后一次输出
        #  只要最后一个lstm单元处理的结果，这里去掉了hidden state
        output = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size, hidden_size*num_direction]

        out = self.fc(output)

        return F.log_softmax(out, dim=-1)