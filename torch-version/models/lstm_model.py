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
        self.softmax = nn.Softmax(dim=1)

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

    def forward_interpet(self,text, seq_len):
        #embedding层
        embedded_text = self.embedding(text)
        embedded_text.retain_grad()

        x, (h_n, c_n) = self.lstm(embedded_text)
        output_fw = h_n[-2, :, :]  # 正向最后一次输出
        # output = x[:,0,hidden_size:]   反向，等同下方
        output_bw = h_n[-1, :, :]  # 反向最后一次输出
        #  只要最后一个lstm单元处理的结果，这里去掉了hidden state
        output = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size, hidden_size*num_direction]

        out = self.fc(output)
        prob = self.softmax(out)

        return embedded_text,prob

'''CNN+biLSTM网络'''
class cnn_biLSTM(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,hidden_size,num_layer,bidirectional,dropout,num_classes):
        super().__init__()
        self.num_embeddings = 12089
        self.cnn = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=1
        )
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self,input):
        '''前向传播'''
        x = torch.unsqueeze(input,dim=1)
        x = torch.as_tensor(x,dtype=torch.float)
        x = self.cnn(x)
        x = torch.squeeze(x,dim=1)
        x = torch.as_tensor(x, dtype=torch.long)
        # print(self.embedding.num_embeddings)
        self.embedding.num_embeddings=self.num_embeddings
        x = self.embedding(x)
        x, (h_n, c_n) = self.lstm(x)
        output_fw = h_n[-2, :, :]  # 正向最后一次输出
        # output = x[:,0,hidden_size:]   反向，等同下方
        output_bw = h_n[-1, :, :]  # 反向最后一次输出
        #  只要最后一个lstm单元处理的结果，这里去掉了hidden state
        output = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size, hidden_size*num_direction]

        out = self.fc(output)

        return F.log_softmax(out, dim=-1)

    def forward_interpet(self,text, seq_len):
        x = torch.unsqueeze(text, dim=1)
        x = torch.as_tensor(x, dtype=torch.float)
        x = self.cnn(x)
        x = torch.squeeze(x, dim=1)
        x = torch.as_tensor(x, dtype=torch.long)
        #embedding层
        embedded_text = self.embedding(x)
        embedded_text.retain_grad()

        x, (h_n, c_n) = self.lstm(embedded_text)
        output_fw = h_n[-2, :, :]  # 正向最后一次输出
        # output = x[:,0,hidden_size:]   反向，等同下方
        output_bw = h_n[-1, :, :]  # 反向最后一次输出
        #  只要最后一个lstm单元处理的结果，这里去掉了hidden state
        output = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size, hidden_size*num_direction]

        out = self.fc(output)
        prob = self.softmax(out)

        return embedded_text,prob

'''CNN+biLSTM网络'''
class cnn_biLSTM2(nn.Module):
    def __init__(self,cnn_outchannels,num_embeddings,embedding_dim,hidden_size,num_layer,bidirectional,dropout,num_classes):
        super().__init__()
        self.cnn = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=cnn_outchannels,
            kernel_size=2,
            stride=1
        )
        self.embedding = nn.Embedding(
            # 字典的长度
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=cnn_outchannels, hidden_size=hidden_size, num_layers=num_layer,
            batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size*2,num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,input):
        '''前向传播'''
        x = self.embedding(input)
        x = x.permute(0,2,1)
        x = self.cnn(x)
        x = x.permute(0,2,1)

        x, (h_n, c_n) = self.lstm(x)
        output_fw = h_n[-2, :, :]  # 正向最后一次输出
        # output = x[:,0,hidden_size:]   反向，等同下方
        output_bw = h_n[-1, :, :]  # 反向最后一次输出
        #  只要最后一个lstm单元处理的结果，这里去掉了hidden state
        output = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size, hidden_size*num_direction]

        out = self.fc(output)

        return F.log_softmax(out, dim=-1)

    def forward_interpet(self,text, seq_len):
        '''前向传播'''
        embedded_text = self.embedding(text)
        embedded_text.retain_grad()
        x = embedded_text.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        x, (h_n, c_n) = self.lstm(x)
        output_fw = h_n[-2, :, :]  # 正向最后一次输出
        # output = x[:,0,hidden_size:]   反向，等同下方
        output_bw = h_n[-1, :, :]  # 反向最后一次输出
        #  只要最后一个lstm单元处理的结果，这里去掉了hidden state
        output = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size, hidden_size*num_direction]

        out = self.fc(output)
        prob = self.softmax(out)

        return embedded_text,prob