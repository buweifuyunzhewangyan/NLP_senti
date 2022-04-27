import torch
from models.lstm_model import biLSTM
from dataset import sentiDataSet,getDataLoader
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import lib

import torch.nn.functional as F

#定义训练-验证的数据
train_loader = getDataLoader('D:\Spyder_project/NLP_senti\ChnSentiCorp/train.tsv',batch_size=128)
eval_loader = getDataLoader('D:\Spyder_project/NLP_senti\ChnSentiCorp/dev.tsv',batch_size=128)

#定义自己的模型
model = biLSTM(
    num_embeddings=len(lib.ws),
    embedding_dim=100,
    hidden_size=128,
    num_layer=2,
    bidirectional=True,
    dropout=0.3,
    num_classes=2
)

#定义优化器
optimizer = Adam(model.parameters(), lr=0.01)


# 训练
def train(epoch):
    for idx, (input, target) in enumerate(train_loader):
        output = model(input)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print(loss.item())
        print('当前第%d轮,idx为%d 损失为:%lf, ' % (epoch, idx, loss.item()))

        # 保存模型
        if idx % 100 == 0:
            torch.save(model.state_dict(), './model/model.pkl')
            torch.save(optimizer.state_dict(), './model/optimizer.pkl')

if __name__ == "__main__":
    epoch = 2
    for i in range(epoch):
        train(i)