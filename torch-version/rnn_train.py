import torch
from tqdm import tqdm

from models.lstm_model import biLSTM
from dataset import sentiDataSet,getDataLoader
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import lib
import numpy as np

import torch.nn.functional as F

#初始化训练参数
epoch = 10
batch_size = 128
device = lib.device

#定义训练-验证的数据
train_loader = getDataLoader('D:\Spyder_project/NLP_senti\ChnSentiCorp/train.tsv',batch_size=batch_size)
eval_loader = getDataLoader('D:\Spyder_project/NLP_senti\ChnSentiCorp/dev.tsv',batch_size=batch_size)

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
    for idx, (input, target) in tqdm(enumerate(train_loader),total=len(train_loader),ascii=True,desc='第{}轮训练'.format(str(epoch))):
        output = model(input)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # print('当前第%d轮,idx为%d 损失为:%lf, ' % (epoch, idx, loss.item()))

        # 保存模型
        if idx % 100 == 0:
            torch.save(model.state_dict(), './model/model.pkl')
            torch.save(optimizer.state_dict(), './model/optimizer.pkl')

def test():
    acc_list = []
    loss_list = []
    # 开启模型评估模式
    model.eval()
    # tqdm(total = 总数,ascii = #,desc=描述)
    for idx, (input, target) in tqdm(enumerate(eval_loader), total=len(eval_loader), ascii=True, desc='评估：'):
        with torch.no_grad():
            output = model(input)
            # 计算当前损失
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss)
            pred = output.max(dim=-1)[-1]
            # 计算当前准确率
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print('准确率为:%lf, 损失为:%lf' % (np.mean(acc_list), np.mean(loss_list)))

if __name__ == "__main__":
    test()
    for i in range(epoch):
        train(i)
        test()