import torch
from tqdm import tqdm

from models.lstm_model import cnn_biLSTM, cnn_biLSTM2
from dataset import sentiDataSet,getDataLoader
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import lib
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

#初始化训练参数
epoch = lib.epoch
batch_size = lib.batch_size
lr = lib.lr
device = lib.device
num_embeddings = len(lib.ws)
embedding_dim = lib.embedding_dim
hidden_size = lib.hidden_size
num_layer = lib.num_layer
bidirectional = lib.bidirectional
dropout = lib.drop_out
num_classes = lib.num_classes
save_path = './model/cnn_lstm_model_{}.pth'


# 训练
def train(epoch,train_loader):
    loss_list = []
    for idx, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader), ascii=True,
                                     desc='第{}轮训练'.format(str(epoch))):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss_list.append(float(loss))
        loss.backward()
        optimizer.step()
        # print('当前第%d轮,idx为%d 损失为:%lf, ' % (epoch, idx, loss.item()))

    # 保存模型
    state = {}
    state['model_state'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict(),
    torch.save(state, save_path.format(str(epoch)))

    # 添加loss到tensorboard中
    writer.add_scalar('train_loss', np.mean(loss_list), epoch)

# 验证
def test(epoch,eval_loader):
    acc_list = []
    loss_list = []
    # 开启模型评估模式
    model.eval()
    # tqdm(total = 总数,ascii = #,desc=描述)
    for idx, (input, target) in tqdm(enumerate(eval_loader), total=len(eval_loader), ascii=True, desc='评估：'):
        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            # 计算当前损失
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss)
            pred = output.max(dim=-1)[-1]
            # 计算当前准确率
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    writer.add_scalar('eval_loss', np.mean(loss_list), epoch)
    writer.add_scalar('eval_acc', np.mean(acc_list), epoch)

if __name__ == "__main__":

    #定义训练-验证的数据
    train_loader = getDataLoader('../ChnSentiCorp/train.tsv',batch_size=batch_size)
    eval_loader = getDataLoader('../ChnSentiCorp/dev.tsv',batch_size=batch_size)

    #定义自己的模型
    model = cnn_biLSTM2(
        cnn_outchannels=256,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layer=num_layer,
        bidirectional=bidirectional,
        dropout=dropout,
        num_classes=num_classes
    )
    model = model.to(device)

    # PATH = './model.pth'  # 加载模型
    # state = torch.load(PATH)
    # model.load_state_dict(state['model_state'])

    #定义优化器
    optimizer = Adam(model.parameters(), lr=lr)

    #定义tensorbo
    writer = SummaryWriter('./cnn_lstm_logs')

    for i in range(epoch):
        train(i,train_loader)
        test(i,eval_loader)
    writer.close()