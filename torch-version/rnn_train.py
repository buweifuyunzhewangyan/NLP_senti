from models.lstm_model import biLSTM
from dataset import sentiDataSet
from torch.utils.data import DataLoader

data = DataLoader(sentiDataSet('D:\Spyder_project/NLP_senti\ChnSentiCorp/dev.tsv'))

#定义自己的模型
model = biLSTM(
    num_embeddings=12,
    embedding_dim=100,
    hidden_size=128,
    num_layer=2,
    bidirectional=True,
    dropout=0.3,
    num_classes=2
)

print(1)