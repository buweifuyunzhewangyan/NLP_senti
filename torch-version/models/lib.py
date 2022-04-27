import torch

max_len = 200
embedding_dim = 100
hidden_size = 128
num_layer = 2
bidirectional = True
drop_out = 0.3
num_classes = 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')