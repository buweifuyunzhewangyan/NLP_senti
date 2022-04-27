import pickle
import sys

import torch

sys.path.append(r'D:\Spyder_project\NLP_senti\torch-version\models')
from word_vocat import Vocab
sys.path.remove(r'D:\Spyder_project\NLP_senti\torch-version\models')


ws = Vocab.load_vocabulary(
       r'D:\Spyder_project\NLP_senti\torch-version\models/vocab.txt' , unk_token='[UNK]', pad_token='[PAD]')
max_len = 200
embedding_dim = 100
hidden_size = 128
num_layer = 2
bidirectional = True
drop_out = 0.3
num_classes = 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')