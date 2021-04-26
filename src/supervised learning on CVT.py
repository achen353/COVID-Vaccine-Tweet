# import pandas as pd
# data = pd.read_csv("result.csv")
# # data
# print(data.shape)
# data = data.drop(index=[25391,35044,49475])
# print(data.shape)
# X = data["text"]
# label = data["compound"]
# print(X.shape)
# print(label.shape)
# import numpy as np
# X = np.array(X)
# label = np.array(label)
# label = np.sign(label)
# print(X)
# print(label)
# newlabel = label + 1
# print(newlabel)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
import torchtext.vocab as torchvocab
from torch.autograd import Variable
import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
import time
import random
#import snowballstemmer
import collections
from collections import Counter
from nltk.corpus import stopwords
from itertools import chain
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, newlabel, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
# import nltk.tokenize as tk
#
# train_tokenized = []
# test_tokenized = []
# val_tokenized = []
# train_cnt = 0
# val_cnt = 0
# test_cnt = 0
# cnt = 0
# train_null = []
# test_null = []
# val_null = []
# tokenizer = tk.WordPunctTokenizer()
# maxlen = 0
# for review in X_train:
#     #     print(type(review))
#     try:
#         tokens = tokenizer.tokenize(review)
#         #         print(cnt)
#         train_cnt += 1
#         train_tokenized.append(tokens)
#         if len(tokens) > maxlen:
#             maxlen = len(tokens)
#         train_null.append(cnt)
#     except:
#
#         print(cnt)
#         print(review)
#     #     print(tokens)
#     cnt += 1
# cnt = 0
# for review in X_test:
#     try:
#         tokens = tokenizer.tokenize(review)
#         test_tokenized.append(tokens)
#         test_cnt += 1
#         test_null.append(cnt)
#     except:
#
#         print(review)
#     cnt += 1
# cnt = 0
# for review in X_val:
#     try:
#         tokens = tokenizer.tokenize(review)
#         val_tokenized.append(tokens)
#         val_cnt += 1
#         val_null.append(cnt)
#     except:
#
#         print(review)
#     cnt += 1
#
# print(maxlen)
# # print(train_tokenized)
# vocab = set(chain(*train_tokenized))
# vocab_size = len(vocab)
# print(vocab_size)
#
# word_to_idx  = {word: i+1 for i, word in enumerate(vocab)}
# word_to_idx['<unk>'] = 0
# idx_to_word = {i+1: word for i, word in enumerate(vocab)}
# idx_to_word[0] = '<unk>'
# import json
# # # cvt
# with open("./word_to_idx.json", 'w') as file_obj:
#     json.dump(word_to_idx, file_obj)
# with open("./idx_to_word.json", 'w') as file_obj:
#     json.dump(idx_to_word, file_obj)
# def encode_samples(tokenized_samples, vocab):
#     features = []
#     for sample in tokenized_samples:
#         feature = []
#         for token in sample:
#             if token in word_to_idx:
#                 feature.append(word_to_idx[token])
#             else:
#                 feature.append(0)
#         features.append(feature)
#     return features
#
# def pad_samples(features, maxlen=50, PAD=0):
#     padded_features = []
#     for feature in features:
#         if len(feature) >= maxlen:
#             padded_feature = feature[:maxlen]
#         else:
#             padded_feature = feature
#             while(len(padded_feature) < maxlen):
#                 padded_feature.append(PAD)
#         padded_features.append(padded_feature)
#     return padded_features
# train_features = torch.tensor(pad_samples(encode_samples(train_tokenized, vocab)))
# train_labels = torch.tensor(y_train)[train_null]
# test_features = torch.tensor(pad_samples(encode_samples(test_tokenized, vocab)))
# test_labels = torch.tensor(y_test)[test_null]
# val_features = torch.tensor(pad_samples(encode_samples(val_tokenized, vocab)))
# val_labels = torch.tensor(y_val)[val_null]
# torch.save(train_features, "./train_features_cvt.pt")
# torch.save(val_features, "./val_features_cvt.pt")
# torch.save(test_features, "./test_features_cvt.pt")
# torch.save(train_labels, "./train_labels_cvt.pt")
# torch.save(val_labels, "./val_labels_cvt.pt")
# torch.save(test_labels, "./test_labels_cvt.pt")
# print(train_features.shape)
# print(train_labels.shape)
# print(test_features.shape)
# print(test_labels.shape)
# print(val_features.shape)
# print(val_labels.shape)
#
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
train_features = torch.load("./train_features_cvt.pt")
val_features = torch.load("./val_features_cvt.pt")
test_features = torch.load("./test_features_cvt.pt")
train_labels = torch.load("./train_labels_cvt.pt")
val_labels = torch.load("./val_labels_cvt.pt")
test_labels = torch.load("./test_labels_cvt.pt")
import json
with open("./word_to_idx.json", 'r') as f:
    word_to_idx = json.load(f)
with open("./idx_to_word.json", 'r') as f:
    idx_to_word = json.load(f)
# 输入文件
# glove_file = './glove'
# # 输出文件
# tmp_file = get_tmpfile("test_word2vec.txt")
# glove2word2vec(glove_file, tmp_file)
# f = open("test_word2vec.txt", 'w', encoding='utf-8') ##ffilename可以是原来的txt文件，也可以没有然后把写入的自动创建成txt文件
# f.write(tmp_file)
# f.close()
with open("test_word2vec.txt", "r") as f:  # 打开文件
    tmp_file = f.read()  # 读取文件
vocab_size = 70736
#
wvmodelwvmodel = KeyedVectors.load_word2vec_format(tmp_file)
class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu,**kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = True
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs
import numpy as np

embed_size = 300
num_hiddens = 256
num_layers = 1
bidirectional = True

labels = 3
lr = 0.1
device = torch.device("cuda:0")
use_gpu = True
weight = torch.zeros(vocab_size+1, embed_size)
cnt = 0
for i in range(len(wvmodelwvmodel.index_to_key)):
    try:
        index = word_to_idx[wvmodelwvmodel.index_to_key[i]]
        cnt += 1
    except:
        continue
    weight[index, :] = torch.from_numpy(np.random.rand(300))#torch.from_numpy(wvmodelwvmodel.get_vector(wvmodelwvmodel.index_to_key[i])) #
print(cnt)
# torch.save(weight, "./weight_cvt.pt")
# weight = torch.load("./weight_cvt.pt")
net = SentimentNet(vocab_size=(vocab_size+1), embed_size=embed_size,
                   num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, weight=weight,
                   labels=labels, use_gpu=use_gpu)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
lr = 0.05
batch_size = 32

train_set = torch.utils.data.TensorDataset(train_features, train_labels)
test_set = torch.utils.data.TensorDataset(test_features, test_labels)
val_set = torch.utils.data.TensorDataset(val_features, val_labels)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                        shuffle=False)
val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=False)
num_epochs = 10
best_res = {}
best_val = 0.0
losses = []
print("training start")
for epoch in range(num_epochs):

    start = time.time()
    train_loss, test_losses, val_loss = 0, 0, 0
    train_acc, test_acc, val_acc = 0, 0, 0

    n, m = 0, 0

    for feature, label in train_iter:
        n += 1
        if n % 1000 == 999:
            print(n)
        net.zero_grad()
        feature = Variable(feature.to(device))
        label = Variable(label.to(device))
        score = net(feature)
        loss = loss_function(score, label.long())
        loss.backward()
        optimizer.step()
        train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                 dim=1), label.cpu())
        train_loss += loss

    train_acc = train_acc / n
    train_loss.data = train_loss.data / n
    losses.append(train_loss.data)
    with torch.no_grad():
        f1_avg = 0
        for test_feature, test_label in test_iter:
            m += 1
            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
            test_score = net(test_feature)
            test_loss = loss_function(test_score, test_label.long())
            test_acc += accuracy_score(torch.argmax(test_score.cpu().data,
                                                    dim=1), test_label.cpu())
            test_losses += test_loss
            f1_avg += f1_score(test_label.cpu(), torch.argmax(test_score.cpu().data,
                                                              dim=1), average='weighted')
        test_acc = test_acc / m
        test_loss.data = test_loss.data / m

        f1_avg = f1_avg / m
        m = 0
        for val_feature, val_label in val_iter:
            m += 1
            val_feature = val_feature.to(device)
            val_label = val_label.to(device)
            val_score = net(val_feature)
            val_loss = loss_function(val_score, val_label.long())
            val_acc += accuracy_score(torch.argmax(val_score.cpu().data,
                                                   dim=1), val_label.cpu())
            val_loss += val_loss
        val_acc = val_acc / m
        val_loss.data = val_loss.data / m
        if val_acc > best_val:
            best_val = val_acc
            best_res["epoch"] = epoch
            best_res["train_acc"] = train_acc
            best_res["test_acc"] = test_acc
            best_res["test_f1"] = f1_avg
            best_res["model"] = net
            best_res["val_acc"] = val_acc
    end = time.time()
    runtime = end - start
    print(
        'epoch: %d, train loss: %.4f, train acc: %.2f, train loss: %.4f, val acc: %.2f, test loss: %.4f, test acc: %.2f, time: %.2f' %
        (epoch, train_loss.data, train_acc, val_loss.data, val_acc, test_losses.data, test_acc, runtime))
print(best_res)
