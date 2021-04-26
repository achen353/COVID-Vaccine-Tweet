# def read(path):
#     data = []
#     label = []
#     with open(path, "r", encoding="utf8") as f:
#         for line in f:
#             # print(line)
#             words = line.split()
#             new_sentence = ""
#             for i in range(len(words)-2):
#                 words[i] = words[i].lower()
#                 new_sentence += words[i] + " "
#             data.append(new_sentence)
#             label.append(words[-1])
#     return data, label
#
#
# train_data, train_label = read('train.txt')
# val_data, val_label = read('valid.txt')
# test_data, test_label = read('test.txt')

import pandas as pd
# # cvt
# data = pd.read_csv("result.csv")
# data = data.drop(index=[25391,35044,49475])
# X = data["text"]
# label = data["compound"]
# import numpy as np
# X = np.array(X)
# label = np.array(label)
# label = np.sign(label)
# newlabel = label + 1
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
# cvt
from sklearn.model_selection import train_test_split
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, newlabel, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
import nltk.tokenize as tk
# cvt
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
# print("train:")
# print(train_cnt)
# print("test:")
# print(test_cnt)
# print("val:")
# print(val_cnt)
# print("max len:")
# print(maxlen)
# # print(train_tokenized)
# X_train = np.array(train_tokenized)#[train_null]
# y_train = y_train[train_null]
# X_val = np.array(val_tokenized)#[val_null]
# y_val = y_val[val_null]
# X_test = np.array(test_tokenized)#[test_null]
# y_test = y_test[test_null]
# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)
# print(X_test.shape)
# print(y_test.shape)
# X_train = X_train.tolist()
# y_train = y_train.tolist()
# X_val = X_val.tolist()
# y_val = y_val.tolist()
# X_test = X_test.tolist()
# y_test = y_test.tolist()
import json
# cvt
# with open("./train_data_cvt.json", 'w') as file_obj:
#     json.dump(X_train, file_obj)
# with open("./train_label_cvt.json", 'w') as file_obj:
#     json.dump(y_train, file_obj)
# with open("./val_data_cvt.json", 'w') as file_obj:
#     json.dump(X_val, file_obj)
# with open("./val_label_cvt.json", 'w') as file_obj:
#     json.dump(y_val, file_obj)
# with open("./test_data_cvt.json", 'w') as file_obj:
#     json.dump(X_test, file_obj)
# with open("./test_label_cvt.json", 'w') as file_obj:
#     json.dump(y_test, file_obj)
# sst
# with open("test_data", 'w') as file_obj:
#     json.dump(test_data, file_obj)
# with open("test_label", 'w') as file_obj:
#     json.dump(test_label, file_obj)

#
from transformers import BertTokenizer
import json
from transformers import BertForSequenceClassification, AdamW, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

with open("train_data_cvt.json") as file_obj:
    train_data = json.load(file_obj)
with open("val_data_cvt.json") as file_obj:
    val_data = json.load(file_obj)
with open("test_data_cvt.json") as file_obj:
    test_data = json.load(file_obj)
with open("train_label_cvt.json") as file_obj:
    train_label = json.load(file_obj)
with open("val_label_cvt.json") as file_obj:
    val_label = json.load(file_obj)
with open("test_label_cvt.json") as file_obj:
    test_label = json.load(file_obj)
for i in range(len(train_label)):
    train_label[i] = int(train_label[i])
for i in range(len(val_label)):
    val_label[i] = int(val_label[i])
for i in range(len(test_label)):
    test_label[i] = int(test_label[i])
train_label = torch.tensor(train_label)
val_label = torch.tensor(val_label)
test_label = torch.tensor(test_label)
max_len = 0
for sent in train_data:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
for sent in val_data:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
# for sent in test_data:
#     input_ids = tokenizer.encode(sent, add_special_tokens=True)
#     max_len = max(max_len, len(input_ids))
max_len += 10
print('Max sentence length: ', max_len)
train_ids = []
train_attention_masks = []
for sent in train_data:
    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    train_ids.append(encoded_dict['input_ids'])
    train_attention_masks.append(encoded_dict['attention_mask'])
print(len(train_ids))
train_ids = torch.cat(train_ids, dim=0)
print(train_ids.shape)
train_attention_masks = torch.cat(train_attention_masks, dim=0)

val_ids = []
val_attention_masks = []
for sent in val_data:
    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    val_ids.append(encoded_dict['input_ids'])
    val_attention_masks.append(encoded_dict['attention_mask'])
val_ids = torch.cat(val_ids, dim=0)
print(val_ids.shape)
val_attention_masks = torch.cat(val_attention_masks, dim=0)

test_ids = []
test_attention_masks = []
for sent in test_data:
    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    test_ids.append(encoded_dict['input_ids'])
    test_attention_masks.append(encoded_dict['attention_mask'])
test_ids = torch.cat(test_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)

torch.save(train_ids, "./train_ids_cvt.pt")
torch.save(val_ids, "./val_ids_cvt.pt")
torch.save(test_ids, "./test_ids_cvt.pt")
torch.save(train_attention_masks, "./train_attention_masks_cvt.pt")
torch.save(val_attention_masks, "./val_attention_masks_cvt.pt")
torch.save(test_attention_masks, "./test_attention_masks_cvt.pt")
torch.save(train_label, "./train_label_cvt.pt")
torch.save(val_label, "./val_label_cvt.pt")
torch.save(test_label, "./test_label_cvt.pt")
