from transformers import BertTokenizer
import json
from transformers import BertForSequenceClassification, AdamW, BertConfig
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import time
import datetime
from sklearn.metrics import accuracy_score, f1_score
import random
import numpy as np
import pandas as pd
# train_ids = torch.load("./train_ids_cvt.pt")
# val_ids = torch.load("./val_ids_cvt.pt")
# test_ids = torch.load("./test_ids.pt")
# train_attention_masks = torch.load("./train_attention_masks_cvt.pt")
# val_attention_masks = torch.load("./val_attention_masks_cvt.pt")
# test_attention_masks = torch.load("./test_attention_masks.pt")
# train_labels = torch.load("./train_label_cvt.pt")
# val_labels = torch.load("./val_label_cvt.pt")
# test_labels = torch.load("./test_label.pt")
# print(val_ids.shape)
# # print(test_labels.shape)
#
# train_ids = val_ids
# # print(test_ids.shape)
# # print(test_labels.shape)
# train_ids1 = torch.load("./val_ids.pt")
# for i in range(int(train_ids.shape[0]/3)):
#     if i % 100 == 99:
#         print(i)
#     if i == 0:
#         all_idx_cvt = set(train_ids[i])
#     else:
#         all_idx_cvt = all_idx_cvt | set(train_ids[i])
# for i in range(int(train_ids1.shape[0]/3)):
#     if i % 100 == 99:
#         print(i)
#     if i == 0:
#         all_idx_sst = set(train_ids1[i])
#     else:
#         all_idx_sst = all_idx_sst | set(train_ids1[i])
# # train_cvt = set(all_idx_cvt)
# # train_sst = set(all_idx_sst)
# num_cvt = len(all_idx_cvt)
# num_sst = len(all_idx_sst)
# print(num_cvt)
# print(num_sst)
# common = all_idx_cvt & all_idx_sst #s3={2,3}
# print(len(common))
# print(len(common)/num_cvt)
#
#
# def change_labels(train_labels):
#     for i in range(train_labels.shape[0]):
#         if train_labels[i] < 2:
#             train_labels[i] = 0
#         elif train_labels[i] == 2:
#             train_labels[i] = 1
#         else:
#             train_labels[i] = 2
#     return train_labels
#
#
# # train_labels = change_labels(train_labels)
# # val_labels = change_labels(val_labels)
# test_labels = change_labels(test_labels)
# batch_size = 32
# train_set = torch.utils.data.TensorDataset(train_ids, train_attention_masks, train_labels)
# test_set = torch.utils.data.TensorDataset(test_ids, test_attention_masks, test_labels)
# val_set = torch.utils.data.TensorDataset(val_ids, val_attention_masks, val_labels)
# train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
# val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
#
# print(f"%d classification {max(test_labels)+1}")
# # model = BertForSequenceClassification.from_pretrained(
# #     "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
# #     num_labels = max(train_labels)+1, # The number of output labels--2 for binary classification.
# #                     # You can increase this for multi-class tasks.
# #     output_attentions = False, # Whether the model returns attentions weights.
# #     output_hidden_states = False, # Whether the model returns all hidden-states.
# # )
# # model.cuda()
# #
# # optimizer = AdamW(model.parameters(),
# #                   lr = 2e-5, # args.learning_rate
# #                   eps = 1e-8 # args.adam_epsilon
# #                 )
# #
# # # Number of training epochs
# # epochs = 3
# #
# # # Total number of training steps is [number of batches] x [number of epochs].
# # total_steps = len(train_iter) * epochs
# #
# # # Create the learning rate scheduler.
# # scheduler = get_linear_schedule_with_warmup(optimizer,
# #                                             num_warmup_steps = 0, # Default value in run_glue.py
# #                                             num_training_steps = total_steps)
# #
#
# # Helper function for formatting elapsed times as hh:mm:ss
# def format_time(elapsed):
#     '''
#     Takes a time in seconds and returns a string hh:mm:ss
#     '''
#     # Round to the nearest second.
#     elapsed_rounded = int(round((elapsed)))
#
#     # Format as hh:mm:ss
#     return str(datetime.timedelta(seconds=elapsed_rounded))
#
#
# device = "cuda:0"
#
#
# def fit_batch(dataloader, model, optimizer, epoch):
#     total_train_loss = 0
#
#     for batch in tqdm(dataloader, desc=f"Training epoch:{epoch + 1}", unit="batch"):
#         # Unpack batch from dataloader.
#         input_ids = batch[0].to(device)
#         attention_masks = batch[1].to(device)
#         # token_type_ids = batch[2].to(device)
#         labels = batch[2].to(device)
#
#         # clear any previously calculated gradients before performing a backward pass.
#         model.zero_grad()
#
#         # Perform a forward pass (evaluate the model on this training batch).
#         outputs = model(input_ids,
#                         attention_mask=attention_masks,
#                         labels=labels)
#         loss = outputs[0]
#         total_train_loss += loss.item()
#
#         # Perform a backward pass to calculate the gradients.
#         loss.backward()
#
#         # normlization of the gradients to 1.0 to avoid exploding gradients
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
#         # Update parameters and take a step using the computed gradient.
#         optimizer.step()
#
#         # Update the learning rate.
#         scheduler.step()
#
#     return total_train_loss
#
#
# def eval_batch(dataloader, model, metric=accuracy_score):
#     total_eval_accuracy = 0
#     total_eval_loss = 0
#     predictions, predicted_labels = [], []
#     f1_avg = 0
#     m = 0
#     for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
#         # Unpack batch from dataloader.
#         input_ids = batch[0].to(device)
#         attention_masks = batch[1].to(device)
#         # token_type_ids = batch[2].to(device)
#         labels = batch[2].to(device)
#
#         # Tell pytorch not to bother with constructing the compute graph during
#         # the forward pass, since this is only needed for backprop (training).
#         with torch.no_grad():
#             # Forward pass, calculate logit predictions.
#             outputs = model(input_ids,
#                             attention_mask=attention_masks,
#                             labels=labels)
#             loss = outputs[0]
#             logits = outputs[1]
#         total_eval_loss += loss.item()
#
#         # Move logits and labels to CPU
#         logits = logits.detach().cpu().numpy()
#         label_ids = labels.to('cpu').numpy()
#
#         # Calculate the accuracy for this batch of validation sentences, and
#         # accumulate it over all batches.
#
#         y_pred = np.argmax(logits, axis=1).flatten()
#         f1_avg += f1_score(label_ids, y_pred, average='weighted')
#         total_eval_accuracy += metric(label_ids, y_pred)
#         m += 1
#         predictions.extend(logits.tolist())
#         predicted_labels.extend(y_pred.tolist())
#     f1_avg = f1_avg / m
#     return total_eval_accuracy, f1_avg, total_eval_loss, predictions, predicted_labels
#
#
# def train(train_dataloader, validation_dataloader, test_dataloader, model, optimizer, epochs):
#     # list to store a number of quantities such as
#     # training and validation loss, validation accuracy, and timings.
#     training_stats = []
#     best_res = {}
#     best_val = 0.0
#     # Measure the total training time for the whole run.
#     total_t0 = time.time()
#
#     for epoch in range(0, epochs):
#         # Measure how long the training epoch takes.
#         t0 = time.time()
#
#         # Reset the total loss for this epoch.
#         total_train_loss = 0
#
#         # Put the model into training mode.
#         model.train()
#
#         total_train_loss = fit_batch(train_dataloader, model, optimizer, epoch)
#
#         # Calculate the average loss over all of the batches.
#         avg_train_loss = total_train_loss / len(train_dataloader)
#
#         # Measure how long this epoch took.
#         training_time = format_time(time.time() - t0)
#
#         t0 = time.time()
#
#         # Put the model in evaluation mode--the dropout layers behave differently
#         # during evaluation.
#         model.eval()
#
#         total_eval_accuracy, f1, total_eval_loss, _, _ = eval_batch(validation_dataloader, model)
#
#         # Report the final accuracy for this validation run.
#         avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
#
#         print("\n")
#         print(f"score: {avg_val_accuracy}")
#
#         # Calculate the average loss over all of the batches.
#         avg_val_loss = total_eval_loss / len(validation_dataloader)
#
#         # Measure how long the validation run took.
#         validation_time = format_time(time.time() - t0)
#
#         print(f"Validation Loss: {avg_val_loss}")
#         print("\n")
#
#         total_test_accuracy, f1_test, total_test_loss, _, _ = eval_batch(test_dataloader, model)
#         total_test_accuracy = total_test_accuracy / len(test_iter)
#         if avg_val_accuracy > best_val:
#             best_val = avg_val_accuracy
#             best_res["epoch"] = epoch
#             best_res["test_acc"] = total_test_accuracy
#             best_res["test_f1"] = f1_test
#             best_res["val_acc"] = avg_val_accuracy
#
#     print("")
#     print("Training complete!")
#
#     print(f"Total training took {format_time(time.time() - total_t0)}")
#     return best_res
#
#
# # Set the seed value all over the place to make this reproducible.
# seed_val = 2020
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
#
# # training_stats = train(train_iter, val_iter, test_iter, model, optimizer, epochs)
# # # df_stats = pd.DataFrame(training_stats).set_index('epoch')
# #
# # torch.save(model, "./model_cvt")
# # print(training_stats)
# model = torch.load("./model_cvt")
# model.cuda()
# total_test_accuracy, f1_test, total_test_loss, _, _ = eval_batch(test_iter, model)
# total_test_accuracy = total_test_accuracy / len(test_iter)
# print(total_test_accuracy)
# print(f1_test)
#
