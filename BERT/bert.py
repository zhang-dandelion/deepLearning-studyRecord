# -*- coding:utf-8 -*-
# bert文本分类baseline模型
# model: bert

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import transformers
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

train_curve = []
val_curve = []
val_accuracy_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义一些参数，模型选择了最基础的bert中文模型
batch_size = 625
epoches = 10
model = "bert-base-chinese"
model_directory = 'D:/pycharmProjects/BERT/bert_chinese_model'
hidden_size = 768
n_class = 2
maxlen = 10

encode_layer=12
filter_sizes = [2, 2, 2]
num_filters = 3

class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        # 确保 sent 是字符串
        if not isinstance(sent, str):
            sent = str(sent)
        #   使用 self.tokenizer 对句子进行编码
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=maxlen,
                                      return_tensors='pt')

        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

        if self.with_labels:
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

class BertClassify(nn.Module):
    def __init__(self):
        super(BertClassify, self).__init__()
        #   加载预训练的 BERT 模型
        self.bert = AutoModel.from_pretrained(model_directory, output_hidden_states=True, return_dict=True)
        #   线性层，将隐藏层大小（hidden_size）映射到分类数（n_class）
        self.linear = nn.Linear(hidden_size, n_class)
        #    dropout 层，防止过拟合，概率为 0.5
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        logits = self.linear(self.dropout(outputs.pooler_output))
        return logits

def train(model,train_loader,loss_fn,optimizer,epoches):
    total_step = len(train_loader)
    for epoch in range(epoches):
        model.train()
        sum_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = tuple(p.to(device) for p in batch)
            pred = model([batch[0], batch[1], batch[2]])
            loss = loss_fn(pred, batch[3])
            sum_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()    # 更新参数
            print('[{}|{}] step:{}/{} loss:{:.4f}'.format(epoch + 1, epoches, i + 1, total_step, loss.item()))
        train_curve.append(sum_loss)
        evaluate(model,eval_loader,loss_fn)
def evaluate(model,test_loader,loss_fn):
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(p.to(device) for p in batch)
            pred = model([batch[0], batch[1], batch[2]])
            loss = loss_fn(pred, batch[3])
            val_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            total += batch[3].size(0)
            correct += (predicted == batch[3]).sum().item()
    val_curve.append(val_loss / len(eval_loader))
    val_accuracy_curve.append(correct / total)
    print('Validation Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss / len(eval_loader), correct / total))

if __name__ == '__main__':
    # 加载数据集
    file_path = './data/ChnSentiCorp_htl_all.csv'
    df = pd.read_csv(file_path)

    # CSV文件包含两列：'review' 和 'label'
    texts = df['review'].values
    labels = df['label'].values

    # 划分训练集和测试集
    train_samples, eval_samples, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42)
    # 加载训练集数据
    train_loader = Data.DataLoader(dataset=MyDataset(train_samples, train_labels), batch_size=batch_size, shuffle=True, num_workers=1)
    # 加载测试集数据
    eval_loader = Data.DataLoader(dataset=MyDataset(eval_samples, eval_labels), batch_size=batch_size, shuffle=False,
                                  num_workers=1)
    #   实例化模型
    bc = BertClassify().to(device)
    #   Adam 优化器，设置学习率为 1e-3，权重衰减为 1e-2
    optimizer = optim.Adam(bc.parameters(), lr=1e-3, weight_decay=1e-2)
    #   交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()

    train(bc,train_loader,loss_fn,optimizer,epoches)
    # 绘制损失和准确率变化的折线图
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epoches + 1), train_curve, label='Training Loss')
    plt.plot(range(1, epoches + 1), val_curve, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(range(1, epoches + 1), val_accuracy_curve, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



