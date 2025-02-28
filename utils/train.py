# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from .data_utils import MyDataSet, make_data
from .model import Transformer

# 训练函数
def train(model, loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        model.train()
        total_loss = 0  # 用于累积当前 epoch 的总损失
        num_batches = len(loader)  # 当前 epoch 的 batch 数量

        for enc_inputs, dec_inputs, dec_outputs in loader:
            optimizer.zero_grad()
            outputs, _, _, _ = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # 累积当前 batch 的损失值
        
        avg_loss = total_loss / num_batches
        print(f'Epoch: {epoch + 1:04d}, Avg Loss: {avg_loss:.6f}')


