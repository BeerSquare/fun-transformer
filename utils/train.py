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
        for enc_inputs, dec_inputs, dec_outputs in loader:
            optimizer.zero_grad()
            outputs, _, _, _ = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1:04d}, Loss: {loss.item():.6f}')


