# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from utils.data_utils import load_sentences_from_txt, build_vocab, make_data, MyDataSet
from utils.train import train
from utils.model import Transformer


# 主函数
def main():
    # 加载句子和生成词汇表
    file_path = 'hongloumeng.txt'  # 替换为你的 txt 文件路径
    sentences = load_sentences_from_txt(file_path)# 使用函数从txt文件中读取sentences
    src_vocab, tgt_vocab = build_vocab(sentences)# 使用函数生成源语言和目标语言的词汇表

    # 如果 vocab.pth 文件存在，则加载词汇表；否则生成并保存词汇表
    if os.path.exists('vocab.pth'):
        vocab_data = torch.load('vocab.pth')
        src_vocab = vocab_data['src_vocab']
        tgt_vocab = vocab_data['tgt_vocab']
        print("词汇表已从 'vocab.pth' 加载")
    else:
        src_vocab, tgt_vocab = build_vocab(sentences)  # 使用函数生成源语言和目标语言的词汇表
        # 保存词汇表到文件
        vocab_data = {
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'src_len': max(len(sentence[0].split()) for sentence in sentences),  # 源句子最大长度
            'tgt_len': max(len(sentence[1].split()) for sentence in sentences)   # 目标句子最大长度
        }
        torch.save(vocab_data, 'vocab.pth')
        print("词汇表已保存到 'vocab.pth'")    


    # 计算 src_len 和 tgt_len的最大长度
    src_len = max(len(sentence[0].split()) for sentence in sentences)  # sentence[0] 是源句子，split() 将其按空格分割成词，len() 计算词的数量
    tgt_len = max(len(sentence[1].split()) for sentence in sentences)  

    # 调用make_data生成Encoder输入、Decoder输入、Decoder输出
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences, src_vocab, tgt_vocab, src_len, tgt_len)

    # 创建数据集和数据加载器
    dataset = MyDataSet(enc_inputs, dec_inputs, dec_outputs)#将 enc_inputs、dec_inputs 和 dec_outputs 封装为一个数据集对象
    loader = Data.DataLoader(dataset, batch_size=2, shuffle=True)#使用 DataLoader 创建数据加载器 loader，用于批量加载数据

    # 定义超参数
    d_model = 512 #词向量维度
    d_ff = 2048 #前馈神经网络的隐藏层维度
    d_k = d_v = 64 #K和V的维度
    n_layers = 6 #tranformer模型的层数
    n_heads = 8 #注意力头数
    src_vocab_size = len(src_vocab)#源语言词汇表的大小
    tgt_vocab_size = len(tgt_vocab)#目标语言词汇表的大小

    # 初始化Transformer模型
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, d_k, d_v, n_heads, d_ff, n_layers)
    
    # 定义损失函数criterion和优化器optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充符的损失
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    # 训练模型
    train(model, loader, criterion, optimizer, epochs=50)
    

# 运行主函数
if __name__ == '__main__':
    main()


