# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.utils.data as Data

# 从 txt 文件中读取句子
def load_sentences_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sentences = []
    for line in lines:
        src, tgt = line.strip().split('|')  # 假设 txt 文件中每行格式为 "源句子|目标句子"
        sentences.append([src, f'S {tgt}', f'{tgt} E'])  # 添加起始符和结束符
    # 生成词汇表
    src_vocab, tgt_vocab = build_vocab(sentences)
    return sentences
    

# 动态生成词汇表
def build_vocab(sentences):
    src_vocab = {'P': 0, 'S': 1, 'E': 2}  # 特殊符号
    tgt_vocab = {'P': 0, 'S': 1, 'E': 2}  # 特殊符号
    src_idx = 3
    tgt_idx = 3

    for src, _, _ in sentences:
        for word in src.split():
            if word not in src_vocab:
                src_vocab[word] = src_idx
                src_idx += 1

    for _, _, tgt in sentences:
        for word in tgt.split():
            if word not in tgt_vocab:
                tgt_vocab[word] = tgt_idx
                tgt_idx += 1

    # 保存词汇表到文件
    vocab_data = {
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'src_len': 10,  # 假设源句子最大长度为 10
        'tgt_len': 10   # 假设目标句子最大长度为 10
    }
    torch.save(vocab_data, 'vocab.pth')
    print("词汇表已保存到 'vocab.pth'")

    return src_vocab, tgt_vocab

# 数据预处理
def make_data(sentences, src_vocab, tgt_vocab, src_len, tgt_len):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for src, dec_input, dec_output in sentences:
        # 将句子转换为索引
        enc_input = [src_vocab[word] for word in src.split()]
        dec_input = [tgt_vocab[word] for word in dec_input.split()]
        dec_output = [tgt_vocab[word] for word in dec_output.split()]
        # 填充到固定长度
        enc_input = enc_input + [src_vocab['P']] * (src_len - len(enc_input))
        dec_input = dec_input + [tgt_vocab['P']] * (tgt_len - len(dec_input))
        dec_output = dec_output + [tgt_vocab['P']] * (tgt_len - len(dec_output))
        # 添加到列表中
        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


#自定义数据集函数
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

# 生成注意力掩码
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0)  # 找到序列中值为0的位置（填充位置）
    pad_attn_mask = pad_attn_mask.unsqueeze(1)  # 扩展维度 [batch_size, 1, len_k]
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)  # 扩展为 [batch_size, len_q, len_k]
    return pad_attn_mask

# 生成上三角掩码（用于屏蔽未来信息）
def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成上三角矩阵，k=1表示不包括对角线
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # 转换为PyTorch张量
    return subsequence_mask