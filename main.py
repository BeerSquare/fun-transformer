# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from utils import make_data, MyDataSet, train, Transformer

# 从 txt 文件中读取句子
def load_sentences_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sentences = []
    for line in lines:
        src, tgt = line.strip().split('|')  # 假设 txt 文件中每行格式为 "源句子|目标句子"
        sentences.append([src, f'S {tgt}', f'{tgt} E'])  # 添加起始符和结束符
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

    return src_vocab, tgt_vocab

def translate(model, src_sentence, src_vocab, tgt_vocab, src_len, tgt_len):
    model.eval()  # 将模型设置为评估模式
    src_tokens = src_sentence.split()
    src_indices = [src_vocab.get(token, src_vocab['P']) for token in src_tokens]  # 将源句子转换为索引
    src_indices = torch.LongTensor(src_indices).unsqueeze(0)  # 添加 batch 维度

    # 初始化 Decoder 输入
    dec_input = torch.LongTensor([[tgt_vocab['S']]])  # 以起始符开始

    # 逐个生成目标句子
    for i in range(tgt_len):
        with torch.no_grad():
            dec_logits, _, _, _ = model(src_indices, dec_input)  # 提取 dec_logits
        pred = dec_logits.argmax(dim=-1)[-1].item()  # 获取最后一个词的预测结果
        if pred == tgt_vocab['E']:  # 如果预测到结束符，停止生成
            break
        dec_input = torch.cat([dec_input, torch.LongTensor([[pred]])], dim=-1)  # 将预测词添加到 Decoder 输入中

    # 将生成的索引转换为目标句子
    tgt_tokens = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(idx)] for idx in dec_input.squeeze().tolist()]
    tgt_sentence = ' '.join(tgt_tokens[1:])  # 去掉起始符

    return tgt_sentence




# 测试函数
def test(model, src_vocab, tgt_vocab, src_len, tgt_len):
    # 测试句子
    test_sentence = "满纸辛酸泪"
    print(f"源句子: {test_sentence}")
    translated_sentence = translate(model, test_sentence, src_vocab, tgt_vocab, src_len, tgt_len)
    print(f"翻译结果: {translated_sentence}")


# 主函数
def main():
    # 加载句子和生成词汇表
    file_path = 'hongloumeng.txt'  # 替换为你的 txt 文件路径
    sentences = load_sentences_from_txt(file_path)
    src_vocab, tgt_vocab = build_vocab(sentences)

    # 计算 src_len 和 tgt_len
    src_len = max(len(sentence[0].split()) for sentence in sentences)  # Encoder 输入的最大长度
    tgt_len = max(len(sentence[1].split()) for sentence in sentences)  # Decoder 输入输出的最大长度

    # 生成数据
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences, src_vocab, tgt_vocab, src_len, tgt_len)

    # 创建数据集和数据加载器
    dataset = MyDataSet(enc_inputs, dec_inputs, dec_outputs)
    loader = Data.DataLoader(dataset, batch_size=2, shuffle=True)

    # 定义超参数
    d_model = 512
    d_ff = 2048
    d_k = d_v = 64
    n_layers = 6
    n_heads = 8
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # 初始化模型
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, d_k, d_v, n_heads, d_ff, n_layers)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充符的损失
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    # 训练模型
    train(model, loader, criterion, optimizer, epochs=50)
    
    # 测试模型
    test(model, src_vocab, tgt_vocab, src_len, tgt_len)

# 运行主函数
if __name__ == '__main__':
    main()


