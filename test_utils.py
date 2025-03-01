# -*- coding: utf-8 -*-
import torch
from utils.model import Transformer

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


def test(model, src_vocab, tgt_vocab, src_len, tgt_len, batch_size=2, epochs=50):
    # 测试句子
    test_sentence = "满纸辛酸泪"
    
    print(f"源句子: {test_sentence}")
    translated_sentence = translate(model, test_sentence, src_vocab, tgt_vocab, src_len, tgt_len)
    print(f"翻译结果: {translated_sentence}")

if __name__ == '__main__':
   if __name__ == '__main__':
    # 加载词汇表和其他参数
    vocab_data = torch.load('vocab.pth')
    src_vocab = vocab_data['src_vocab']
    tgt_vocab = vocab_data['tgt_vocab']
    src_len = vocab_data['src_len']
    tgt_len = vocab_data['tgt_len']

    # 根据词汇表大小初始化模型
    src_vocab_size = len(src_vocab)  # 使用实际词汇表大小
    tgt_vocab_size = len(tgt_vocab)  # 使用实际词汇表大小
    model = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, d_model=512, d_k=64, d_v=64, n_heads=8, d_ff=2048, n_layers=6)

    # 加载模型权重
    state_dict = torch.load('model.pth')

    # 过滤不匹配的权重（可选）
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}

    # 加载匹配的权重
    model.load_state_dict(filtered_state_dict, strict=False)  # strict=False 允许部分加载
    model.eval()  # 设置为评估模式

    # 测试模型
    test(model, src_vocab, tgt_vocab, src_len, tgt_len)
