# Fun Transformer
## 项目描述
本项目基于深度学习平台Datawhale于2025年2月发布的开源课程“零基础实践Transformer模型”进行修改。项目上传出于教学用途，旨在加深初学者对Transformer模型的理解。
- 开源教程网址：[fun-transformer](https://github.com/datawhalechina/fun-transformer)
- 在线学习网站：[Datawhale](http://www.datawhale.cn/learn/summary/87)

## 功能特性
- 完整的 Transformer 实现：包括 Encoder、Decoder、多头注意力机制、位置编码等。
- 中英文翻译：支持从中文到英文的翻译任务。
- 自定义数据集：支持用户自定义训练数据。
- 训练和测试：提供完整的训练和测试流程。
- 注意力可视化：支持查看 Encoder 和 Decoder 的注意力权重。

## 修改
- 分离了训练模块和测试模块
- 删除了显式指明sentence并静态生成词汇表模块
- 支持从 txt 文件中读取sentence并动态生成词汇表

## 安装指南
### 依赖环境
- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
### 安装步骤
- 克隆本仓库
  ```bash
  git clone https://github.com/BeerSquare/fun-transformer.git
- 进入项目目录
   ```bash
  cd fun-transformer
- 安装依赖
  ```bash
  pip install torch numpy

## 使用说明
### 数据准备
- 文件内提供了两个包含中英文句子词库。其中[hongloumeng.txt](https://github.com/BeerSquare/fun-transformer/blob/main/hongloumeng.txt)仅有四个句子，适合快速验证代码；[ChineseToEnglish.txt](https://github.com/BeerSquare/fun-transformer/blob/main/ChineseToEnglish.txt)有4096个句子，适合验证训练质量。
- 你可以通过[Tatoeba多语言句子库](https://tatoeba.org/zh-cn/downloads)等开放语言库获更大的数据集。数据格式如下：
  ```txt
  满纸荒唐言|Full of nonsense
  一把辛酸泪|A handful of bitter tears
  都言作者痴|They say the author is foolish
  谁解其中味|Who understands the true meaning

### 训练模型
- 运行[main.py](https://github.com/BeerSquare/fun-transformer/blob/main/main.py)开始训练：
  ```python
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
  
### 测试模型
- 运行[test_utils.py](https://github.com/BeerSquare/fun-transformer/blob/main/test_utils.py)进行测试
  ```python
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
  
      # 过滤不匹配的权重
      filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
  
      # 加载匹配的权重
      model.load_state_dict(filtered_state_dict, strict=False)  # strict=False 允许部分加载
      model.eval()  # 设置为评估模式
  
      # 测试模型
      test(model, src_vocab, tgt_vocab, src_len, tgt_len)
  
### 使用[hongloumeng.txt](https://github.com/BeerSquare/fun-transformer/blob/main/hongloumeng.txt)作为数据集
- 测试：翻译“满纸辛酸泪”，batch_size=2, epochs=50
- 测试结果：
  ```txt
  Epoch: 0001, Avg Loss: 3.338455
  Epoch: 0002, Avg Loss: 3.004684
  Epoch: 0003, Avg Loss: 2.488638
  Epoch: 0004, Avg Loss: 2.109935
  Epoch: 0005, Avg Loss: 1.764553
  Epoch: 0006, Avg Loss: 1.393467
  Epoch: 0007, Avg Loss: 1.174642
  Epoch: 0008, Avg Loss: 1.005366
  Epoch: 0009, Avg Loss: 0.873622
  Epoch: 0010, Avg Loss: 0.711802
  Epoch: 0011, Avg Loss: 0.572180
  Epoch: 0012, Avg Loss: 0.443027
  Epoch: 0013, Avg Loss: 0.366159
  Epoch: 0014, Avg Loss: 0.293282
  Epoch: 0015, Avg Loss: 0.191762
  Epoch: 0016, Avg Loss: 0.124517
  Epoch: 0017, Avg Loss: 0.089571
  Epoch: 0018, Avg Loss: 0.059290
  Epoch: 0019, Avg Loss: 0.048592
  Epoch: 0020, Avg Loss: 0.040043
  Epoch: 0021, Avg Loss: 0.034745
  Epoch: 0022, Avg Loss: 0.023343
  Epoch: 0023, Avg Loss: 0.021037
  Epoch: 0024, Avg Loss: 0.018994
  Epoch: 0025, Avg Loss: 0.018008
  Epoch: 0026, Avg Loss: 0.013908
  Epoch: 0027, Avg Loss: 0.014778
  Epoch: 0028, Avg Loss: 0.013300
  Epoch: 0029, Avg Loss: 0.010145
  Epoch: 0030, Avg Loss: 0.011074
  Epoch: 0031, Avg Loss: 0.010475
  Epoch: 0032, Avg Loss: 0.012780
  Epoch: 0033, Avg Loss: 0.011357
  Epoch: 0034, Avg Loss: 0.010622
  Epoch: 0035, Avg Loss: 0.010350
  Epoch: 0036, Avg Loss: 0.010978
  Epoch: 0037, Avg Loss: 0.013631
  Epoch: 0038, Avg Loss: 0.010630
  Epoch: 0039, Avg Loss: 0.009026
  Epoch: 0040, Avg Loss: 0.008175
  Epoch: 0041, Avg Loss: 0.006770
  Epoch: 0042, Avg Loss: 0.006879
  Epoch: 0043, Avg Loss: 0.005603
  Epoch: 0044, Avg Loss: 0.004413
  Epoch: 0045, Avg Loss: 0.004573
  Epoch: 0046, Avg Loss: 0.003865
  Epoch: 0047, Avg Loss: 0.003989
  Epoch: 0048, Avg Loss: 0.003490
  Epoch: 0049, Avg Loss: 0.002911
  Epoch: 0050, Avg Loss: 0.002583
  Model saved to 'model.pth'
  源句子: 满纸辛酸泪
  翻译结果: Full of nonsense
- 翻译结果：张冠李戴。txt内是所有词汇对应表。训练数据太少，只能缘木求鱼；凡有修改，就会张冠李戴。
### 使用[ChineseToEnglish.txt](https://github.com/BeerSquare/fun-transformer/blob/main/ChineseToEnglish.txt)作为数据集
- 测试：翻译“满纸辛酸泪”，batch_size=32, epochs=10
- 测试结果：
  ```txt
  Epoch: 0001, Avg Loss: 7.172143
  Epoch: 0002, Avg Loss: 6.510853
  Epoch: 0003, Avg Loss: 6.210540
  Epoch: 0004, Avg Loss: 5.940146
  Epoch: 0005, Avg Loss: 5.701981
  Epoch: 0006, Avg Loss: 5.484452
  Epoch: 0007, Avg Loss: 5.219655
  Epoch: 0008, Avg Loss: 4.949931
  Epoch: 0009, Avg Loss: 4.694051
  Epoch: 0010, Avg Loss: 4.447792
  Model saved to 'model.pth'
  源句子: 满纸辛酸泪
  翻译结果: I don't like to do that.
- 翻译结果：意译。勉强“信”，不“达雅”。数据集内现代表达居多。


## 设计文档
### 项目架构
- 项目从零构建了 Transformer 模型，完成端到端的机器翻译任务，便于初学者深入理解Attention 机制、Encoder-Decoder 架构及其与CNN/RNN的差异。
- 项目分为四个模块：数据预处理模块、编码器-解码器模块、训练模块和测试模块。
### 模块说明
- data_utils.py为数据预处理模块。此模块主要包含五个函数和一个类。
  - load_sentences_from_txt负责从 txt 文件中读取句子对。其输入为txt文件路径。该函数逐句处理文件，分离源语言和目标语言，为源语言添加起始符和结束符，输出符合要求的句子对列表（sentences），为下一步动态生成词汇表做准备；
    
  - build_vocab负责由txt文件动态生成词汇表。其输入为经处理的句子对列表（sentences）。该函数首先初始化词汇表，先设置了三个特殊符号，即词汇表中索引0、1、2的三个位置分别为特殊符号P（填充）、S（开始）和E（结束）；其次该函数遍历句子对，将源句子和目标句子拆分为单词，不重复地添加到源词汇表（src_vocab）和目标词汇表（tgt_vocab）中；最后保存两张词汇表到vocab.pth中，输出构建好的 源词汇表（src_vocab）和目标词汇表（tgt_vocab）；
    
  - make_data负责对数据进行预处理。其输入为句子对列表（sentences）、源语言和目标语言的词汇表及其长度。该函数经过索引化和填充，将源语言句子转化为编码器输入（enc_inputs），将目标语言句子转化为解码器输入（dec_inputs）和输出（dec_outputs）， 将输入的句子转化成模型训练所需要的张量格式。
 
  - get_attn_pad_mask负责
    
  - get_attn_subsequence_mask负责
    
- 编码器-解码器模块包含layers.py和model.py两个文件。其中，model.py包含Encoder、Decoder和Transformer的整体设计，主要依赖layers.py中的PositionEncoding、MultipleAttention、FF（前馈层）来实现。
  - PositionEncoding
  - MultipleAttention
  - FF
  - Encoder
  - Decoder
  - Transformer
- train.py为训练模块。
- test.py为测试模块。

### 流程图
txt文件输入->句子对列表（sentences）->源词汇表（src_vocab）和目标词汇表（tgt_vocab）->编码器输入（enc_inputs）、解码器输入（dec_inputs）、输出（dec_outputs）

### 常见问题QA
- 为什么数据预处理需要填充？
  - 确保所有输入具有相同的形状，这是深度学习模型的常见要求。
- 为什么目标语言既是解码器的输入、又是解码器的输出？
  - 这种设计是为了实现自**回归Auto-regressive**的训练方式。在训练阶段，解码器的目标是逐步生成目标序列。为了实现这一点，解码器在每一步的输入是前一步的输出。
  - 例如，解码器首先生成第一个词 "Full"，然后以 "Full" 作为输入生成第二个词 "of"，依此类推。

## 许可证
本项目基于 [fun-transformer](https://github.com/datawhalechina/fun-transformer) 的 Apache License 2.0许可证。详情请参阅 [LICENSE](https://github.com/BeerSquare/fun-transformer/blob/main/LICENSE.txt) 文件。

## 联系方式
如有任何问题，请联系：
- 邮箱：1259712042@qq.com
- GitHub:[BeerSquare](https://github.com/BeerSquare)


