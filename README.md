# Fun Transformer
## 项目描述
本项目基于深度学习平台Datawhale于2025年1月发布的开源课程“零基础实践Transformer模型”进行修改。项目上传出于教学用途，旨在加深初学者对Transformer模型的理解。
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
- [data_utils.py](https://github.com/BeerSquare/fun-transformer/blob/main/utils/data_utils.py)为数据预处理模块。此模块主要包含五个函数和一个类。
  - **函数load_sentences_from_txt**负责从 txt 文件中读取句子对。其输入为txt文件路径。该函数逐句处理文件，分离源语言和目标语言，为源语言添加起始符和结束符，输出符合要求的句子对列表（sentences），为下一步动态生成词汇表做准备；
    
  - **函数build_vocab**负责由txt文件动态生成词汇表。其输入为经处理的句子对列表（sentences）。该函数首先初始化词汇表，先设置了三个特殊符号，即词汇表中索引0、1、2的三个位置固定为为特殊符号P（填充）、S（开始）和E（结束）；其次该函数遍历句子对列表（sentences），将源句子和目标句子拆分为单词，不重复地添加到源词汇表（src_vocab）和目标词汇表（tgt_vocab）中；最后保存两张词汇表到vocab.pth中，输出构建好的源词汇表（src_vocab）和目标词汇表（tgt_vocab）；
    
  - **函数make_data**负责对数据进行预处理。其输入为句子对列表（sentences）、源语言和目标语言的词汇表及其长度。该函数经过索引化和填充，将源语言句子转化为编码器输入（enc_inputs），将目标语言句子转化为解码器输入（dec_inputs）和输出（dec_outputs）， 将输入的句子转化成模型训练所需要的张量格式。
    
  - **类MyDataSet**类封装了编码器输入（enc_inputs）、解码器输入（dec_inputs）和解码器输出（dec_outputs），以便在训练或推理过程中方便地加载和处理数据。
    
  - **函数get_attn_pad_mask**负责屏蔽填充位置（Padding Mask），防止模型关注填充符号 'P'。其输入为查询序列（seq_q）和键序列（Key）。函数使用 seq_k.data.eq(0) 找到序列中值为 0 的位置，使用 unsqueeze(1) 将掩码扩展为 [batch_size, 1, len_k]，再使用expand 将掩码扩展为 [batch_size, len_q, len_k]，使其与注意力权重的形状一致。
    
  - **函数get_attn_subsequence_mask**负责防止解码器在生成当前词时看到未来的词。该函数使用 np.triu() 生成一个上三角矩阵，使用 torch.from_numpy().byte() 将 NumPy 数组转换为 PyTorch 张量。函数输出一个布尔张量，形状为 [batch_size, len_seq, len_seq]，其中上三角部分为 1，其余部分为 0。
    
- 编码器-解码器模块包含[layers.py](https://github.com/BeerSquare/fun-transformer/blob/main/utils/layers.py)和[model.py](https://github.com/BeerSquare/fun-transformer/blob/main/utils/model.py)两个文件。其中，[model.py](https://github.com/BeerSquare/fun-transformer/blob/main/utils/model.py)包含Encoder、Decoder和Projection的三个核心，这三个核心依赖[layers.py](https://github.com/BeerSquare/fun-transformer/blob/main/utils/layers.py)中的PositionEncoding、MultipleAttention、FF（前馈神经网络层）实现。
  - **类PositionEncoding**负责为 Transformer 模型中的输入序列添加位置信息。其输入为编码器输入（enc_inputs）。该类定义了两个方法，**init方法**负责初始化位置编码模块，生成位置编码表 pos_table。具体的生成公式推导参见**附录**。该方法生成位置编码表后，将其转换为 PyTorch 张量，最后初始化 Dropout 层。**forward 方法**负责将位置编码添加到输入序列中。该方法添加位置编码后应用 Dropout。

  - **类ScaledDotProductAttention**实现了缩放点积注意力机制（Scaled Dot-Product Attention）。该机制的公式和具体作用详见**附录**。其中**init方法**负责初始化缩放点积注意力模块，保存 d_k 的值，用于后续的缩放操作。**forward 方法**负责计算缩放点积注意力，返回上下文表示 context 和注意力权重 attn。
    
  - **类MultiHeadAttention**负责实现多头注意力机制。该机制的公式和具体作用详见**附录**。其中**init方法**定义线性变换矩阵 W_Q、W_K、W_V，用于将输入映射到多个注意力头的查询、键和值；定义线性变换矩阵 fc，用于将多个注意力头的输出拼接并映射回 d_model 维度。**forward 方法**计算了多头注意力。

  - **类FF**负责实现前馈神经网络。init初始化前馈神经网络模块。 forward 方法对输入进行前馈神经网络处理，残差连接和层归一化：
  - **类EncoderLayer**负责编码器的单层结构，包含多头自注意力和前馈神经网络。
  - **类Encoder**负责将输入序列编码为上下文表示。
  - **类DecoderLayer**负责解码器的单层结构，包含多头自注意力、编码器-解码器注意力和前馈神经网络。
  - **类Decoder**负责基于编码器的输出和之前的解码器输出，生成目标序列。
  - **类Transformer**为完整的 Transformer 模型，包括编码器和解码器。
- train.py为训练模块。
- test.py为测试模块。

## 附录：核心机制的可视化
### 词嵌入与词向量
- 词汇映射
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  plt.rcParams['font.sans-serif']=['SimSun']
  from mpl_toolkits.mplot3d import Axes3D
  
  words=["猫","狗","爱","跑"]
  vectors=[[0.3,0.4,0.25],[0.35,0.45,0.3],[0.8,0.8,0.8],[0.1,0.2,0.6]]
  
  #将三维向量转化为numpy数组
  vectors=np.array(vectors)
  
  #创建3维坐标轴底图
  fig=plt.figure(figsize=(15,7))#图像大小10*7
  ax=fig.add_subplot(111,projection='3d')#在图像fig（1，1，1）处添加子图并指定为3d投影
  
  #将每个词汇向量描点在3d底图上
  scatter=ax.scatter(vectors[:,0],vectors[:,1],vectors[:,2],c=['blue','green','red','purple'],s=100)#散点们的x、y、z坐标分别从vector数组的第0、1、2列获取，设置颜色，设置散点大小为100
  
  #为每个点添加标签
  for i,word in enumerate(words):#遍历words列表，每次迭代返回一个元组(i,word),其中word是返回单词，i是索引
      ax.text(vectors[i,0],vectors[i,1],vectors[i,2],word,size=15,zorder=1)#zorder是文本位置，越大越靠上显示
  
  ax.set_xlabel('X轴',fontsize=14)
  ax.set_ylabel('Y轴',fontsize=14)
  ax.set_zlabel('Z轴',fontsize=14)
  
  ax.set_xticklabels([])
  ax.set_yticklabels([])
![image](https://github.com/user-attachments/assets/4ca96882-2674-43ff-adec-bc476dd56735)
- 词向量转化
  ```python
  import jieba
  import re
  import numpy as np
  from sklearn.decomposition import PCA
  import gensim
  from gensim.models import Word2Vec
  import matplotlib.pyplot as plt
  import matplotlib
  
  #r是原始字符串，避免\被解释为转义字符
  f=open(r"C:\...\hongloumeng.txt",encoding='gb2312',errors='ignore')
  #创建一个空列表 lines，用于存储处理后的文本行
  lines=[]
  #遍历文件的每一行，line 是当前行的字符串
  for line in f:
      #将字符串 line 分词，返回一个分词后的列表 temp
      temp=jieba.lcut(line)
      #创建一个空列表 words，用于存储当前行处理后的词语
      words=[]
      #遍历分词后的列表 temp，i 是当前词语
      for i in temp:
          #去除不必要字符
          i=re.sub("[\s+\.\!\/_.$%^*(++\"\'“”《》]+|[+——！，。？、\
                              ~·@#￥%……&* ( ) '------------'；：‘]+","",i)
          #如果处理后的词语 i 不为空（len(i) > 0）
          #则将其添加到列表 words 中
          if len(i)>0:
              words.append(i)
      #如果当前行处理后的词语列表 words 不为空，则将其添加到 lines 中
      if len(words)>0:
          lines.append(words)
  #展示前三段分词结束
  print(lines[:3])
  
  #模型会遍历整个训练数据集的次数为7，负样本（目标词和随机词）来学习词向量
  #Word2Vec 模型通过对比正样本（目标词和上下文词）和负样本（目标词和随机词）来学习词向量
  model = Word2Vec(lines, vector_size = 20, window=3, min_count=3, \
                   epochs=7,negative=10)
  
  # 输入一个路径，保存训练好的模型
  model.save("C:/.../word2vec_gensim")
  
  #可视化
  import numpy as np
  import matplotlib.pyplot as plt
  #将高维数据（如文本、图像、特征向量）转换为低维数据
  from sklearn.decomposition import PCA
  import matplotlib.font_manager as fm
  
  #定义函数用于绘制词向量，model是训练好的词模型，words_to_plot是需要特别标注的词列表
  def plot_word_vectors(model, words_to_plot):
      rawWorVec = []#用于存储所有词向量的列表
      word2ind = {}#用于存储词语到索引的映射词典
      #enumerate可以返回model.wv.index_to_key里的索引和值，即词汇表的索引和值
      #i是索引，即第一次循环i=0，w=model.wv.index_to_key[0]，w是一个词
      for i, w in enumerate(model.wv.index_to_key):
          rawWorVec.append(model.wv[w])#w的词向量存储到rawWorVec中
          word2ind[w] = i#key=w的位置存一下i这个value，即词汇w和其对应的索引存储到word2ind中
  
      rawWorVec = np.array(rawWorVec)#Word2Vec本身默认是Numpy，但为了统一格式，会显式转换一下
      X_reduced = PCA(n_components=2).fit_transform(rawWorVec)#将高维词向量降维到 2 维，便于可视化
  
      fig = plt.figure(figsize=(16, 16))#创建画布16*16
      ax = fig.gca()#获取当前坐标系，即获取 Axes 对象，才能直接对坐标系进行自定义设置
      ax.set_facecolor('white')#设置背景颜色
      #绘制散点图
      #X_reduced[:, 0] 和 X_reduced[:, 1]分别表示降维后的数据的第一列映射到x轴、第二列映射到y轴
      #点的颜色为黑色，大小为 1，透明度为 0.3。
      ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.3, color='black')
  
      # 查找系统中可用的中文字体
      #fontpaths=None表示在所有默认字体路径中查找
      #表示查找扩展名为 .ttf 的字体文件
      font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
      for font in font_list:
          #如果找到 "SimHei" 字体，则使用 fm.FontProperties() 加载该字体
          if 'SimHei' in font:
              zhfont1 = fm.FontProperties(fname=font, size=10)
              break
  
      for w in words_to_plot:
          if w in word2ind:
              ind = word2ind[w]#获取词 w 的索引
              xy = X_reduced[ind]#获取词在降维空间中的二维坐标 [x, y]
              plt.plot(xy[0], xy[1], '.', alpha=1, color='green', markersize=10)
              plt.text(xy[0], xy[1], w, alpha=1, color='blue', fontproperties=zhfont1)
  
      plt.show()
  
  # 假设已经加载了词向量模型
  # 这里需要根据实际情况加载模型，例如：
  # from gensim.models import Word2Vec
  # model = Word2Vec.load('your_model.bin')
  
  words = ['紫鹃', '香菱', '王熙凤', '林黛玉', '贾宝玉']
  model=Word2Vec.load(r'C:\...\word2vec_gensim')
  #调用了plot_word_vectors(model, words_to_plot)函数
  plot_word_vectors(model, words)
![image](https://github.com/user-attachments/assets/ffb3672e-0cfd-49ad-adf2-857d2bc36a8c)
### 位置编码
![image](https://github.com/user-attachments/assets/0916d559-b5ad-4876-a67e-5222212883ae)
### 注意力机制
![image](https://github.com/user-attachments/assets/83321008-7d1e-4cfe-b1b7-876ba99831bd)
![image](https://github.com/user-attachments/assets/9baac572-c115-45d2-9f6a-d50da8354e7c)

### 编码器-解码器
![image](https://github.com/user-attachments/assets/026c1f5c-f6b7-434a-be1c-41912577a1d2)

## 常见问题QA
### 为什么数据预处理需要填充？
- 在Seq2Seq问题中，输入序列的长度通常不一致。为了将多个序列打包成一个批次（batch），通常会将较短的序列填充到固定长度，填充符号通常用 0 表示，以便后续统一作为张量处理。
### 为什么目标语言既是解码器的输入、又是解码器的输出？
- 这种设计是为了实现自**回归Auto-regressive**的训练方式。在训练阶段，解码器的目标是逐步生成目标序列。为了实现这一点，解码器在每一步的输入是前一步的输出。
- 例如，解码器首先生成第一个词 "Full"，然后以 "Full" 作为输入生成第二个词 "of"，依此类推。
### 位置编码的具体过程是？
- 经典的编码过程（摘自《Attention is All You Need》）如下。设定d为位置向量的维度。在Transformer中，词向量和位置向量是逐元素相加的，因此位置向量的维度通常与词向量维度相同。 设定i为位置编码向量的维度的索引，其取值为0到d/2-1。
- 当i为偶数是，即2i=(0,2,4,…)$时，位置编码应当使用正弦函数；当i为奇数时，即2i+1=(1, 3, 5, …)时，位置编码应当使用余弦函数。因此位置编码公式为：
- $PE(pos，2i)=sin(pos/10000^{2i/d})$
- $PE(pos，2i+1)=cos(pos/10000^{2i/d})$

- 举例来说，假设d=4，则i取值为0到1。对于pos=1，位置编码计算如下：当i=0时，2i=0，则
- $PE(1，0)=sin(1/10000^{0/4})$；
- $PE(1，1)=cos(1/10000^{0/4})$。
- 当i=1时，2i=2，则
- $PE(1，2)=sin(1/10000^{1/4})$；
- $PE(1，3)=cos(1/10000^{1/4})$。
### 为什么要用Dropout在训练时随机丢弃部分位置编码信息？
- Dropout 是一种正则化技术，用于防止神经网络过拟合。它的核心思想是在训练过程中，随机丢弃神经网络中的一部分神经元或特征，模拟噪声，从而减少神经元之间的相互依赖，增强模型的泛化能力，防止过拟合。
### 多头注意力机制的具体实现过程是？
- 它由多个头组成，每个头都有自己的Q、K、V。其中，Q是我们需要关注的内容，K是序列中每个位置的特征，V是序列中每个位置的实际内容。
- 1. 输入x分别与Q的权重矩阵、K的权重矩阵、V的权重矩阵相乘，得到Query、Key、Value。
- 2. 自注意力输出$Z=softmax（QK^T/\sqrt{\smash[b]{d_k}}）V$
### K和V的区别在于？
- *K* 关注的是 **内容的相似性**：它捕捉的是输入序列中每个位置的特征，动态地匹配 Query 与输入序列中的内容的相似度，用于计算注意力分数。
- V 关注的是 **实际的信息**：它捕捉的是输入序列中每个位置的实际信息，用于加权求和
### 为什么要缩放点积注意力？
- 在输出公式中，$QK^T$计算了 Query 和 Key 之间的相似度，被称为点积注意力。可以观察到当查询Q和键值K相似度越高，点积结果越大。
- 其中，$d_k$ 是每个头的维度。通常$d_k=d_{model}/h$，其中$d_{model}$是词向量的维度（也是位置编码的维度），$h$是多头注意力的头数。$\sqrt{\smash[b]{d_k}}$被称为缩放因子，其作用是防止内积过大。
- 其原理在于，当每个注意力头的维度越大，点积注意力$QK^T$这个矩阵内部元素的最大值也会很大，大到在softmax（指数求概率）时，结果会接近一个独热向量，完全不存在其他位置的注意力信息，反正都是1或者0。此时除以一个$\sqrt{\smash[b]{d_k}}$，可以有效缩小$QK^T$softmax后的值。
### 残差连接的目的是？
- 残差连接的目的是缓解梯度消失和梯度爆炸问题。其原理在于，x+y对反向传播时梯度的影响。
- $\frac{\partial {L}}{\partial{x}}={\frac{\partial {L}}{\partial{y}}}.\frac{\partial {y}}{\partial{x}}={\frac{\partial {L}}{\partial{y}}}.\frac{\partial {(F(x)+x)}}{\partial{x}}={\frac{\partial {L}}{\partial{y}}}.{(\frac{\partial {(F(x))}}{\partial{x}}+1)}$
- 即使$\frac{\partial {F(x)}}{\partial{x}}$很小，残差连接仍然可以保证至少有一部分梯度（即 +1）能够传播回去。
### 层归一化的目的是？
- 加速训练过程并提高模型的稳定性。
- 为什么使输出均值为0，方差为1就能加速训练过程并提高模型的稳定性呢？因为输出均值为0，方差为1，可使每一层的输入分布更加稳定，使输入数据集中在激活函数的敏感区域，梯度更容易传播。

## 许可证
本项目基于 [fun-transformer](https://github.com/datawhalechina/fun-transformer) 的 Apache License 2.0许可证。详情请参阅 [LICENSE](https://github.com/BeerSquare/fun-transformer/blob/main/LICENSE.txt) 文件。

## 联系方式
如有任何问题，请联系：
- 邮箱：1259712042@qq.com
- GitHub:[BeerSquare](https://github.com/BeerSquare)


