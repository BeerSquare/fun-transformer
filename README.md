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
### 删除了以下模块：
- 显式指明sentence，静态生成词汇表
### 添加了以下模块：
- 从 txt 文件中读取句子
- 动态生成词汇表

## 安装指南
### 依赖环境
- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib（可选，用于可视化注意力权重）
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
- 数据格式：[hongloumeng.txt](https://github.com/BeerSquare/fun-transformer/blob/main/hongloumeng.txt)是一个包含中英文句子的列表，格式如下：
  ```txt
  满纸荒唐言|Full of nonsense
  一把辛酸泪|A handful of bitter tears
  都言作者痴|They say the author is foolish
  谁解其中味|Who understands the true meaning
- 或者，你可以选择更大的数据集[ChineseToEnglish.txt](https://github.com/BeerSquare/fun-transformer/blob/main/ChineseToEnglish.txt)。此数据集来源[Tatoeba多语言句子库](https://tatoeba.org/zh-cn/downloads)。

### 训练模型
- 运行以下代码开始训练：
  ```python
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
    loader = Data.DataLoader(dataset, batch_size=32, shuffle=True)

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
    train(model, loader, criterion, optimizer, epochs=10)
    
    # 测试模型
    test(model, src_vocab, tgt_vocab, src_len, tgt_len)

### 使用[hongloumeng.txt](https://github.com/BeerSquare/fun-transformer/blob/main/hongloumeng.txt)作为数据集训练模型
- 测试：翻译“满纸辛酸泪”：
  ```python
  # 测试函数
  def test(model, src_vocab, tgt_vocab, src_len, tgt_len):
    # 测试句子
    test_sentence = "满纸辛酸泪"
    print(f"源句子: {test_sentence}")
    translated_sentence = translate(model, test_sentence, src_vocab, tgt_vocab, src_len, tgt_len)
    print(f"翻译结果: {translated_sentence}")
- 测试结果：
  ```txt
  Epoch: 0001, Avg Loss: 3.185951
  Epoch: 0002, Avg Loss: 3.095047
  Epoch: 0003, Avg Loss: 2.966973
  Epoch: 0004, Avg Loss: 2.807616
  Epoch: 0005, Avg Loss: 2.617177
  Epoch: 0006, Avg Loss: 2.440889
  Epoch: 0007, Avg Loss: 2.246624
  Epoch: 0008, Avg Loss: 2.162881
  Epoch: 0009, Avg Loss: 1.898559
  Epoch: 0010, Avg Loss: 1.730894
  源句子: 满纸辛酸泪
  翻译结果: They say the author
- 翻译结果：张冠李戴。txt内是所有词汇对应表。训练数据太少，只能缘木求鱼；凡有修改，就会张冠李戴。
### 使用[ChineseToEnglish.txt](https://github.com/BeerSquare/fun-transformer/blob/main/ChineseToEnglish.txt)作为数据集训练模型
- 测试：翻译“满纸辛酸泪”：
  ```python
  # 测试函数
  def test(model, src_vocab, tgt_vocab, src_len, tgt_len):
    # 测试句子
    test_sentence = "满纸辛酸泪"
    print(f"源句子: {test_sentence}")
    translated_sentence = translate(model, test_sentence, src_vocab, tgt_vocab, src_len, tgt_len)
    print(f"翻译结果: {translated_sentence}")
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
  源句子: 满纸辛酸泪
  翻译结果: I don't like to do that.
- 翻译结果：意译。勉强“信”，不“达雅”。数据集内现代表达居多。

### 注意力可视化
- 可以通过 enc_self_attns 和 dec_self_attns 查看 Encoder 和 Decoder 的注意力权重。

## 许可证
本项目基于 [fun-transformer](https://github.com/datawhalechina/fun-transformer) 的 Apache License 2.0许可证。详情请参阅 [LICENSE](https://github.com/BeerSquare/fun-transformer/blob/main/LICENSE.txt) 文件。

## 联系方式
如有任何问题，请联系：
- 邮箱：1259712042@qq.com
- GitHub:[BeerSquare](https://github.com/BeerSquare)


