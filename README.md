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
- 数据格式：[hongloumeng.txt](https://github.com/BeerSquare/fun-transformer/hongloumeng.txt)是一个包含中英文句子的列表，格式如下：
  ```txt
  满纸荒唐言|Full of nonsense
  一把辛酸泪|A handful of bitter tears
  都言作者痴|They say the author is foolish
  谁解其中味|Who understands the true meaning

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

### 测试模型
- 使用以下代码进行翻译测试：
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
  Epoch: 0001, Loss: 3.222407
  Epoch: 0001, Loss: 3.209087
  Epoch: 0002, Loss: 2.994991
  Epoch: 0002, Loss: 2.858533
  Epoch: 0003, Loss: 2.630063
  Epoch: 0003, Loss: 2.376590
  Epoch: 0004, Loss: 2.329458
  Epoch: 0004, Loss: 1.806459
  Epoch: 0005, Loss: 1.912214
  Epoch: 0005, Loss: 1.548342
  Epoch: 0006, Loss: 1.447533
  Epoch: 0006, Loss: 1.313579
  Epoch: 0007, Loss: 1.453127
  Epoch: 0007, Loss: 1.045131
  Epoch: 0008, Loss: 1.095424
  Epoch: 0008, Loss: 0.956225
  Epoch: 0009, Loss: 0.847677
  Epoch: 0009, Loss: 0.862410
  Epoch: 0010, Loss: 0.761644
  Epoch: 0010, Loss: 0.608952
  Epoch: 0011, Loss: 0.427852
  Epoch: 0011, Loss: 0.623979
  Epoch: 0012, Loss: 0.510841
  Epoch: 0012, Loss: 0.247995
  Epoch: 0013, Loss: 0.329883
  Epoch: 0013, Loss: 0.314782
  Epoch: 0014, Loss: 0.238684
  Epoch: 0014, Loss: 0.197691
  Epoch: 0015, Loss: 0.173405
  Epoch: 0015, Loss: 0.211747
  Epoch: 0016, Loss: 0.157756
  Epoch: 0016, Loss: 0.129617
  Epoch: 0017, Loss: 0.058498
  Epoch: 0017, Loss: 0.088644
  Epoch: 0018, Loss: 0.066126
  Epoch: 0018, Loss: 0.050853
  Epoch: 0019, Loss: 0.046373
  Epoch: 0019, Loss: 0.031356
  Epoch: 0020, Loss: 0.036603
  Epoch: 0020, Loss: 0.022961
  Epoch: 0021, Loss: 0.027571
  Epoch: 0021, Loss: 0.012210
  Epoch: 0022, Loss: 0.026893
  Epoch: 0022, Loss: 0.019911
  Epoch: 0023, Loss: 0.021833
  Epoch: 0023, Loss: 0.015205
  Epoch: 0024, Loss: 0.014717
  Epoch: 0024, Loss: 0.026177
  Epoch: 0025, Loss: 0.007949
  Epoch: 0025, Loss: 0.018996
  Epoch: 0026, Loss: 0.013227
  Epoch: 0026, Loss: 0.016381
  Epoch: 0027, Loss: 0.010039
  Epoch: 0027, Loss: 0.022546
  Epoch: 0028, Loss: 0.016663
  Epoch: 0028, Loss: 0.014132
  Epoch: 0029, Loss: 0.014592
  Epoch: 0029, Loss: 0.009989
  Epoch: 0030, Loss: 0.013257
  Epoch: 0030, Loss: 0.011373
  Epoch: 0031, Loss: 0.014153
  Epoch: 0031, Loss: 0.012010
  Epoch: 0032, Loss: 0.015713
  Epoch: 0032, Loss: 0.010886
  Epoch: 0033, Loss: 0.018617
  Epoch: 0033, Loss: 0.011008
  Epoch: 0034, Loss: 0.011933
  Epoch: 0034, Loss: 0.016848
  Epoch: 0035, Loss: 0.019637
  Epoch: 0035, Loss: 0.006750
  Epoch: 0036, Loss: 0.021592
  Epoch: 0036, Loss: 0.007617
  Epoch: 0037, Loss: 0.005679
  Epoch: 0037, Loss: 0.021839
  Epoch: 0038, Loss: 0.019522
  Epoch: 0038, Loss: 0.004970
  Epoch: 0039, Loss: 0.007639
  Epoch: 0039, Loss: 0.016362
  Epoch: 0040, Loss: 0.008814
  Epoch: 0040, Loss: 0.012206
  Epoch: 0041, Loss: 0.005462
  Epoch: 0041, Loss: 0.010372
  Epoch: 0042, Loss: 0.008653
  Epoch: 0042, Loss: 0.006187
  Epoch: 0043, Loss: 0.004641
  Epoch: 0043, Loss: 0.006426
  Epoch: 0044, Loss: 0.004145
  Epoch: 0044, Loss: 0.005483
  Epoch: 0045, Loss: 0.005588
  Epoch: 0045, Loss: 0.004220
  Epoch: 0046, Loss: 0.004442
  Epoch: 0046, Loss: 0.004004
  Epoch: 0047, Loss: 0.004562
  Epoch: 0047, Loss: 0.004575
  Epoch: 0048, Loss: 0.002863
  Epoch: 0048, Loss: 0.005559
  Epoch: 0049, Loss: 0.002542
  Epoch: 0049, Loss: 0.006132
  Epoch: 0050, Loss: 0.003476
  Epoch: 0050, Loss: 0.003182
  源句子: 满纸辛酸泪
  翻译结果: They say the author is foolish
- 翻译结果：张冠李戴。txt内是所有词汇对应表。训练数据太少，只能缘木求鱼；凡有修改，就会张冠李戴。

### 注意力可视化
- 可以通过 enc_self_attns 和 dec_self_attns 查看 Encoder 和 Decoder 的注意力权重。

## 许可证
本项目基于 [fun-transformer](https://github.com/datawhalechina/fun-transformer) 的 Apache License 2.0许可证。详情请参阅 [LICENSE](https://github.com/BeerSquare/fun-transformer/LICENSE.txt) 文件。

## 联系方式
如有任何问题，请联系：
- 邮箱：1259712042@qq.com
- GitHub:[BeerSquare](https://github.com/BeerSquare)


