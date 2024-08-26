Transform:

视频学习链接：https://www.bilibili.com/video/BV12L4y157MS/?p=7&spm_id_from=pageDriver&vd_source=714d825f236a68ff0eac19dda098f7e1

解码器：

 

位置编码（Positional Encoding）的实例化理解：https://www.bilibili.com/video/BV1yG411Z78H/?spm_id_from=333.337.search-card.all.click&vd_source=714d825f236a68ff0eac19dda098f7e1

<img src="https://raw.githubusercontent.com/Benluncy/PicGo/main/202408261157894.png" alt="image-20240825144218288" style="zoom: 15%;" />

| 我   | Pos  | dim=6              | (pos,2i or 2i+1)   | (i的最大值能取dim/2) |                  |      |      |
| ---- | ---- | ------------------ | ------------------ | -------------------- | ---------------- | ---- | ---- |
| 有   | 0    | if i=0：i=0,2i+1=2 | 可算出：0和0+1的值 | if i=1：2i=2,2i+1=3  | 可算出：2和3的值 |      |      |
| 一   | 1    |                    |                    |                      |                  |      |      |
| 只   | 2    |                    |                    |                      |                  |      |      |
| 猫   | 3    |                    |                    |                      |                  |      |      |
| 。   | 4    |                    |                    |                      |                  |      |      |

理解：首先对数据进行编码，以上面的文本为例，将每个汉字编码为1*6维度的向量。然后利用位置编码可以得到加入对应位置编码信息以后的向量。

注：有很多论文在研究：加入的位置编码信息在transformer中到达多头注意力机制的时候已经没有位置编码信息了，若有时间可测试研究（待研究）



编码器：

1.由N个编码器层堆叠而成

2.每个编码器层由两个子层连接结构组成

3.第一个子层连接结构包括一个多头自注意力子层的规范化层以及一个残差连接；

4.第二个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接

<img src="../../../AppData/Roaming/Typora/typora-user-images/image-20240819214950232.png" alt="image-20240819214950232" style="zoom: 67%;" />



解码器部分：

1.由N个解码器层堆叠而成

2.每个解码器层由三个子层连接结构组成；

3.第一个子层连接结构包括一个**自**多头注意力子层（加了一个掩码）和规范化层以及一个残差连接

4.第二个子层连接结构包括一个多头注意力子层和规范化层以及一个残差连接

5.第三个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接

<img src="C:/Users/2023/AppData/Roaming/Typora/typora-user-images/image-20240820211658640.png" alt="image-20240820211658640" style="zoom:50%;" />

注：解码器比编码器多了：1.masked 多头注意力层；2.交互注意力层（也就是和解码器的输出有联系的层）

训练时解码器的输入是要预测的序列

1. 遮盖的多头注意力层的目的：masked的意义是为了将未来信心掩盖住，使得训练出来的模型更加准确。

   - For example：输入“我爱新中国”，当预测“中”时，模型获得的信息应该是“我爱新”这三个字，即符合因果论。但是Transformer的attn层做计算时是一整个seqs输入的，若不加masked那么在预测“中”时，所获得的的前置信息是“我爱新 国”。“国”对于“中”而言属于未来信息。古在训练室需要掩盖未来信息，这就是需要masked的原因

   - 

2. 可能还存在防止过拟合和提高模型泛化能力的考虑在里面；（code中用torch.triu(）实现<img src="https://raw.githubusercontent.com/Benluncy/PicGo/main/202408252238883.png" alt="image-20240825150821863" style="zoom: 50%;" />

   - 具体实例化理解：预测"我爱新中国时"，预测”我“的时候只有encode那边的信息过来，解码器的”output“的信息全部被遮盖掉（也就是第一行的全部false）	，预测第二个字的放出前面预测的”我“这个字。放在结构中就是把Q，K向量遮盖掉。



1. 交互注意力层：
   - 解码器中的交互注意力层与编码器当中的注意力层唯一区别在于，前者计算Query向量的输入是编码器的输出。编码器的注意力层实际上被称为子注意力层
   - 两者的计算方式：
     - 自注意力：<img src="https://raw.githubusercontent.com/Benluncy/PicGo/main/202408252238689.png" alt="image-20240825151851329" style="zoom:33%;" />
     - 交叉注意力：<img src="https://raw.githubusercontent.com/Benluncy/PicGo/main/202408252238897.png" alt="image-20240825151915052" style="zoom:33%;" />，其中$Out_{input}$为解码器的输出





#### 总结：在不考虑修改Transformer进行相关研究的情况下可将transformer作为一个Seq2seq模型进行实例化应用，处理好输入端和输出端就行。

#后工作--VIT#



### Diffusion-TS：Interpretable Diffusion for General Time Series Generation（from 合肥工业大学）——在时序语音信号中的应用

<img src="https://raw.githubusercontent.com/Benluncy/PicGo/main/202408252238893.png" alt="image-20240825154543136" style="zoom:50%;" />

- [ ] 时序生成（Time Series Generation）-基于diff的条件生成 
- [ ] 可解释性（interpretable Diff）--结合趋势分解，学习有意义的时间属性

##### 创行点：

- 在DDPM的框架上，重新设计去噪网络（和AGD-Adapt有相似性）
- 去噪网络基于transformer（encoder-decoder），引入可解释的分解结构
- 训练一个无条件的模型，兼容不同的条件任务



##### 算法理解描述：

1. 前向没变，反向感觉是将ddpm的学习噪声改为了从$x_t$直接学习$x_{t-1}$，没有调成噪声然后剪掉噪声的这么一个过程了。
2. 学习一个近似器，让模型的输出尽可能的逼近输入
3. 需要对transformer有一定的理解（可基于transformer开展一些工作加深理解）

关键词：梯度引导，训练分类器

### Vision Transformer(ViT)

<img src="https://raw.githubusercontent.com/Benluncy/PicGo/main/202408261157616.png" alt="image-20240825172123631" style="zoom: 50%;" />

- 文章对比模型：Vit，Hybrid，resnet



##### 结构分解：

1. Linear Projection of Flattened Patches (Embedding层)
2. Transformer Encoder（这个就是原Trans的结构没变）
3. MLP Head （最终用于分类的层结构）

<img src="https://upload-images.jianshu.io/upload_images/23551183-6279f24de7aba178.gif?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" alt="img" style="zoom:33%;" />

Transformer模块的输入要求是token序列，即二维矩阵[num_token,token_dim]。

- 在代码的实现中，直接通过一个卷积层来实现，e.g.，ViT-16，使用卷积核大小为16*16，步长为16（无栅格窗口），得出的卷积核个数为768。
- [224,224,3]->[14,14,768]->[196,768]----------$[14\times 14;16\times16\times3]$
- 在输入Transformer Encoder之前需要加上[class]token 以及Position Embedding，都是可训练参数
  - 拼接后：[class]token:Cat([1,768,[196,768]])->最后变成了[197,768]，相当于前面加了一个位置标签
  - 然后叠加位置编码：[197,768]->[197,768]

​			注：使用位置编码确实有提升整体性能，原论文中最终采取的1-d位置编码

<img src="https://raw.githubusercontent.com/Benluncy/PicGo/main/202408252245855.png" alt="image-20240825211703700" style="zoom: 50%;" /><img src="https://raw.githubusercontent.com/Benluncy/PicGo/main/202408252239480.png" alt="image-20240825211953730" style="zoom:33%;" />















