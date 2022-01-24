# Image Captioning (图像描述生成)

## 简介

Image Captioning 指的是输入一张图像，生成由若干个单词组成的对图像内容的文本描述。  
典型应用场景包括：图文互搜、图像检索、辅助视觉障碍者等。示例如下图所示:

![Image Captioning Example](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220112-ZuoYihao-ImageCaptioning/captions-splash.jpg)

### 常用数据集

 - [Flickr30k Entities](https://github.com/BryanPlummer/flickr30k_entities)
 - [Microsoft COCO Caption](https://cocodataset.org/#download)
 
    注：在 Image Captioning 领域，通常使用此数据集的 [Karpathy Split](https://www.zhihu.com/question/283314344/answer/700488776)


### 常用评估指标

- BLEU-n（Bilingual Evaluation Understudy，双语评估辅助工具）：比较候选译文和参考译文里的 n-gram 的重合程度，重合程度越高就认为译文质量越高。把 sentence 划分成长度为n个单词的短语，统计它们在标准译文中的出现次数，除以划分总数。
(mBleu-4, best-k)：对于一个 image ，选择生成的最好的 k 个 captions，对于每个 caption，计算其与其他 (k-1) 个 captions 的 BLEU-4 值，再取平均。值越低，多样性越高。
- METEOR（Metric for Evaluation of Translation with Explicit ORdering，显式排序的翻译评估指标）：计算生成结果和参考译文之间的准确率和召回率的调和平均。
- ROUGE（Recall-Oriented Understudy for Gisting Evaluation，面向召回率的摘要评估辅助工具）：大致分为四种：- ROUGE-N，ROUGE-L，ROUGE-W，ROUGE-S。常用的是前两种（ -N 与 -L）。ROUGE-N中的“N”指的是 N-gram，其计算方式与BLEU类似，只是BLEU基于精确率，而 ROUGE 基于召回率。ROUGE-L 中的“L”指的是Longest Common Subsequence，计算的是候选摘要与参考摘要的最长公共子序列长度，长度越长，得分越高，基于F值。
- CIDEr（Consensus-based Image Description Evaluation，基于共识的图像描述评估）：把每个句子看成文档，然后计算其 TF-IDF 向量（注意向量的每个维度表示的是n-gram 而不一定是单词）的余弦夹角，据此得到候选句子和参考句子的相似度。
- SPICE（Semantic Propositional Image Caption Evaluation，语义命题图像标题评估）：SPICE 使用基于图的语义表示来编码 caption 中的 objects, attributes 和 relationships。它先将待评价 caption 和参考 captions 用 Probabilistic Context-Free Grammar (PCFG) dependency parser parse 成 syntactic dependencies trees，然后用基于规则的方法把 dependency tree 映射成 scene graphs。最后计算待评价的 caption 中 objects, attributes 和 relationships 的 F-score 值。

关于以上评估指标的详细定义，请参阅：[文本生成评价方法 ](https://zhuanlan.zhihu.com/p/108630305?utm_source=wechat_timeline)


## 项目说明

本项目使用 PyTorch 深度学习框架实现了 [Oscar](https://arxiv.org/abs/2004.06165) 模型，支持在Microsoft COCO Caption数据集上进行训练和推理。支持使用 BLEU、METEOR、ROUGE、CIDEr、SPICE 指标进行模型精度评估。

## 项目文件结构
```
├─oscar               Oscar核心定义代码
├─oscar_dependencies  模型依赖
├─pretrained_models   预训练模型
├─inference_models    推理时需要的模型 (包括提取图像特征的Bottom Up Attenion以及Oscar本身)
├─README.assets       README.md引用的图片资源
├─scripts             准备环境、训练、推理对应的Shell脚本
└─objects_vocab.txt   COCO Caption数据集的类别名称
```


### 准备环境

在项目根目录下执行：
```
bash scripts/prepare.sh
```
此脚本将完成准备数据集、安装必要的依赖等一系列工作。

### 在 COCO Caption 的训练集上训练

在项目根目录下执行：
```
bash scripts/train.sh
```

### 在 COCO Caption 的测试集上评估

在项目根目录下执行：
```
bash scripts/eval.sh
```

### 推理
在项目根目录下执行：
```
python oscar/infer_on_single.py --image_path <图像文件的路径>
```


## 参考
- [Microsoft-Oscar](https://github.com/microsoft/Oscar)
- [Microsoft COCO Caption Evaluation](https://github.com/LuoweiZhou/coco-caption)
- [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention)
