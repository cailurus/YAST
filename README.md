# YAST: Yet Another Short Text classification toolkit

[![Build Status](https://travis-ci.org/ailurus1991/YAST.svg?branch=master)](https://travis-ci.org/ailurus1991/YAST)
[![Documentation Status](https://readthedocs.org/projects/yast-doc/badge/?version=latest)](http://yast-doc.readthedocs.io/en/latest/?badge=latest)

YAST 是一个简易的文本分类项目，基于 [LibLinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/)，使用 [jieba](https://github.com/fxsjy/jieba) 作为中文分词。

## Getting started

```python
#!/usr/bin/env python
# encoding: utf-8

from yast import Yast

sample = Yast('sample')

sample.train([
    ('stock','英国脱欧与德银危机施压美股收跌'),
    ('stock','港股缺资金难闯24000点 美大选困扰后市'),
    ('f1', '2016丝绸之路拉力赛收官 标致道达尔汽车组夺冠'),
    ('f1','保时捷超级杯霍根海姆站 中国车手张大胜再出击'),
    ('basketball','林书豪透露生涯两低谷：效力湖人勇士令人失望'),
    ('basketball','后场双星合砍27分10助 开拓者全队发挥战胜爵士')])

print sample_2.predict_single('队内对抗曝光湖人新阵容 阿联或任内线主力替补').predicted_y
# basketball
print sample_2.predict_single('再出悲剧！ 达喀尔拉力赛后勤车肇事致1死10伤').predicted_y
# f1

# customize configuration
configs = {
    'grid': 0, # 网格搜索开关。0 为关闭网格搜索，1 为开启。默认关闭。
    'feature': 3, # 特征表达。0 为 Binary feature，1 为 word count，2 为词频，3 为TF-IDF。
    'classifier': 0 # 分类器选择。0 为 Crammer and Singer SVM multiclass，1 为 L1 损失分类 one-vs-rest，2 为 L2损失分类 one-vs-rest，3 为逻辑回归 one-vs-rest。
}

another_sample = Yast('another_sample', configs)
# 可以自定义配置文件
another_sample.train('./train_file.txt')
#训练文件格式为：label 和 text，以分隔符分开，测试文件同。
#label1  text
#label2  text

another_sample.test('./test_file.txt')

print another_sample.analyze('都说苹果的创新力越来越差了，根据您的了解，苹果有哪些外行看不到内行却深感振奋的黑科技？')
# 打印 query 的每个向量的权重，用作分析
```

## Features

- [x] 支持多种分类器
- [x] 支持多种特征表达
- [x] 支持结果分析
- [x] 支持 grid-search 并行搜索 RBF 核函数全局最佳参数 ![equation](http://latex.codecogs.com/gif.latex? \gamma) 和 ![equation](http://latex.codecogs.com/gif.latex? C)
- [x] 自动获取标签

## How to get

```python
pip install yast
```
