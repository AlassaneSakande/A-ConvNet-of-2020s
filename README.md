## A ConvNet of the 2020s

<p align="center">
<img src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png" alt="Python" height="40" style="vertical-align:top; margin:4px">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252"/>
</p>


![c2](https://user-images.githubusercontent.com/84173235/177842088-e5929e0e-36f4-4636-b3f9-3918cea9ae36.png)

This repo is about an implementation of the prestigious research paper "A ConvNet of the 2020s" published by Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie, Facebook AI Research (FAIR), UC Berkeley on the 10 January 2022.

Note that the python code files in /src are the final codes expected by the novel ConvNeXt architecture
If you are looking for a step by step process following the paper just look to the notebook file



As described in the original papers we'll start from a simple ResNet and move gradually to our ConvNext basing on the archictecture of the Swin Transformer. Please refer to the research papers : https://arxiv.org/abs/2201.03545*

Let's sum up things to make them easier:

The architecture is as follow:


> ResNet-50

> Macro Design

> ResNext

> Inverted Bottleneck

> Large Kernel

> Micro Design

1.   We'll begin with the original ResNet-50 while considering an enhancement of capabilities by using some modern training procedures. Refer to this paper: https://openreview.net/pdf?id=NG6MJnVl6M5

2.   Next, we'll adjust the design of the number of blocks in each stage from regular ResNet(3-4-6-3) to 3-3-9-3 to approach similar FLOPs as in Swin Transformer and also replace the ResNet stem cell of (7,7 stride=2) with one of (4,4 stride=4) following the Swin Transformer.

3.   After that, we will mind use depthwise convolution effectively to reduce the network FLOPs by increasing the wifth if the network from 64 to 96.
     We'll also use **Inverted Bottleneck** by inversing the ResNet bottleneckt structure

4.   Following this we must increase the kernel of the bottleneck from 3,3 to a larger one: 7,7**

5.   Finally we must use fewer activation functions and replace ReLU by GELU.

  Use fewer normalization layers by replacing BachNorm (BN) with   Layer Normalization (LN).

  And Add a (2,2 stride=2) spatial downsampling at the start of each state.
    
