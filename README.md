## RepVGG实现10分类

### data：kaggle

### RepVGGpaper：[arxiv.org](https://arxiv.org/abs/2101.03697)

### RepVgg author：[DingXiaoH/RepVGG: RepVGG: Making VGG-style ConvNets Great Again (github.com)](https://github.com/DingXiaoH/RepVGG)

### Introduced

简单实现了下，但是训练效果不是很好，大概60准确率

train.py是直接用单gpu跑，如果有多gpus，可以直接DP加速。

trainDateParrelle.py是用DDP加速

运行方式

```
 python -m torch.distributed.launch --use_env --standalone --nnodes 1 --nproc_per_node [nums of gpus] trainDateParrelle.py
```
pt file:
https://drive.google.com/file/d/13GFn_UnUCMaUfkKQntUwaeIq7zEg3EPW/view?usp=sharing, https://drive.google.com/file/d/1WhaUcuiOcWY1ZFb9Mc1r_IAson8Hr6rW/view?usp=sharing
