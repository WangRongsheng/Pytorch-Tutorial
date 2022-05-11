# -*- coding: utf-8 -*-
# 作者：王荣胜
# 时间：2022-05-10

import torch

'''
- 输出：pytorch版本和是否支持GPU（True为支持，Flase为不支持）
'''
print('pytorch版本：',torch.__version__)
print('是否支持GPU：',torch.cuda.is_available())

'''
两个非常重要的方法：dir()和help()的使用
1. dir()：dir(torch)、dir(torch.cuda)、dir(torch.cuda.is_available())
2. help()：help(torch.cuda.is_available)

- 输出：dir()输出方法，help()输出帮助信息
'''
print(dir(torch))
print(dir(torch.cuda))
print(dir(torch.cuda.is_available))
print(help(torch.cuda.is_available))