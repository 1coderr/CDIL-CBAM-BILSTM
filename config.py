# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   超惨配置
# ----------------------------------------------------#
epochs = 50  # 训练轮数
batch_size = 1  # 批次量LR = 1e-3
LR = 1e-3

name2number = {
    '噪声': 0,
    '挖掘': 1,
    '泄漏': 2,
}
number2name = {}
for key, val in name2number.items():
    number2name[val] = key



