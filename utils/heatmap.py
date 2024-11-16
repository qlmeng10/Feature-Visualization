# Created by MengQingluo on 2024/11/12 18:41
import os.path

import numpy as np
import matplotlib.pyplot as plt

def show_and_save_featuremap(features, save_path, feat_type="feat_res1"):
    """
    :param features: OrderedDict([["feat_res1", layer1], ["feat_res2", layer2],
                    ["feat_res3", layer3], ["feat_res4", layer4]])
    :param save_path:
    :param feat_type:
    :return: heatmap
    """
    feature = features[feat_type]
    feature = feature.detach().numpy()
    # 以下参数的设置是经过反复调试，得到的较好结果
    feature_ = np.where(feature < 0.5, 0, feature)
    feature_ = np.where(feature_ > 0.8, feature_ + 0.1, feature_)
    feature_ = np.where((feature_ > 0.5) | (feature_ < 0.7), feature_ + 0.2, feature_)
    feature_ = np.where(feature_ > 1, 1, feature_)

    # 创建一个单通道的heatmap，将关键特征映射到红色
    heatmap = np.sum(feature_, axis=1)  # 将通道维度合并
    # 归一化heatmap到[0, 1]范围
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    plt.imshow(heatmap[0], cmap='jet', interpolation='nearest')
    plt.xticks([])  # 禁用 x 轴刻度
    plt.yticks([])  # 禁用 y 轴刻度
    plt.show()
    plt.imsave(save_path, heatmap[0], cmap='jet')
