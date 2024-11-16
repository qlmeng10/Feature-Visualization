# Created by MengQingluo on 2024/11/16 16:43
import pdb

import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_cam(feature_map, model, feat_type):
    """

    :param feature_map: C
    :param model:
    :param feat_type:
    :return:
    """
    feat = model.gap(feature_map)           # [1, C,1,1]
    feat = feat.view(feat.shape[0], -1)     # [1,C]
    if feat_type == "feat_res1":
        classifier = model.classifier256
    elif feat_type == "feat_res2":
        classifier = model.classifier512
    elif feat_type == "feat_res3":
        classifier = model.classifier1024
    else:
        classifier = model.classifier

    output = classifier(feat)         # [1, 1000]
    _, class_index = output.max(1)          # [1]
    weight = classifier.weight[class_index[0]]  # [C]

    # pdb.set_trace()
    cam = feature_map[0] * weight[:, None, None]  # [C,H,W]
    cam = cam.sum(axis=0)  # [H,W]
    # ReLU激活和归一化
    cam = F.relu(cam)  # [H,W]
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # [H,W]

    cam = cam.detach().numpy()
    return cam


def show_and_save_cam(features, model, img_path, save_path, feat_type="feat_res4"):
    """

    :param features: OrderedDict([["feat_res1", layer1], ["feat_res2", layer2],
                    ["feat_res3", layer3], ["feat_res4", layer4]])
    :param model:
    :param img_path:
    :param save_path:
    :param feat_type:
    :return:
    """
    feature = features[feat_type]
    cam = get_cam(feature, model, feat_type)

    img = cv2.imread(img_path)  # 用cv2加载原始图像 [H, W, C]
    H, W, _ = img.shape

    # 将热力图的大小调整为与原始图像相同
    cam = cv2.resize(cam, (W, H))
    cam = np.uint8(255 * cam)  # 将热力图转换为RGB格式
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # 将热力图应用于原始图像

    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # 转换颜色映射

    # 图像混合
    superimposed_img = cv2.addWeighted(cam, 0.4, img, 0.6, 0)

    plt.imshow(superimposed_img)  # 使用默认颜色映射
    plt.xticks([])  # 禁用 x 轴刻度
    plt.yticks([])  # 禁用 y 轴刻度
    plt.show()
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, superimposed_img)
