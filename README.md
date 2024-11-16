# 特征可视化

## 效果展示1
|<img src="https://github.com/qlmeng10/Feature-Visualization/blob/master/data/veri/0001_c001_00016490_0_c01_t4.jpg" alt="原图" width="256" height="256">|<img src="https://github.com/qlmeng10/Feature-Visualization/blob/master/data_output/veri/cam/0001_c001_00016490_0_c01_t4_feat_res4.jpg" alt="CAM" width="256" height="256"> | <img src="https://github.com/qlmeng10/Feature-Visualization/blob/master/data_output/veri/heatmap/0001_c001_00016490_0_c01_t4_feat_res1.jpg" alt="Heatmap" width="256" height="256">|

## 效果展示2
|<img src=https://github.com/qlmeng10/Feature-Visualization/blob/master/data/veri/0137_c013_00046940_0_c07_t8.jpg alt="原图" width="256" height="256">|
<img src="https://github.com/qlmeng10/Feature-Visualization/blob/master/data_output/veri/cam/0137_c013_00046940_0_c07_t8_feat_res4.jpg" alt="CAM" width="256" height="256"> | 
<img src="https://github.com/qlmeng10/Feature-Visualization/blob/master/data_output/veri/heatmap/0137_c013_00046940_0_c07_t8_feat_res1.jpg" alt="Heatmap" width="256" height="256">|

## 替换数据集
1. 在data目录下，任意添加一个文件目录（如XXX）,里面若干张图片。
2. 修改 baseline.yaml 文件，DATASETS:NAMES:"XXX"即可
3. 运行 python main.py 

