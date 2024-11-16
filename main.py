# Created by MengQingluo on 2024/11/12 18:21
import argparse
import os
import pdb

from tqdm import tqdm
from configs.defaults import _C as cfg
from PIL import Image
from models.transformer import build_trainsformer
from models.backbone import build_resnet
from utils.heatmap import show_and_save_featuremap
from utils.cam import show_and_save_cam


def make_output_dir(cfg):
    output_root = cfg.OUTPUT.OUTPUT_ROOT
    output_dir = os.path.join(output_root, cfg.DATASETS.NAMES)
    heatmap_dir = os.path.join(output_dir, cfg.OUTPUT.HEATMAP)
    cam_dir = os.path.join(output_dir, cfg.OUTPUT.CAM)
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(heatmap_dir):
        os.mkdir(heatmap_dir)
    if not os.path.exists(cam_dir):
        os.mkdir(cam_dir)
    return heatmap_dir, cam_dir


def main():
    parser = argparse.ArgumentParser(description="Image Processing")
    parser.add_argument("--config_file", default="configs/baseline.yml", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # input
    data_root = cfg.DATASETS.ROOT_DIR     # data
    images_root = os.path.join(data_root, cfg.DATASETS.NAMES)

    # output
    heatmap_dir, cam_dir = make_output_dir(cfg)

    # model
    model = build_resnet(cfg)
    transform = build_trainsformer(cfg)

    for image_name in tqdm(os.listdir(images_root), desc="Processing"):
        # input
        img_path = os.path.join(images_root, image_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        # output
        heatmap_save_path = os.path.join(heatmap_dir, image_name)
        cam_save_path = os.path.join(cam_dir, image_name)

        # get features
        features = model(img_tensor)

        for key in features.keys():
            # processing
            result_name = "_"+key+".jpg"
            show_and_save_featuremap(features, heatmap_save_path[:-4]+result_name, feat_type=key)
            show_and_save_cam(features, model, img_path, cam_save_path[:-4]+result_name, feat_type=key)

    print("Done!")


if __name__ == '__main__':
    main()