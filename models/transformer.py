# Created by MengQingluo on 2024/11/12 18:24
import torchvision.transforms as T

def build_trainsformer(cfg):
    if cfg.DATASETS.NAMES == 'vsw':
        image_size = cfg.INPUT.IMAGE_SIZE_LARGE
    else:
        image_size = cfg.INPUT.IMAGE_SIZE
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        normalize_transform,
    ])
    return transform
