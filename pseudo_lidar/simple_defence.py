import numpy as np
import torch
from torchvision.transforms import transforms
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image as pil
import io
from scipy import ndimage


def bitdepth_defense(img_rgb: torch.Tensor, depth=5):
    max_value = np.rint(2 ** depth - 1)
    img_rgb = torch.round(img_rgb * max_value)  # int 0-max_val
    img_rgb = img_rgb / max_value  # float 0-1
    return img_rgb

def jpeg_defense(img: torch.Tensor, quality=20):
    tmp = io.BytesIO()
    transforms.ToPILImage()(img.squeeze(0).cpu()).save(tmp, format='jpeg', quality=quality)
    # transforms.ToPILImage()(img.squeeze(0).cpu()).save(tmp, format='png')
    _img = transforms.ToTensor()(pil.open(tmp)).cuda().unsqueeze(0)
    return _img

def blurring_defense(img: torch.Tensor, ksize=5):
    img_np = img.cpu().squeeze(0).numpy() * 255
    img_np = img_np.astype(np.uint8)
    rgb = ndimage.filters.median_filter(img_np, size=(1,ksize, ksize), mode='reflect')
    # rgb = img_np
    rgb = rgb.astype(np.uint8)
    rgb_tensor = torch.from_numpy(rgb).float() / 255
    rgb_tensor = rgb_tensor.unsqueeze(0).cuda()
    return rgb_tensor

def gaussian_noise(img: torch.Tensor, std=0.1):
    noise = torch.normal(torch.zeros_like(img), torch.ones_like(img) * std).cuda()
    img += noise
    img.clamp_(0, 1)
    return img

class Magnet_defence:
    def __init__(self, model_type) -> None:
        from magnet import DenoisingAutoEncoder, MAP_MAGNET, IMG_CROP_HEIGHT, IMG_CROP_WIDTH
        tmp = DenoisingAutoEncoder((IMG_CROP_HEIGHT, IMG_CROP_WIDTH, 3),                                                                                                                                           
                            MAP_MAGNET[model_type]['model'],     
                            model_dir=MAP_MAGNET[model_type]['path'],                                                                                                                            
                            v_noise=0.1,                                                                                                                                                                    
                            activation='relu',                                                                                                                                                              
                            reg_strength=1e-9)
        tmp.load('best_weights.hdf5')
        self.magnet_model = tmp.model
    
    def magnet_defense(self, img: torch.Tensor):
        img_np = img.cpu().numpy().transpose((0, 2, 3, 1))
        _img = self.magnet_model.predict(img_np)[0]
        _img = np.expand_dims(_img.transpose((2, 0, 1)), axis=0)
        _img = torch.from_numpy(_img).clamp_(0, 1).cuda()
        return _img



# def magnet_defense(img, patch):
#     _img = np.where(np.isnan(patch), img.reshape(IMG_INPUT_SHAPE)[0].transpose((1, 2, 0)), patch)
#     rgb = yuv2rgb(_img) / 255 #.astype(np.uint8)
#     rgb = self.magnet_model.predict(np.expand_dims(rgb, axis=0))[0] * 255
#     yuv = rgb2yuv(rgb.astype(np.uint8))
#     _patch = np.where(~np.isnan(patch), yuv, np.nan)
#     return yuv.transpose((2, 0, 1)).flatten(), _patch