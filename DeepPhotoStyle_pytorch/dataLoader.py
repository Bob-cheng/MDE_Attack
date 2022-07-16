from PIL import Image
from scipy import interpolate
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import random
import torch.nn.functional as F
import utils

def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            return img.convert('RGB')
        else:
            return img.convert('I')


def seperate_sets(root_dir, fn):
    file_path = os.path.join(root_dir, fn)
    with open(file_path) as f:
        names = f.readlines()
    folder_name = 'training'
    vehicle_types = ['Car', 'Van', 'Truck']
    vehicle_fns = []
    non_vehicle_fns = []

    for name in names:
        name = name.rstrip()
        label_path = os.path.join(root_dir, folder_name, 'label_2', name+'.txt')
        with open(label_path) as label_f:
            lines = label_f.readlines()
            vehicle_found = False
            for line in lines:
                type = line.split(' ')[0]
                if type in vehicle_types:
                    vehicle_fns.append(name)
                    vehicle_found = True
                    break
            if not vehicle_found:
                non_vehicle_fns.append(name)
    sub_folder = 'vehicle_detection'
    
    with open(os.path.join(root_dir, sub_folder, 'trainval_vehicle.txt'), 'w') as f:
        f.write('\n'.join(vehicle_fns) + '\n')
    
    with open(os.path.join(root_dir, sub_folder, 'trainval_no_vehicle.txt'), 'w') as f:
        f.write('\n'.join(non_vehicle_fns) + '\n')

    class_num = len(non_vehicle_fns)
    random.Random(1234).shuffle(vehicle_fns)
    selected_vehicle_fns = vehicle_fns[0:class_num]
    output_list = []
    for name in non_vehicle_fns:
        output_list.append((name, 0))
    for name in selected_vehicle_fns:
        output_list.append((name, 1))
    random.Random(1234).shuffle(output_list)
    training_num = int(class_num * 2 * 0.8)
    training_list = output_list[:training_num]
    testing_list = output_list[training_num:]

    with open(os.path.join(root_dir, sub_folder, 'training.txt'), 'w') as f:
        for item in training_list:
            f.write(item[0] + ' ' + str(item[1]) + '\n')
    
    with open(os.path.join(root_dir, sub_folder, 'testing.txt'), 'w') as f:
        for item in testing_list:
            f.write(item[0] + ' ' + str(item[1]) + '\n')


def readPathFiles(root_dir, list_name):
    file_path = os.path.join(root_dir, list_name)
    base_path = os.path.join(root_dir, 'training', 'image_2')
    filename_list = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            if len(items) == 2:
                filename_list.append((os.path.join(base_path, items[0].rstrip()+'.png'), int(items[1])))
            else:
                filename_list.append((os.path.join(base_path, items[0].rstrip()+'.png'), 1))
    # print(filename_list)
    return filename_list



def lin_interp(sparse_depth):
    # modified from https://github.com/hunse/kitti
    m, n = sparse_depth.shape
    ij = np.zeros((len(sparse_depth[sparse_depth>0]), 2))
    x, y = np.where(sparse_depth>0)
    ij[:,0] = x
    ij[:,1] = y
    d = sparse_depth[x,y]
    f = interpolate.LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    interp_depth = f(IJ).reshape(sparse_depth.shape)
    return interp_depth


class KittiLoader(Dataset):
    """
        RGB image path:
        kitti_raw_data/2011_xx_xx/2011_xx_xx_drive_xxxx_sync/image_0x/data/xxxxxxxxxx.png
        
        Depth path:
        train: train/2011_xx_xx/2011_xx_xx_drive_xxxx_sync/proj_depth/groundtruth/image_0x/xxxxxxxxxx.png
        val: val/2011_xx_xx/2011_xx_xx_drive_xxxx_sync/proj_depth/groundtruth/image_0x/xxxxxxxxxx.png
        
        KITTI mean & std
        self.mean = torch.Tensor([0.3864, 0.4146, 0.3952])
        self.std = torch.Tensor([0.2945, 0.3085, 0.3134])
        
        ImageNet mean & std
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])
        
    """
    
    def __init__(self, root_dir=utils.kitti_object_path,
                 mode='train', loader=pil_loader, size=(1024, 320), 
                 train_list='vehicle_detection/training.txt', 
                 val_list='vehicle_detection/testing.txt', data_limit:int=-1):
        super(KittiLoader, self).__init__()
        self.root_dir = root_dir

        self.mode = mode
        # self.filepaths = None
        self.loader = loader
        self.size = size
        self.datalimit = data_limit
        
        # set ImageNet mean and std for image normalization
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])
        self.uni_std = torch.Tensor([1, 1, 1])      
        
        # set color jitter parameter
        self.brightness =0.2
        self.contrast = 0.2 
        self.saturation = 0.2
        self.hue = 0.1

        if self.mode == 'train':
            self.filepaths = readPathFiles(root_dir, train_list)
        elif self.mode == 'val':
            self.filepaths = readPathFiles(root_dir, val_list)    

    def __len__(self):
        if self.datalimit == -1:
            return len(self.filepaths)
        else:
            return self.datalimit
    

    def get_color(self, color_path):
        color = self.loader(color_path, rgb=True)
        
        return color

    def get_depth(self, depth_path):
        sparse_depth = self.loader(depth_path, rgb=False)
        sparse_depth = np.asarray(sparse_depth) / 256.
        interp_depth = lin_interp(sparse_depth)

        return sparse_depth, interp_depth
    
    def train_transform(self, color):
        
        # augmentation parameters
        rotation_angle = 5.0 # random rotation degrees
        flip_p = 0.5  # random horizontal flip
        color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) # adjust color for input RGB image

        original_w, original_h = color.size
        new_w, new_h = self.size

        # garg/eigen crop, x=153:371, y=44:1197
        CROP_LEFT = (original_w - new_w)//2
        CROP_TOP = original_h - new_h
        CROP_RIGHT = CROP_LEFT + new_w
        CROP_BOTTOM = original_h
        _color = color.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)) 
    
        # transform = T.Compose([
        #     T.Resize((385, CROP_RIGHT-CROP_LEFT), T.InterpolationMode.BILINEAR), # resize x-axis, and remain y-axis
        #     T.CenterCrop(self.size),
        # ])
        
        # _color = transform(_color)
        # _color = color_jitter(_color)
       
        
        _color = np.array(_color).astype(np.float32) / 256.0
        
        _color = T.ToTensor()(_color)
#         if self.norm:
#             if self.uni:
#                 im_ = T.Normalize(mean=self.mean, std=self.std)(im_)
#             else:
#                 im_ = T.Normalize(mean=self.mean, std=self.uni_std)(im_)

        return _color
    
    def val_transform(self, color, sparse_depth, dense_depth):
        
        sparse_depth = Image.fromarray(sparse_depth)
        dense_depth = Image.fromarray(dense_depth)
        
        # garg/eigen crop, x=153:371, y=44:1197
        CROP_LEFT = 44
        CROP_TOP = 153
        CROP_RIGHT = 1197
        CROP_BOTTOM = 371
        _color = color.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)) 
        _sparse_depth = sparse_depth.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)) 
        _dense_depth = dense_depth.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM))
    
        transform = T.Compose([
            T.Resize((385, CROP_RIGHT-CROP_LEFT), T.InterpolationMode.BILINEAR), # resize x-axis, and remain y-axis
            T.CenterCrop(self.size),
        ])
        
        _color = transform(_color)
        _sparse_depth = transform(_sparse_depth)
        _dense_depth = transform(_dense_depth)
        
        _color = np.array(_color).astype(np.float32) / 256.0
        _sparse_depth = np.array(_sparse_depth).astype(np.float32)
        _dense_depth = np.array(_dense_depth).astype(np.float32)  

        _color = T.ToTensor()(_color)
        _sparse_depth = T.ToTensor()(_sparse_depth)
        _dense_depth = T.ToTensor()(_dense_depth)
        
#         if self.norm:
#             if self.uni:
#                 im_ = T.Normalize(mean=self.mean, std=self.std)(im_)
#             else:
#                 im_ = T.Normalize(mean=self.mean, std=self.uni_std)(im_)

        return _color, _sparse_depth, _dense_depth

    def __getitem__(self, idx):
        color_path, label = self.filepaths[idx]
        color = self.get_color(color_path)
        color = self.train_transform(color)
        # target = F.one_hot(torch.tensor(label,dtype=torch.int64), num_classes=2)
        target = torch.tensor(label,dtype=torch.int64)
        
        return color, target

        # if self.mode == 'train':
        #     color, sparse_depth, interp_depth = self.train_transform(color, sparse_depth, interp_depth)
        #     return color, sparse_depth, interp_depth
        # elif self.mode == 'val':
        #     color, sparse_depth, interp_depth = self.val_transform(color, sparse_depth, interp_depth)
        #     return color, sparse_depth, interp_depth


if __name__ == "__main__":
    
    import config
    kitti_loader_train = KittiLoader(mode='train', train_list='trainval.txt', val_list='val.txt')
    kitti_loader_eval = KittiLoader(mode='val', train_list='trainval.txt', val_list='val.txt')
    train_loader = DataLoader(kitti_loader_train, batch_size=3, shuffle=True, num_workers=10, pin_memory=True)
    test_loader = DataLoader(kitti_loader_eval, batch_size=3, shuffle=False, num_workers=10, pin_memory=True)
    for color, target in train_loader:
        color.to(config.device0)
        target.to(config.device0)
        utils.save_pic(color[0], 3, utils.project_root + 'DeepPhotoStyle_pytorch/')
        print(color.size())
        print(target.size())
        break