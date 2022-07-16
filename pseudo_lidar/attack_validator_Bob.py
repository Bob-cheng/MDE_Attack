from builtins import print
import os
import sys
import PIL.Image as pil
from PIL import ImageOps
import numpy as np
import torch
from torchvision.transforms import transforms
sys.path.append(".")
from DeepPhotoStyle_pytorch.depth_model import import_depth_model
import random
from preprocessing.kitti_util import Calibration,point_cloud_adjustment
from Open3D_ML.open3d_visualization import get_pointpillars_pipeline, make_obj_pred, visualize_frame, visualize_dataset, visulize_data
from DeepPhotoStyle_pytorch.dataLoader import DataLoader, KittiLoader
from PIL import Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from  torch.nn.functional import interpolate
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import datetime
import laspy
import glob
from numpy.core.fromnumeric import sort
from open3d.ml.vis import LabelLUT, Visualizer
from simple_defence import Magnet_defence, bitdepth_defense, gaussian_noise, jpeg_defense, blurring_defense

kitti_labels = {
    0: 'unlabeled',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign'
}


def eval_depth_diff(img1: torch.tensor, img2: torch.tensor, depth_model, filename):
    disp1 = depth_model(img1).detach().cpu().squeeze().numpy()
    disp2 = depth_model(img2).detach().cpu().squeeze().numpy()
    image1 = transforms.ToPILImage()(img1.squeeze())
    image2 = transforms.ToPILImage()(img2.squeeze())
    diff_disp = np.abs(disp1 - disp2)
    vmax = np.percentile(disp1, 95)
    
    fig: Figure = plt.figure(figsize=(12, 7)) # width, height
    plt.subplot(321); plt.imshow(image1); plt.title('Image 1'); plt.axis('off')
    plt.subplot(322); plt.imshow(image2); plt.title('Image 2'); plt.axis('off')
    plt.subplot(323)
    plt.imshow(disp1, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 1'); plt.axis('off')
    plt.subplot(324)
    plt.imshow(disp2, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 2'); plt.axis('off')
    plt.subplot(325)
    plt.imshow(diff_disp, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference'); plt.axis('off')
    plt.subplot(326)
    plt.imshow(diff_disp, cmap='magma'); plt.title('Disparity difference (scaled)'); plt.axis('off')
    fig.canvas.draw()
    # plt.savefig('temp_' + filename + '.png')
    pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return pil_image, disp1, disp2

def disp_to_depth(disp,min_depth,max_depth):
# """Convert network's sigmoid output into depth prediction
# The formula for this conversion is given in the 'additional considerations'
# section of the paper.
# """
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp=min_disp+(max_disp-min_disp)*disp
    depth=1/scaled_disp
    return scaled_disp,depth

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_original_meshgrid(disp_size, original_size):
    rows, cols = original_size
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    crop_rows, crop_cols = disp_size
    left = (cols - crop_cols) // 2
    right = left + crop_cols
    top = rows - crop_rows
    bottom = rows
    h_range = slice(top, bottom)
    w_range = slice(left, right)
    c_cropped = c[h_range, w_range]
    r_cropped = r[h_range, w_range]
    assert c_cropped.shape == disp_size
    return c_cropped, r_cropped

def random_drop_points(keep_ratio, lidar):
    num = lidar.shape[0]
    left = int(num * keep_ratio)
    indices = np.random.choice(num, left, replace=False)
    return lidar[indices, :]

def create_pseudo_lidar_dataset(index_range, depth_model, split, output_name, sparse=0):
    """
    split = ["training", "testing"]
    """
    original_size = (1242, 375)
    scene_size = (1024, 320)
    output_dir = "/data/cheng443/kitti/object/{}/{}".format(split, output_name)
    print("output dir: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # output_dir = '/data/cheng443/kitti/object/training/velodyne'
    mode = 'train' if split == 'training' else 'val'
    kitti_loader_train = KittiLoader(mode=mode,  train_list='trainval.txt', val_list='test.txt', size=original_size)
    device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans = transforms.Resize([int(scene_size[1]), int(scene_size[0])])
    for i in range(index_range):
        calib_path = "/data/cheng443/kitti/object/{}/calib/{:0>6}.txt".format(split, i)
        scene_tensor, _ = kitti_loader_train[i]
        scene_tensor = scene_tensor.unsqueeze(0).to(device0, torch.float)
        scene_tensor = trans(scene_tensor)
        with torch.no_grad():
            scene_disp   = depth_model(scene_tensor)
            scene_disp   = interpolate(scene_disp, (original_size[1], original_size[0]), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        lidar = generate_point_cloud(scene_disp, calib_path, None, 2)
        if sparse != 0:
            lidar = random_drop_points(sparse, lidar)
        lidar = lidar.astype(np.float32)
        lidar.tofile('{}/{:0>6}.bin'.format(output_dir, i))
        if i % 20 == 0:
            print('current index: {}/{}'.format(i, index_range))
    return

def train_pointpillar_pipeline():
    pipeline, obj_model, dataset = get_pointpillars_pipeline(pretrain=False)
    pipeline.run_train()
    pipeline.run_valid()


def generate_point_cloud(disp_map, calib_path, output_path, max_height, is_sparse=False):
    calib = Calibration(calib_path)
    disp_map = (disp_map).astype(np.float32)
    # print(disp_map.shape)
    lidar = project_disp_to_points(calib, disp_map, max_height)
    # pad 1 in the indensity dimension
    lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
    # print(lidar.shape)
    lidar = lidar.astype(np.float32)
    if is_sparse:
        lidar = random_drop_points(0.1, lidar)
    if output_path!=None:
        out_fn = output_path+'.npy'
        np.save(out_fn, lidar)
    return lidar


def compose_vis_from_dir(lidar_dir, pipeline, frame_idx=None):
    if not os.path.isdir(lidar_dir):
        print("please give a diretory.")
        return
    paths = glob.glob(os.path.join(lidar_dir, '*.{}'.format('npy')))
    paths = sort(paths)
    all_frame = []
    for lidar_path in paths:
        if frame_idx != None and str(frame_idx) not in lidar_path:
            continue
        filename = os.path.splitext(os.path.basename(lidar_path))[0]
        point_cloud = np.load(lidar_path)
        point_cloud = point_cloud_adjustment(point_cloud, rot_y=5)
        bboxes = filter_car(make_obj_pred(pipeline, point_cloud), confidence_thre=0.3)
        data = {
            'name': filename,
            'points': point_cloud,
            'bounding_boxes': bboxes
        }
        all_frame.append(data)
    return all_frame


def project_disp_to_points(calib : Calibration, disp, max_high):
    disp[disp < 0] = 0
    mask = disp > 0
    # baseline = 0.54
    # depth = calib.f_u * baseline / (disp + 1. - mask)
    SCALE_FACTOR = 5.4
    depth = disp_to_depth(disp, 0.1, 100)[1] * SCALE_FACTOR
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    # original_size = (370, 1226)
    # c, r = get_original_meshgrid(depth.shape, original_size)
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] <= 100) & (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

def fromNumpy2laspy(lidar, out_path): # the output file should end with ".las"
    lidar = lidar*1000
    header = laspy.header.Header()
    outfile = laspy.file.File(out_path, mode="w", header=header)
    outfile.header.offset = [np.floor(np.min(lidar[:, 0])),\
        np.floor(np.min(lidar[:, 1])),\
        np.floor(np.min(lidar[:, 2]))]
    outfile.header.scale = [0.01,0.01,0.01]  
    outfile.X = lidar[:, 0].flatten()
    outfile.Y = lidar[:, 1].flatten()
    outfile.Z = lidar[:, 2].flatten()
    outfile.Intensity = lidar[:, 3].flatten()
    
    outfile.close()

def filter_car(bboxes, confidence_thre=0.3):
        output = []
        filter_catigories = ['Car']
        for bbox_obj in bboxes:
            if bbox_obj.label_class in filter_catigories and bbox_obj.confidence > confidence_thre:
                output.append(bbox_obj)
        return output

class AttackValidator():
    def __init__(self, root_path,save_path,car_name, adv_no, scene_name, depth_model, scene_dataset=False, scene_index=35) -> None:
        self.carname=car_name
        self.adv_car_img_path = os.path.join(root_path, "Adv_car", f"{car_name}_{adv_no}.png")
        self.ben_car_img_path = os.path.join(root_path, "Ben_car", f"{car_name}.png")
        self.car_mask_img_path = os.path.splitext(self.ben_car_img_path)[0] + '_CarMask.png'
        self.scene_img_path = os.path.join(root_path, "Scene",  f"{scene_name}.png")
        self.device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = depth_model
        
        self.pc_dir = os.path.join(save_path, 'PointCloud')
        self.ben_pc_path = os.path.join(save_path, 'PointCloud', f'{car_name}_{adv_no}_ben')
        self.adv_pc_path = os.path.join(save_path, 'PointCloud', f'{car_name}_{adv_no}_adv')
        self.scene_pc_path = os.path.join(save_path, 'PointCloud', f'{car_name}_{adv_no}_sce')
        self.compare_path = os.path.join(save_path, 'PointCloud', f'{car_name}_{adv_no}_compare.png')
        if scene_dataset:
            kitti_loader_train = KittiLoader(mode='train',  train_list='trainval.txt', val_list='val.txt', size=(1242, 375))
            self.train_loader = kitti_loader_train # DataLoader(kitti_loader_train, batch_size=1, shuffle=False, pin_memory=True)
            self.dataset_idx = scene_index
            self.calib_path = "/data/cheng443/kitti/object/training/calib/{:0>6}.txt".format(self.dataset_idx)
        else:
            self.train_loader = None
        self.scene_size = (1024, 320)
        self.pipeline, self.obj_model, self.dataset = get_pointpillars_pipeline()

    def set_scene_index(self, scene_idx):
        self.dataset_idx = scene_idx
        self.calib_path = "/data/cheng443/kitti/object/training/calib/{:0>6}.txt".format(self.dataset_idx)
    
    def read_dataset_data(self, split, index):
        """
        data = {
            'point': reduced_pc,
            'full_point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': label,
        }
        """
        split = self.dataset.get_split(split)
        return split.get_data(index)
    
    def crop_scene_img(self, scene_img):
        original_w, original_h = scene_img.size
        new_w, new_h = self.scene_size
        left = (original_w - new_w)//2
        right = left + new_w
        top = original_h - new_h
        bottom = original_h
        scene_img_crop = scene_img.crop((left, top, right, bottom))
        assert self.scene_size == scene_img_crop.size
        # scene_img_crop.save(os.path.join(gen_scene_path, img_name))
        return scene_img_crop
    
    def load_imgs(self):
        ben_car_img = pil.open(self.ben_car_img_path)
        adv_car_img = pil.open(self.adv_car_img_path)
        self.ben_car_tensor = transforms.ToTensor()(ben_car_img)[:3,:,:].unsqueeze(0).to(self.device0, torch.float)
        self.adv_car_tensor = transforms.ToTensor()(adv_car_img)[:3,:,:].unsqueeze(0).to(self.device0, torch.float)
        if self.train_loader == None:
            scene_car_img = pil.open(self.scene_img_path)
            # scene_car_img = self.crop_scene_img(scene_car_img)
            self.scene_tensor = transforms.ToTensor()(scene_car_img)[:3,:,:].unsqueeze(0).to(self.device0, torch.float) # use original size
        else:
            self.scene_tensor, _ = self.train_loader[self.dataset_idx]
            self.scene_tensor = self.scene_tensor.unsqueeze(0).to(self.device0, torch.float)
        _, _, ori_H, ori_W = self.scene_tensor.size()
        self.original_size = (ori_W, ori_H)
        img_mask = ImageOps.grayscale(pil.open(self.car_mask_img_path))
        img_mask_np = np.array(img_mask) / 255.0
        img_mask_np[img_mask_np > 0.5] = 1
        img_mask_np[img_mask_np <= 0.5] = 0
        img_mask_np = img_mask_np.astype(int)
        self.car_mask_tensor = torch.from_numpy(img_mask_np).unsqueeze(0).float().to(self.device0).requires_grad_(False)
        # print("ben car size: {} \n adv car size: {} \n scene image size: {} \n car mask size: {}".format(
        #     self.ben_car_tensor.size(), self.adv_car_tensor.size(), self.scene_tensor.size(), self.car_mask_tensor.size()))
    
    def run_defence_on_image(self, defence_args, rgb_img):
        if defence_args['type'] == "bitdepth":
            return bitdepth_defense(rgb_img, depth=defence_args['depth'])
        elif defence_args['type'] == "jpeg_compress":
            return jpeg_defense(rgb_img, quality=defence_args['quality'])
        elif defence_args['type'] == "smooth":
            return blurring_defense(rgb_img, ksize=defence_args['size'])
        elif defence_args['type'] == "guassian":
            return gaussian_noise(rgb_img, std=defence_args['std'])
        elif defence_args['type'] == "autoencoder":
            magnet = Magnet_defence(defence_args['model_type'])
            return magnet.magnet_defense(rgb_img)

    
    def get_depth_data(self, is_sparse, defence_args=None):
        self.load_imgs()
        trans = transforms.Resize([int(self.scene_size[1]), int(self.scene_size[0])])
        ## scale with adv car
        # scene_car_mask=self.attach_car_to_scene(1)
        # self.adv_scene_tensor = trans(self.adv_scene_tensor)
        # self.ben_scene_tensor = trans(self.ben_scene_tensor)
        # self.scene_tensor = trans(self.scene_tensor)
        ## scale scene only
        self.scene_tensor = trans(self.scene_tensor)
        scene_car_mask=self.attach_car_to_scene(1)

        if defence_args!=None:
            self.ben_scene_tensor = self.run_defence_on_image(defence_args, self.ben_scene_tensor)
            self.adv_scene_tensor = self.run_defence_on_image(defence_args, self.adv_scene_tensor)

        target_size = (self.original_size[1], self.original_size[0])
        with torch.no_grad():
            adv_scene_disp = self.depth_model(self.adv_scene_tensor)
            ben_scene_disp = self.depth_model(self.ben_scene_tensor)
            scene_disp     = self.depth_model(self.scene_tensor)
            adv_scene_disp = interpolate(adv_scene_disp, target_size, mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            ben_scene_disp = interpolate(ben_scene_disp, target_size, mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            scene_disp     = interpolate(scene_disp,     target_size, mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        
        self.adv_scene_lidar = generate_point_cloud(adv_scene_disp, self.calib_path, self.adv_pc_path, 2, is_sparse=is_sparse)
        self.ben_scene_lidar = generate_point_cloud(ben_scene_disp, self.calib_path, self.ben_pc_path, 2, is_sparse=is_sparse)
        data_frame = self.read_dataset_data('training', self.dataset_idx)
        self.gt_scene_lidar = data_frame['point']
        self.gt_bboxes = data_frame['bounding_boxes']
        self.scene_lidar = generate_point_cloud(scene_disp, self.calib_path, self.scene_pc_path, 2, is_sparse=is_sparse)
        pil_image,disp1,disp2 = eval_depth_diff(self.ben_scene_tensor, self.adv_scene_tensor, self.depth_model, '')
        pil_image.save(self.compare_path)
        # print("image saved to: ", self.compare_path)
        return disp1,disp2,scene_car_mask

    def get_depth_data2(self,i):
        scene_car_mask,adv_scene_img =self.attach_car_to_scene2(1,i)
        with torch.no_grad():
            adv_scene_disp = self.depth_model(self.adv_scene_tensor).squeeze().cpu().numpy()
            ben_scene_disp = self.depth_model(self.ben_scene_tensor).squeeze().cpu().numpy()
            scene_disp = self.depth_model(self.scene_tensor).squeeze().cpu().numpy()
        self.adv_scene_lidar = generate_point_cloud(adv_scene_disp, self.calib_path, self.adv_pc_path, 2)
        self.ben_scene_lidar = generate_point_cloud(ben_scene_disp, self.calib_path, self.ben_pc_path, 2)
        self.scene_lidar = generate_point_cloud(scene_disp, self.calib_path, self.scene_pc_path, 2)
        pil_image,disp1,disp2 = eval_depth_diff(self.ben_scene_tensor, self.adv_scene_tensor, self.depth_model, '')
        pil_image.save(self.compare_path)
        return disp1,disp2,scene_car_mask, adv_scene_img

    def run_obj_det_model(self):
        self.adv_scene_bboxes = filter_car(make_obj_pred(self.pipeline, self.adv_scene_lidar))
        self.ben_scene_bboxes = filter_car(make_obj_pred(self.pipeline, self.ben_scene_lidar))
        self.scene_bboxes     = make_obj_pred(self.pipeline, self.scene_lidar)
        # self.gt_bboxes     = self.gt_bboxes
    
    def compose_vis_data(self):
        ben_data = {
            'name': '{:0>6}_ben'.format(self.dataset_idx),
            'points': self.ben_scene_lidar,
            'bounding_boxes': self.ben_scene_bboxes
        }

        adv_data = {
            'name': '{:0>6}_adv'.format(self.dataset_idx),
            'points': self.adv_scene_lidar,
            'bounding_boxes': self.adv_scene_bboxes
        }

        return ben_data, adv_data

    def vis_frame(self, type):
        lut = LabelLUT()
        for val in sorted(kitti_labels.keys()):
            lut.add_label(kitti_labels[val], val)
        if type == 'scene':
            visualize_frame(0, [self.gt_scene_lidar, self.scene_lidar], self.gt_bboxes, lut=lut)
        elif type == 'benign':
            visualize_frame(1, [self.gt_scene_lidar, self.ben_scene_lidar], self.ben_scene_bboxes, lut=lut)
        elif type == 'adv':
            visualize_frame(2, [self.gt_scene_lidar, self.adv_scene_lidar], self.adv_scene_bboxes, lut=lut)
        elif type == 'all':
            visualize_frame(3, [self.gt_scene_lidar, self.ben_scene_lidar, self.adv_scene_lidar], 
            bboxes_array=[self.gt_bboxes, self.ben_scene_bboxes, self.adv_scene_bboxes], lut=lut)
        elif type == 'dataset':
            visualize_dataset(self.dataset, 'training', [1,2,3,4,5], lut=lut)


    def attach_car_to_scene2(self, batch_size,i):
        """
        Attach the car image and adversarial car image to the given scene with random position. 
        The scene could have multiple images (batch size > 1)
        scene_img: B * C * H * W
        car_img:   1 * C * H * W
        car_mask:      1 * H * W
        """
        scene_img = self.scene_tensor
        adv_car_img = self.adv_car_tensor
        car_img = self.ben_car_tensor
        car_mask = self.car_mask_tensor
        _, _, H, W = adv_car_img.size()
        if scene_img.size()[0] == batch_size:
            adv_scene = scene_img.clone()
            car_scene = scene_img.clone()
        else:
            adv_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
            car_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
        scene_car_mask = torch.zeros(adv_scene.size()).float().to(self.device0)
        
        B_Sce, _, H_Sce, W_Sce = adv_scene.size()
        
        for idx_Bat in range(B_Sce):
            # scale = 0.7 # 600 -- 0.4/0.3, 300 -- 0.7
            scale = 0.5 - i*0.45/500 if self.carname != 'p2' else 0.10 - i*0.09/500

            # scale = (scale_upper - scale_lower) * torch.rand(1) + scale_lower
            # Do some transformation on the adv_car_img together with car_mask
            trans_seq = transforms.Compose([ 
                # transforms.RandomRotation(degrees=3),
                transforms.Resize([int(H * scale), int(W * scale)])
                ])
            # trans_seq_color = transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1)

            # adv_car_img_trans = trans_seq_color(trans_seq(adv_car_img)).squeeze(0)
            # car_img_trans = trans_seq_color(trans_seq(car_img)).squeeze(0)

            adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
            car_img_trans = trans_seq(car_img).squeeze(0)
            
            car_mask_trans = trans_seq(car_mask)

            # paste it on the scene
            _, H_Car, W_Car = adv_car_img_trans.size()
            
            left_range = W_Sce - W_Car
            bottom_range = int((H_Sce - H_Car)/2)

            bottom_height = int(bottom_range - scale * max(bottom_range, 0) - 15 +i/10)  if self.carname != 'p2' else int(bottom_range - scale * max(bottom_range, 0) - 80 +i/4)# random.randint(min(10, bottom_range), bottom_range) # 20 
            left =  390 + int(i/5) if self.carname != 'p2' else 500 + int(i/20) # random.randint(50, left_range-50) 
            h_index = H_Sce - H_Car - bottom_height
            w_index = left
            h_range = slice(h_index, h_index + H_Car)
            w_range = slice(w_index, w_index + W_Car)

            car_area_in_scene = adv_scene[idx_Bat, :, h_range, w_range]
            adv_scene[idx_Bat, :, h_range, w_range] = \
                adv_car_img_trans * car_mask_trans + car_area_in_scene * (1- car_mask_trans)
            car_scene[idx_Bat, :, h_range, w_range] = \
                    car_img_trans * car_mask_trans + car_area_in_scene * (1 - car_mask_trans)
            scene_car_mask[idx_Bat, :, h_range, w_range] = car_mask_trans

            

        self.adv_scene_tensor = adv_scene
        self.ben_scene_tensor = car_scene
        unloader = transforms.ToPILImage() # tensor to PIL image
        adv_scene_img = unloader(self.adv_scene_tensor.cpu().squeeze(0))
        ben_scene_img = unloader(self.ben_scene_tensor.cpu().squeeze(0))
        adv_scene_img.save(self.adv_pc_path + '.png')
        ben_scene_img.save(self.ben_pc_path + '.png')
        return scene_car_mask, adv_scene


    def attach_car_to_scene(self, batch_size):
        """
        Attach the car image and adversarial car image to the given scene with random position. 
        The scene could have multiple images (batch size > 1)
        scene_img: B * C * H * W
        car_img:   1 * C * H * W
        car_mask:      1 * H * W
        """
        scene_img = self.scene_tensor
        adv_car_img = self.adv_car_tensor
        car_img = self.ben_car_tensor
        car_mask = self.car_mask_tensor
        _, _, H, W = adv_car_img.size()
        if scene_img.size()[0] == batch_size:
            adv_scene = scene_img.clone()
            car_scene = scene_img.clone()
        else:
            adv_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
            car_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
        scene_car_mask = torch.zeros(adv_scene.size()).float().to(self.device0)
        
        B_Sce, _, H_Sce, W_Sce = adv_scene.size()
        
        for idx_Bat in range(B_Sce):
            # scale = 0.7 # 600 -- 0.4/0.3, 300 -- 0.7
            scale_upper = 0.5
            scale_lower = 0.4
            # scale = (scale_upper - scale_lower) * torch.rand(1) + scale_lower
            # scale=0.37 if self.carname != 'p2' else 0.13
            scale=0.42 if self.carname != 'p2' else 0.13
            # Do some transformation on the adv_car_img together with car_mask
            trans_seq = transforms.Compose([ 
                # transforms.RandomRotation(degrees=3),
                transforms.Resize([int(H * scale), int(W * scale)])
                ])
            # trans_seq_color = transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1)

            # adv_car_img_trans = trans_seq_color(trans_seq(adv_car_img)).squeeze(0)
            # car_img_trans = trans_seq_color(trans_seq(car_img)).squeeze(0)

            adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
            car_img_trans = trans_seq(car_img).squeeze(0)
            
            car_mask_trans = trans_seq(car_mask)

            # paste it on the scene
            _, H_Car, W_Car = adv_car_img_trans.size()
            
            left_range = W_Sce - W_Car
            bottom_range = int((H_Sce - H_Car)/2)

            # bottom_height = int(bottom_range - scale * max(bottom_range +120 , 0))  # random.randint(min(10, bottom_range), bottom_range) # 20 
            bottom_height = 0
            left =  400 # random.randint(50, left_range-50)
            h_index = H_Sce - H_Car - bottom_height
            w_index = left
            h_range = slice(h_index, h_index + H_Car)
            w_range = slice(w_index, w_index + W_Car)

            car_area_in_scene = adv_scene[idx_Bat, :, h_range, w_range]
            adv_scene[idx_Bat, :, h_range, w_range] = \
                adv_car_img_trans * car_mask_trans + car_area_in_scene * (1- car_mask_trans)
            car_scene[idx_Bat, :, h_range, w_range] = \
                    car_img_trans * car_mask_trans + car_area_in_scene * (1 - car_mask_trans)
            scene_car_mask[idx_Bat, :, h_range, w_range] = car_mask_trans

            

        self.adv_scene_tensor = adv_scene
        self.ben_scene_tensor = car_scene
        unloader = transforms.ToPILImage() # tensor to PIL image
        adv_scene_img = unloader(self.adv_scene_tensor.cpu().squeeze(0))
        ben_scene_img = unloader(self.ben_scene_tensor.cpu().squeeze(0))
        adv_scene_img.save(self.adv_pc_path + '.png')
        ben_scene_img.save(self.ben_pc_path + '.png')
        return scene_car_mask

def cal_mean_depth_error(disp1, disp2, scene_car_mask):
    scaler = 5.4
    dep1 = torch.clamp(disp_to_depth(torch.abs(torch.tensor(disp1)), 0.1, 100)[
                        1]*scene_car_mask.unsqueeze(0).cpu()*scaler, max=100)
    dep2 = torch.clamp(disp_to_depth(torch.abs(torch.tensor(disp2)), 0.1, 100)[
                        1]*scene_car_mask.unsqueeze(0).cpu()*scaler, max=100)
    mean_depth_diff = torch.sum(torch.abs(dep1-dep2)) / \
        (torch.sum(scene_car_mask)+1)
    return mean_depth_diff

def eval_depth_defence(validator):
    ben_disp1,adv_disp2,scene_car_mask = validator.get_depth_data(is_sparse=True)
    ben_error = cal_mean_depth_error(ben_disp1, ben_disp1, scene_car_mask)
    atk_error = cal_mean_depth_error(ben_disp1, adv_disp2, scene_car_mask)
    print(f"Original, benign error: {ben_error}, attack error: {atk_error}")
    # defence_args = {
    #     "type": "bitdepth",
    #     "depth": 1
    # }

    # for q in range(10, 91, 10):
    #     defence_args = {
    #         "type": "jpeg_compress",
    #         "quality":q
    #     }
    #     ben_disp1_def,adv_disp2_def,scene_car_mask = validator.get_depth_data(is_sparse=True, defence_args=defence_args)
    #     ben_error = cal_mean_depth_error(ben_disp1, ben_disp1_def, scene_car_mask)
    #     atk_error = cal_mean_depth_error(ben_disp1, adv_disp2_def, scene_car_mask)
    #     print(f"level: {q}, benign error: {ben_error}, attack error: {atk_error}")
    
    for k in range(1, 36, 2):
        defence_args = {
            "type": "smooth",
            "size":k
        }
        ben_disp1_def,adv_disp2_def,scene_car_mask = validator.get_depth_data(is_sparse=True, defence_args=defence_args)
        ben_error = cal_mean_depth_error(ben_disp1, ben_disp1_def, scene_car_mask)
        atk_error = cal_mean_depth_error(ben_disp1, adv_disp2_def, scene_car_mask)
        print(f"level: {k}, benign error: {ben_error}, attack error: {atk_error}")
    
    # for k in [0.1, 0.05, 0.02, 0.01]:
    #     defence_args = {
    #         "type": "guassian",
    #         "std": k
    #     }
    #     ben_disp1_def,adv_disp2_def,scene_car_mask = validator.get_depth_data(is_sparse=True, defence_args=defence_args)
    #     ben_error = cal_mean_depth_error(ben_disp1, ben_disp1_def, scene_car_mask)
    #     atk_error = cal_mean_depth_error(ben_disp1, adv_disp2_def, scene_car_mask)
    #     print(f"level: {k}, benign error: {ben_error}, attack error: {atk_error}")

    # for model_type in ['param1', 'mnist', 'cifar',  'param2']:
    #     defence_args = {
    #         "type": "autoencoder",
    #         "model_type": model_type
    #     }
    #     ben_disp1_def,adv_disp2_def,scene_car_mask = validator.get_depth_data(is_sparse=True, defence_args=defence_args)
    #     ben_error = cal_mean_depth_error(ben_disp1, ben_disp1_def, scene_car_mask)
    #     atk_error = cal_mean_depth_error(ben_disp1, adv_disp2_def, scene_car_mask)
    #     print(f"level: {model_type}, benign error: {ben_error}, attack error: {atk_error}")

def get_cdf_data(car_name, adv_no):
    validator = AttackValidator(generated_root_path, save_path, car_name, adv_no, scene_name, depth_model, scene_dataset=True, scene_index=72)
    ben_disp1,adv_disp2,scene_car_mask = validator.get_depth_data(is_sparse=True)
    scaler = 5.4
    dep1 = torch.clamp(disp_to_depth(torch.abs(torch.tensor(ben_disp1)), 0.1, 100)[
                        1]*scene_car_mask.unsqueeze(0).cpu()*scaler, max=100)
    dep2 = torch.clamp(disp_to_depth(torch.abs(torch.tensor(adv_disp2)), 0.1, 100)[
                        1]*scene_car_mask.unsqueeze(0).cpu()*scaler, max=100)
    depth_diff = torch.abs(dep1-dep2).numpy().flatten()
    x_data = np.sort(depth_diff[depth_diff != 0])
    N = len(x_data)
    y_data = np.arange(N) / float(N)

    # getting data of the histogram
    count, bins_count = np.histogram(x_data, bins=500)
    
    # finding the PDF of the histogram using count values
    y_pdf = count / sum(count)
    x_pdf = bins_count[1:]

    return x_data, y_data, x_pdf, y_pdf

if __name__ == '__main__':
    generated_root_path = "/home/cheng443/projects/Monodepth/Monodepth2_official/pseudo_lidar/figures/GeneratedAtks/"
    save_path = '/data/cheng443/depth_atk'
    setup_seed(18)
    car_name = 'BMW'
    # adv_no = 'style_lambda1'
    # adv_no = 'cloud_comp'
    # adv_no = 'noise_style'
    # adv_no = 'car_style'
    # adv_no = 'Mono_1_9_Rob'
    adv_no = '102'
    
    model='monodepth2'
    depth_model = import_depth_model((1024, 320), model).to(torch.device("cuda")).eval()
    # create_pseudo_lidar_dataset(7481, depth_model, 'training', 'velodyne_pseudoSparse04', sparse=0.4) # 7481 from training
    # exit(0)
    scene_name='000027'
    
    ## process and visualize folder data
    # lidar_dir = '/data/cheng443/depth_atk/videos/09-30-2021/IMG_3596_Processed/Lidar'
    # validator = AttackValidator(generated_root_path, save_path, car_name, adv_no, scene_name, depth_model, scene_dataset=True, scene_index=72) # 210 is good
    # all_frames = compose_vis_from_dir(lidar_dir, validator.pipeline, frame_idx=None)
    # visulize_data(all_frames)
    # exit(0)

    ## process defence data
    # validator = AttackValidator(generated_root_path, save_path, car_name, adv_no, scene_name, depth_model, scene_dataset=True, scene_index=72) 
    # eval_depth_defence(validator)

    # ## draw error cdf
    compare_adv_no = ['101', '102']
    x_data1, y_data1, x_pdf1, y_pdf1 = get_cdf_data(car_name, compare_adv_no[0])
    x_data2, y_data2, x_pdf2, y_pdf2= get_cdf_data(car_name, compare_adv_no[1])

    plt.rcParams['font.size'] = '20'
    fig, ax1 = plt.subplots(figsize=(6,4))
    plt.xlabel('Depth Estimation Error (m)')
    ax1.plot(x_data1, y_data1)
    ax1.plot(x_data2, y_data2)
    ax1.legend(['Baseline', 'Ours'], loc='center right')
    ax1.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    ax1.set_ylabel("CDF")
    plt.xlim([0, 40])

    ax2 = ax1.twinx()
    plt.rcParams['font.size'] = '20'
    ax2.hist(x_data1, 200, alpha=.5, density=True)
    ax2.hist(x_data2, 200, alpha=.5, density=True)
    ax2.set_ylabel("Frequency")
    plt.ylim([0, 0.3])
    plt.savefig("error_pdf_cdf.png", bbox_inches='tight')





    
    


    ## process single data                                                         
    # validator = AttackValidator(generated_root_path, save_path, car_name, adv_no, scene_name, depth_model, scene_dataset=True, scene_index=72) # 210 is good
    # validator.get_depth_data(is_sparse=True)
    # validator.run_obj_det_model()
    # # validator.vis_frame('adv')
    # # fromNumpy2laspy(validator.ben_scene_lidar, validator.ben_pc_path + '.las')
    # # fromNumpy2laspy(validator.adv_scene_lidar, validator.adv_pc_path + '.las')
    # validator.vis_frame('adv')
    
    ## process multiple data

    # lut = LabelLUT()
    # for val in sorted(kitti_labels.keys()):
    #     lut.add_label(kitti_labels[val], val)
    # from pseudo_lidar.eval_scene_indices import scene_indices
    # scene_indices = [72]
    # ben_frames = []
    # adv_frames = []
    # all_frames = []
    # validator = AttackValidator(generated_root_path, save_path, car_name, adv_no, scene_name, depth_model, scene_dataset=True, scene_index=1)
    # for scene_no in scene_indices:
    #     print("current scene no: ", scene_no)
    #     validator.set_scene_index(scene_no)
    #     validator.get_depth_data(is_sparse=True)
    #     validator.run_obj_det_model()
    #     ben_data, adv_data = validator.compose_vis_data()
    #     ben_frames.append(ben_data)
    #     adv_frames.append(adv_data)
    #     all_frames.append(ben_data)
    #     all_frames.append(adv_data)
    # # fromNumpy2laspy(validator.ben_scene_lidar, validator.ben_pc_path + '.las')
    # # fromNumpy2laspy(validator.adv_scene_lidar, validator.adv_pc_path + '.las')
    # visulize_data(all_frames)