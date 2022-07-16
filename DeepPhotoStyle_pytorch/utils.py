from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from torchvision.transforms.functional import InterpolationMode
from matting import *
import config
import scipy.ndimage as spi

kitti_object_path = '/data/cheng443/kitti/object/'
project_root = '/home/cheng443/projects/Monodepth/Depth_attack/'
log_dir = '/data/cheng443/depth_atk'



def load_image(path, size):
    image = Image.open(path)
    if size is None:
        pass
    else:
        image = image.resize((size, size), Image.BICUBIC)

    return image

def texture_to_car_size(texture, car_size):
    _, _, i_h, i_w = texture.size()
    _, _, c_h, c_w = car_size
    assert i_w == c_w
    if c_h != i_h:
        input_img_resize = transforms.Resize([c_h, c_w])(texture)
        # if i_h > c_h:
        #     input_img_resize = texture[:, :, i_h-c_h: i_h, :]
        # else:
        #     input_img_resize =
    else:
        input_img_resize = texture
    return input_img_resize


def image_to_tensor(img):
    transform_ = transforms.Compose([transforms.ToTensor()])
    return transform_(img)


def show_pic(tensor, title=None):
    plt.figure()
    unloader = transforms.ToPILImage() # tensor to PIL image
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    plt.title(title)

def save_pic(tensor, i, log_dir=''):
    unloader = transforms.ToPILImage() # tensor to PIL image
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if log_dir != '':
        file_path = os.path.join(log_dir, "{}.png".format(i))
    else:
        file_path = "{}.png".format(i)
    image.save(file_path, "PNG")


def from_mask_to_inf(mask: torch.Tensor):
    epsilon = 1e-7
    mask = mask + torch.distributions.Uniform(low=-epsilon, high=epsilon).sample(mask.shape).to(config.device0)
    mask = torch.clip(mask, 0.0, 1.0)
    mask = torch.arctanh((mask - 0.5) * (2 - epsilon))
    return mask

def from_inf_to_mask(values: torch.Tensor, mask_size):
    epsilon = 1e-7
    # values = transforms.Resize(mask_size[1:3])(values)
    mask = (torch.tanh(values) / (2 - epsilon) + 0.5)
    mask = transforms.Resize(mask_size[1:3], Image.NEAREST)(mask)
    return mask

def extract_patch(adv_car, paint_mask):
    _, _, H, W = adv_car.size()
    paint_mask_2D = paint_mask.squeeze()
    last_row = False
    last_col = False
    h_range = []
    w_range = []
    for i in range(H):
        has_one = False
        for j in range(W):
            if abs(paint_mask_2D[i, j] - 1)  < 0.5:
                has_one = True
                if not last_row:
                    h_range.append(i)
                    last_row = True
                break
        if not has_one and last_row:
            h_range.append(i)
            last_row = False
    if len(h_range) == 1:
        h_range.append(H)
    
    for j in range(W):
        has_one = False
        for i in range(H):
            if abs(paint_mask_2D[i, j] - 1)  < 0.5:
                has_one = True
                if not last_col:
                    w_range.append(j)
                    last_col = True
                break
        if not has_one and last_col:
            w_range.append(j)
            last_col = False
    if len(w_range) == 1:
        w_range.append(W)
    
    return adv_car[:, :, h_range[0] : h_range[1], w_range[0] : w_range[1]]

import torch

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor

def bilinear_interpolate_torch(im, x, y):
    print(im.size())
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    
    Ia = im[ y0, x0 ][0]
    Ib = im[ y1, x0 ][0]
    Ic = im[ y0, x1 ][0]
    Id = im[ y1, x1 ][0]
    
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))
    
    
    return torch.t(torch.t(Ia)*wa) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)

def nearest_interpolate(array, height, width):
    channel, ori_h, ori_w = array.shape
    ratio_h = ori_h / height
    ratio_w = ori_w / width
    # target_array = torch.zeros((channel, height, width))
    target_array = torch.cuda.FloatTensor(channel, height, width).fill_(0)
    for i in range(height):
        for j in range(width):
            th = int(i * ratio_h)
            tw = int(j * ratio_w)
            target_array[:, i, j] = array[:, th, tw]

    return target_array    


def compute_lap(path_img):
    '''
    input: image path
    output: laplacian matrix of the input image, format is sparse matrix of pytorch in gpu
    '''
    image = cv2.imread(path_img, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = 1.0 * image / 255.0
    h, w, _ = image.shape
    const_size = np.zeros(shape=(h, w))
    M = compute_laplacian(image)    
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long().cuda()
    values = torch.from_numpy(M.data).cuda()
    shape = torch.Size(M.shape)
    Ms = torch.sparse_coo_tensor(indices, values, shape, device=torch.device('cuda'))
    return Ms

def make_square_mask(mask_size, boarders, soft=True):
    """
    boarders: 0: left, 1: right, 2: top, 3: bottom
    mask_size: 0: channel, 1: height, 2: width
    """
    if soft:
        x = torch.arange(0, mask_size[1])
        y = torch.arange(0, mask_size[2])
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x.requires_grad = False
        grid_y.requires_grad = False
        grid_x = grid_x.to(config.device0)
        grid_y = grid_y.to(config.device0)
        l, r, t, b = boarders[0], boarders[1], boarders[2], boarders[3]
        mask = 0.25 * (-torch.tanh(grid_x-t) * torch.tanh(grid_x-b) + 1) * (-torch.tanh(grid_y-l) * torch.tanh(grid_y-r) + 1)
        mask = mask.clamp(0, 1).unsqueeze(0)
    else:
        mask = torch.zeros((mask_size[1], mask_size[2]))
        if boarders[2] < 0 or boarders[3] > mask_size[1] or boarders[0] < 0 or boarders[1] > mask_size[2]:
            mask = None
        else:
            l, r, t, b = int(boarders[0].item()), int(boarders[1].item()), int(boarders[2].item()), int(boarders[3].item())
            mask[t:b, l:r] = 1
            mask = mask.unsqueeze(0).to(config.device0)
    return mask

def get_mask_source(mask_type, full_size, paint_mask_np: np.ndarray, args):
    obj_type = args["vehicle"]
    if mask_type == '-2':
        paint_mask_init = torch.tensor([0, full_size[2], 0, full_size[1]-40]).float().to(config.device0).requires_grad_(True)
    elif mask_type == '-3':
        # 2 * 1
        # paint_mask_init = torch.tensor([[0, full_size[2], 0, full_size[1]//2], [0, full_size[2], full_size[1]//2, full_size[1]]]).float().to(config.device0).requires_grad_(True)
        grid_layout = [5, 5]
        W, H = full_size[2], full_size[1]
        W_stride, H_stride = W // grid_layout[1], H // grid_layout[0]
        mask_list = []
        for i in range(grid_layout[0]):
            for j in range(grid_layout[1]):
                l = j * W_stride
                r = l + W_stride
                t = i * H_stride
                b = t + H_stride
                mask_list.append([l, r, t, b])
        # multiple overall patch
        # mask_list = [[0, W-10, 0, H-50], [10, W, 10, H-40]]
        paint_mask_init = torch.tensor(mask_list).float().to(config.device0).requires_grad_(True)
        args['mask_weight'] = args['mask_weight'] / (grid_layout[0] * grid_layout[1])
        
    elif mask_type == "-4":
        W, H = full_size[2], full_size[1]
        if "BMW" in obj_type:
            paint_mask_init = torch.tensor([W // 3, W // 3 * 2, H // 3, H // 3 * 2]).float().to(config.device0).requires_grad_(True)
        elif "Pedestrain" in obj_type:
            paint_mask_init = torch.tensor([0, W, H // 3, H // 3 * 2]).float().to(config.device0).requires_grad_(True)
        elif "TrafficBarrier" in obj_type:
            paint_mask_init = torch.tensor([W // 3, W // 3 * 2, 0, H // 2]).float().to(config.device0).requires_grad_(True)
        else:
            raise NotImplementedError("obj type not implemented")
    elif mask_type == '-1':
        paint_mask_np_inf = np.arctanh((paint_mask_np - 0.5) * (2 - 1e-7))
        paint_mask_init = torch.from_numpy(paint_mask_np_inf).unsqueeze(0).float().to(config.device0).requires_grad_(True)
    else:
        paint_mask_init = torch.from_numpy(paint_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    return paint_mask_init

def   get_mask_target(mask_type, full_size, mask_source: torch.Tensor):
    if mask_type == '-2':
        paint_mask = make_square_mask(full_size, mask_source)
    elif mask_type == '-3':
        paint_mask_list = []
        for mk_part in mask_source:
            paint_mask_list.append(make_square_mask(full_size, mk_part))
        paint_mask = torch.stack(paint_mask_list, dim=0).sum(dim=0)
        paint_mask.clamp_(0, 1)
    elif mask_type == '-4':
        paint_mask = make_square_mask(full_size, mask_source, soft=False)
    elif mask_type == '-1':
        paint_mask = from_inf_to_mask(mask_source, full_size)
    else:
        paint_mask = mask_source
    return paint_mask


