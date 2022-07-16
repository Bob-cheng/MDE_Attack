#%%
# from torch import cuda
# from depth_networks.manydepth import manydepth
# sys.path.append('depth_networks/manydepth/')
# import manydepth
import os
import sys
import torch
import torch.nn
import json
import argparse
import numpy as np

file_dir = os.path.dirname(os.path.realpath(__file__))
file_parent_dir = os.path.dirname(file_dir)
md2_model_dir = os.path.join(file_parent_dir, 'models')
DH_model_dir = os.path.join(file_dir, 'depth_networks', 'depth-hints', 'models')
manyd_model_dir = os.path.join(file_dir, 'depth_networks', 'manydepth', 'manydepth', 'models')
# print(depth_model_dir)

class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        return disp

class ManyDepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder, encoder_dict) -> None:
        super(ManyDepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.zero_pose = torch.zeros([1,1,4,4])
        self.encoder_dict = encoder_dict

        # sys.path.append('depth_networks/manydepth/')
        # from manydepth import networks
        # from layers import transformation_from_parameters

        # model_name = 'KITTI_HR'
        # depth_model_dir = manyd_model_dir
        # self.model_path = os.path.join(depth_model_dir, model_name)
        # encoder_path = os.path.join(self.model_path, "encoder.pth")
        intrinsics_json_path=os.path.join(file_dir, 'depth_networks', 'manydepth', 'assets','test_sequence_intrinsics.json')
        # self.encoder_dict = torch.load(encoder_path, map_location='cpu')

        self.K, self.invK = load_and_preprocess_intrinsics(intrinsics_json_path,
                                             resize_width=self.encoder_dict['width'],
                                             resize_height=self.encoder_dict['height'])
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()

    
    def forward(self, input_image):

        # if torch.cuda.is_available():
        #     self.encoder.cuda()
        #     self.decoder.cuda()

        features, lowest_cost, _ = self.encoder(current_image=input_image,
                                         lookup_images=input_image.unsqueeze(1)*0,
                                         poses=self.zero_pose,
                                         K=self.K,
                                         invK=self.invK,
                                         min_depth_bin=self.encoder_dict['min_depth_bin'],
                                         max_depth_bin=self.encoder_dict['max_depth_bin'])

        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        return disp/8.6437

def load_and_preprocess_image(image_path, resize_width, resize_height):
    image = pil.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = image.resize((resize_width, resize_height), pil.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda(), (original_height, original_width)
    return image, (original_height, original_width)

def load_and_preprocess_intrinsics(intrinsics_path, resize_width, resize_height):
    K = np.eye(4)
    with open(intrinsics_path, 'r') as f:
        K[:3, :3] = np.array(json.load(f))

    # Convert normalised intrinsics to 1/4 size unnormalised intrinsics.
    # (The cost volume construction expects the intrinsics corresponding to 1/4 size images)
    K[0, :] *= resize_width // 4
    K[1, :] *= resize_height // 4

    invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
    K = torch.Tensor(K).unsqueeze(0)

    if torch.cuda.is_available():
        return K.cuda(), invK.cuda()
    return K, invK

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth



def import_depth_model(scene_size, model_type='monodepth2'):
    """
    import different depth model to attack:
    possible choices: monodepth2, depthhints
    """
    if scene_size == (1024, 320):
        if model_type == 'monodepth2':
            model_name = 'mono+stereo_1024x320'
            depth_model_dir = md2_model_dir
            sys.path.append(file_parent_dir)
            # from .. import networks # for lint perpose
            import networks
        elif model_type == 'depthhints':
            model_name = 'DH_MS_320_1024'
            depth_model_dir = DH_model_dir
            sys.path.append(file_parent_dir)
            # from .. import networks # for lint perpose
            import networks
        elif model_type == 'manydepth':
            # sys.path.append(file_dir)
            sys.path.append(os.path.join(file_dir, 'depth_networks/manydepth/manydepth/'))
            import networks
            model_name = 'KITTI_HR'
            depth_model_dir = manyd_model_dir
        else:
            raise RuntimeError("depth model unfound")
    else:
        raise RuntimeError("scene size undefined!")
    model_path = os.path.join(depth_model_dir, model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    if model_type == 'manydepth':
        ## Manydepth encoder and poses network
        encoder = networks.ResnetEncoderMatching(18, False,
                                            input_width=loaded_dict_enc['width'],
                                            input_height=loaded_dict_enc['height'],
                                            adaptive_bins=True,
                                            min_depth_bin=loaded_dict_enc['min_depth_bin'],
                                            max_depth_bin=loaded_dict_enc['max_depth_bin'],
                                            depth_binning='linear',
                                            num_depth_bins=96)


    else:
        encoder = networks.ResnetEncoder(18, False)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    if model_type == 'manydepth':
        depth_model = ManyDepthModelWrapper(encoder, depth_decoder, loaded_dict_enc)
    else:
        depth_model = DepthModelWrapper(encoder, depth_decoder)
    return depth_model


#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from PIL import Image as pil
    from torchvision import transforms
    model='manydepth'
    depth_model = import_depth_model((1024, 320), model).to(torch.device("cuda")).eval()
    img = pil.open('/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/gen_img/scene/0000000017.png').convert('RGB')
    assert img.size == (1024, 320)
    img = transforms.ToTensor()(img).unsqueeze(0).to(torch.device("cuda"))
    with torch.no_grad():
        disp = depth_model(img)
        print(disp.size())
        disp_np = disp.squeeze().cpu().numpy()
    
    vmax = np.percentile(disp_np, 95)
    plt.figure(figsize=(5,5))
    plt.imshow(disp_np, cmap='magma', vmax=vmax)
    plt.title('Disparity')
    plt.axis('off')
    plt.savefig('temp_test.png')
    
# %%
