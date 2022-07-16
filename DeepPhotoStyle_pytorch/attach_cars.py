import torch
from torchvision.transforms import transforms
import random
import config

def attach_car_to_scene_fixed(scene_img, adv_car_img, car_img, car_mask, object_name):
    """
    Attach the car image and adversarial car image to the given scene with fixed position. 
    The scene could have multiple images (batch size > 1)
    scene_img: B * C * H * W
    car_img:   1 * C * H * W
    car_mask:      1 * H * W
    """
    _, _, H, W = adv_car_img.size()
    
    adv_scene = scene_img.clone()
    car_scene = scene_img.clone()
    scene_car_mask = torch.zeros(adv_scene.size()).float().to(config.device0)
    
    B_Sce, _, H_Sce, W_Sce = adv_scene.size()
    
    for idx_Bat in range(B_Sce):
        if object_name == "BMW.png" or object_name == "TrafficBarrier2.png":
            scale = 0.4 # 600 -- 0.4, 300 -- 0.7
        elif object_name == 'Pedestrain2.png':
            scale = 0.14
        else:
            raise NotImplementedError("object_name unseen")
        # Do some transformation on the adv_car_img together with car_mask
        trans_seq = transforms.Compose([ 
            transforms.Resize([int(H * scale), int(W * scale)])
            ])

        adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
        car_img_trans = trans_seq(car_img).squeeze(0)

        # adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
        # car_img_trans = trans_seq(car_img).squeeze(0)
        
        car_mask_trans = trans_seq(car_mask)

        # paste it on the scene
        _, H_Car, W_Car = adv_car_img_trans.size()
        
        left_range = W_Sce - W_Car
        bottom_range = int((H_Sce - H_Car)/2)
        bottom_height = int(bottom_range - scale * max(bottom_range - 10, 0))  # random.randint(min(10, bottom_range), bottom_range) # 20 
        left = left_range // 2
        # left = random.randint(50, left_range-50)

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
        # utils.save_pic(adv_scene[idx_Bat,:,:,:], f'attached_adv_scene_{idx_Bat}')
        # utils.save_pic(car_scene[idx_Bat,:,:,:], f'attached_car_scene_{idx_Bat}')
        # utils.save_pic(scene_car_mask[idx_Bat,:,:,:], f'attached_scene_mask_{idx_Bat}')
    return adv_scene, car_scene, scene_car_mask

def attach_car_to_scene(scene_img, adv_car_img, car_img, car_mask, batch_size,paint_mask,object_name):
    """
    Attach the car image and adversarial car image to the given scene with random position. 
    The scene could have multiple images (batch size > 1)
    scene_img: B * C * H * W
    car_img:   1 * C * H * W
    car_mask:      1 * H * W
    """
    _, _, H, W = adv_car_img.size()
    if scene_img.size()[0] == batch_size:
        adv_scene = scene_img.clone()
        car_scene = scene_img.clone()
    else:
        adv_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
        car_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
    scene_car_mask = torch.zeros(adv_scene.size()).float().to(config.device0)
    scene_paint_mask = torch.zeros(adv_scene.size()).float().to(config.device0)
    
    B_Sce, _, H_Sce, W_Sce = adv_scene.size()
    
    for idx_Bat in range(B_Sce):
        # scale = 0.7 # 600 -- 0.4, 300 -- 0.7
        # car 0.5/0.4 people 0.07/0.035 tb 0.3/0.2 tb2 0.5/0.4 pedestrain2 0.15/0.13
        scale_upper = 0.5
        scale_lower = 0.4

        if object_name == 'Pedestrain2.png':
            scale_upper = 0.15
            scale_lower = 0.13
            # scale_upper = 0.10
            # scale_lower = 0.07

        scale = (scale_upper - scale_lower) * torch.rand(1) + scale_lower
        # Do some transformation on the adv_car_img together with car_mask
        trans_seq = transforms.Compose([ 
            transforms.RandomRotation(degrees=3),
            transforms.Resize([int(H * scale), int(W * scale)])
            ])
        trans_seq_color = transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1)

        adv_car_img_trans = trans_seq_color(trans_seq(adv_car_img)).squeeze(0)
        car_img_trans = trans_seq_color(trans_seq(car_img)).squeeze(0)

        # adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
        # car_img_trans = trans_seq(car_img).squeeze(0)
        
        car_mask_trans = trans_seq(car_mask)
        paint_mask_trans = trans_seq(paint_mask)

        # paste it on the scene
        _, H_Car, W_Car = adv_car_img_trans.size()
        
        left_range = W_Sce - W_Car
        bottom_range = int((H_Sce - H_Car)/2)
        ## tb2/car +10 p2 +100
        scale2=10
        if object_name == 'Pedestrain2.png':
            scale2=100
        bottom_height = int(bottom_range - scale * max(bottom_range + scale2, 0))  # random.randint(min(10, bottom_range), bottom_range) # 20 
        left = random.randint(50, left_range-50)
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
        scene_paint_mask[idx_Bat, :, h_range, w_range] = paint_mask_trans
        # utils.save_pic(adv_scene[idx_Bat,:,:,:], f'attached_adv_scene_{idx_Bat}')
        # utils.save_pic(car_scene[idx_Bat,:,:,:], f'attached_car_scene_{idx_Bat}')
        # utils.save_pic(scene_car_mask[idx_Bat,:,:,:], f'attached_scene_mask_{idx_Bat}')
    return adv_scene, car_scene, scene_car_mask, scene_paint_mask

def attach_car_to_scene_Robustness_training(scene_img, adv_car_img, car_img, car_mask, batch_size,paint_mask,object_name):
    """
    Attach the car image and adversarial car image to the given scene with random position. 
    The scene could have multiple images (batch size > 1)
    scene_img: B * C * H * W
    car_img:   1 * C * H * W
    car_mask:      1 * H * W
    """
    _, _, H, W = adv_car_img.size()
    i=random.randint(0,300)
    if scene_img.size()[0] == batch_size:
        adv_scene = scene_img.clone()
        car_scene = scene_img.clone()
    else:
        adv_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
        car_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
    scene_car_mask = torch.zeros(adv_scene.size()).float().to(config.device0)
    scene_paint_mask = torch.zeros(adv_scene.size()).float().to(config.device0)
    B_Sce, _, H_Sce, W_Sce = adv_scene.size()
        
    for idx_Bat in range(B_Sce):
        # scale = 0.7 # 600 -- 0.4/0.3, 300 -- 0.7
        scale = 0.5 - i*0.45/500 if object_name != 'p2' else 0.10 - i*0.09/500

        trans_seq = transforms.Compose([ 
            transforms.RandomRotation(degrees=3),
            transforms.Resize([int(H * scale), int(W * scale)])
            ])
        trans_seq_color = transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1)

        adv_car_img_trans = trans_seq_color(trans_seq(adv_car_img)).squeeze(0)
        car_img_trans = trans_seq_color(trans_seq(car_img)).squeeze(0)

        # adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
        # car_img_trans = trans_seq(car_img).squeeze(0)
        
        car_mask_trans = trans_seq(car_mask)
        paint_mask_trans = trans_seq(paint_mask)

        # paste it on the scene
        _, H_Car, W_Car = adv_car_img_trans.size()
        
        left_range = W_Sce - W_Car
        bottom_range = int((H_Sce - H_Car)/2)

        bottom_height = int(bottom_range - scale * max(bottom_range, 0) - 15 +i/10)  # random.randint(min(10, bottom_range), bottom_range) # 20 
        left =  390 + int(i/5) # random.randint(50, left_range-50)
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
        scene_paint_mask[idx_Bat, :, h_range, w_range] = paint_mask_trans

    return adv_scene, car_scene, scene_car_mask, scene_paint_mask
    
def attach_car_to_scene_validator(scene_img, adv_car_img, car_img, car_mask, batch_size,paint_mask,mask):
    """
    Attach the car image and adversarial car image to the given scene with random position. 
    The scene could have multiple images (batch size > 1)
    scene_img: B * C * H * W
    car_img:   1 * C * H * W
    car_mask:      1 * H * W
    """
    _, _, H, W = adv_car_img.size()
    if scene_img.size()[0] == batch_size:
        adv_scene = scene_img.clone()
        car_scene = scene_img.clone()
    else:
        adv_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
        car_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
    scene_car_mask = torch.zeros(adv_scene.size()).float().to(config.device0)
    scene_paint_mask = torch.zeros(adv_scene.size()).float().to(config.device0)
    
    B_Sce, _, H_Sce, W_Sce = adv_scene.size()
    
    for idx_Bat in range(B_Sce):
        # car/tb2 0.4496 p2 0.15
        scale = 0.4496
        if mask == 'Pedestrain2.png':
            scale=0.15
            # scale=0.10
        # Do some transformation on the adv_car_img together with car_mask
        trans_seq = transforms.Compose([ 
            # transforms.RandomRotation(degrees=3),
            transforms.Resize([int(H * scale), int(W * scale)])
            ])

        adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
        car_img_trans = trans_seq(car_img).squeeze(0)

        # adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
        # car_img_trans = trans_seq(car_img).squeeze(0)
        
        car_mask_trans = trans_seq(car_mask)
        paint_mask_trans = trans_seq(paint_mask)

        # paste it on the scene
        _, H_Car, W_Car = adv_car_img_trans.size()
        
        left_range = W_Sce - W_Car
        bottom_range = int((H_Sce - H_Car)/2)
        ## car +50 p2 +100
        scale2=50
        if mask == 'Pedestrain2.png':
            scale2=160
        bottom_height = int(bottom_range - scale * max(bottom_range + scale2, 0))  # random.randint(min(10, bottom_range), bottom_range) # 20 
        left = 380
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
        scene_paint_mask[idx_Bat, :, h_range, w_range] = paint_mask_trans
        # utils.save_pic(adv_scene[idx_Bat,:,:,:], f'attached_adv_scene_{idx_Bat}')
        # utils.save_pic(car_scene[idx_Bat,:,:,:], f'attached_car_scene_{idx_Bat}')
        # utils.save_pic(scene_car_mask[idx_Bat,:,:,:], f'attached_scene_mask_{idx_Bat}')
    return adv_scene, car_scene, scene_car_mask, scene_paint_mask