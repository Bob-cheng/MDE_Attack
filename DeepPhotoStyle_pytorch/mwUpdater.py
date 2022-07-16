import torch
import utils
# from model import get_adv_loss
import config


class MaskWeightUpdater():
    def __init__(self, initweight, maskloss_thresh, init_ratio, total_steps, upscaler=1.5,  downscaler=0.7, interval=20) -> None:
        self.upscaler = upscaler
        self.downscaler = downscaler
        self.interval = interval
        self.init_weight = initweight
        self.init_ratio = init_ratio
        self.total_steps = total_steps
        self.current_step = 0
        self.maskloss_thresh  = maskloss_thresh
        self.mask_weight = initweight
    
    def get_target_ratio(self):
        end_step = 0.6 * self.total_steps
        slope = (self.init_ratio - self.maskloss_thresh) / end_step
        if self.current_step > end_step:
            return self.maskloss_thresh
        else:
            return self.init_ratio - slope * self.current_step

    def get_target_ratio2(self):
        m = self.init_ratio
        f = 0.6 * self.total_steps
        g = self.maskloss_thresh
        x = self.current_step
        if self.current_step > f:
            return self.maskloss_thresh
        else:
            return (m-g)/(f*f)*x*x + 2 * (g-m)/f * x + m

    
    def step(self, mask_ratio):
        ref_value = mask_ratio
        self.current_step += 1
        if self.current_step % self.interval == 0:
            # if maskloss < self.maskloss_thresh:
            #     self.mask_weight *= self.downscaler
            # else:
            #     self.mask_weight *= self.upscaler

            # self.mask_weight = self.init_weight

            ## try to fix the mask area to maskloss_thresh
            # bound = 2.0
            # if (ref_value > self.maskloss_thresh * bound):
            #     self.mask_weight = self.init_weight
            # elif (ref_value < self.maskloss_thresh / bound):
            #     self.mask_weight = - self.init_weight
            # elif (ref_value >= self.maskloss_thresh):
            #     self.mask_weight =  (ref_value / self.maskloss_thresh -1)  / (bound - 1) * self.init_weight
            # else:
            #     self.mask_weight = - (self.maskloss_thresh / ref_value - 1)  / (bound - 1) * self.init_weight

            # target_ratio = self.get_target_ratio()
            target_ratio = self.get_target_ratio2() # square
            print("target_ratio:", target_ratio, "current_ratio", ref_value)
            if ref_value >= target_ratio:
                self.mask_weight *= self.upscaler
            else:
                self.mask_weight *= self.downscaler

        return self.mask_weight

    def get_mask_weight(self):
        return self.mask_weight

def get_mask_ratio(paint_mask,mask, mk_init=None):
    # mapped_mask = 0.5 * torch.tanh(20 * paint_mask - 2) + 0.5
    # loss = torch.sum(torch.abs(mapped_mask))
    # mapped_mask = paint_mask
    # loss = torch.mean(torch.abs(mapped_mask))
    if mk_init == None:
        ratio = torch.sum(paint_mask)/torch.sum(mask)
    elif len(mk_init.size()) == 1:
        ratio = (mk_init[1] - mk_init[0]) * (mk_init[3] - mk_init[2]) / torch.sum(mask)
    return ratio

def mask_loss_fucntion(size,threshold,mask_weight):
    # loss=torch.exp(torch.abs(size-threshold))*mask_weight
    # loss=torch.tanh(abs(size-threshold))*mask_weight # not good
    # loss=torch.sqrt(abs(size-threshold))*mask_weight # good
    # loss=torch.abs(size-threshold/2)*mask_weight # not good
    loss=torch.abs(size-threshold)*mask_weight # okay
    return loss

MASK_INIT_EDGES_SUM = None
def mask_loss_fucntion2(mk_init, car_mask, mask_weight):
    # this loss function can avoid influence of rectangular shape making each edge have same weight
    # which is required in single edge optimization each time
    global MASK_INIT_EDGES_SUM
    C, H, W = car_mask.size()
    loss = 0
    if len(mk_init.size()) == 1:
        loss= (torch.abs(mk_init[1] - mk_init[0]) + torch.abs(mk_init[3] - mk_init[2])) / (H + W) * mask_weight
    elif len(mk_init.size()) == 2: # require the mask to be horizontal
        num_masks = mk_init.size()[0]
        for i in range(num_masks):
            loss += (torch.abs(mk_init[i][1] - mk_init[i][0]) + torch.abs(mk_init[i][3] - mk_init[i][2]))
        if MASK_INIT_EDGES_SUM == None:
            MASK_INIT_EDGES_SUM = loss.item()
        loss = loss / MASK_INIT_EDGES_SUM * mask_weight
    return loss


def edge_based_update(paint_mask_init, mask_optimizer, run):
    if len(paint_mask_init.size()) == 1:
        mask_grads = paint_mask_init.grad.clone().detach()
        # mask_grads[0] = - mask_grads[0]
        # mask_grads[2] = - mask_grads[2]
        mask_grads = torch.abs(mask_grads)
        optim_idxs = []
        optim_idxs.append(torch.max(mask_grads, dim=0)[1].item()) ## for one edge training
        # optim_idxs.extend([0,1,2,3]) ## for 4 edge training
        # if run[0] % 20 == 0:
        #     print("paint mask grad: ", paint_mask_init.grad, " max index", optim_idxs)
        paint_mask_init_old = paint_mask_init.clone()
        mask_optimizer.step()
        with torch.no_grad():
            for pm_idx in range(4):
                if pm_idx not in optim_idxs:
                    paint_mask_init[pm_idx] = paint_mask_init_old[pm_idx]
    else:
        num_masks = paint_mask_init.size()[0]
        optim_idxs = []
        mask_grads = paint_mask_init.grad.clone().detach()
        # mask_grads[:, 0] = - mask_grads[:, 0]
        # mask_grads[:, 2] = - mask_grads[:, 2]
        mask_grads = torch.abs(mask_grads)
        optim_idxs = torch.max(mask_grads, dim=1)[1].unsqueeze(1)
        # if run[0] % 20 == 0:
        #     print("paint mask grad: ", paint_mask_init.grad, " max index", optim_idxs)
        paint_mask_init_old = paint_mask_init.clone()
        mask_optimizer.step()
        with torch.no_grad():
            for mk_idx in range(num_masks):
                for pm_idx in range(4):
                    if pm_idx not in optim_idxs[mk_idx]:
                        paint_mask_init[mk_idx][pm_idx] = paint_mask_init_old[mk_idx][pm_idx]