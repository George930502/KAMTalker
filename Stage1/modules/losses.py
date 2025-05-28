import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from utils import *

class _PerceptualNetwork(nn.Module):
    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        self.network = network.cuda()
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                output[layer_name] = x
        return output


def _vgg19(layers):
    network = torchvision.models.vgg19()
    state_dict = torch.utils.model_zoo.load_url(
        "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth", map_location=torch.device("cuda:0"), progress=True)
    network.load_state_dict(state_dict)
    network = network.features
    layer_name_mapping = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        17: "relu_3_4",
        20: "relu_4_1",
        22: "relu_4_2",
        24: "relu_4_3",
        26: "relu_4_4",
        29: "relu_5_1",
    }
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg_face(layers):
    network = torchvision.models.vgg16(num_classes=2622)
    state_dict = torch.utils.model_zoo.load_url(
        "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/" "vgg_face_dag.pth", map_location=torch.device("cuda:0"), progress=True)
    feature_layer_name_mapping = {
        0: "conv1_1",
        2: "conv1_2",
        5: "conv2_1",
        7: "conv2_2",
        10: "conv3_1",
        12: "conv3_2",
        14: "conv3_3",
        17: "conv4_1",
        19: "conv4_2",
        21: "conv4_3",
        24: "conv5_1",
        26: "conv5_2",
        28: "conv5_3",
    }
    new_state_dict = {}
    for k, v in feature_layer_name_mapping.items():
        new_state_dict["features." + str(k) + ".weight"] = state_dict[v + ".weight"]
        new_state_dict["features." + str(k) + ".bias"] = state_dict[v + ".bias"]
    classifier_layer_name_mapping = {0: "fc6", 3: "fc7", 6: "fc8"}
    for k, v in classifier_layer_name_mapping.items():
        new_state_dict["classifier." + str(k) + ".weight"] = state_dict[v + ".weight"]
        new_state_dict["classifier." + str(k) + ".bias"] = state_dict[v + ".bias"]
    network.load_state_dict(new_state_dict)
    layer_name_mapping = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        18: "relu_4_1",
        20: "relu_4_2",
        22: "relu_4_3",
        25: "relu_5_1",
    }
    return _PerceptualNetwork(network.features, layer_name_mapping, layers)


class PerceptualLoss(nn.Module):
    def __init__(self, layers_weight={"relu_1_1": 0.03125, "relu_2_1": 0.0625, "relu_3_1": 0.125, "relu_4_1": 0.25, "relu_5_1": 1.0}, n_scale=3):
        super().__init__()
        self.vgg19 = _vgg19(layers_weight.keys())
        self.vggface = _vgg_face(layers_weight.keys())
        self.criterion = nn.L1Loss()
        self.layers_weight, self.n_scale = layers_weight, n_scale

    def forward(self, input, target):
        self.vgg19.eval()
        self.vggface.eval()

        loss = 0.0

        features_vggface_input = self.vggface(apply_vggface_normalization(input))
        features_vggface_target = self.vggface(apply_vggface_normalization(target))

        features_vgg19_input = self.vgg19(apply_imagenet_normalization(input))
        features_vgg19_target = self.vgg19(apply_imagenet_normalization(target))

        for layer, weight in self.layers_weight.items():
            loss += weight * self.criterion(features_vggface_input[layer], features_vggface_target[layer].detach()) / 255
            loss += weight * self.criterion(features_vgg19_input[layer], features_vgg19_target[layer].detach())

        for i in range(self.n_scale):
            input = F.interpolate(input, mode="bilinear", scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            target = F.interpolate(target, mode="bilinear", scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            features_vgg19_input = self.vgg19(input)
            features_vgg19_target = self.vgg19(target)
            for layer, weight in self.layers_weight.items():
                loss += weight * self.criterion(features_vgg19_input[layer], features_vgg19_target[layer].detach())

        return loss


def fuse_math_min_mean_pos(x):
    """Fuse operation min mean for hinge loss computation of positive samples"""
    minval = torch.min(x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss

def fuse_math_min_mean_neg(x):
    """
    Fuse operation min mean for hinge loss computation of negative amples"""
    minval = torch.min(-x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


class GANLoss(nn.Module):
    # Update generator: gan_loss(fake_output, True, False) + other losses
    # Update discriminator: gan_loss(fake_output(detached), False, True) + gan_loss(real_output, True, True)
    def __init__(self):
        super().__init__()

    def forward(self, dis_output, t_real, dis_update=True):
        """GAN loss computation.
        Args:
            dis_output (tensor or list of tensors): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): If ``True``, the loss will be used to update the
                discriminator, otherwise the generator.
        Returns:
            loss (tensor): Loss value.
        """

        if dis_update:
            if t_real:
                loss = fuse_math_min_mean_pos(dis_output)
            else:
                loss = fuse_math_min_mean_neg(dis_output)
        else:
            loss = -torch.mean(dis_output)
        return loss

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, fake_features, real_features):
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)  # dtype and device are the same
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j], real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss

class EquivarianceLoss(nn.Module):
    '''
    Affine transformations and randomly sampled thin plate splines are used to perform the transformation. 
    Since all these are 2D transformations, we project our 3D keypoints to 2D by simply dropping the z values before computing the losses
    '''
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, kp_d, reverse_kp):
        loss = self.criterion(kp_d[:, :, :2], reverse_kp)
        return loss

class KeypointPriorLoss(nn.Module):
    def __init__(self, Dt=0.1, zt=0.33):
        super().__init__()
        self.Dt = Dt
        self.zt = zt

    def forward(self, kp_d):
        # kp_d: [B, K, 3] 
        dist_matrix = torch.cdist(kp_d, kp_d).square()  # shape [B, K, K]
        dist_penalty = torch.clamp(self.Dt - dist_matrix, min=0)
        loss_dist = dist_penalty.sum((1, 2)).mean()
        z_mean = kp_d[:, :, 2].mean(1)  # shape: [B]
        loss_depth = torch.abs(z_mean - self.zt).mean()
        loss = loss_dist + loss_depth
        return loss

class HeadPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, yaw, pitch, roll, real_yaw, real_pitch, real_roll):
        loss = (self.criterion(yaw, real_yaw.detach()) + self.criterion(pitch, real_pitch.detach()) + self.criterion(roll, real_roll.detach())) / 3
        return loss / np.pi * 180

        
class DeformationPriorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, deformation):
        loss = deformation.abs().mean()
        return loss
    

class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas = [0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K = (0.01, 0.03),
                 alpha = 0.84):
        
        super().__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda()

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype = torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR

        return loss_mix.mean()

    
if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256).cuda()
    target = torch.randn(1, 3, 256, 256).cuda()
    loss = FeatureMatchingLoss().cuda()
    output = loss(input, target)
    print(output)