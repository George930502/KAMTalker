import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import face_alignment
import sys, os

from modules.Eapp import AppearanceFeatureExtraction
from modules.MotionFieldEstimator import MotionFieldEstimator
from modules.Generator import Generator
from modules.Emtn_gt import Hopenet
from modules.Discriminator import Discriminator
from modules.utils import *
from modules.losses import *


class VideoSynthesisModel(nn.Module):
    def __init__(self, 
                 AFE: AppearanceFeatureExtraction,
                 MFE: MotionFieldEstimator,
                 Generator: Generator,
                 Discriminator: Discriminator,
                 pretrained_path = "pretrained/hopenet_robust_alpha1.pkl",
                 num_bins = 66):
        
        super().__init__()
        self.fa_ckd = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
        self.AFE = AFE
        self.MFE = MFE
        self.Generator = Generator
        self.Discriminator = Discriminator

        self.weights = {
            "P": 5,
            "G": 1,
            "F": 10,
            "MIX": 10
        }

        self.losses = {
            "P": PerceptualLoss(),
            "G": GANLoss(),
            "F": FeatureMatchingLoss(),
            "MIX": MS_SSIM_L1_LOSS()
        }

        pretrained_HPNet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins).cuda()
        pretrained_HPNet.load_state_dict(torch.load(pretrained_path, map_location = torch.device("cuda:0"), weights_only = True))
        for parameter in pretrained_HPNet.parameters():
            parameter.requires_grad = False   # for evaluation only
        self.pretrained_HPNet = pretrained_HPNet

    def extract_canonical_kp(self, image_tensor):
        B = image_tensor.shape[0]
        keypoints_list = []
        for i in range(B):
            img_tensor = image_tensor[i]  # (3, H, W)
            image_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255
            image_np = image_np.astype(np.uint8)
            landmarks = self.fa_ckd.get_landmarks(image_np)
            if landmarks is None or len(landmarks) == 0:
                # raise ValueError(f"No face detected in sample {i}")
                return None
            keypoints_list.append(torch.tensor(landmarks[0], dtype=torch.float32))
        keypoints_batch = torch.stack(keypoints_list, dim=0).cuda()  # shape (B, 68, 3)
        return keypoints_batch

    def forward(self, source_img, drive_img):
        fs = self.AFE(source_img)
        cated = torch.cat([source_img, drive_img], dim = 0)

        kp_s = self.extract_canonical_kp(source_img)    
        kp_d = self.extract_canonical_kp(drive_img)

        # skip this batch if face detection failed
        if kp_s is None or kp_d is None:
            return None, None, None, None, None   

        with torch.no_grad():
            self.pretrained_HPNet.eval()
            real_yaw, real_pitch, real_roll = self.pretrained_HPNet(F.interpolate(apply_imagenet_normalization(cated), size = (224, 224)))
        
        [yaw_s, yaw_d], [pitch_s, pitch_d], [roll_s, roll_d] = (
            torch.chunk(real_yaw, 2, dim = 0),
            torch.chunk(real_pitch, 2, dim = 0),
            torch.chunk(real_roll, 2, dim = 0),
        )

        Rs = getRotationMatrix(yaw_s, pitch_s, roll_s)
        Rd = getRotationMatrix(yaw_d, pitch_d, roll_d)
    
        composited_flow_field, occlusion_mask = self.MFE(fs, kp_s, kp_d, Rs, Rd)
        generated_drive_img = self.Generator(fs, composited_flow_field, occlusion_mask)
        _, features_real_drive = self.Discriminator(drive_img, kp_d)
        output_fake_drive, features_fake_drive = self.Discriminator(generated_drive_img, kp_d)
        
        # calculate loss function
        loss = {
            "P": self.weights["P"] * self.losses["P"](generated_drive_img, drive_img),
            "G": self.weights["G"] * self.losses["G"](output_fake_drive, True, False),
            "F": self.weights["F"] * self.losses["F"](features_fake_drive, features_real_drive),
            "MIX": self.weights["MIX"] * self.losses["MIX"](generated_drive_img, drive_img),
        }

        return loss, generated_drive_img, kp_s, kp_d, occlusion_mask

class DiscriminatorFull(nn.Module):
    def __init__(self, Discriminator: Discriminator):
        super().__init__()
        self.Discriminator = Discriminator
        self.weights = {
            "G": 1,
        }
        self.losses = {
            "G": GANLoss(),
        }

    def forward(self, drive_img, generated_drive_img, kp_d):
        output_real_drive, _ = self.Discriminator(drive_img, kp_d)
        output_fake_drive, _ = self.Discriminator(generated_drive_img.detach(), kp_d)
        loss = {
            "G1": self.weights["G"] * self.losses["G"](output_fake_drive, False, True),
            "G2": self.weights["G"] * self.losses["G"](output_real_drive, True, True),
        }
        return loss


def print_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    size_in_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    size_in_mb = size_in_bytes / (1024 * 1024)
    print(f"Total number of parameters: {num_params}")
    print(f"Size of the model: {size_in_mb:.2f} MB")


if __name__ == '__main__':
    g_models = {"AFE": AppearanceFeatureExtraction(), "MFE": MotionFieldEstimator(num_kp=68), "Generator": Generator()}
    d_models = {"Discriminator": Discriminator(num_kp=68)}

    from skimage import io
    from torchvision import transforms

    def load_real_face_image(path):
        img = io.imread(path)  # numpy: (H, W, 3)
        transform = transforms.Compose([
            transforms.ToTensor(),  # -> (3, H, W), range [0,1]
        ])
        return transform(img).unsqueeze(0).cuda()  # -> (1, 3, 256, 256)

    source_img = load_real_face_image("demo/source.jpg")
    drive_img = load_real_face_image("demo/source.jpg")

    model = VideoSynthesisModel(**g_models, **d_models).cuda()
    output = model(source_img, drive_img)
    print(output[0])
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    print(output[4].shape)
    print_model_size(model)


