import numpy as np
import torch
import torch.nn.functional as F
import os
from skimage.draw import disk
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
import sys

from modules.Eapp import AppearanceFeatureExtraction
from modules.MotionFieldEstimator import MotionFieldEstimator
from modules.Generator import Generator
from modules.Discriminator import Discriminator
from trainer import *

def to_cpu(losses):
    return {key: value.detach().data.cpu().numpy() for key, value in losses.items()}

def save_comparison_image(source, driving, generated, save_path):
    """Save a side-by-side comparison of source, driving, and generated images."""
    # Convert tensors to numpy arrays for visualization
    source = source.detach().cpu().numpy().transpose(1, 2, 0)
    driving = driving.detach().cpu().numpy().transpose(1, 2, 0)
    generated = generated.detach().cpu().numpy().transpose(1, 2, 0)

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [source, driving, generated]
    titles = ["Source", "Driving", "Generated"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.clip(0, 1))
        ax.set_title(title)
        ax.axis("off")

    # Save the comparison image
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class Logger:
    def __init__(
        self,
        num_kp,
        ckp_dir,
        vis_dir,
        dataloader,
        lr,
        checkpoint_freq= 6250,   # 450000
        visualizer_params={"kp_size": 3, "draw_border": True, "colormap": "gist_rainbow"},
        zfill_num=8,
        log_file_name="record/log.txt",
        visualize_interval= 1250,  # Visualize every 1000 iterations
    ):
        self.visualize_interval = visualize_interval
        self.g_losses, self.d_losses = [], []
        self.ckp_dir = ckp_dir
        self.vis_dir = vis_dir

        self.num_kp = num_kp

        if not os.path.exists(self.ckp_dir):
            os.makedirs(self.ckp_dir)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        self.log_file = open(log_file_name, "a")

        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 1
        self.best_loss = float("inf")

        self.g_models = {
            "AFE": AppearanceFeatureExtraction(),
            "MFE": MotionFieldEstimator(num_kp = self.num_kp),
            "Generator": Generator()
        }
        self.d_models = {"Discriminator": Discriminator(num_kp = self.num_kp)}

        for name, model in self.g_models.items():
            self.g_models[name] = model.cuda()
        for name, model in self.d_models.items():
            self.d_models[name] = model.cuda()

        self.g_optimizers = {
            name: torch.optim.Adam(self.g_models[name].parameters(), lr=lr, betas=(0.5, 0.999))
            for name in self.g_models.keys()
        }
        self.d_optimizers = {
            name: torch.optim.Adam(self.d_models[name].parameters(), lr=lr, betas=(0.5, 0.999))
            for name in self.d_models.keys()
        }
        self.g_full = VideoSynthesisModel(**self.g_models, **self.d_models)
        self.d_full = DiscriminatorFull(**self.d_models)
        self.g_loss_names, self.d_loss_names = None, None
        self.dataloader = dataloader

    def log_scores(self):
        loss_mean = np.array(self.g_losses).mean(axis=0)
        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(self.g_loss_names, loss_mean)])
        loss_string = "G" + str(self.epoch).zfill(self.zfill_num) + ") " + loss_string
        print(loss_string, file=self.log_file)
        self.g_losses = []
        loss_mean = np.array(self.d_losses).mean(axis=0)
        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(self.d_loss_names, loss_mean)])
        loss_string = "D" + str(self.epoch).zfill(self.zfill_num) + ") " + loss_string
        print(loss_string, file=self.log_file)
        self.d_losses = []
        self.log_file.flush()

    def visualize_rec(self, s, d, generated_d, kp_s, kp_d, occlusion):
        image, titles = self.visualizer.visualize(s, d, generated_d, kp_s, kp_d, occlusion)

        per_plot_width = 12
        per_plot_height = 12  
        num_titles = len(titles)
        
        fig, ax = plt.subplots(1, num_titles, figsize=(per_plot_width, per_plot_height))
        
        for i in range(num_titles):
            start_col = i * image.shape[1] // num_titles
            end_col = (i + 1) * image.shape[1] // num_titles
            ax[i].imshow(image[:, start_col:end_col])  
            ax[i].set_title(titles[i], fontsize=14)
            ax[i].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def save_cpk(self, iteration):
        ckp = {
            **{k: v.state_dict() for k, v in self.g_models.items()},
            **{k: v.state_dict() for k, v in self.d_models.items()},
            **{"optimizer_" + k: v.state_dict() for k, v in self.g_optimizers.items()},
            **{"optimizer_" + k: v.state_dict() for k, v in self.d_optimizers.items()},
            "epoch": self.epoch,
        }
        filename = f"{str(self.epoch).zfill(self.zfill_num)}-iter{iteration:06d}-checkpoint.pth.tar"
        ckp_path = os.path.join(self.ckp_dir, filename)
        torch.save(ckp, ckp_path)

    def load_cpk(self, epoch, iteration):
        filename = f"{str(epoch).zfill(self.zfill_num)}-iter{iteration:06d}-checkpoint.pth.tar"
        ckp_path = os.path.join(self.ckp_dir, filename)
        checkpoint = torch.load(ckp_path, map_location=torch.device("cuda"), weights_only=True)
        for k, v in self.g_models.items():
            v.load_state_dict(checkpoint[k])
        for k, v in self.d_models.items():
            v.load_state_dict(checkpoint[k])
        for k, v in self.g_optimizers.items():
            v.load_state_dict(checkpoint["optimizer_" + k])
        for k, v in self.d_optimizers.items():
            v.load_state_dict(checkpoint["optimizer_" + k])
        self.epoch = checkpoint["epoch"] + 1

    def log_iter(self, g_losses, d_losses, iteration, s, d, generated_d):
        g_losses = collections.OrderedDict(g_losses.items())
        d_losses = collections.OrderedDict(d_losses.items())
        if self.g_loss_names is None:
            self.g_loss_names = list(g_losses.keys())
        if self.d_loss_names is None:
            self.d_loss_names = list(d_losses.keys())
        self.g_losses.append(list(g_losses.values()))
        self.d_losses.append(list(d_losses.values()))

        # Save images periodically
        if iteration % self.visualize_interval == 0:
            save_dir = os.path.join(self.vis_dir, f"Epochs_{self.epoch}_iteration_{iteration}")
            os.makedirs(save_dir, exist_ok=True)
            # Save the comparison image
            comparison_path = os.path.join(save_dir, "comparison.png")
            save_comparison_image(s[0], d[0], generated_d[0], comparison_path)
            print(f"Comparison image saved at iteration {iteration} in {comparison_path}")

    def log_info(self, s, d, generated_d, kp_s, kp_d, occlusion, iteration):
        if iteration % self.checkpoint_freq == 0:
            self.save_cpk(iteration)
            self.log_scores()
            self.visualize_rec(s, d, generated_d, kp_s, kp_d, occlusion)

    def step(self):
        print("Epoch", self.epoch)
        with tqdm(total=len(self.dataloader.dataset)) as progress_bar:
            for iteration, (s, d) in enumerate(self.dataloader, start=1):
                s = s.cuda()
                d = d.cuda()

                for optimizer in self.g_optimizers.values():
                    optimizer.zero_grad()
                for optimizer in self.d_optimizers.values():
                    optimizer.zero_grad()

                losses_g, generated_d, kp_s, kp_d, occlusion = self.g_full(s, d)
                
                if losses_g is None:
                    continue 

                loss_g = sum(losses_g.values())
                loss_g.backward()

                for optimizer in self.g_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()
                
                losses_d = self.d_full(d, generated_d, kp_d)
                loss_d = sum(losses_d.values())
                loss_d.backward()

                for optimizer in self.d_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()

                self.log_iter(to_cpu(losses_g), to_cpu(losses_d), iteration, s, d, generated_d)
                self.log_info(s, d, generated_d, kp_s, kp_d, occlusion, iteration)
                progress_bar.update(len(s))

        self.epoch += 1

class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap="gist_rainbow"):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            center = (int(kp[1]), int(kp[0]))  
            rr, cc = disk(center, self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3] 
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            for i in range(images.shape[0]):
                images[i, :, [0, -1]] = 1 
                images[i, [0, -1], :] = 1  
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, s, d, generated_d, kp_s, kp_d, occlusion):
        images = []
        titles = [] 
        # Source image with keypoints
        source = s.data.cpu()
        kp_source = kp_s.data.cpu().numpy()[:, :, :2]
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))
        titles.append("Source Image")

        # Driving image with keypoints
        kp_driving = kp_d.data.cpu().numpy()[:, :, :2]
        driving = d.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))
        titles.append("Driving Image")

        # Result with and without keypoints
        prediction = generated_d.data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)
        titles.append("Generated Image")
        
        # Occlusion map
        occlusion_map = occlusion.data.cpu().repeat(1, 3, 1, 1)
        occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3])
        occlusion_map = occlusion_map.detach().cpu().numpy()
        occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
        images.append(occlusion_map)
        titles.append("Occlusion Map")

        image = self.create_image_grid(*images)
        image = image.clip(0, 1)
        image = (255 * image).astype(np.uint8)
        return image, titles
    

if __name__ == "__main__":
    # 假設圖像大小為 3 x 64 x 64（模擬彩色影像）
    B, C, H, W = 4, 3, 64, 64
    num_kp = 10

    # 模擬輸入資料
    source = torch.rand(B, C, H, W)
    driving = torch.rand(B, C, H, W)
    generated = torch.rand(B, C, H, W)

    # 模擬關鍵點座標（標準化 -1 到 1）
    kp_source = torch.rand(B, num_kp, 3)[..., :2] * 2 - 1
    kp_driving = torch.rand(B, num_kp, 3)[..., :2] * 2 - 1

    # 模擬 occlusion map（單通道）
    occlusion = torch.rand(B, 1, H, W)

    # 初始化 visualizer
    visualizer = Visualizer(kp_size=4, draw_border=True)

    # 可視化
    image, titles = visualizer.visualize(source, driving, generated, kp_source, kp_driving, occlusion)

    # 顯示結果
    plt.figure(figsize=(12, 12))
    num_sections = len(titles)
    for i in range(num_sections):
        start = i * image.shape[1] // num_sections
        end = (i + 1) * image.shape[1] // num_sections
        plt.subplot(1, num_sections, i + 1)
        plt.imshow(image[:, start:end])
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()