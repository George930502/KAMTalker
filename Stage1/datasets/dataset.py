import os
from skimage import io
from skimage.util import img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from skimage.transform import resize
from imageio import get_reader

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
import shutil

import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

def read_video(name):
    """
    Read video which can be:
      - '.mp4' and'.gif'
    """

    if name.lower().endswith(".gif") or name.lower().endswith(".mp4"):
        reader = get_reader(name)
        video_frames = []
        for frame in reader:
            frame = img_as_float32(frame)
            if frame.shape[-1] == 4:  # RGBA to RGB
                frame = frame[..., :3]
            if len(frame.shape) == 2:  # Grayscale to RGB
                frame = gray2rgb(frame)
            video_frames.append(frame)
        video_array = np.array(video_frames)

    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(
        self,
        root_dir,
        frame_shape=(256, 256, 3),
        is_train=True,
        random_seed=0,
        augmentation_params={
            "flip_param": {"horizontal_flip": True, "time_flip": True},
            "jitter_param": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.1},
        },
    ):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            # print("Use predefined train-test split.")
            self.train_target_directory = os.path.join(root_dir, 'train')
            self.test_target_directory = os.path.join(root_dir, 'test')

        else:   
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.1)
            self.train_target_directory = os.path.join(self.root_dir, "train")
            self.test_target_directory = os.path.join(self.root_dir, "test")

            os.makedirs(self.train_target_directory, exist_ok=True)
            os.makedirs(self.test_target_directory, exist_ok=True)

            for video in train_videos:
                video_path = os.path.join(self.root_dir, video)
                shutil.move(video_path, self.train_target_directory)  # Copy the video to the target directory
            for video in test_videos:
                video_path = os.path.join(self.root_dir, video)
                shutil.move(video_path, self.test_target_directory)  # Copy the video to the target directory

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def update_videos_list(self):
        if self.is_train:
            self.videos = os.listdir(self.train_target_directory)
        else:
            self.videos = os.listdir(self.test_target_directory)

    def __len__(self):
        self.update_videos_list()
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train:
            self.videos = os.listdir(self.train_target_directory)
            name = self.videos[idx]
            path = os.path.join(self.train_target_directory, name)
        else:
            self.videos = os.listdir(self.test_target_directory)
            name = self.videos[idx]
            path = os.path.join(self.test_target_directory, name)

        video_array = read_video(path)
        num_frames = len(video_array)

        # if num_frames >= 2:
        #     frame_idx = np.sort(np.random.choice(num_frames, replace = False, size=2))
        # else:
        #     frame_idx = [0, 0]

        if self.is_train:
             frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) 
        else:
             frame_idx = range(num_frames)
        
        video_array = video_array[frame_idx]

        if self.transform is not None:
             video_array = self.transform(video_array)
        
        if self.is_train:
            source = np.array(video_array[0], dtype="float32")
            driving = np.array(video_array[1], dtype="float32")

            driving = driving.transpose((2, 0, 1))
            source = source.transpose((2, 0, 1))
            return source, driving
        else:
            video = np.array(video_array, dtype="float32")
            video = video.transpose((3, 0, 1, 2))
            return video
            # source = np.array(video_array[0], dtype="float32")
            # driving = np.array(video_array[1], dtype="float32")

            # driving = driving.transpose((2, 0, 1))
            # source = source.transpose((2, 0, 1))
            # return source, driving

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=75):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
    

if __name__ == "__main__":
    root_dir = r'C:\Users\george\VG_Project\video_enhancement\GFPGAN\pure_talking_faces'  # 您實際的資料集目錄路徑

    # 測試 train dataset
    train_dataset = FramesDataset(
        root_dir=root_dir, 
        frame_shape=(256, 256, 3), 
        is_train=True,
        random_seed=0, 
        augmentation_params={}
    )
    print("Train dataset length:", train_dataset.__len__())

    for i in range(5):
        sample = train_dataset[i]
        print(sample[0].shape)
        print(sample[1].shape)

    # 測試 test dataset
    test_dataset = FramesDataset(
        root_dir=root_dir, 
        frame_shape=(256, 256, 3), 
        is_train=False,
        random_seed=0, 
        augmentation_params={}
    )

    print("Test dataset length:", len(test_dataset))
    for i in range(5):
        sample = test_dataset[i]
        print(sample.shape)
