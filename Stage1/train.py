import os
import argparse
import sys
import torch.utils.data as data
from logger import Logger
from datasets.dataset import FramesDataset, DatasetRepeater
from distributed import init_seeds  

def main(args):
    init_seeds(not args.benchmark)

    trainset = DatasetRepeater(FramesDataset(root_dir = r'C:\Users\george\VG_Project\video_enhancement\GFPGAN\pure_talking_faces\train'), num_repeats=50)
    trainloader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,  
    )

    logger = Logger(args.num_kp, args.ckp_dir, args.vis_dir, trainloader, args.lr)

    if args.ckp_epoch and args.ckp_iter > 0:
        logger.load_cpk(args.ckp_epoch, args.ckp_iter)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + args.ckp_epoch + 1}/{args.num_epochs}")
        logger.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--benchmark", type=str2bool, default=True, help="Turn on CUDNN benchmarking")
    parser.add_argument("--gpu_id", default=0, type=int, help="ID of the GPU to use")
    parser.add_argument("--lr", default=2.0e-4, type=float, help="Learning rate")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs to train")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of data loader threads")
    parser.add_argument("--num_kp", type=int, default=68, help="Number of keypoints")
    parser.add_argument("--ckp_dir", type=str, default="record/ckp", help="Checkpoint directory")
    parser.add_argument("--vis_dir", type=str, default="record/vis", help="Visualization directory")
    parser.add_argument("--ckp_epoch", type=int, default=6, help="Checkpoint epoch")
    parser.add_argument("--ckp_iter", type=int, default=56250, help="Checkpoint iteration")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    main(args)
