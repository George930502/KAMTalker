import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.linalg import sqrtm
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

#####################################
# Function to compute PSNR between two images
#####################################
def compute_psnr(gt_img, gen_img):
    # Convert PIL images to numpy arrays (uint8)
    gt = np.array(gt_img)
    gen = np.array(gen_img)
    # Debug prints for image shapes
    #print("GT Image Shape:", gt.shape)
    #print("Generated Image Shape:", gen.shape)
    # Set data_range to 255 (8-bit images)
    return peak_signal_noise_ratio(gt, gen, data_range=255)

#####################################
# Function to compute SSIM between two images
#####################################
def compute_ssim(gt_img, gen_img):
    # Convert PIL images to numpy arrays
    gt = np.array(gt_img)
    gen = np.array(gen_img)
    # For color images, use channel_axis=2 (newer versions of scikit-image)
    return structural_similarity(gt, gen, channel_axis=2, win_size=7)

#####################################
# Get Inception v3 model for feature extraction
# This model is used to extract features for FID calculation.
#####################################
def get_inception_model(device):
    # Load pre-trained Inception v3 model from torchvision and set it to eval mode
    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # Remove final classification layer to get features
    model.to(device)
    model.eval()
    return model

#####################################
# Compute activation statistics (mean and covariance) for a list of images
#####################################
def compute_activation_statistics(image_list, model, device, batch_size=32):
    model.eval()
    activations = []
    with torch.no_grad():
        for i in range(0, len(image_list), batch_size):
            batch_imgs = image_list[i:i+batch_size]
            batch = torch.stack(batch_imgs).to(device)
            pred = model(batch)
            activations.append(pred.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

#####################################
# Calculate FID given the activation statistics of two image sets
#####################################
def calculate_fid(mu1, sigma1, mu2, sigma2):
    # Compute the difference between the means
    diff = mu1 - mu2
    #print("mu_gen shape:", mu1.shape)
    #print("sigma_gen shape:", sigma1.shape)
    #print("mu_gt shape:", mu2.shape)
    #print("sigma_gt shape:", sigma2.shape)
    # Compute the square root of the product of covariances
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # Handle potential numerical error resulting in complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid

#####################################
# Compute FID for two sets of images
#####################################
def compute_fid(gen_images, gt_images, device):
    # Define the transformations required by Inception v3: resize to 299x299, convert to tensor, and normalize
    transform_inception = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Convert all images to tensors using the defined transformation
    gen_tensors = [transform_inception(img) for img in gen_images]
    gt_tensors = [transform_inception(img) for img in gt_images]
    
    # Load the Inception model on the specified device
    model = get_inception_model(device)
    
    # Compute activation statistics for generated and ground truth images
    mu_gen, sigma_gen = compute_activation_statistics(gen_tensors, model, device)
    mu_gt, sigma_gt = compute_activation_statistics(gt_tensors, model, device)
    
    fid_value = calculate_fid(mu_gen, sigma_gen, mu_gt, sigma_gt)
    return fid_value

#####################################
# Main workflow: process image pairs from subfolders,
# compute PSNR, SSIM, and FID, then calculate average metrics.
#####################################
def main(data_root):
    # Lists to store metrics for each image pair
    psnr_list = []
    ssim_list = []
    
    # Lists to store images for FID calculation
    gen_imgs_all = []
    gt_imgs_all = []
    
    # Get all subfolders in the data root directory
    subfolders = [os.path.join(data_root, d) for d in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, d))]
    print(f"Found {len(subfolders)} subfolders (pairs).")
    
    for folder in subfolders:
        # Assuming each folder contains 'driving.jpg' (ground truth) and 'output.jpg' (generated)
        gt_path = os.path.join(folder, "driving.jpg")
        gen_path = os.path.join(folder, "output.jpg")
        
        # Open images and convert to RGB
        gt_img = Image.open(gt_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")
        
        # Compute PSNR and SSIM for the current image pair
        psnr = compute_psnr(gt_img, gen_img)
        ssim = compute_ssim(gt_img, gen_img)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        
        # Collect images for FID computation
        gt_imgs_all.append(gt_img)
        gen_imgs_all.append(gen_img)
    
    # Calculate average PSNR and SSIM
    avg_psnr = np.nanmean(psnr_list)
    avg_ssim = np.nanmean(ssim_list)

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_value = compute_fid(gen_imgs_all, gt_imgs_all, device)
    
    # Output the average results
    print("Average Results:")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"FID: {fid_value:.4f}")

if __name__ == "__main__":
    # Set the root directory containing the subfolders with image pairs
    data_root = "results/1"
    main(data_root)
