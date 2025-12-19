import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

# Add dino repo to path to import vision_transformer
sys.path.append(os.path.join(os.path.dirname(__file__), 'dino'))
import vision_transformer as vits

from ncut_pytorch import Ncut

def load_model(checkpoint_path, arch='vit_small', patch_size=16):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.cuda()
    
    if os.path.exists(checkpoint_path):
        # Allow loading numpy scalars etc by setting weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Load teacher weights
        if 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (DDP)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Handle potential prefix mismatch
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {checkpoint_path} with msg: {msg}")
    else:
        print(f"Checkpoint {checkpoint_path} not found")
        return None
        
    return model

def extract_features(model, images):
    # images: (B, C, H, W) tensor
    # DINO v1 feature extraction
    with torch.no_grad():
        # get_intermediate_layers returns list of features from last n blocks
        # We usually want the last block's output
        # For DINO v1 ViT, the output of forward is class token? 
        # get_intermediate_layers(x, n=1) returns the last layer output including CLS token
        
        # We need the spatial tokens.
        # DINO's vision_transformer.py implementation of get_intermediate_layers:
        # returns [x] where x is (B, N, D)
        
        output = model.get_intermediate_layers(images, n=1)[0]
        # output shape: (B, N_tokens + 1, Dim) (assuming +1 for CLS)
        
        # Remove CLS token (index 0)
        output = output[:, 1:, :]
        
        # Reshape to (B, H, W, D)
        # Patch size 16, img 224 -> 14x14
        w, h = images.shape[2] // 16, images.shape[3] // 16
        output = output.reshape(output.shape[0], w, h, output.shape[2])
        
    return output

def visualize_ncut(features, save_path, epoch):
    # features: (B, H, W, D)
    B, H, W, D = features.shape
    
    # Flatten for Ncut: (N, D)
    # We process each image separately for visualization
    
    for i in range(B):
        feat = features[i].reshape(-1, D) # (H*W, D)
        
        # Run Ncut
        # Using Ncut with default settings (Nystrom approximation if large, or exact)
        # Here 14x14=196 is small, exact is fine, but library handles it.
        eigvecs = Ncut(n_eig=3).fit_transform(feat) # (N, 3)
        
        # Visualize top 3 eigenvectors as RGB
        # Normalize to 0-1
        eigvecs = (eigvecs - eigvecs.min(0, keepdim=True)[0]) / (eigvecs.max(0, keepdim=True)[0] - eigvecs.min(0, keepdim=True)[0] + 1e-8)
        
        eig_img = eigvecs.reshape(H, W, 3)
        
        # Upsample for better view
        eig_img_np = eig_img.cpu().numpy()
        
        plt.figure(figsize=(4, 4))
        plt.imshow(eig_img_np)
        plt.axis('off')
        plt.title(f'Epoch {epoch} - Img {i}')
        plt.savefig(f"{save_path}_img{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./dino_experiment', type=str)
    parser.add_argument('--data_path', default='./cifar10/val/airplane', type=str) # Use one class for demo
    args = parser.parse_args()

    # Load a few test images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    image_files = glob.glob(os.path.join(args.data_path, '*.png'))[:4]
    if not image_files:
        print("No images found in data path")
        return
        
    images = []
    original_images = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        original_images.append(img)
        images.append(transform(img))
        
    images = torch.stack(images).cuda()
    
    # Create output dir for vis
    vis_dir = os.path.join(args.output_dir, 'ncut_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Find checkpoints
    checkpoints = glob.glob(os.path.join(args.output_dir, 'checkpoint*.pth'))
    # Sort by number
    checkpoints.sort(key=lambda x: int(x.split('checkpoint')[-1].split('.pth')[0]) if x.split('checkpoint')[-1].split('.pth')[0].isdigit() else 999999)
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    for ckpt in checkpoints:
        if 'checkpoint.pth' in ckpt and len(checkpoints) > 1:
            continue # Skip the 'latest' symlink-like file if we have numbered ones, or process it last
            
        epoch_num = ckpt.split('checkpoint')[-1].split('.pth')[0]
        if not epoch_num.isdigit(): epoch_num = "latest"
        
        print(f"Processing {ckpt}...")
        model = load_model(ckpt, arch='vit_small', patch_size=16)
        if model is None: continue
        
        feats = extract_features(model, images)
        visualize_ncut(feats, os.path.join(vis_dir, f'epoch_{epoch_num}'), epoch_num)

if __name__ == '__main__':
    main()

