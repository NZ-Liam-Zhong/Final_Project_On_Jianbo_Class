import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np

# Add dino repo to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dino'))
import vision_transformer as vits
from ncut_pytorch import Ncut

def align_eigenvectors(curr_eig, ref_eig):
    """
    Align current eigenvectors to reference eigenvectors (previous epoch)
    to ensure temporal consistency (prevent color flipping).
    Args:
        curr_eig: (N, K)
        ref_eig: (N, K)
    """
    if ref_eig is None:
        return curr_eig
        
    K = curr_eig.shape[1]
    aligned_eig = curr_eig.clone()
    
    # We want to find a sign flip for each column that maximizes correlation with reference
    # Assuming the order of eigenvectors stays roughly the same (sorted by eigenvalue)
    
    for k in range(K):
        # Dot product with reference
        dot_prod = torch.dot(curr_eig[:, k], ref_eig[:, k])
        if dot_prod < 0:
            aligned_eig[:, k] = -aligned_eig[:, k]
            
    return aligned_eig

def generate_aligned_ncut(project_dir, output_dir):
    print("Starting generation of Aligned Ncut (Batch + Temporal Alignment) montages...")
    
    # --- 1. Setup paths and data ---
    data_path = os.path.join(project_dir, 'cifar10/val/airplane')
    img_files = sorted(glob.glob(os.path.join(data_path, '*.png')))[:100]
    
    if len(img_files) < 100:
        print(f"Warning: Only found {len(img_files)} images, processing all of them.")
    
    # Output directory
    montage_dir = os.path.join(output_dir, 'airplane_evolution_100_aligned')
    os.makedirs(montage_dir, exist_ok=True)
    
    # --- 2. Checkpoints ---
    checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint*.pth'))
    checkpoints.sort(key=lambda x: int(x.split('checkpoint')[-1].split('.pth')[0]) if x.split('checkpoint')[-1].split('.pth')[0].isdigit() else 999999)
    
    valid_checkpoints = []
    for ckpt in checkpoints:
        if 'checkpoint.pth' in ckpt: continue
        epoch_num = int(ckpt.split('checkpoint')[-1].split('.pth')[0])
        if epoch_num <= 56: 
            valid_checkpoints.append((epoch_num, ckpt))
            
    print(f"Found {len(valid_checkpoints)} checkpoints.")

    # --- 3. Load Images & Preprocess ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    original_images = []
    images_tensor = []
    
    for img_path in img_files:
        img = Image.open(img_path).convert('RGB')
        original_images.append(img.resize((224, 224)))
        images_tensor.append(transform(img))
    
    images_tensor = torch.stack(images_tensor).cuda()
    N_IMGS = len(images_tensor)
    
    # --- 4. Processing Loop ---
    results = {i: {} for i in range(N_IMGS)}
    prev_eigvecs = None # To store previous epoch's eigenvectors
    
    for epoch_num, ckpt_path in valid_checkpoints:
        print(f"Processing Epoch {epoch_num}...")
        model = load_model(ckpt_path)
        
        # Extract features for all images
        batch_size = 25 
        all_feats = []
        
        with torch.no_grad():
            for i in range(0, N_IMGS, batch_size):
                batch = images_tensor[i:i+batch_size]
                output = model.get_intermediate_layers(batch, n=1)[0]
                output = output[:, 1:, :] 
                all_feats.append(output)
        
        feats = torch.cat(all_feats, dim=0) 
        B_total, N_patches, D_dim = feats.shape
        
        # Flatten for Ncut
        feats_flat = feats.reshape(-1, D_dim) 
        
        # Run Ncut
        print(f"  Running Ncut on {feats_flat.shape[0]} nodes...")
        eigvecs = Ncut(n_eig=3).fit_transform(feats_flat)
        
        # --- TEMPORAL ALIGNMENT ---
        if prev_eigvecs is not None:
            eigvecs = align_eigenvectors(eigvecs, prev_eigvecs)
        
        prev_eigvecs = eigvecs.clone()
        
        # Normalize for visualization
        eigvecs_disp = (eigvecs - eigvecs.min(0, keepdim=True)[0]) / (eigvecs.max(0, keepdim=True)[0] - eigvecs.min(0, keepdim=True)[0] + 1e-8)
        
        # Reshape back to images
        eigvecs_reshaped = eigvecs_disp.reshape(B_total, 14, 14, 3).cpu().numpy()
        
        for i in range(B_total):
            eig_img_np = eigvecs_reshaped[i]
            # Upsample
            eig_pil = Image.fromarray((eig_img_np * 255).astype(np.uint8)).resize((224, 224), resample=Image.NEAREST)
            results[i][epoch_num] = eig_pil

    # --- 5. Generate Montages ---
    print(f"Creating montages in {montage_dir}...")
    for i in range(N_IMGS):
        img_list = [original_images[i]]
        sorted_epochs = sorted(results[i].keys())
        for ep in sorted_epochs:
            img_list.append(results[i][ep])
        
        cols = 6
        rows = (len(img_list) + cols - 1) // cols
        
        w, h = 224, 224
        montage = Image.new('RGB', (w * cols, h * rows))
        
        for idx, img in enumerate(img_list):
            r = idx // cols
            c = idx % cols
            
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            label = "Original" if idx == 0 else f"Ep {sorted_epochs[idx-1]}"
            
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            draw.rectangle((5, 5, 5 + (right-left) + 10, 5 + (bottom-top) + 10), fill="black")
            draw.text((10, 10), label, font=font, fill="white")
            
            montage.paste(img, (c * w, r * h))
            
        save_name = os.path.basename(img_files[i]).replace('.png', '_evolution_aligned.png')
        montage.save(os.path.join(montage_dir, save_name))
        
    print(f"Done! Saved aligned montages to {montage_dir}")

def load_model(checkpoint_path):
    model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.cuda()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['teacher'] if 'teacher' in checkpoint else checkpoint
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model

if __name__ == '__main__':
    PROJECT_DIR = "/mnt/disk1/ilykyleliam/liam/dino_ncut_project"
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "dino_experiment")
    generate_aligned_ncut(PROJECT_DIR, OUTPUT_DIR)

