#!/usr/bin/env python3
"""
OpenVLA Vision Tower NCut可视化脚本 (修复版)
- 处理全部帧 (不限制max_frames)
- 改进可视化效果

输出:
1. 无时序对齐版本: openvla_ncut_batch/
2. 有时序对齐版本: openvla_ncut_aligned/
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Set environment
os.environ['HF_HOME'] = '/mnt/disk1/ilykyleliam/public'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/disk1/ilykyleliam/public'

from transformers import AutoModelForVision2Seq, AutoProcessor
from ncut_pytorch import Ncut


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VisionFeatureHook:
    """Hook to capture vision encoder features"""
    def __init__(self):
        self.features = None
        
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            self.features = output[0].detach()
        else:
            self.features = output.detach()


def extract_all_frames(video_path):
    """Extract ALL frames from video (no limit)"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video has {total_frames} frames at {fps:.1f} fps")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames, fps


def align_eigenvectors(curr_eig, ref_eig):
    """
    Align current eigenvectors to reference eigenvectors
    to ensure temporal consistency (prevent color flipping).
    """
    if ref_eig is None:
        return curr_eig
        
    K = curr_eig.shape[1]
    aligned_eig = curr_eig.clone()
    
    for k in range(K):
        dot_prod = torch.dot(curr_eig[:, k], ref_eig[:, k])
        if dot_prod < 0:
            aligned_eig[:, k] = -aligned_eig[:, k]
            
    return aligned_eig


def compute_batch_ncut(features_list, n_eig=3, device='cuda'):
    """
    Compute batched NCut on a list of features.
    
    Args:
        features_list: list of tensors, each (N, D)
        n_eig: number of eigenvectors
        device: cuda device
    
    Returns:
        colors_list: list of RGB colors, each (N, 3)
    """
    # Concatenate all features
    all_features = []
    frame_lengths = []
    
    for feat in features_list:
        if feat.dim() == 3:
            feat = feat.squeeze(0)
        all_features.append(feat.float())
        frame_lengths.append(len(feat))
    
    # Concatenate along sequence dimension
    batched_features = torch.cat(all_features, dim=0).to(device)  # (total_N, D)
    
    print(f"    Batched features shape: {batched_features.shape}")
    
    try:
        # Compute NCut with proper parameters
        ncut = Ncut(n_eig=n_eig, device=device)
        eigvecs = ncut.fit_transform(batched_features)
        
        # Normalize to 0-1 for RGB visualization
        eigvecs_norm = (eigvecs - eigvecs.min(0, keepdim=True)[0]) / \
                       (eigvecs.max(0, keepdim=True)[0] - eigvecs.min(0, keepdim=True)[0] + 1e-8)
        
        # Split colors back to individual frames
        colors_list = []
        start_idx = 0
        for length in frame_lengths:
            colors_list.append(eigvecs_norm[start_idx:start_idx + length].cpu())
            start_idx += length
        
        # Clear GPU memory
        del batched_features, eigvecs, eigvecs_norm
        torch.cuda.empty_cache()
        
        return colors_list
    except Exception as e:
        print(f"    Batched NCut failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def compute_aligned_ncut(features_list, n_eig=3, device='cuda'):
    """
    Compute NCut with temporal alignment.
    Same as batch NCut but with sign correction for temporal consistency.
    """
    # First do batch NCut
    colors_list = compute_batch_ncut(features_list, n_eig, device)
    
    if colors_list is None:
        return None
    
    # Apply temporal alignment
    aligned_colors = []
    prev_colors = None
    
    for colors in colors_list:
        if prev_colors is not None:
            aligned = colors.clone()
            for c in range(colors.shape[1]):
                curr_channel = colors[:, c]
                prev_channel = prev_colors[:, c]
                
                # Check correlation
                corr = torch.corrcoef(torch.stack([curr_channel, prev_channel]))[0, 1]
                if corr < 0:
                    aligned[:, c] = 1.0 - aligned[:, c]
            
            aligned_colors.append(aligned)
            prev_colors = aligned
        else:
            aligned_colors.append(colors)
            prev_colors = colors
    
    return aligned_colors


def create_side_by_side_video(original_frames, ncut_colors, grid_shape, output_path, fps=30):
    """
    Create a side-by-side comparison video.
    Left: original frame, Right: NCut visualization
    """
    if len(original_frames) == 0 or ncut_colors is None:
        print(f"    Skipping video creation: no frames or colors")
        return False
    
    h, w = grid_shape
    first_frame = original_frames[0]
    frame_h, frame_w = first_frame.size[1], first_frame.size[0]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w * 2, frame_h))
    
    for i, (orig_frame, colors) in enumerate(zip(original_frames, ncut_colors)):
        # Convert original to numpy
        orig_np = np.array(orig_frame)
        
        # Reshape colors to grid
        expected_tokens = h * w
        
        # Features should already be exactly 256 tokens (CLS removed before NCut)
        # But add safety check just in case
        if len(colors) != expected_tokens:
            print(f"    Warning frame {i}: expected {expected_tokens} tokens, got {len(colors)}")
            if len(colors) > expected_tokens:
                colors = colors[:expected_tokens]
            else:
                padding = torch.zeros(expected_tokens - len(colors), 3)
                colors = torch.cat([colors, padding], dim=0)
        
        # Reshape to grid and convert to image
        colors_grid = colors.reshape(h, w, 3).numpy()
        colors_grid = (colors_grid * 255).astype(np.uint8)
        
        # Resize to match original frame size using bilinear interpolation for smoother look
        colors_resized = cv2.resize(colors_grid, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        
        # Concatenate side by side
        combined = np.concatenate([orig_np, colors_resized], axis=1)
        
        # Convert RGB to BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)
    
    out.release()
    print(f"    ✓ Saved {len(original_frames)} frames to {output_path.name}")
    return True


def process_video(model, processor, hook, video_path, output_dir_batch, output_dir_aligned, device):
    """Process a single video and create both batch and aligned NCut videos"""
    
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {video_name[:60]}...")
    print(f"{'='*60}")
    
    # Extract ALL frames (no limit)
    print(f"  Extracting frames...")
    frames, fps = extract_all_frames(video_path)
    print(f"  Extracted {len(frames)} frames")
    
    if len(frames) == 0:
        print(f"  ERROR: No frames extracted!")
        return False
    
    # Extract task description from filename
    if '--task=' in video_name:
        task_desc = video_name.split('--task=')[1].replace('_', ' ')
    else:
        task_desc = "pick up the black bowl and place it on the plate"
    
    prompt = f"In: What action should the robot take to {task_desc}?\nOut:"
    
    # Collect features for all frames
    print(f"  Extracting vision features for {len(frames)} frames...")
    all_features = []
    grid_shape = None
    
    for i, frame in enumerate(tqdm(frames, desc="  Features")):
        inputs = processor(prompt, frame).to(device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            _ = model(**inputs)
        
        feat = hook.features
        if feat is not None:
            # feat shape is typically (1, N, D) where N = H*W or H*W + extra tokens
            if feat.dim() == 3:
                feat = feat.squeeze(0)  # (N, D)
            
            # Determine grid shape from first frame
            if grid_shape is None:
                N = feat.shape[0]
                print(f"  Raw feature shape: {feat.shape}, N tokens: {N}")
                
                # OpenVLA uses DINOv2 with 16x16 = 256 patches
                # Plus CLS + 4 register tokens = 5 extra tokens at the beginning
                # Total: 261 tokens
                grid_shape = (16, 16)
                n_patches = grid_shape[0] * grid_shape[1]  # 256
                n_extra = N - n_patches  # Should be 5 (CLS + 4 register)
                print(f"  Grid shape: {grid_shape}, patches: {n_patches}, extra tokens: {n_extra}")
            
            # CRITICAL: Remove CLS + register tokens BEFORE NCut
            # Keep only the last 256 tokens (patch tokens)
            n_patches = grid_shape[0] * grid_shape[1]
            if feat.shape[0] > n_patches:
                feat = feat[-n_patches:]  # Take last 256 tokens (patches only)
            
            all_features.append(feat.cpu())
        
        # Clear GPU cache periodically
        if i % 50 == 0:
            torch.cuda.empty_cache()
    
    if len(all_features) == 0:
        print(f"  ERROR: No features extracted!")
        return False
    
    print(f"  Collected {len(all_features)} feature tensors")
    
    # Compute Batch NCut (no alignment)
    print(f"  Computing Batch NCut...")
    batch_colors = compute_batch_ncut(all_features, n_eig=3, device=device)
    
    if batch_colors is not None:
        # Create batch video
        output_batch = output_dir_batch / f"{video_name}_ncut_batch.mp4"
        print(f"  Creating batch NCut video...")
        success = create_side_by_side_video(frames, batch_colors, grid_shape, output_batch, fps=fps)
    else:
        print(f"  ERROR: Batch NCut failed!")
    
    # Compute Aligned NCut
    print(f"  Computing Aligned NCut...")
    aligned_colors = compute_aligned_ncut(all_features, n_eig=3, device=device)
    
    if aligned_colors is not None:
        # Create aligned video
        output_aligned = output_dir_aligned / f"{video_name}_ncut_aligned.mp4"
        print(f"  Creating aligned NCut video...")
        success = create_side_by_side_video(frames, aligned_colors, grid_shape, output_aligned, fps=fps)
    else:
        print(f"  ERROR: Aligned NCut failed!")
    
    # Clean up
    del all_features, batch_colors, aligned_colors
    torch.cuda.empty_cache()
    
    return True


def main():
    seed_everything(42)
    
    # Paths
    video_dir = Path("/mnt/disk1/ilykyleliam/liam/openvla_repro/openvla/rollouts/2025_12_08/failure")
    output_base = Path("/mnt/disk1/ilykyleliam/liam/openvla_repro")
    output_dir_batch = output_base / "openvla_ncut_batch"
    output_dir_aligned = output_base / "openvla_ncut_aligned"
    
    # Clean and recreate output directories
    import shutil
    if output_dir_batch.exists():
        shutil.rmtree(output_dir_batch)
    if output_dir_aligned.exists():
        shutil.rmtree(output_dir_aligned)
    
    output_dir_batch.mkdir(parents=True, exist_ok=True)
    output_dir_aligned.mkdir(parents=True, exist_ok=True)
    
    # Find videos (take first 6)
    videos = sorted(video_dir.glob("*.mp4"))[:6]
    print(f"Found {len(videos)} videos to process")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n" + "="*60)
    print("Loading OpenVLA model...")
    print("="*60)
    
    model_name = "openvla/openvla-7b-finetuned-libero-spatial"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded")
    
    # Register hook for vision encoder (last layer)
    hook = VisionFeatureHook()
    
    # Find vision backbone and register hook
    if hasattr(model, 'vision_backbone'):
        vision = model.vision_backbone
        if hasattr(vision, 'featurizer') and hasattr(vision.featurizer, 'blocks'):
            # Register on the last block
            last_block = vision.featurizer.blocks[-1]
            handle = last_block.register_forward_hook(hook)
            print(f"✓ Registered hook on vision encoder layer {len(vision.featurizer.blocks)-1}")
        else:
            print("WARNING: Could not find vision blocks")
            return
    else:
        print("WARNING: Could not find vision_backbone")
        return
    
    # Process videos
    print("\n" + "="*60)
    print(f"Processing {len(videos)} videos (ALL frames)...")
    print("="*60)
    
    for video_path in videos:
        try:
            process_video(
                model, processor, hook, 
                video_path, 
                output_dir_batch, 
                output_dir_aligned,
                device
            )
        except Exception as e:
            print(f"  ERROR processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    handle.remove()
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nBatch NCut videos saved to: {output_dir_batch}")
    print(f"Aligned NCut videos saved to: {output_dir_aligned}")


if __name__ == '__main__':
    main()
