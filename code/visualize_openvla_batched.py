#!/usr/bin/env python3
"""
OpenVLA 中间层可视化 (Batched NCut)
对每个视频的所有帧一起做 batched NCut
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
from PIL import Image

# Set environment
os.environ['HF_HOME'] = '/mnt/disk1/ilykyleliam/public'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/disk1/ilykyleliam/public'

from transformers import AutoModelForVision2Seq, AutoProcessor

# NCut
from ncut_pytorch import Ncut
from ncut_pytorch.color import umap_color
from ncut_pytorch.ncut import rbf_affinity


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OpenVLAVisualizationHook:
    """提取 OpenVLA 所有层的特征用于可视化"""
    def __init__(self, capture_vision=True, capture_llm=True):
        self.vision_features = {}
        self.llm_features = {}
        self.vision_handles = []
        self.llm_handles = []
        self.capture_vision = capture_vision
        self.capture_llm = capture_llm
        
    def get_vision_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.vision_features[name] = output[0].detach().cpu()
            else:
                self.vision_features[name] = output.detach().cpu()
        return hook
    
    def get_llm_hook(self, name):
        def hook(module, input, output):
            # 只捕获 prefill 阶段
            if name in self.llm_features:
                return
            if isinstance(output, tuple):
                self.llm_features[name] = output[0].detach().cpu()
            else:
                self.llm_features[name] = output.detach().cpu()
        return hook
    
    def register_hooks(self, model):
        # Vision Encoder - DinoV2
        if self.capture_vision and hasattr(model, 'vision_backbone'):
            vb = model.vision_backbone
            if hasattr(vb, 'featurizer') and hasattr(vb.featurizer, 'blocks'):
                for idx, block in enumerate(vb.featurizer.blocks):
                    handle = block.register_forward_hook(self.get_vision_hook(f'vision_layer_{idx}'))
                    self.vision_handles.append(handle)
                print(f"✓ Registered {len(vb.featurizer.blocks)} Vision Encoder hooks")
        
        # LLM Backbone
        if self.capture_llm and hasattr(model, 'language_model'):
            llm = model.language_model
            if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
                for idx, layer in enumerate(llm.model.layers):
                    handle = layer.register_forward_hook(self.get_llm_hook(f'llm_layer_{idx}'))
                    self.llm_handles.append(handle)
                print(f"✓ Registered {len(llm.model.layers)} LLM Backbone hooks")
        
        return self
    
    def remove_hooks(self):
        for handle in self.vision_handles + self.llm_handles:
            handle.remove()
        self.vision_handles = []
        self.llm_handles = []
    
    def clear(self):
        self.vision_features = {}
        self.llm_features = {}


def compute_batched_ncut_colors(features_list, n_eig=20, device='cuda'):
    """
    对一批特征一起计算 NCut 颜色
    features_list: list of tensors, each (N, D)
    returns: list of colors, each (N, 3)
    """
    # Concatenate all features
    all_features = []
    frame_lengths = []
    
    for feat in features_list:
        if feat.dim() == 3:
            feat = feat.squeeze(0)
        all_features.append(feat.float().to(device))
        frame_lengths.append(len(feat))
    
    # Concatenate along sequence dimension
    batched_features = torch.cat(all_features, dim=0)  # (total_N, D)
    
    try:
        # Use batched NCut with safer parameters
        ncut = Ncut(n_eig=n_eig, affinity_fn=rbf_affinity, device=device)
        
        # Compute eigenvectors for all frames together
        eigvecs = ncut.fit_transform(batched_features)
        
        # Get RGB colors
        rgb_colors = umap_color(eigvecs[:, :min(3, n_eig)])
        
        # Convert to CPU immediately
        rgb_colors = rgb_colors.cpu()
        
        # Split colors back to individual frames
        colors_list = []
        start_idx = 0
        for length in frame_lengths:
            colors_list.append(rgb_colors[start_idx:start_idx + length])
            start_idx += length
        
        # Clear GPU memory
        del batched_features, eigvecs
        torch.cuda.empty_cache()
        
        return colors_list
    except Exception as e:
        print(f"Batched NCut failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def visualize_and_save_layer(features, colors, output_path, layer_name, grid_shape):
    """可视化并保存单个层的特征"""
    h, w = grid_shape
    expected = h * w
    
    # Skip CLS token if present
    if len(features) == expected + 1:
        features = features[1:]
        colors = colors[1:]
    elif len(features) > expected:
        features = features[:expected]
        colors = colors[:expected]
    
    try:
        if len(colors) >= expected:
            colors_grid = colors[:expected].reshape(h, w, 3).numpy()
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(colors_grid, interpolation='nearest')
            ax.set_title(f"{layer_name}", fontsize=12)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"  Failed {layer_name}: {e}")


def extract_video_frames(video_path, num_frames=5):
    """从视频中均匀提取指定数量的帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Warning: Video {video_path} has 0 frames")
        return []
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames


def process_video_batched(model, processor, video_path, task_description, output_dir, device):
    """对单个视频的所有帧进行批处理 NCut 可视化"""
    
    # Extract frames
    frames = extract_video_frames(str(video_path), num_frames=5)
    print(f"  Extracted {len(frames)} frames")
    
    if len(frames) == 0:
        return
    
    # Collect features for all frames
    all_vision_features = {f'frame_{i+1:02d}': {} for i in range(len(frames))}
    all_llm_features = {f'frame_{i+1:02d}': {} for i in range(len(frames))}
    
    # Register hooks
    viz_hook = OpenVLAVisualizationHook(capture_vision=True, capture_llm=True)
    viz_hook.register_hooks(model)
    
    # Process each frame and collect features
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    
    for frame_idx, frame in enumerate(frames):
        frame_name = f"frame_{frame_idx+1:02d}"
        print(f"  Processing {frame_name}...")
        
        # Save original frame
        frame_output_dir = os.path.join(output_dir, frame_name)
        os.makedirs(frame_output_dir, exist_ok=True)
        frame.save(os.path.join(frame_output_dir, 'original_frame.png'))
        
        # Run inference
        inputs = processor(prompt, frame).to(device, dtype=torch.bfloat16)
        
        viz_hook.clear()
        with torch.no_grad():
            try:
                # Only do forward pass to collect features, not full action prediction
                _ = model(**inputs, output_hidden_states=True)
                print(f"    Features collected")
                torch.cuda.empty_cache()  # Clear cache after each frame
            except Exception as e:
                print(f"    Warning: Forward pass failed: {e}")
                torch.cuda.empty_cache()
        
        # Collect features
        all_vision_features[frame_name] = dict(viz_hook.vision_features)
        all_llm_features[frame_name] = dict(viz_hook.llm_features)
    
    viz_hook.remove_hooks()
    
    # Process Vision Encoder with batched NCut
    if all_vision_features[list(all_vision_features.keys())[0]]:
        print(f"  Computing batched NCut for Vision Encoder...")
        layer_names = sorted(all_vision_features[list(all_vision_features.keys())[0]].keys(),
                           key=lambda x: int(x.split('_')[-1]))
        
        for layer_name in tqdm(layer_names, desc="  Vision layers"):
            # Collect features for this layer across all frames
            layer_features_list = []
            for frame_name in sorted(all_vision_features.keys()):
                feat = all_vision_features[frame_name][layer_name]
                if feat.dim() == 3:
                    feat = feat[0]
                layer_features_list.append(feat)
            
            # Compute batched NCut colors
            colors_list = compute_batched_ncut_colors(layer_features_list, n_eig=100, device=device)
            
            if colors_list:
                # Save visualization for each frame
                for frame_idx, (frame_name, colors) in enumerate(zip(sorted(all_vision_features.keys()), colors_list)):
                    output_path = os.path.join(output_dir, frame_name, 'vision_encoder', f'{layer_name}.png')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    visualize_and_save_layer(
                        layer_features_list[frame_idx],
                        colors,
                        output_path,
                        f"Vision {layer_name}",
                        (16, 16)
                    )
    
    # Process LLM Backbone with batched NCut
    if all_llm_features[list(all_llm_features.keys())[0]]:
        print(f"  Computing batched NCut for LLM Backbone...")
        layer_names = sorted(all_llm_features[list(all_llm_features.keys())[0]].keys(),
                           key=lambda x: int(x.split('_')[-1]))
        
        for layer_name in tqdm(layer_names, desc="  LLM layers"):
            # Collect features for this layer across all frames
            layer_features_list = []
            for frame_name in sorted(all_llm_features.keys()):
                feat = all_llm_features[frame_name][layer_name]
                if feat.dim() == 3:
                    feat = feat[0]
                # Extract first 256 tokens (image tokens)
                if len(feat) >= 256:
                    feat = feat[:256]
                layer_features_list.append(feat)
            
            # Compute batched NCut colors
            colors_list = compute_batched_ncut_colors(layer_features_list, n_eig=100, device=device)
            
            if colors_list:
                # Save visualization for each frame
                for frame_idx, (frame_name, colors) in enumerate(zip(sorted(all_llm_features.keys()), colors_list)):
                    output_path = os.path.join(output_dir, frame_name, 'llm_backbone', f'{layer_name}.png')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    visualize_and_save_layer(
                        layer_features_list[frame_idx],
                        colors,
                        output_path,
                        f"LLM {layer_name}",
                        (16, 16)
                    )
    
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='OpenVLA 中间层可视化 (Batched NCut)')
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_success', type=int, default=3)
    parser.add_argument('--num_failure', type=int, default=2)
    parser.add_argument('--model_checkpoint', type=str,
                       default='openvla/openvla-7b-finetuned-libero-spatial')
    args = parser.parse_args()
    
    seed_everything(42)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load OpenVLA model
    print("\n" + "="*60)
    print("Loading OpenVLA model...")
    print("="*60)
    
    processor = AutoProcessor.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded")
    
    # Find videos
    video_dir = Path(args.video_dir)
    success_videos = sorted(list((video_dir / 'success').glob('*.mp4')))[:args.num_success]
    failure_videos = sorted(list((video_dir / 'failure').glob('*.mp4')))[:args.num_failure]
    
    print(f"\nFound {len(success_videos)} success videos, {len(failure_videos)} failure videos")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process videos
    all_videos = [
        ('success', vid, idx+1) for idx, vid in enumerate(success_videos)
    ] + [
        ('failure', vid, idx+1) for idx, vid in enumerate(failure_videos)
    ]
    
    print("\n" + "="*60)
    print("Processing videos with batched NCut...")
    print("="*60)
    
    for status, video_path, video_idx in all_videos:
        print(f"\n[{status.upper()} Video {video_idx}] {video_path.name}")
        
        # Extract task description from filename
        filename = video_path.stem
        task_parts = filename.split('--task=')
        if len(task_parts) > 1:
            task_desc = task_parts[1].replace('_', ' ')
        else:
            task_desc = "pick up the black bowl and place it on the plate"
        
        print(f"  Task: {task_desc}")
        
        video_output_dir = os.path.join(args.output_dir, f"{status}_video_{video_idx}")
        
        try:
            process_video_batched(
                model, processor, video_path, task_desc,
                video_output_dir, device
            )
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Batched NCut visualization completed!")
    print("="*60)
    print(f"\nOutput directory: {args.output_dir}")


if __name__ == '__main__':
    main()

