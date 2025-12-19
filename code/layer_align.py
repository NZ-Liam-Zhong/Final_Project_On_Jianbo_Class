# %%

import os
# 设置使用第2块GPU（索引为1）
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from ncut_pytorch.predictor import NcutDinov3Predictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'monospace'

import glob
from PIL import Image

# 创建输出目录
output_dir = "./layer_align_outputs"
os.makedirs(output_dir, exist_ok=True)
print(f"图片将保存到: {os.path.abspath(output_dir)}")

image_root = "/mnt/disk1/ilykyleliam/liam/ncut_tutorial_demo/images"
images = glob.glob(os.path.join(image_root, "**", "*.*p*g"), recursive=True)
images = sorted(images)

print(f"共找到 {len(images)} 张图片：")
for im_path in images[:5]:
    print(im_path)

images = [Image.open(img).convert("RGB") for img in images]


# %%

# load the model
predictor = NcutDinov3Predictor(
    input_size=(512, 512),
    model_cfg="dinov3_vitl16",
)
predictor = predictor.to('cuda')
model = predictor.model.model
transform = predictor.transform
L = len(model.blocks)


# %%

# Load SAM model for feature visualization
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import Sam

URL_DICT = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

model_cfg = 'vit_b'
statedict = torch.hub.load_state_dict_from_url(URL_DICT[model_cfg], map_location='cuda')
sam_model: Sam = sam_model_registry[model_cfg]()
sam_model.load_state_dict(statedict)
sam_model = sam_model.eval()
sam_model = sam_model.to('cuda')
print(f"SAM model loaded: {model_cfg}")


# %%

# Extract SAM features (multi-layer)
@torch.no_grad()
def extract_sam_features_multilayer(sam_model, images, batch_size=8):
    all_layers_features = []
    n_layers = len(sam_model.image_encoder.blocks)
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        # Prepare images for SAM (1024x1024)
        batch_tensors = []
        for img in batch_images:
            img_array = np.array(img.resize((1024, 1024)))
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            # SAM uses pixel mean/std normalization
            pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1) / 255.0
            pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) / 255.0
            img_tensor = (img_tensor - pixel_mean) / pixel_std
            batch_tensors.append(img_tensor)
        
        batch_tensors = torch.stack(batch_tensors).to('cuda')
        
        # Extract multi-layer features
        layer_features = []
        x = sam_model.image_encoder.patch_embed(batch_tensors)
        if sam_model.image_encoder.pos_embed is not None:
            x = x + sam_model.image_encoder.pos_embed
        
        for idx, blk in enumerate(sam_model.image_encoder.blocks):
            x = blk(x)
            # Handle different possible shapes
            if len(x.shape) == 4:  # (b, h, w, c)
                layer_feat = x.permute(0, 3, 1, 2)
            elif len(x.shape) == 3:  # (b, hw, c) or (b, c, hw)
                if x.shape[1] > x.shape[2]:  # (b, hw, c)
                    b, hw, c = x.shape
                    h = w = int(hw ** 0.5)
                    layer_feat = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (b, c, h, w)
                else:  # (b, c, hw)
                    b, c, hw = x.shape
                    h = w = int(hw ** 0.5)
                    layer_feat = x.reshape(b, c, h, w)
            else:
                raise ValueError(f"Unexpected shape at layer {idx}: {x.shape}")
            layer_features.append(layer_feat.cuda())
        
        # 注意：SAM 的 neck 会将通道数降到 256，无法与前面层堆叠，这里不再追加。
        
        all_layers_features.append(torch.stack(layer_features))  # (n_layers, b, c, h, w)
    
    all_layers_features = torch.cat(all_layers_features, dim=1)  # (n_layers, total_b, c, h, w)
    return all_layers_features

sam_features = extract_sam_features_multilayer(sam_model, images)
print(f"SAM multi-layer features shape: {sam_features.shape}")


# %%

# Reshape SAM features
from einops import rearrange
sam_l, sam_b, sam_c, sam_h, sam_w = sam_features.shape
sam_features_flat = rearrange(sam_features, 'l b c h w -> l (b h w) c')
print(f"SAM features flat shape: {sam_features_flat.shape}")


# %%

## SAM: sub-sample pixels using FPS
from ncut_pytorch.utils.sample import farthest_point_sampling
from ncut_pytorch import ncut_fn

def ncut_fps(features, n_eig=9, n_sample=1000):
    eigvecs, eigvals = ncut_fn(features, n_eig=n_eig)
    return farthest_point_sampling(eigvecs[:, 1:], n_sample=n_sample)

def ncut_fps_multiple_layers(feature_list, n_eig=9, n_sample=1000):
    pixel_fps_indices = []
    for feature in feature_list:
        pixel_fps_indices.append(ncut_fps(feature, n_eig=n_eig, n_sample=n_sample))
    pixel_fps_indices = np.concatenate(pixel_fps_indices, axis=0)
    pixel_fps_indices = np.unique(pixel_fps_indices)
    print(f"n unique pixel fps indices: {pixel_fps_indices.shape[0]}")
    random_draw = np.random.RandomState(42).choice(pixel_fps_indices, size=n_sample, replace=False)
    return random_draw

sam_feature_list = [sam_features_flat[0], sam_features_flat[-3], sam_features_flat[-1]]
sam_pixel_fps_indices = ncut_fps_multiple_layers(sam_feature_list, n_eig=9, n_sample=1000)

# show SAM pixel fps indices
sam_pixel_mask = np.zeros((sam_b*sam_h*sam_w))
sam_pixel_mask[sam_pixel_fps_indices] = 1
sam_pixel_mask = sam_pixel_mask.reshape(sam_b, sam_h, sam_w)
sam_pixel_mask = sam_pixel_mask.astype(np.uint8)
# show pixel mask
fig, axs = plt.subplots(2, 15, figsize=(15, 2))
for i in range(min(15, sam_b)):
    axs[0, i].imshow(images[i])
    axs[0, i].axis('off')
    axs[1, i].imshow(sam_pixel_mask[i])
    axs[1, i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_sam_pixel_mask.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"保存: 01_sam_pixel_mask.png")


# %%

## SAM: compute affinity features
from ncut_pytorch.utils.math import rbf_affinity
from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps
degree = 0.1

sam_affinity_features = []
sam_original_feature = []
for i_layer in range(sam_features_flat.shape[0]):
    _feature = sam_features_flat[i_layer]
    gamma = find_gamma_by_degree_after_fps(_feature, d_gamma=degree)
    print(f"SAM layer {i_layer}, gamma: {gamma}")
    affinity_feature = rbf_affinity(_feature, _feature[sam_pixel_fps_indices], gamma=gamma)
    sam_affinity_features.append(affinity_feature)
    sam_original_feature.append(_feature)
sam_affinity_features = torch.stack(sam_affinity_features, dim=0)
print(f"SAM affinity features shape: {sam_affinity_features.shape}")


# %%

# SAM: choose some layers and compute NCUT
sam_affinity_selected = sam_affinity_features[6:]  # 选择后面的层
sam_original_selected = sam_original_feature[6:]
sam_l_selected = len(sam_affinity_selected)
sam_p = sam_affinity_selected.shape[1]
sam_c = sam_affinity_selected.shape[2]
print(f"SAM selected layers: {sam_l_selected}")

sam_affinity_flat = rearrange(sam_affinity_selected, 'l p c -> (l p) c')
sam_original_flat = rearrange(sam_original_selected, 'l p c -> (l p) c')

sam_eigvecs, sam_eigvals = ncut_fn(sam_affinity_flat, n_eig=100)
sam_eigvecs_original, sam_eigvals_original = ncut_fn(sam_original_flat, n_eig=100)
print(f"SAM eigvecs shape: {sam_eigvecs.shape}")


# %%

# SAM: color the eigenvectors
from ncut_pytorch.color import mspace_color


def safe_mspace_color(features, *, n_dim=3, training_steps=1000, description=""):
    try:
        return mspace_color(features, n_dim=n_dim, training_steps=training_steps)
    except torch._C._LinAlgError as err:
        print(f"[WARN] mspace_color 在 {description or 'unknown'} 上 SVD 失败，将采用简单归一化颜色映射。具体错误: {err}")
    except RuntimeError as err:
        if "svd" not in str(err).lower():
            raise
        print(f"[WARN] mspace_color 在 {description or 'unknown'} 上 SVD 失败，将采用简单归一化颜色映射。具体错误: {err}")

    with torch.no_grad():
        vecs = features[:, :n_dim].float().clone()
        if vecs.shape[1] < n_dim:
            pad = torch.zeros(vecs.shape[0], n_dim - vecs.shape[1], device=vecs.device)
            vecs = torch.cat([vecs, pad], dim=1)
        vecs = torch.nn.functional.normalize(vecs, dim=1)
        vecs = vecs.nan_to_num(0.0)
        vecs = vecs - vecs.min(dim=0, keepdim=True).values
        denom = vecs.max(dim=0, keepdim=True).values.clamp_min(1e-6)
        vecs = vecs / denom
        return vecs


sam_rgb = safe_mspace_color(sam_eigvecs[:, :20], n_dim=3, training_steps=1000, description="SAM affinity")
sam_rgb_original = safe_mspace_color(sam_eigvecs_original[:, :20], n_dim=3, training_steps=1000, description="SAM original")


# %%

# SAM: plot affinity features
def plot_sam_rgb(rgb, title_prefix='SAM'):
    _rgb = rearrange(rgb, '(l p) c -> l p c', l=sam_l_selected, p=sam_p)
    _rgb = rearrange(_rgb, 'l (b h w) c -> l b h w c', l=sam_l_selected, b=sam_b, h=sam_h, w=sam_w)
    s = 2
    n_rows = sam_l_selected + 1
    n_cols = min(15, sam_b)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(s * n_cols, s * n_rows))
    for j in range(n_cols):
        ax = axes[0, j]
        ax.imshow(images[j])
        ax.axis('off')
        for i in range(sam_l_selected):
            ax = axes[i+1, j]
            ax.imshow(_rgb[i, j])
            ax.axis('off')
    for i in range(sam_l_selected):
        axes[i+1, 0].text(-0.1, 0.5, f'layer {i+6}', fontsize=16, rotation=0, ha='right', va='center', transform=axes[i+1, 0].transAxes)
    fig.tight_layout()
    return fig

fig = plot_sam_rgb(sam_rgb, 'SAM Affinity')
fig.suptitle(f'SAM layer aligned (affinity features) ncut + mspace color', fontsize=24, y=1.01)
fig.tight_layout()
plt.savefig(os.path.join(output_dir, '02_sam_affinity_ncut.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"保存: 02_sam_affinity_ncut.png")

fig = plot_sam_rgb(sam_rgb_original, 'SAM Original')
fig.suptitle(f'SAM layer original ncut mspace color', fontsize=24, y=1.01)
fig.tight_layout()
plt.savefig(os.path.join(output_dir, '02_sam_original_ncut.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"保存: 02_sam_original_ncut.png")


# %%

@torch.no_grad()
def extract_features(model, transform, images, batch_size=8):
    device = next(model.parameters()).device
    images = [transform(image) for image in images]
    images = torch.stack(images).to(device)
    n_layers = len(model.blocks)
    features = []
    for i in range(0, images.shape[0], batch_size):
        chunk = images[i:i+batch_size].to(device, dtype=torch.float32)
        feature = model.get_intermediate_layers(chunk, n=n_layers, reshape=True)
        feature = torch.stack(feature) # l, b, c, h, w
        feature = feature.to('cpu')
        features.append(feature)
    features = torch.cat(features, dim=1) # l, b, c, h, w
    return features


# %%

features = extract_features(model, transform, images)

# %%

l, b, c, h, w = features.shape
print(features.shape)

# %%

features = rearrange(features, 'l b c h w -> l (b h w) c')
print(features.shape)


# %%

## sub-sample the pixels
feature_list = [features[0], features[-10], features[-1]]
pixel_fps_indices = ncut_fps_multiple_layers(feature_list, n_eig=9, n_sample=1000)


# %%

# show pixel fps indices
pixel_mask = np.zeros((b*h*w))
pixel_mask[pixel_fps_indices] = 1
pixel_mask = pixel_mask.reshape(b, h, w)
pixel_mask = pixel_mask.astype(np.uint8)
# show pixel mask
n_cols = min(15, len(images))
fig, axs = plt.subplots(2, 15, figsize=(15, 2))
for i in range(n_cols):
    axs[0, i].imshow(images[i])
    axs[0, i].axis('off')
    axs[1, i].imshow(pixel_mask[i])
    axs[1, i].axis('off')
for i in range(n_cols, 15):
    axs[0, i].axis('off')
    axs[1, i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_dinov3_pixel_mask.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"保存: 03_dinov3_pixel_mask.png")


# %%

## compute self affinity features
affinity_features = []
original_feature = []
for i_layer in range(features.shape[0]):
    _feature = features[i_layer]
    gamma = find_gamma_by_degree_after_fps(_feature, d_gamma=degree)
    print(f"layer {i_layer}, gamma: {gamma}")
    affinity_feature = rbf_affinity(_feature, _feature[pixel_fps_indices], gamma=gamma)
    affinity_features.append(affinity_feature)
    original_feature.append(_feature)
affinity_features = torch.stack(affinity_features, dim=0)
print(affinity_features.shape)




# %%

# choose some layers
dino_layer_start = 6
dino_layer_end = min(12, affinity_features.shape[0])
affinity_features = affinity_features[dino_layer_start:dino_layer_end]
original_feature = original_feature[dino_layer_start:dino_layer_end]
l, p, c = affinity_features.shape
print(l)
affinity_flat = rearrange(affinity_features, 'l p c -> (l p) c')
original_flat = rearrange(original_feature, 'l p c -> (l p) c')
eigvecs, eigvals = ncut_fn(affinity_flat, n_eig=100)
eigvecs_original, eigvals_original = ncut_fn(original_flat, n_eig=100)
print(eigvecs.shape)



# %%

rgb = safe_mspace_color(eigvecs[:, :20], n_dim=3, training_steps=1000, description="DINO affinity")
rgb_original = safe_mspace_color(eigvecs_original[:, :20], n_dim=3, training_steps=1000, description="DINO original")



# %%

def plot_rgb(rgb, layer_offset):
    _rgb = rearrange(rgb, '(l p) c -> l p c', l=l, p=p)
    _rgb = rearrange(_rgb, 'l (b h w) c -> l b h w c', l=l, b=b, h=h, w=w)
    s = 2
    n_rows = l + 1
    n_cols = max(1, min(15, len(images)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(s * n_cols, s * n_rows))
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    for j in range(n_cols):
        ax = axes[0, j]
        ax.imshow(images[j])
        ax.axis('off')
        for i in range(l):
            ax = axes[i+1, j]
            ax.imshow(_rgb[i, j])
            ax.axis('off')
    for i in range(l):
        axes[i+1, 0].text(-0.1, 0.5, f'layer {layer_offset + i}', fontsize=16, rotation=0, ha='right', va='center', transform=axes[i+1, 0].transAxes)
    fig.tight_layout()
    return fig




# %%

fig = plot_rgb(rgb, dino_layer_start)
fig.suptitle(f'DINOv3 layer aligned ncut mspace color', fontsize=28, y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(output_dir, '04_dinov3_affinity_ncut.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"保存: 04_dinov3_affinity_ncut.png")



# %%

def plot_rgb_original(rgb_original, layer_offset):
    _rgb = rearrange(rgb_original, '(l p) c -> l p c', l=l, p=p)
    _rgb = rearrange(_rgb, 'l (b h w) c -> l b h w c', l=l, b=b, h=h, w=w)
    s = 2
    n_rows = l + 1
    n_cols = max(1, min(15, len(images)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(s * n_cols, s * n_rows))
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    for j in range(n_cols):
        ax = axes[0, j]
        ax.imshow(images[j])
        ax.axis('off')
        for i in range(l):
            ax = axes[i+1, j]
            ax.imshow(_rgb[i, j])
            ax.axis('off')
    for i in range(l):
        axes[i+1, 0].text(-0.1, 0.5, f'layer {layer_offset + i}', fontsize=16, rotation=0, ha='right', va='center', transform=axes[i+1, 0].transAxes)
    fig.tight_layout()
    return fig


# %%

fig = plot_rgb_original(rgb_original, dino_layer_start)
fig.suptitle(f'DINOv3 layer original ncut mspace color', fontsize=28, y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(output_dir, '05_dinov3_original_ncut.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"保存: 05_dinov3_original_ncut.png")

print(f"\n所有图片已保存到: {os.path.abspath(output_dir)}")