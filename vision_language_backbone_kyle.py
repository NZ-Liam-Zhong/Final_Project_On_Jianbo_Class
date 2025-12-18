import argparse
import html
import os
import random
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from io import BytesIO
import imageio
from matplotlib.font_manager import FontProperties

# Use non-interactive backend for faster rendering
import matplotlib
matplotlib.use('Agg')


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_frames_as_video_fast(frames: List[np.ndarray], output_path: str, fps: int = 10, use_gpu: bool = True) -> None:
    """Save frames as MP4 video using ffmpeg for fast processing.
    
    Attempts GPU encoding (NVENC) first, falls back to CPU (libx264).
    Falls back to GIF via imageio if ffmpeg is not available.
    """
    if not frames:
        return
    
    output_path = str(output_path)
    # Force .mp4 extension
    if output_path.lower().endswith('.gif'):
        output_path = output_path[:-4] + '.mp4'
    elif not output_path.lower().endswith('.mp4'):
        output_path = output_path + '.mp4'
    
    height, width = frames[0].shape[:2]
    # Ensure dimensions are even (required for most video codecs)
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    
    # Try ffmpeg with GPU encoding first, then CPU
    encoders = []
    if use_gpu:
        # NVIDIA GPU encoding (much faster)
        encoders.append(('h264_nvenc', ['-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', '23']))
    # CPU encoding (always available with ffmpeg)
    encoders.append(('libx264', ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']))
    
    for encoder_name, encoder_args in encoders:
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-',  # Read from stdin
                *encoder_args,
                '-pix_fmt', 'yuv420p',  # Compatibility
                output_path
            ]
            
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Write frames directly to ffmpeg stdin
            for frame in frames:
                # Crop to even dimensions if needed
                cropped = frame[:height, :width, :]
                proc.stdin.write(cropped.tobytes())
            
            proc.stdin.close()
            proc.wait()
            
            if proc.returncode == 0:
                print(f"  [{encoder_name}] Saved video: {output_path}")
                return
            else:
                stderr = proc.stderr.read().decode()
                if 'nvenc' in stderr.lower() or 'cuda' in stderr.lower():
                    # GPU not available, try next encoder
                    continue
                    
        except (subprocess.CalledProcessError, FileNotFoundError, BrokenPipeError) as e:
            continue
    
    # Final fallback: imageio (slower, creates GIF)
    print(f"  [ffmpeg unavailable, falling back to imageio GIF]")
    gif_path = output_path.replace('.mp4', '.gif')
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"  [imageio] Saved GIF: {gif_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize language/vision backbone layers with batch NCut coloring.")
    parser.add_argument("--images", type=str, default=str(Path(__file__).parent / "images"), help="Directory of images to load (jpg/png).")
    parser.add_argument("--max-images", type=int, default=4, help="Max number of images to load for a batch after sampling.")
    parser.add_argument("--frame-start", type=int, default=0, help="Index (0-based) of the first frame to include after sorting.")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame (2 means every other frame).")
    parser.add_argument("--prompt", type=str, default="What does this image show?", help="User prompt to pair with each image.")
    parser.add_argument("--openvla-model-id", type=str, default="openvla/openvla-7b", help="HF model id for OpenVLA (vision-language).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--device-map", type=str, default="", help="HuggingFace device_map strategy, e.g. 'auto' or 'balanced'.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Torch dtype for model.")
    parser.add_argument("--resize-long-edge", type=int, default=0, help="If >0, resize each image so its longer side is at most this many pixels.")
    parser.add_argument("--crop-box", type=str, default="", help="Optional crop box after resize as 'left,top,right,bottom'. Values can be pixels or percentages (e.g., '10%').")
    parser.add_argument("--layer", type=int, default=None, help="Shared layer index for both language and vision (-1 for last).")
    parser.add_argument("--lang-layer", type=int, default=10, help="DEPRECATED: If --layer is set, this is ignored.")
    parser.add_argument("--vision-layer", type=int, default=10, help="DEPRECATED: If --layer is set, this is ignored.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for processing.")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / "viz_outputs"), help="Directory to save visualizations.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--n-segments", type=int, default=5, help="Number of NCut clusters for vision channel attribution.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k channels to visualize per cluster.")
    parser.add_argument("--gif-out", type=str, default="", help="If set, write a GIF with per-image composites to this path.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for GIF output.")
    parser.add_argument("--gif-chunk-size", type=int, default=250, help="Max frames per GIF before rolling over (0=single file).")
    parser.add_argument("--skip-clusters", action="store_true", help="Skip cluster visualization (much faster, only generates main video).")
    parser.add_argument("--mirror-output-root", type=str, default="", help="If set, mirror the input images path hierarchy under this root for outputs. Creates: root/[relative_path]/ncut.mp4, html/, png/")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


@dataclass
class Batch:
    images: List[Image.Image]
    prompts: List[str]


@dataclass(frozen=True)
class CropValue:
    value: float
    is_percent: bool


def parse_crop_box(crop_str: str) -> Optional[Tuple[CropValue, CropValue, CropValue, CropValue]]:
    if not crop_str:
        return None
    parts = [p.strip() for p in crop_str.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--crop-box must have four comma-separated values: left,top,right,bottom")
    values: List[CropValue] = []
    for part in parts:
        is_percent = part.endswith("%")
        numeric = part[:-1] if is_percent else part
        try:
            val = float(numeric)
        except ValueError as exc:
            raise ValueError(f"Invalid crop value '{part}'. Use numbers or percentages.") from exc
        if is_percent and not (0.0 <= val <= 100.0):
            raise ValueError(f"Percentage crop values must be between 0 and 100, got {val}%.")
        values.append(CropValue(value=val, is_percent=is_percent))
    return tuple(values)  # type: ignore[return-value]


def apply_crop_box(img: Image.Image, crop_spec: Tuple[CropValue, CropValue, CropValue, CropValue]) -> Image.Image:
    width, height = img.size
    coords: List[int] = []
    for idx, spec in enumerate(crop_spec):
        base = width if idx in (0, 2) else height
        raw = spec.value / 100.0 * base if spec.is_percent else spec.value
        coord = int(round(raw))
        if idx in (0, 2):
            coord = max(0, min(width, coord))
        else:
            coord = max(0, min(height, coord))
        coords.append(coord)
    left, top, right, bottom = coords
    if right - left < 4 or bottom - top < 4:
        raise ValueError("Crop box is too small or inverted; ensure right > left and bottom > top.")
    return img.crop((left, top, right, bottom))


def load_images(
    image_dir: str,
    max_images: int,
    frame_start: int = 0,
    frame_step: int = 1,
    resize_long_edge: int = 0,
    crop_spec: Optional[Tuple[CropValue, CropValue, CropValue, CropValue]] = None,
) -> List[Image.Image]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in sorted(Path(image_dir).glob("*")) if p.suffix.lower() in exts]
    if frame_start > 0:
        paths = paths[frame_start:]
    if frame_step > 1:
        paths = paths[::frame_step]
    paths = paths[:max_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    images: List[Image.Image] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if resize_long_edge and resize_long_edge > 0:
            w, h = img.size
            long_edge = max(w, h)
            if long_edge > resize_long_edge:
                scale = resize_long_edge / float(long_edge)
                new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                img = img.resize(new_size, Image.LANCZOS)
        if crop_spec is not None:
            img = apply_crop_box(img, crop_spec)
        images.append(img)
    return images


def build_batch(images: List[Image.Image], prompt: str) -> Batch:
    prompts = []
    for _ in images:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompts.append(conversation)
    return Batch(images=images, prompts=prompts)


def color_from_ncut(features: torch.Tensor, n_eig: int = 10) -> np.ndarray:
    """Run NCut on features [num_tokens, hidden] and return RGB colors [num_tokens, 3].

    Always casts to float32 on CPU to avoid half-precision CUDA ops (e.g., cdist) not implemented.
    """
    try:
        from ncut_pytorch import ncut_fn, tsne_color
    except Exception as e:
        raise RuntimeError(
            "ncut_pytorch is required. Install via `pip install ncut-pytorch`"
        ) from e

    with torch.no_grad():
        f32 = features.detach().to(device="cpu", dtype=torch.float32)
        eigvecs, _ = ncut_fn(f32, n_eig=n_eig)
    colors = tsne_color(eigvecs)
    return colors.cpu().numpy()


def differentiable_kway_ncut(features_flat: torch.Tensor, n_segment: int):
    """Run NCut with gradient tracking on CPU float32.

    features_flat: [N, C] requires_grad=True will be honored.
    Returns (eigvec, kway_eigvec) both tensors on CPU.
    """
    from ncut_pytorch import ncut_fn, kway_ncut
    f32 = features_flat.to(device="cpu", dtype=torch.float32)
    eigvec, _ = ncut_fn(f32, n_eig=n_segment, track_grad=True)
    kway_eigvec = kway_ncut(eigvec)
    return eigvec, kway_eigvec


def channel_gradient_from_cluster(features_flat: torch.Tensor, cluster_mask_flat: torch.Tensor, kway_eigvec: torch.Tensor, cluster_idx: int) -> torch.Tensor:
    """Compute average channel gradient for a specific cluster index.

    features_flat: [N, C] with requires_grad=True (CPU float32)
    cluster_mask_flat: [N] bool tensor (CPU)
    returns grad: [C] (CPU)
    """
    assert features_flat.requires_grad is True
    if features_flat.grad is not None:
        features_flat.grad.zero_()
    # ensure mask device matches features
    cluster_mask_flat = cluster_mask_flat.to(device=features_flat.device)
    loss = - kway_eigvec[cluster_mask_flat, cluster_idx].abs().mean()
    loss.backward(retain_graph=True)
    grad = features_flat.grad[cluster_mask_flat].mean(0)
    return grad.detach()


def html_color_tokens(tokens: List[str], colors: np.ndarray, special_tokens: List[str]) -> str:
    special = set(special_tokens)
    spans: List[str] = []
    for tok, rgb in zip(tokens, colors):
        if tok in special:
            continue
        r, g, b = (int(max(0, min(1, ch)) * 255) for ch in rgb)
        # LLaMA tokenizers often use U+2581 as whitespace marker, keep it visible but safe
        safe = html.escape(tok)
        spans.append(f'<span style="background-color: #{r:02x}{g:02x}{b:02x}">{safe}</span>')
    return "".join(spans)


def draw_language_tokens_panel(
    ax,
    tokens: List[str],
    colors: np.ndarray,
    special_tokens: List[str],
    title: str = "language tokens",
    max_tokens: int = 60,
) -> None:
    ax.set_title(title)
    ax.axis("off")

    if tokens is None:
        tokens = []
    if colors is None or len(colors) == 0:
        colors = np.zeros((0, 3), dtype=np.float32)

    display_tokens: List[str] = []
    display_colors: List[np.ndarray] = []
    for tok, color in zip(tokens, colors):
        if tok in special_tokens:
            continue
        display_tokens.append(tok)
        display_colors.append(color)
        if len(display_tokens) >= max_tokens:
            break

    if not display_tokens:
        ax.text(0.5, 0.5, "(no tokens)", ha="center", va="center", fontsize=10, alpha=0.7)
        return

    fig = ax.figure
    renderer = None
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None

    font_props = FontProperties(size=11)
    token_pad_px = 4.0
    token_gap_px = 4.0
    line_spacing_scale = 1.05

    margin_left = 0.02
    margin_right = 0.02
    margin_bottom = 0.02
    y_cursor = 0.96
    x_cursor = margin_left
    gap_x = 0.006
    max_width = 1.0 - margin_right

    if renderer is not None:
        ax_bbox = ax.get_window_extent(renderer)
        ax_width = max(ax_bbox.width, 1.0)
        ax_height = max(ax_bbox.height, 1.0)
        sample_width, sample_height, _ = renderer.get_text_width_height_descent("Ag", font_props, ismath=False)
        pad_px = token_pad_px
        gap_x = max((token_gap_px) / ax_width, 0.001)
        line_height = ((sample_height + pad_px * 2.0) / ax_height) * line_spacing_scale
    else:
        pad_px = token_pad_px
        line_height = 0.07

    for tok, color in zip(display_tokens, display_colors):
        text = tok
        color_tuple = tuple(np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0))

        if renderer is not None:
            width_px, height_px, _ = renderer.get_text_width_height_descent(text, font_props, ismath=False)
            token_width = (width_px + pad_px * 2.0) / ax_width
            token_height = (height_px + pad_px * 2.0) / ax_height
        else:
            token_width = 0.025 * max(len(text), 1)
            token_height = 0.06

        if x_cursor + token_width > max_width:
            x_cursor = margin_left
            y_cursor -= line_height

        if y_cursor - token_height < margin_bottom:
            ax.text(0.5, margin_bottom, "…", ha="center", va="bottom", fontsize=12, transform=ax.transAxes)
            break

        ax.text(
            x_cursor,
            y_cursor,
            text,
            transform=ax.transAxes,
            fontproperties=font_props,
            fontsize=font_props.get_size_in_points(),
            color="black",
            ha="left",
            va="top",
            bbox=dict(
                facecolor=color_tuple,
                edgecolor="none",
                alpha=0.88,
                boxstyle="round,pad=0.3",
            ),
        )
        x_cursor += token_width + gap_x


def render_cluster_row(
    axes,
    image,
    base_rgb: Optional[np.ndarray],
    tokens: List[str],
    token_colors: np.ndarray,
    special_tokens: List[str],
    mask: Optional[np.ndarray],
    cluster_idx: int,
    eig_entries: List[Optional[Tuple[int, np.ndarray]]],
) -> None:
    if isinstance(axes, np.ndarray):
        ax_list = list(np.ravel(axes))
    elif isinstance(axes, (list, tuple)):
        ax_list = list(axes)
    else:
        ax_list = [axes]

    if not ax_list:
        return

    def _to_rgb_array(img):
        if img is None:
            return None
        try:
            arr = np.array(img)
        except Exception:
            return None
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    image_arr = _to_rgb_array(image)
    mask_arr = None
    if mask is not None:
        mask_arr = mask if isinstance(mask, np.ndarray) else np.array(mask)

    # Image
    img_ax = ax_list[0]
    img_ax.set_title("image")
    img_ax.axis("off")
    if image_arr is not None:
        img_ax.imshow(image_arr)
    else:
        fallback_shape = (mask_arr.shape[0], mask_arr.shape[1]) if mask_arr is not None else (64, 64)
        img_ax.imshow(np.zeros((*fallback_shape, 3), dtype=np.float32))

    # NCut base
    base_ax = ax_list[1] if len(ax_list) > 1 else None
    if base_ax is not None:
        base_ax.set_title("NCut base")
        base_ax.axis("off")
        if isinstance(base_rgb, np.ndarray) and base_rgb.size > 0:
            base_ax.imshow(np.clip(base_rgb, 0.0, 1.0))
        else:
            base_ax.imshow(np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.float32) if mask_arr is not None else np.zeros((64, 64, 3)))

    # Language tokens
    if len(ax_list) > 2:
        draw_language_tokens_panel(
            ax_list[2],
            tokens or [],
            token_colors,
            special_tokens,
            title="language tokens",
        )

    # Cluster mask
    if len(ax_list) > 3:
        mask_ax = ax_list[3]
        mask_ax.set_title(f"cluster {cluster_idx}")
        mask_ax.axis("off")
        if mask_arr is not None:
            mask_ax.imshow(mask_arr, cmap="viridis")
        else:
            mask_ax.imshow(np.zeros((64, 64), dtype=np.float32), cmap="viridis")

    # Eigenmaps
    start_idx = 4
    for idx, entry in enumerate(eig_entries):
        ax_idx = start_idx + idx
        if ax_idx >= len(ax_list):
            break
        eig_ax = ax_list[ax_idx]
        eig_ax.axis("off")
        if entry is None:
            eig_ax.set_title("")
            continue
        eig_idx, eig_map = entry
        eig_ax.imshow(np.clip(eig_map, 0.0, 1.0), cmap="viridis")
        eig_ax.set_title(f"eig {eig_idx}")

    # Any remaining axes
    for ax in ax_list[start_idx + len(eig_entries):]:
        ax.axis("off")


# -------------------- OpenVLA helpers --------------------
def ensure_hf_cache_env() -> None:
    # Default to user's shared cache if available to avoid scattering downloads
    os.environ.setdefault("HF_HOME", "/mnt/disk1/ilykyleliam/public")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/disk1/ilykyleliam/public")


def openvla_load(model_id: str, device: Optional[torch.device], dtype: torch.dtype, device_map: Optional[str] = None):
    ensure_hf_cache_env()
    from transformers import AutoProcessor, AutoModelForVision2Seq

    # Try to load from the provided id/path. If the id is not a valid HF model
    # identifier (e.g. user provided a short name but the repo exists only in
    # the local cache), attempt to find a local checkout under HF_HOME and
    # load from there.
    hf_home = os.environ.get("HF_HOME", "")
    local_path: Optional[str] = None

    # If model_id is an explicit local directory, prefer it.
    if os.path.isdir(model_id):
        local_path = model_id

    processor = None
    # First attempt: try loading using the model_id directly (this will hit
    # the Hub or a local path if model_id points to one).
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        # If that fails, try to locate a matching repo dir inside HF_HOME
        # (e.g., models--owner--repo style directories). We search for any
        # directory name that contains the model_id string or ends with it.
        if hf_home and os.path.isdir(hf_home):
            for root, dirs, _ in os.walk(hf_home):
                for d in dirs:
                    if model_id in d or d.endswith(model_id):
                        cand = os.path.join(root, d)
                        if os.path.isdir(cand):
                            # found a candidate; prefer ones beginning with models--
                            local_path = cand
                            break
                if local_path:
                    break
        if local_path is None:
            # re-raise the original error to surface the failure
            raise

        # Transformers expects config files at the repo root (or a snapshot
        # folder). If the cached repo dir only contains a `snapshots/<sha>/`
        # subdir (common for HF local cache), prefer that snapshot directory
        # which contains `config.json`.
        def _find_snapshot_with_config(base_dir: str) -> Optional[str]:
            cfg = os.path.join(base_dir, "config.json")
            if os.path.isfile(cfg):
                return base_dir
            snaps = os.path.join(base_dir, "snapshots")
            if os.path.isdir(snaps):
                for entry in sorted(os.listdir(snaps)):
                    cand = os.path.join(snaps, entry)
                    if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
                        return cand
            return None

        snapshot_candidate = _find_snapshot_with_config(local_path)
        if snapshot_candidate:
            local_path = snapshot_candidate

        # load processor from the resolved local path
        processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True, local_files_only=True)

    # Now load the model weights. Use the discovered local_path if present,
    # otherwise load from model_id (which may be a HF repo id).
    model_source = local_path if local_path is not None else model_id
    # Log where we're loading from for easier debugging
    try:
        print(f"[openvla_load] loading model from: {model_source}")
    except Exception:
        pass

    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if device_map:
        load_kwargs["device_map"] = device_map

    # If we're using a local path, ensure we only load local files.
    local_only = local_path is not None
    if local_only:
        model = AutoModelForVision2Seq.from_pretrained(model_source, local_files_only=True, **load_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_source, **load_kwargs)

    if not device_map and device is not None:
        model.to(device)
    model.eval()
    return processor, model


def openvla_get_vision_tokens(model, inputs, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        pixel_values = inputs["pixel_values"] if isinstance(inputs, dict) else inputs.pixel_values
        pixel_values = pixel_values.to(device, dtype=dtype)
        # Heuristic: OpenVLA is PrismaticVLM-like; try common attributes
        if hasattr(model, "vision_backbone"):
            vision_features = model.vision_backbone(pixel_values)
        else:
            # Fallback to encoder/backbone candidates
            candidates = [a for a in dir(model) if ("vision" in a.lower() or "encoder" in a.lower()) and not a.startswith("_")]
            for name in candidates:
                try:
                    comp = getattr(model, name)
                    vision_features = comp(pixel_values)
                    break
                except Exception:
                    continue
            else:
                raise ValueError("OpenVLA vision component not found")

        if hasattr(vision_features, "last_hidden_state"):
            tokens = vision_features.last_hidden_state  # [B, N, C]
        elif isinstance(vision_features, torch.Tensor):
            tokens = vision_features
        elif isinstance(vision_features, (tuple, list)):
            tokens = vision_features[0]
        elif hasattr(vision_features, "hidden_states"):
            tokens = vision_features.hidden_states[-1]
        else:
            raise ValueError(f"Unexpected vision feature type: {type(vision_features)}")
    return tokens  # [B, N, C] or [N, C]


def openvla_get_language_layers(model, inputs) -> List[torch.Tensor]:
    """Attempt to retrieve language hidden states layers for OpenVLA.

    Tries generate(..., output_hidden_states=True) first; falls back to forward().
    Returns a tuple/list where each entry is [B, T, H] tensor for a layer.
    """
    with torch.no_grad():
        # ensure pixel dtype matches model
        if inputs.get("pixel_values") is not None:
            target_dtype = next(model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(model.device, dtype=target_dtype)
        try:
            gen_out = model.generate(
                input_ids=inputs.get("input_ids"),
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            if getattr(gen_out, "hidden_states", None):
                hs = gen_out.hidden_states[0]
                if hs is not None:
                    return hs
        except Exception:
            pass

        fwd = model(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
        for key in ["hidden_states", "decoder_hidden_states"]:
            if hasattr(fwd, key) and getattr(fwd, key) is not None:
                return getattr(fwd, key)
        if hasattr(fwd, "language_model_output") and getattr(fwd, "language_model_output") is not None:
            lmo = getattr(fwd, "language_model_output")
            if hasattr(lmo, "hidden_states") and lmo.hidden_states is not None:
                return lmo.hidden_states
    raise RuntimeError("Could not retrieve OpenVLA language hidden states.")



def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if args.frame_start < 0:
        raise ValueError("--frame-start must be >= 0")
    crop_spec = parse_crop_box(args.crop_box) if args.crop_box else None
    seed_everything(args.seed)

    # Handle mirrored output structure
    mp4_output_dir = args.output_dir
    html_output_dir = args.output_dir
    png_output_dir = args.output_dir
    
    if args.mirror_output_root:
        # Extract relative path from images input (strip the Ily_Dataset base and front_rgb... suffix)
        images_path = Path(args.images).resolve()
        # Try to find common dataset markers to extract relative path
        images_str = str(images_path)
        
        # Look for "labelled - " pattern to extract hierarchy
        # e.g., /mnt/disk1/.../labelled - Cheez/labelled - 100_432/right/success - 0/front_rgb...
        # becomes: labelled - Cheez/labelled - 100_432/right/success - 0
        relative_parts = []
        parts = images_path.parts
        capture = False
        for part in parts:
            if part.startswith("labelled - ") or part.startswith("labelled -") or capture:
                if part.startswith("front_rgb") or part.endswith("_extracted"):
                    break
                relative_parts.append(part)
                capture = True
        
        if relative_parts:
            relative_path = Path(*relative_parts)
            mirror_base = Path(args.mirror_output_root) / relative_path
        else:
            # Fallback: use last few directory components
            mirror_base = Path(args.mirror_output_root) / images_path.parent.name
        
        mp4_output_dir = str(mirror_base)
        html_output_dir = str(mirror_base / "html")
        png_output_dir = str(mirror_base / "png")
        
        print(f"Mirror output mode enabled:")
        print(f"  MP4 output: {mp4_output_dir}")
        print(f"  HTML output: {html_output_dir}")
        print(f"  PNG output: {png_output_dir}")
    
    os.makedirs(mp4_output_dir, exist_ok=True)
    os.makedirs(html_output_dir, exist_ok=True)
    os.makedirs(png_output_dir, exist_ok=True)

    dtype = get_dtype(args.dtype)
    device_arg = args.device.strip()
    visible_devices: List[str] = []
    primary_device_str = device_arg

    if "," in device_arg:
        raw_devices = [d.strip() for d in device_arg.split(",") if d.strip()]
        if not raw_devices:
            raise ValueError("No valid devices provided via --device.")
        normalized: List[str] = []
        for entry in raw_devices:
            if entry.startswith("cuda") or entry.startswith("cpu"):
                normalized.append(entry)
            elif entry.isdigit():
                normalized.append(f"cuda:{entry}")
            else:
                raise ValueError(f"Unsupported device format: {entry}")
        visible_devices = normalized
        primary_device_str = visible_devices[0]
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
    elif device_arg == "":
        primary_device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    primary_device = torch.device(primary_device_str)

    device_map = args.device_map.strip()
    if not device_map and len(visible_devices) > 1:
        device_map = "auto"

    if primary_device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    # OpenVLA backend
    from transformers import set_seed as hf_set_seed
    hf_set_seed(args.seed)
    processor, model = openvla_load(
        args.openvla_model_id,
        None if device_map else primary_device,
        dtype,
        device_map or None,
    )
    images = load_images(
        args.images,
        args.max_images,
        frame_start=args.frame_start,
        frame_step=args.frame_step,
        resize_long_edge=args.resize_long_edge,
        crop_spec=crop_spec,
    )
    proc_inputs = [processor(args.prompt, img, return_tensors="pt") for img in images]
    if len(proc_inputs) == 0:
        raise RuntimeError("No processor inputs were generated for the provided images.")
    keys = set().union(*[set(pi.keys()) for pi in proc_inputs])
    batch_size = max(1, args.batch_size)

    tok = getattr(processor, "tokenizer", processor)
    special_tokens_list = list(getattr(tok, "all_special_tokens", []))

    lang_features_per_sample: List[torch.Tensor] = []
    lang_text_per_sample: List[List[str]] = []
    prompt_colors_per_sample: List[np.ndarray] = []
    vision_token_pool: List[torch.Tensor] = []
    vision_token_counts: List[int] = []
    per_sample_base_maps: List[torch.Tensor] = []  # [C, h, w] or empty
    per_sample_base_colors: List[np.ndarray] = []
    per_sample_highres_maps: List[torch.Tensor] = []  # [C, H', W'] or empty
    lang_layer_idx: Optional[int] = None
    total_lang_layers: Optional[int] = None
    grid_shape: Optional[Tuple[int, int]] = None
    vision_title_layer = None

    for start in range(0, len(proc_inputs), batch_size):
        end = min(start + batch_size, len(proc_inputs))
        if start >= end:
            continue
        batch_indices = range(start, end)
        batch_inputs = {}
        for k in keys:
            tensors = [proc_inputs[idx][k] for idx in batch_indices if k in proc_inputs[idx]]
            if tensors:
                batch_inputs[k] = torch.cat(tensors, dim=0).to(primary_device)

        lang_layers_batch = openvla_get_language_layers(model, batch_inputs)
        if lang_layer_idx is None:
            total_lang_layers = len(lang_layers_batch)
            if args.layer is not None:
                lang_layer_idx = (total_lang_layers - 1) if args.layer == -1 else args.layer
            else:
                lang_layer_idx = args.lang_layer if args.lang_layer != -1 else (total_lang_layers - 1)
            if not (0 <= lang_layer_idx < total_lang_layers):
                raise ValueError(
                    f"Resolved lang-layer index {lang_layer_idx} is out of range [0, {total_lang_layers - 1}]"
                )
        lang_features_layer = lang_layers_batch[lang_layer_idx]

        ids_batch = batch_inputs.get("input_ids")
        if ids_batch is None:
            raise RuntimeError("Processor inputs missing input_ids; cannot tokenize language features.")
        for local_idx, global_idx in enumerate(range(start, end)):
            ids_b = ids_batch[local_idx]
            feats_b = lang_features_layer[local_idx]
            tokens = tok.convert_ids_to_tokens(ids_b.tolist())
            m = min(len(tokens), feats_b.size(0))
            lang_features_per_sample.append(feats_b[:m].detach().cpu())
            lang_text_per_sample.append(tokens[:m])

        ovla_tokens_batch = openvla_get_vision_tokens(model, batch_inputs, dtype, primary_device)
        if ovla_tokens_batch.dim() == 2:
            ovla_tokens_batch = ovla_tokens_batch.unsqueeze(0)
        B, N, C = ovla_tokens_batch.shape
        if grid_shape is None:
            s = int(round(float(N) ** 0.5))
            grid_shape = (s, s) if s * s == N else None
        if vision_title_layer is None:
            vision_title_layer = "last"

        for local_idx in range(B):
            base_feat = ovla_tokens_batch[local_idx].detach().cpu()
            vision_token_pool.append(base_feat)
            vision_token_counts.append(base_feat.size(0))
            if grid_shape is not None:
                gh, gw = grid_shape
                base_map = base_feat.view(gh, gw, C).permute(2, 0, 1).contiguous()
            else:
                base_map = torch.empty(0, 0, 0)
            per_sample_base_maps.append(base_map)
            per_sample_highres_maps.append(torch.empty(0, 0, 0))

        for key in list(batch_inputs.keys()):
            batch_inputs[key] = batch_inputs[key].cpu()
        del lang_layers_batch, ovla_tokens_batch, batch_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if lang_layer_idx is None:
        raise RuntimeError("Failed to resolve language layer index; batching loop did not execute.")

    if len(lang_features_per_sample) == 0 or sum(x.size(0) for x in lang_features_per_sample) == 0:
        raise RuntimeError("No language tokens found after batching; check tokenizer special tokens.")

    total_samples = len(lang_text_per_sample)

    # Concatenate all language features across batch for batch-NCut
    lang_pool = torch.cat(lang_features_per_sample, dim=0)  # [N_lang_tokens, H]
    lang_colors_pool = color_from_ncut(lang_pool, n_eig=10)  # [N, 3]

    # Split back per sample for HTML coloring (input tokens)
    lang_offsets = np.cumsum([0] + [x.size(0) for x in lang_features_per_sample]).tolist()
    prompt_colors_per_sample.clear()
    for i in range(len(lang_text_per_sample)):
        start, end = lang_offsets[i], lang_offsets[i + 1]
        colors_i = lang_colors_pool[start:end]
        prompt_colors_per_sample.append(colors_i.copy())

    # ----- Vision features (target layer) -----
    if len(vision_token_pool) == 0:
        raise RuntimeError("No vision tokens were collected; check model outputs.")

    vision_pool = torch.cat(vision_token_pool, dim=0)  # [N_vision_tokens, C]
    vision_colors_pool = color_from_ncut(vision_pool, n_eig=10)

    # Split back and render per-sample grids colored by NCut colors
    import matplotlib.pyplot as plt

    offset = 0
    # Determine output path for main video
    if args.mirror_output_root:
        # In mirror mode, put ncut.mp4 directly in mp4_output_dir
        out_gif_base = Path(mp4_output_dir) / "ncut.mp4" if args.gif_out else None
    else:
        out_gif_base = Path(args.gif_out) if args.gif_out else None
    
    # Collect all frames first, then batch save at the end (MUCH faster)
    main_gif_frames: List[np.ndarray] = []
    
    print(f"Rendering {len(per_sample_base_maps)} frames...")
    for i in range(len(per_sample_base_maps)):
        if i % 20 == 0:
            print(f"  Processing frame {i}/{len(per_sample_base_maps)}...")
        n_base = vision_token_counts[i] if i < len(vision_token_counts) else 0
        cols_base = vision_colors_pool[offset : offset + n_base]
        offset += n_base
        base_map = per_sample_base_maps[i]
        if base_map.numel() == 0 or n_base == 0:
            per_sample_base_colors.append(np.zeros((0, 0, 3), dtype=np.float32))
            continue
        h = base_map.size(1)
        w = base_map.size(2)
        base_rgb = cols_base.reshape(h, w, 3)
        per_sample_base_colors.append(base_rgb.copy())
        plt.figure(figsize=(5, 5))
        plt.imshow(base_rgb)
        plt.title(f"Vision base patches NCut colors (layer {vision_title_layer})")
        plt.axis("off")
        plt.tight_layout()
        out_img = Path(png_output_dir) / f"vision_base_layer_{vision_title_layer}_sample_{i}.png"
        plt.savefig(out_img, dpi=160)
        plt.close()

        # Compose side-by-side frame: original image + NCut base grid + language tokens with matching colors
        try:
            if out_gif_base is not None:
                fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
                axes[0].imshow(images[i])
                axes[0].set_title("image")
                axes[0].axis('off')
                axes[1].imshow(base_rgb)
                axes[1].set_title("NCut base")
                axes[1].axis('off')
                tokens_all = lang_text_per_sample[i] if i < len(lang_text_per_sample) else []
                colors_all = prompt_colors_per_sample[i] if i < len(prompt_colors_per_sample) else np.zeros((0, 3), dtype=np.float32)
                draw_language_tokens_panel(
                    axes[2],
                    tokens_all,
                    colors_all,
                    special_tokens_list,
                    title="language tokens",
                )

                plt.tight_layout()
                buf = BytesIO()
                # Reduced DPI for faster rendering (100 instead of 140)
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                frame = Image.open(buf).convert('RGB')
                main_gif_frames.append(np.array(frame))
                buf.close()
        except Exception as e:
            print(f"  Warning: Failed to create composite frame {i}: {e}")

        if per_sample_highres_maps[i].numel() > 0:
            high = per_sample_highres_maps[i]
            H, W = high.size(1), high.size(2)
            n_high = H * W
            cols_high = vision_colors_pool[offset : offset + n_high]
            offset += n_high
            high_rgb = cols_high.reshape(H, W, 3)
            # Empirical alignment tweak observed in anyres packing (roll by 4 pixels)
            try:
                import numpy as _np
                high_rgb = _np.roll(high_rgb, shift=-4, axis=1)
            except Exception:
                pass
            plt.figure(figsize=(5, 5))
            plt.imshow(high_rgb)
            plt.title(f"Vision high-res NCut colors (layer {vision_title_layer})")
            plt.axis("off")
            plt.tight_layout()
            out_hr = Path(png_output_dir) / f"vision_highres_layer_{vision_title_layer}_sample_{i}.png"
            plt.savefig(out_hr, dpi=160)
            plt.close()

    print(f"Frame rendering complete. Collected {len(main_gif_frames)} frames for video.")
    
    # Save main video FIRST (before cluster computation)
    if main_gif_frames and out_gif_base is not None:
        print(f"Saving main video with {len(main_gif_frames)} frames...")
        out_gif_base.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle chunking if requested
        if args.gif_chunk_size and args.gif_chunk_size > 0:
            for part_idx, start in enumerate(range(0, len(main_gif_frames), args.gif_chunk_size), 1):
                chunk = main_gif_frames[start:start + args.gif_chunk_size]
                name = f"{out_gif_base.stem}_part{part_idx:03d}.mp4"
                path = out_gif_base.parent / name
                save_frames_as_video_fast(chunk, str(path), fps=max(1, args.fps))
        else:
            save_frames_as_video_fast(main_gif_frames, str(out_gif_base), fps=max(1, args.fps))
        # Free memory
        del main_gif_frames
    elif out_gif_base is not None:
        print("No frames collected for main video.")
    
    print(f"Saving HTML language visualizations for {total_samples} samples...")
    
    for idx in range(total_samples):
        prompt_tokens_all = lang_text_per_sample[idx]
        prompt_colors = prompt_colors_per_sample[idx] if idx < len(prompt_colors_per_sample) else np.zeros((0, 3), dtype=np.float32)
        if prompt_colors.shape[0] != len(prompt_tokens_all):
            min_len = min(prompt_colors.shape[0], len(prompt_tokens_all))
            prompt_tokens = prompt_tokens_all[:min_len]
            prompt_colors_use = prompt_colors[:min_len]
        else:
            prompt_tokens = prompt_tokens_all
            prompt_colors_use = prompt_colors
        prompt_html = html_color_tokens(prompt_tokens, prompt_colors_use, special_tokens_list)

        out_path = Path(html_output_dir) / f"lang_layer_{lang_layer_idx}_sample_{idx}.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"<div id=\"lang-viz-{idx}\">")
            f.write(f"<h3>Language tokens (layer {lang_layer_idx})</h3>")
            f.write(prompt_html)
            f.write("</div>")

    print("HTML language visualizations saved.")
    
    # ----- Vision: cluster mask + top-k channels per cluster (like notebook) -----
    if args.skip_clusters:
        print("Skipping cluster visualization (--skip-clusters flag set).")
    else:
        # Build per-image feature maps [B, h, w, C]; only for samples with known grid
        print("Building feature maps for cluster visualization...")
        feat_maps: List[torch.Tensor] = []
        valid_indices: List[int] = []
        for idx, fmap in enumerate(per_sample_base_maps):
            if fmap.numel() == 0:
                continue
            # fmap: [C, h, w] -> [h, w, C]
            feat_maps.append(fmap.permute(1, 2, 0).contiguous())
            valid_indices.append(idx)

        cluster_frames: Dict[int, List[np.ndarray]] = {}
        if len(feat_maps) > 0:
            print(f"Running differentiable k-way NCut on {len(feat_maps)} feature maps...")
            Bv = len(feat_maps)
            h0, w0, C0 = feat_maps[0].shape
            fm_stack = torch.stack(feat_maps, dim=0).to(dtype=torch.float32)  # [Bv, h, w, C]
            features_flat = rearrange(fm_stack, 'b h w c -> (b h w) c').contiguous()
            features_flat = features_flat.to(device='cpu', dtype=torch.float32)
            features_flat.requires_grad_(True)

            eigvec, kway_eigvec = differentiable_kway_ncut(features_flat, n_segment=args.n_segments)
            print(f"NCut complete. Processing {args.n_segments} clusters...")
            cluster_idx_all = kway_eigvec.argmax(1)  # [B*h*w]
            eig_components = eigvec.size(1) if eigvec.dim() == 2 else 0
            eig_maps_np: Optional[np.ndarray] = None
            if eig_components > 0:
                try:
                    eig_maps_np = eigvec.reshape(Bv, h0, w0, eig_components).detach().cpu().numpy()
                except Exception:
                    eig_maps_np = None

            grad_topk = args.topk
            for cl in range(args.n_segments):
                print(f"  Processing cluster {cl + 1}/{args.n_segments}...")
                cluster_mask_flat = (cluster_idx_all == cl)
                if cluster_mask_flat.dtype != torch.bool:
                    cluster_mask_flat = cluster_mask_flat.to(dtype=torch.bool)

                topk_eig_indices: List[int] = []
                if eig_components > 0 and cluster_mask_flat.any():
                    cluster_eigs = eigvec[cluster_mask_flat]
                    if cluster_eigs.numel() > 0:
                        eig_scores = cluster_eigs.abs().mean(dim=0)
                        topk_count = min(grad_topk, eig_scores.size(0))
                        if topk_count > 0:
                            topk_eig_indices = torch.topk(eig_scores, topk_count).indices.tolist()
                if not topk_eig_indices and eig_components > 0:
                    topk_eig_indices = list(range(min(grad_topk, eig_components)))

                cols = 4 + grad_topk
                masks = cluster_mask_flat.reshape(Bv, h0, w0)
                cluster_frames.setdefault(cl, [])

                # Process frames for this cluster (skip the giant combined figure - too slow)
                for r in range(Bv):
                    if r % 20 == 0:
                        print(f"    Rendering cluster {cl + 1} frame {r}/{Bv}...")
                    img_idx = valid_indices[r]
                    image_obj = images[img_idx] if 0 <= img_idx < len(images) else None
                    base_rgb = per_sample_base_colors[img_idx] if img_idx < len(per_sample_base_colors) else None
                    tokens = lang_text_per_sample[img_idx] if img_idx < len(lang_text_per_sample) else []
                    token_colors_src = prompt_colors_per_sample[img_idx] if img_idx < len(prompt_colors_per_sample) else np.zeros((0, 3), dtype=np.float32)
                    mask_np = masks[r].cpu().numpy()

                    row_eig_entries: List[Optional[Tuple[int, np.ndarray]]] = []
                    for k in range(grad_topk):
                        entry: Optional[Tuple[int, np.ndarray]] = None
                        if eig_maps_np is not None and k < len(topk_eig_indices):
                            eig_idx = topk_eig_indices[k]
                            if eig_idx < eig_components:
                                eig_map = eig_maps_np[r, :, :, eig_idx]
                                eig_min = float(np.min(eig_map))
                                eig_max = float(np.max(eig_map))
                                if eig_max > eig_min:
                                    eig_norm = (eig_map - eig_min) / (eig_max - eig_min)
                                else:
                                    eig_norm = np.zeros_like(eig_map)
                                entry = (eig_idx, eig_norm)
                        row_eig_entries.append(entry)

                    # Only create ONE figure per frame (not two like before)
                    row_fig, row_axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
                    render_cluster_row(
                        row_axes,
                        image_obj,
                        base_rgb if isinstance(base_rgb, np.ndarray) and base_rgb.size > 0 else None,
                        tokens,
                        token_colors_src,
                        special_tokens_list,
                        mask_np,
                        cl,
                        row_eig_entries,
                    )
                    row_fig.tight_layout()
                    buf = BytesIO()
                    row_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    try:
                        frame_arr = np.array(Image.open(buf).convert('RGB'))
                        cluster_frames[cl].append(frame_arr)
                    finally:
                        buf.close()
                    plt.close(row_fig)

                # Save one representative image for this cluster (first frame)
                if cluster_frames[cl]:
                    out_fig_path = Path(png_output_dir) / f"vision_cluster_{cl}_top{grad_topk}.png"
                    Image.fromarray(cluster_frames[cl][0]).save(out_fig_path)

            print(f"Cluster visualization complete. Generated {sum(len(f) for f in cluster_frames.values())} cluster frames.")
            
            # Save cluster videos using fast method
            print(f"Saving {len(cluster_frames)} cluster videos...")
            for cl_idx, frames in cluster_frames.items():
                if not frames:
                    continue
                cluster_video_path = Path(mp4_output_dir) / f"vision_cluster_{cl_idx}_top{grad_topk}.mp4"
                save_frames_as_video_fast(frames, str(cluster_video_path), fps=max(1, args.fps))
    
    print("Done!")
    print(f"Saved visualizations to: {mp4_output_dir}")


if __name__ == "__main__":
    # Allow module execution via `python visualize_language_backbone.py`.
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

