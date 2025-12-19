import argparse
import math
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def collect_layers(video_dir: Path, layer_type: str):
    # Determine layers by checking first frame directory
    first_frame = sorted(video_dir.glob('frame_*'))[0]
    layer_dir = first_frame / layer_type
    layers = sorted([p.name for p in layer_dir.glob('*.png')])
    return layers

def collect_frames(video_dir: Path):
    return sorted(video_dir.glob('frame_*'))

def build_grid(images, cols=10, bg_color=(255, 255, 255)):
    if not images:
        return None
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    canvas = Image.new('RGB', (cols * w, rows * h), color=bg_color)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        canvas.paste(img, (c * w, r * h))
    return canvas

def process_video(video_dir: Path, out_root: Path, layer_type: str, cols: int):
    layers = collect_layers(video_dir, layer_type)
    frames = collect_frames(video_dir)
    target_dir = out_root / video_dir.name / layer_type
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{video_dir.name}] {layer_type}: {len(frames)} frames, {len(layers)} layers")

    for layer in tqdm(layers, desc=f"{video_dir.name}-{layer_type}"):
        imgs = []
        for f in frames:
            img_path = f / layer_type / layer
            if img_path.exists():
                imgs.append(Image.open(img_path).convert('RGB'))
        grid = build_grid(imgs, cols=cols)
        if grid is None:
            continue
        grid.save(target_dir / f"{layer.replace('.png','')}_all_frames.png")


def main():
    parser = argparse.ArgumentParser(description="Combine per-frame NCut visualizations into per-layer grids")
    parser.add_argument('--input_dir', required=True, help='Path to openvla_ncut_visualizations')
    parser.add_argument('--output_dir', required=True, help='Where to save combined grids')
    parser.add_argument('--cols', type=int, default=10, help='Number of columns in the grid')
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    videos = sorted([p for p in input_root.iterdir() if p.is_dir()])

    for video in videos:
        process_video(video, output_root, 'vision_encoder', cols=args.cols)
        process_video(video, output_root, 'llm_backbone', cols=args.cols)

    print("Done. Output at", output_root)

if __name__ == '__main__':
    main()
