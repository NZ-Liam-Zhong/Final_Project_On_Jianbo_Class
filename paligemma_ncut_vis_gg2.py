import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import html

# ==========================================
# 1. Setup & NCut Utilities
# ==========================================

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def color_from_ncut(features: torch.Tensor, n_eig: int = 4) -> np.ndarray:
    """
    Runs NCut on features [N, D] and returns RGB colors [N, 3].
    Uses the ncut-pytorch library.
    """
    try:
        from ncut_pytorch import ncut_fn, tsne_color
    except ImportError:
        print("Error: ncut-pytorch not installed. Please run `pip install ncut-pytorch`")
        sys.exit(1)

    # Normalize features for better cosine similarity graph construction
    features = F.normalize(features, p=2, dim=-1)
    
    with torch.no_grad():
        # Move to CPU float32 for NCut calculation
        f32 = features.detach().to(device="cpu", dtype=torch.float32)
        # Calculate eigenvectors
        eigvecs, _ = ncut_fn(f32, n_eig=n_eig)
    
    # Map eigenvectors to RGB (t-SNE style coloring)
    colors = tsne_color(eigvecs)
    return colors.cpu().numpy()

def generate_html_output(tokens, colors, output_path):
    """
    Generates an HTML snippet showing text tokens with their background colors.
    """
    spans = []
    for tok, rgb in zip(tokens, colors):
        # Convert 0-1 float RGB to 0-255 int
        r, g, b = (int(c * 255) for c in rgb)
        # Determine text color (black/white) based on background brightness
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "black" if brightness > 128 else "white"
        
        safe_tok = html.escape(tok).replace(' ', '&nbsp;')
        span = f'<span style="background-color: rgb({r},{g},{b}); color: {text_color}; padding: 2px; border-radius: 3px; margin-right: 2px;">{safe_tok}</span>'
        spans.append(span)
    
    html_content = f"""
    <div style="font-family: monospace; line-height: 1.5; padding: 20px; background: #f0f0f0; border-radius: 8px;">
        <h3>Vision-Driven Semantic Alignment</h3>
        <p><b>Logic:</b> Text tokens are colored based on the <i>Raw Vision Segment</i> they align with most strongly.</p>
        <p>{''.join(spans)}</p>
    </div>
    """
    
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved HTML visualization to {output_path}")

# ==========================================
# 2. Model Logic
# ==========================================

def load_paligemma(model_path, device):
    print(f"Loading model from {model_path}...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    try:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            revision="float16", 
        ).eval()
        processor = PaliGemmaProcessor.from_pretrained(model_path)
        return model, processor
    except OSError as e:
        print(f"Failed to load from local path: {e}")
        print("Please ensure the path is correct and accessible.")
        sys.exit(1)

def extract_features(model, processor, image, prompt, device):
    """
    Runs the forward pass to extract:
    1. Raw SigLIP features (for pure vision segmentation)
    2. Projected Vision features (aligned with LLM)
    3. Language Token features (for semantic coloring)
    """
    model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_ids = model_inputs.input_ids.to(device)
    pixel_values = model_inputs.pixel_values.to(device).to(model.dtype)
    
    with torch.no_grad():
        # A. Raw Vision Extraction (SigLIP)
        vision_outputs = model.vision_tower(pixel_values, output_attentions=False, output_hidden_states=False)
        raw_vision_features = vision_outputs.last_hidden_state # [1, 256, 1152]
        
        # B. Projected Vision Extraction (Aligned space)
        projected_vision_features = model.multi_modal_projector(raw_vision_features) # [1, 256, 2048]
        
        # C. Language Extraction
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Slice text features (everything after image tokens)
        num_image_tokens = model.config.vision_config.num_image_tokens
        full_sequence_features = outputs.hidden_states[-1] # [1, Seq_Len, 2048]
        text_features = full_sequence_features[:, num_image_tokens:, :]
        
        # Get corresponding text tokens
        text_ids = input_ids[0, num_image_tokens:]
        tokens = processor.tokenizer.convert_ids_to_tokens(text_ids)
        
    return raw_vision_features, projected_vision_features, text_features, tokens

# ==========================================
# 3. Main Execution Pipeline
# ==========================================

def main():
    # ==========================================
    # USER CONFIGURATION SECTION
    # ==========================================
    MODEL_PATH = "/mnt/disk1/ilykyleliam/public/paligemma-3b-pt-224"
    IMAGE_PATH = "/mnt/disk1/ilykyleliam/frame060.png"  # <--- REPLACE WITH YOUR IMAGE PATH
    PROMPT = "detect the robot arm"
    OUTPUT_DIR = "./paligemma_viz"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESIZE_DIM = 224
    NCUT_SEGMENTS = 8  # Number of colors/segments to generate
    # ==========================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seed_everything(42)
    
    # 1. Load Resources
    model, processor = load_paligemma(MODEL_PATH, DEVICE)
    
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not find image at {IMAGE_PATH}")
        return

    image_viz = image.resize((RESIZE_DIM, RESIZE_DIM)) 
    
    print(f"Processing image: {IMAGE_PATH}")
    print(f"Using prompt: '{PROMPT}'")
    
    # 2. Extract Features
    raw_vis, proj_vis, text_feat, tokens = extract_features(model, processor, image, PROMPT, DEVICE)
    
    # Remove batch dims
    raw_vis = raw_vis.squeeze(0)    # [256, 1152]
    proj_vis = proj_vis.squeeze(0)  # [256, 2048]
    text_feat = text_feat.squeeze(0)# [T, 2048]
    
    # 3. Compute Colors (The Vision-Driven Approach)
    print("Computing Vision Clusters...")
    
    # A. Generate Palette from Raw Vision (SigLIP)
    # This creates the "ground truth" colors for the image segments
    raw_vis_colors = color_from_ncut(raw_vis, n_eig=NCUT_SEGMENTS) # [256, 3]
    
    print("Mapping Text to Vision Segments...")
    
    # B. Align Text to Vision
    # We use projected features for the matching math (because dimensions must match),
    # but we assign the colors from the Raw Vision calculation.
    proj_vis_norm = F.normalize(proj_vis, p=2, dim=-1)
    text_feat_norm = F.normalize(text_feat, p=2, dim=-1)
    
    # Similarity: [Text_Tokens, Image_Patches]
    sim_matrix = torch.matmul(text_feat_norm, proj_vis_norm.T) 
    
    # Find best matching image patch for each text token
    best_patch_indices = sim_matrix.argmax(dim=1).cpu().numpy() # [T]
    
    # Assign the color of that patch to the text token
    text_colors = raw_vis_colors[best_patch_indices] # [T, 3]

    # ==========================================
    # 4. Generate Visualizations
    # ==========================================
    
    grid_size = int(np.sqrt(raw_vis.shape[0])) # 16 for 224px input
    raw_map = raw_vis_colors.reshape(grid_size, grid_size, 3)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Original Image
    axes[0].imshow(image_viz)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    # Plot 2: Raw Vision Segmentation (The Color Legend)
    # This map explains the colors used in the HTML text
    axes[1].imshow(raw_map)
    axes[1].set_title(f"Raw Vision Segmentation (NCut k={NCUT_SEGMENTS})\n(These colors map to the text below)")
    axes[1].axis("off")
    
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "paligemma_segmentation_legend.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved Segmentation Legend to {out_png}")
    
    # Generate HTML Text Visualization
    out_html = os.path.join(OUTPUT_DIR, "vision_driven_text.html")
    generate_html_output(tokens, text_colors, out_html)
    
    print("Done.")

if __name__ == "__main__":
    main()