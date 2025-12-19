import os

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


MASK_COLOR_LIST = ["jet", "ocean", "Spectral", "spring", "cool"]
MASK_ALPHA = 0.5


def inverse_normalize(img: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Inverse normalize an image tensor.
    Args:
        img: Image tensor of shape (C, H, W) or (B, C, H, W)
        dim: Channel dimension (not used, kept for compatibility)
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    
    # Reshape mean and std to match image dimensions
    if img.ndim == 3:  # (C, H, W)
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    elif img.ndim == 4:  # (B, C, H, W)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    
    return std * img + mean


def visualize_image_with_masks_and_bboxes(
        img: torch.Tensor,
        labels: list[torch.Tensor],
        masks: list[torch.Tensor],
        bboxes: list[torch.Tensor],
) -> None:
    ####################################################################################################################
    # TODO: Visualize the image with the mask and bounding box for each of its objects. Use a different color for
    # different classes of objects. The output should be similar to the example in Figure ???.
    ####################################################################################################################
    import matplotlib.patches as patches
    import numpy as np
    
    # Convert image to displayable format
    img_display = inverse_normalize(img.cpu())  # Denormalize
    img_display = torch.clamp(img_display, 0, 1)  # Clamp to [0,1]
    
    # Remove padding (from 1088 to 1066)
    img_display = img_display[:, :, :1066]
    
    # Convert to numpy and transpose to HWC format
    img_np = img_display.permute(1, 2, 0).numpy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_np)
    
    # Define colors for different classes
    class_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    class_names = ['Background', 'Vehicle', 'People', 'Animal']  # Label 1=Vehicle, 2=People, 3=Animal
    
    if len(labels) > 0:
        # Process masks and bboxes
        masks_np = masks.squeeze(1).cpu().numpy()  # Remove channel dimension
        bboxes_np = bboxes.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Remove padding from masks
        masks_np = masks_np[:, :, :1066]
        
        for i in range(len(labels)):
            label = labels_np[i]
            mask = masks_np[i]
            bbox = bboxes_np[i]
            
            # Get color for this class
            color = class_colors[label % len(class_colors)]
            
            # Overlay mask with transparency
            mask_colored = np.zeros((*mask.shape, 4))
            mask_colored[:, :, :3] = plt.cm.get_cmap(MASK_COLOR_LIST[label % len(MASK_COLOR_LIST)])(0.5)[:3]
            mask_colored[:, :, 3] = mask * MASK_ALPHA  # Alpha channel
            
            ax.imshow(mask_colored)
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label text
            ax.text(x1, y1 - 5, f'{class_names[label]}', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                   fontsize=10, color='white', weight='bold')
    
    ax.set_title('Image with Masks and Bounding Boxes')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    ####################################################################################################################


def visualize_ground_truth(
    images: torch.Tensor,
    instance_ground_truth_list: list[list[torch.Tensor]],
    instance_index_ground_truth_list: list[list[torch.Tensor]],
    category_ground_truth_list: list[list[torch.Tensor]],
) -> None:
    """
    Process single image to generate target labels for each feature pyramid level.
    Input:
    - images (tensor): Shape (bsz, C, H, W)
    - instance_ground_truth_list (list[list[torch.LongTensor]]): bsz x (num_fpn_layers x (n_obj, H, W))
    - instance_index_ground_truth_list (list[list[torch.LongTensor]]): bsz x (num_fpn_layers x (n_obj))
    - category_ground_truth_list (list[list[torch.LongTensor]]): bsz x (num_fpn_layers x (n_obj))
    """
    ####################################################################################################################
    # TODO: Visualize the images at each FPN layer with the correct masks for that layer. Use a different color for
    #  different classes of objects. The output should be similar to the example in Figure ???.
    ####################################################################################################################
    import numpy as np
    
    batch_size = images.shape[0]
    num_fpn_layers = len(instance_ground_truth_list[0])
    
    # Define colors and names
    class_colors = ['red', 'blue', 'green', 'yellow', 'purple']
    class_names = ['Background', 'Vehicle', 'People', 'Animal']  # Label 1=Vehicle, 2=People, 3=Animal
    fpn_names = ['P1 (stride 8)', 'P2 (stride 8)', 'P3 (stride 16)', 'P4 (stride 32)', 'P5 (stride 32)']
    
    # Visualize each image in the batch
    for img_idx in range(batch_size):  # Show all images
        img = images[img_idx]
        
        # Convert image to displayable format
        img_display = inverse_normalize(img.cpu())
        img_display = torch.clamp(img_display, 0, 1)
        img_display = img_display[:, :, :1066]  # Remove padding
        img_np = img_display.permute(1, 2, 0).numpy()
        
        # Create subplots for each FPN level
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Show original image
        axes[0].imshow(img_np)
        axes[0].set_title(f'Original Image {img_idx}')
        axes[0].axis('off')
        
        # Show each FPN level
        for level_idx in range(num_fpn_layers):
            ax = axes[level_idx + 1]
            ax.imshow(img_np)
            
            # Get ground truth data for this level
            instance_masks = instance_ground_truth_list[img_idx][level_idx]  # (S^2, H, W)
            instance_indices = instance_index_ground_truth_list[img_idx][level_idx]  # (S^2,)
            category_targets = category_ground_truth_list[img_idx][level_idx]  # (S, S)
            
            # Find active grid positions
            active_positions = torch.where(instance_indices > 0)[0]
            
            if len(active_positions) > 0:
                # Get the corresponding masks
                active_masks = instance_masks[active_positions]  # (n_active, H, W)
                
                # Resize masks to image size for visualization
                # Note: The masks are at shape (200, 272) from construct_targets
                # We need to resize to (800, 1088) first to match the padded image, then crop
                if active_masks.shape[1:] != (800, 1088):
                    active_masks = F.interpolate(
                        active_masks.unsqueeze(1), 
                        size=(800, 1088), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(1)
                
                # Remove padding from masks to match the displayed image (800, 1066)
                active_masks = active_masks[:, :, :1066]
                
                # Find corresponding categories
                grid_size = int(np.sqrt(len(instance_indices)))
                
                for i, pos in enumerate(active_positions):
                    mask = active_masks[i].cpu().numpy()
                    
                    # Find the grid position
                    grid_y = pos // grid_size
                    grid_x = pos % grid_size
                    
                    # Get category at this position
                    if grid_y < category_targets.shape[0] and grid_x < category_targets.shape[1]:
                        category = category_targets[grid_y, grid_x].item()
                        
                        if category >= 0:  # Include all categories
                            # Create colored mask overlay
                            mask_colored = np.zeros((*mask.shape, 4))
                            color_map = plt.cm.get_cmap(MASK_COLOR_LIST[(category + 1) % len(MASK_COLOR_LIST)])
                            mask_colored[:, :, :3] = color_map(0.7)[:3]
                            mask_colored[:, :, 3] = mask * MASK_ALPHA
                            
                            ax.imshow(mask_colored)
                            
                            # Add grid position indicator
                            img_h, img_w = 800, 1066
                            center_x = (grid_x + 0.5) * img_w / grid_size
                            center_y = (grid_y + 0.5) * img_h / grid_size
                            
                            ax.plot(center_x, center_y, 'o', 
                                   color=class_colors[(category + 1) % len(class_colors)], 
                                   markersize=8, markeredgecolor='white', markeredgewidth=2)
                            
                            # Add text label
                            ax.text(center_x, center_y - 20, f'{class_names[category + 1]}',
                                   ha='center', va='bottom',
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor=class_colors[(category + 1) % len(class_colors)], 
                                           alpha=0.7),
                                   fontsize=8, color='white', weight='bold')
            
            ax.set_title(f'{fpn_names[level_idx]}')
            ax.axis('off')
        
        plt.suptitle(f'Ground Truth Targets for Image {img_idx}', fontsize=16)
        plt.tight_layout()
        plt.show()
    ####################################################################################################################


def visualize_inference(
    images: torch.Tensor,
    nms_sorted_category_list: list[torch.Tensor],
    nms_sorted_instance_mask_list: list[torch.Tensor],
) -> None:
    """
    Process single image to generate target labels for each feature pyramid level.
    Input:
    - images (tensor): Shape (bsz, C, H, W)
    - nms_sorted_category_list (list[torch.LongTensor]): bsz x (n_obj, H, W)
    - nms_sorted_instance_mask_list (list[torch.LongTensor]): bsz x (n_obj)
    """
    ####################################################################################################################
    # Visualize final instance masks overlaid on the original images, colored by predicted categories
    ####################################################################################################################
    import numpy as np

    batch_size = images.shape[0]
    class_names = ['Vehicle', 'People', 'Animal']
    color_cycle = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    for i in range(batch_size):
        img = images[i]
        img_display = inverse_normalize(img.cpu())
        img_display = torch.clamp(img_display, 0, 1)
        img_display = img_display[:, :, :1066]  # Remove padding for display
        img_np = img_display.permute(1, 2, 0).numpy()

        masks = nms_sorted_instance_mask_list[i]
        labels = nms_sorted_category_list[i]

        # Resize masks to display size if needed
        if masks.numel() > 0 and masks.shape[-1] != 1066:
            masks = F.interpolate(masks.unsqueeze(1), size=(800, 1066), mode='nearest').squeeze(1)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_np)

        if masks.numel() > 0:
            masks_np = masks.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for j in range(masks_np.shape[0]):
                mask = masks_np[j]
                cls = int(labels_np[j])
                color = color_cycle[cls % len(color_cycle)]
                mask_colored = np.zeros((*mask.shape, 4))
                cmap = plt.cm.get_cmap(MASK_COLOR_LIST[(cls + 1) % len(MASK_COLOR_LIST)])
                mask_colored[:, :, :3] = cmap(0.7)[:3]
                mask_colored[:, :, 3] = mask * MASK_ALPHA
                ax.imshow(mask_colored)

                ax.text(10, 20 + 20 * j, class_names[cls], color='white', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

        ax.set_title('Inference Result')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    ####################################################################################################################


def plot_loss_curves(log_dir: str) -> None:
    # Find all event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_files.append(os.path.join(root, file))

    # Initialize dictionaries to store data
    data = {
        "train_total_loss": {"steps": [], "values": []},
        "train_dice_loss": {"steps": [], "values": []},
        "train_focal_loss": {"steps": [], "values": []},
        "val_total_loss_epoch": {"steps": [], "values": []},
        "val_dice_loss_epoch": {"steps": [], "values": []},
        "val_focal_loss_epoch": {"steps": [], "values": []}
    }

    # Process each event file
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        # Get all scalar tags
        tags = ea.Tags()["scalars"]

        # Extract data for each loss type
        for tag in data.keys():
            if tag in tags:
                scalars = ea.Scalars(tag)
                data[tag]["steps"].extend([s.step for s in scalars])
                data[tag]["values"].extend([s.value for s in scalars])

    # Ensure that the data is sorted by steps and safely handle empty series
    for key in list(data.keys()):
        steps_list = data[key]["steps"]
        values_list = data[key]["values"]
        if not steps_list or not values_list:
            # leave empty; nothing to plot for this key
            data[key]["steps"] = []
            data[key]["values"] = []
            continue
        sorted_pairs = sorted(zip(steps_list, values_list))
        if len(sorted_pairs) == 0:
            data[key]["steps"] = []
            data[key]["values"] = []
            continue
        steps, values = map(list, zip(*sorted_pairs))
        data[key]["steps"] = steps
        data[key]["values"] = values

    # Plotting each loss curve individually
    DEFAULT_COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, loss_type in enumerate(data.keys()):
        title = " ".join([s.capitalize() for s in loss_type.split("_")[:3]])
        plt.figure(figsize=(10, 6))
        if len(data[loss_type]["steps"]) == 0:
            plt.title(f"{title} (no data)")
            plt.close()
            continue
        plt.plot(list(data[loss_type]["steps"]), list(data[loss_type]["values"]), label=title, color=DEFAULT_COLOR_CYCLE[i])
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()





