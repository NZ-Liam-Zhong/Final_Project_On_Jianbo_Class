import torch
import numpy as np
from ncut_pytorch import Ncut, kway_ncut
from ncut_pytorch.utils.math import rbf_affinity
from scipy.optimize import linear_sum_assignment


def compute_entropy_metrics(features1, features2, n_eig=20, mode: str = 'discrete', temperature: float = 1.0):
    """
    Compute conditional entropy and mutual information between two feature layers.
    
    This function takes raw features from two layers, processes them through the 
    normalized cut pipeline (flatten -> aligned -> Ncut -> kway_ncut), normalizes
    them to probability distributions, and computes various entropy-based metrics
    for each channel.
    
    All entropy and mutual information values are measured in bits (using log2).
    
    Args:
        features1: Features from the first layer, shape (H, W, D)
        features2: Features from the second layer, shape (H, W, D)
        n_eig: Number of eigenvectors, default 20
        mode: 'discrete' (argmax + confusion matrix) or 'soft' (row-softmax with temperature)
        temperature: Softmax temperature when mode == 'soft'. Higher -> softer.
    
    Returns:
        metrics: Dictionary containing the following keys (all values in bits):
            - avg_entropy_layer1: Average entropy of layer 1 across all channels
            - avg_entropy_layer2: Average entropy of layer 2 across all channels
            - avg_conditional_entropy_layer1_given_layer2: Average H(Layer1|Layer2)
            - avg_conditional_entropy_layer2_given_layer1: Average H(Layer2|Layer1)
            - avg_mutual_information: Average I(Layer1;Layer2)
            - channel_entropies_1: List of entropies for each channel in layer 1
            - channel_entropies_2: List of entropies for each channel in layer 2
            - channel_conditional_entropies_1_given_2: List of H(1|2) for each channel
            - channel_conditional_entropies_2_given_1: List of H(2|1) for each channel
            - channel_mutual_information: List of mutual information for each channel
    """
    # Process features1: flatten -> aligned -> Ncut -> kway_ncut
    h1, w1, d1 = features1.shape
    flattened1 = features1.reshape(h1 * w1, d1)
    aligned_features1 = rbf_affinity(flattened1)
    eigvecs1 = Ncut(n_eig=n_eig).fit_transform(aligned_features1)
    kway_eigvecs1 = kway_ncut(eigvecs1)
    
    # Process features2: flatten -> aligned -> Ncut -> kway_ncut
    h2, w2, d2 = features2.shape
    flattened2 = features2.reshape(h2 * w2, d2)
    aligned_features2 = rbf_affinity(flattened2)
    eigvecs2 = Ncut(n_eig=n_eig).fit_transform(aligned_features2)
    kway_eigvecs2 = kway_ncut(eigvecs2)
    
    # Convert to numpy
    if isinstance(kway_eigvecs1, torch.Tensor):
        kway_eigvecs1 = kway_eigvecs1.numpy()
    if isinstance(kway_eigvecs2, torch.Tensor):
        kway_eigvecs2 = kway_eigvecs2.numpy()
    
    # Use the helper function which already handles channel pairing
    metrics = _compute_metrics_from_kway(kway_eigvecs1, kway_eigvecs2, mode=mode, temperature=temperature)
    
    return metrics


def compute_entropy_metrics_group(features_list_layer1, features_list_layer2, n_eig=20, mode: str = 'discrete', temperature: float = 1.0):
    """
    Compute conditional entropy and mutual information for a group of images.
    Features from all images are concatenated and processed together (group ncut),
    then metrics are computed for each image individually and averaged.
    
    All entropy and mutual information values are measured in bits (using log2).
    
    Args:
        features_list_layer1: List of features from layer 1, each shape (H, W, D)
        features_list_layer2: List of features from layer 2, each shape (H, W, D)
        n_eig: Number of eigenvectors, default 20
        mode: 'discrete' (argmax + confusion matrix) or 'soft' (row-softmax with temperature)
        temperature: Softmax temperature when mode == 'soft'. Higher -> softer.
    
    Returns:
        avg_metrics: Dictionary containing average metrics across all images (in bits)
        individual_metrics: List of dictionaries containing metrics for each image (in bits)
    """
    num_images = len(features_list_layer1)
    assert len(features_list_layer2) == num_images, "Both lists must have the same length"
    
    # Flatten all features and concatenate them
    flattened_list1 = []
    flattened_list2 = []
    image_sizes = []
    
    for i in range(num_images):
        h, w, d = features_list_layer1[i].shape
        flattened1 = features_list_layer1[i].reshape(h * w, d)
        flattened2 = features_list_layer2[i].reshape(h * w, d)
        flattened_list1.append(flattened1)
        flattened_list2.append(flattened2)
        image_sizes.append(h * w)
    
    # Concatenate all features
    all_features1 = torch.cat(flattened_list1, dim=0)  # shape (N_total, D)
    all_features2 = torch.cat(flattened_list2, dim=0)  # shape (N_total, D)
    
    print(f"Group features shape: {all_features1.shape}")
    
    # Perform group ncut on concatenated features
    print("Performing group Ncut on layer 1...")
    aligned_features1 = rbf_affinity(all_features1)
    eigvecs1 = Ncut(n_eig=n_eig).fit_transform(aligned_features1)
    kway_eigvecs1 = kway_ncut(eigvecs1)
    
    print("Performing group Ncut on layer 2...")
    aligned_features2 = rbf_affinity(all_features2)
    eigvecs2 = Ncut(n_eig=n_eig).fit_transform(aligned_features2)
    kway_eigvecs2 = kway_ncut(eigvecs2)
    
    # Convert to numpy
    if isinstance(kway_eigvecs1, torch.Tensor):
        kway_eigvecs1 = kway_eigvecs1.numpy()
    if isinstance(kway_eigvecs2, torch.Tensor):
        kway_eigvecs2 = kway_eigvecs2.numpy()
    
    # Split kway_eigvecs back to individual images and compute metrics
    individual_metrics = []
    start_idx = 0
    
    for i in range(num_images):
        end_idx = start_idx + image_sizes[i]
        
        # Extract kway_eigvecs for this image
        kway_img1 = kway_eigvecs1[start_idx:end_idx, :]
        kway_img2 = kway_eigvecs2[start_idx:end_idx, :]
        
        # Compute metrics for this image using the helper function
        metrics = _compute_metrics_from_kway(kway_img1, kway_img2, mode=mode, temperature=temperature)
        individual_metrics.append(metrics)
        
        start_idx = end_idx
    
    # Average metrics across all images
    avg_metrics = {
        'avg_entropy_layer1': np.mean([m['avg_entropy_layer1'] for m in individual_metrics]),
        'avg_entropy_layer2': np.mean([m['avg_entropy_layer2'] for m in individual_metrics]),
        'avg_conditional_entropy_layer1_given_layer2': np.mean([m['avg_conditional_entropy_layer1_given_layer2'] for m in individual_metrics]),
        'avg_conditional_entropy_layer2_given_layer1': np.mean([m['avg_conditional_entropy_layer2_given_layer1'] for m in individual_metrics]),
        'avg_mutual_information': np.mean([m['avg_mutual_information'] for m in individual_metrics]),
    }
    
    return avg_metrics, individual_metrics


def _compute_metrics_from_kway(kway_eigvecs1, kway_eigvecs2, mode: str = 'discrete', temperature: float = 1.0):
    """
    Helper function to compute entropy metrics from kway_eigvecs.
    First finds optimal channel pairing by maximizing mutual information,
    then computes metrics for the paired channels.
    
    All entropy and mutual information values are measured in bits (using log2).
    
    Args:
        kway_eigvecs1: kway_eigvecs from layer 1, shape (N, K)
        kway_eigvecs2: kway_eigvecs from layer 2, shape (N, K)
        mode: 'discrete' or 'soft'
        temperature: Softmax temperature when mode == 'soft'. Higher -> softer.
    
    Returns:
        metrics: Dictionary containing entropy metrics (in bits)
    """
    N, K = kway_eigvecs1.shape
    assert kway_eigvecs2.shape == (N, K), "Both kway_eigvecs must have the same shape"
    eps = 1e-10
    
    def H(p):
        p = np.clip(p, eps, 1.0)
        return -np.sum(p * np.log2(p))
    
    if mode == 'discrete':
        # Argmax labels per pixel
        labels1 = np.argmax(kway_eigvecs1, axis=1)
        labels2 = np.argmax(kway_eigvecs2, axis=1)
        
        # Confusion matrix -> joint distribution
        confusion = np.zeros((K, K), dtype=np.float64)
        for a, b in zip(labels1, labels2):
            confusion[a, b] += 1.0
        P_joint = confusion / float(N)
        P1 = P_joint.sum(axis=1)
        P2 = P_joint.sum(axis=0)
        
        # Pairing by maximizing diagonal (Hungarian) — for reference/inspection only
        row_ind, col_ind = linear_sum_assignment(-P_joint)
        
        # K-way entropies and MI
        H1 = H(P1)
        H2 = H(P2)
        H_joint = H(P_joint.reshape(-1))
        H1_given_2 = H_joint - H2
        H2_given_1 = H_joint - H1
        MI = H1 + H2 - H_joint
        
        # Per-channel contributions (diagnostics)
        channel_entropies_1 = (-P1 * np.log2(np.clip(P1, eps, 1.0))).tolist()
        channel_entropies_2 = (-P2 * np.log2(np.clip(P2, eps, 1.0))).tolist()
        # Diagonal MI contributions (not summing to MI in general, diagnostic only)
        diag = np.clip(np.diag(P_joint), 0.0, 1.0)
        denom = np.clip(P1 * P2, eps, 1.0)
        channel_mutual_information = (diag * np.log2((diag + eps) / denom)).tolist()
        # For conditional entropies per-channel，均分展示，避免误导
        channel_conditional_entropies_1_given_2 = [H1_given_2 / K] * K
        channel_conditional_entropies_2_given_1 = [H2_given_1 / K] * K
        
        metrics = {
            'avg_entropy_layer1': float(H1),
            'avg_entropy_layer2': float(H2),
            'avg_conditional_entropy_layer1_given_layer2': float(H1_given_2),
            'avg_conditional_entropy_layer2_given_layer1': float(H2_given_1),
            'avg_mutual_information': float(MI),
            'channel_entropies_1': channel_entropies_1,
            'channel_entropies_2': channel_entropies_2,
            'channel_conditional_entropies_1_given_2': channel_conditional_entropies_1_given_2,
            'channel_conditional_entropies_2_given_1': channel_conditional_entropies_2_given_1,
            'channel_mutual_information': channel_mutual_information,
            'channel_pairing': list(zip(row_ind.tolist(), col_ind.tolist())),
            'pairing_matrix': P_joint,
        }
        return metrics
    
    # soft mode
    def softmax_rows(x, tau=1.0):
        x = x / max(tau, eps)
        x = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x)
        denom = np.sum(ex, axis=1, keepdims=True) + eps
        return ex / denom
    
    p1 = softmax_rows(kway_eigvecs1, tau=temperature)
    p2 = softmax_rows(kway_eigvecs2, tau=temperature)
    
    # Cluster-level soft joint
    P_joint = (p1[:, :, None] * p2[:, None, :]).sum(axis=0) / float(N)
    P1 = P_joint.sum(axis=1)
    P2 = P_joint.sum(axis=0)
    
    row_ind, col_ind = linear_sum_assignment(-P_joint)
    
    H1 = H(P1)
    H2 = H(P2)
    H_joint = H(P_joint.reshape(-1))
    H1_given_2 = H_joint - H2
    H2_given_1 = H_joint - H1
    MI = H1 + H2 - H_joint
    
    channel_entropies_1 = (-P1 * np.log2(np.clip(P1, eps, 1.0))).tolist()
    channel_entropies_2 = (-P2 * np.log2(np.clip(P2, eps, 1.0))).tolist()
    diag = np.clip(np.diag(P_joint), 0.0, 1.0)
    denom = np.clip(P1 * P2, eps, 1.0)
    channel_mutual_information = (diag * np.log2((diag + eps) / denom)).tolist()
    channel_conditional_entropies_1_given_2 = [H1_given_2 / K] * K
    channel_conditional_entropies_2_given_1 = [H2_given_1 / K] * K
    
    metrics = {
        'avg_entropy_layer1': float(H1),
        'avg_entropy_layer2': float(H2),
        'avg_conditional_entropy_layer1_given_layer2': float(H1_given_2),
        'avg_conditional_entropy_layer2_given_layer1': float(H2_given_1),
        'avg_mutual_information': float(MI),
        'channel_entropies_1': channel_entropies_1,
        'channel_entropies_2': channel_entropies_2,
        'channel_conditional_entropies_1_given_2': channel_conditional_entropies_1_given_2,
        'channel_conditional_entropies_2_given_1': channel_conditional_entropies_2_given_1,
        'channel_mutual_information': channel_mutual_information,
        'channel_pairing': list(zip(row_ind.tolist(), col_ind.tolist())),
        'pairing_matrix': P_joint,
    }
    return metrics

