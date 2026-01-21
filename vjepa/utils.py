"""
Utility functions for V-JEPA experiments
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from pathlib import Path
from typing import Dict, Optional
import json


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_str: str = "cuda") -> torch.device:
    """
    Get torch device with fallback to CPU if CUDA unavailable.

    Args:
        device_str: Device string ('cuda' or 'cpu')

    Returns:
        torch.device
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        if device_str == "cuda":
            print("CUDA not available, falling back to CPU")
        return torch.device("cpu")


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.

        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:
    """
    Tracks multiple metrics across training.
    """

    def __init__(self):
        self.metrics = {}
        self.history = {}

    def update(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics.

        Args:
            metrics_dict: Dictionary of metric names to values
            step: Optional step number for history tracking
        """
        for name, value in metrics_dict.items():
            if name not in self.metrics:
                self.metrics[name] = AverageMeter(name)
                self.history[name] = []

            self.metrics[name].update(value)

            if step is not None:
                self.history[name].append((step, value))

    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {name: meter.avg for name, meter in self.metrics.items()}

    def reset(self):
        """Reset all meters (but keep history)."""
        for meter in self.metrics.values():
            meter.reset()

    def __str__(self) -> str:
        return " | ".join(str(meter) for meter in self.metrics.values())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: str,
    is_best: bool = False
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    # Create directory if needed
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Save best model separately
    if is_best:
        best_path = str(Path(checkpoint_path).parent / "best_model.pt")
        torch.save(checkpoint, best_path)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to

    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {})
    }


def save_config(config, path: str):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration object (dataclass)
        path: Path to save to
    """
    from dataclasses import asdict

    config_dict = asdict(config)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved config to {path}")


def compute_embedding_statistics(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about embeddings to monitor representation quality.

    Args:
        embeddings: Tensor of embeddings (B, D)

    Returns:
        Dictionary of statistics
    """
    with torch.no_grad():
        # Normalize embeddings
        norm = embeddings.norm(dim=1, keepdim=True)
        normalized = embeddings / (norm + 1e-8)

        # Compute statistics
        stats = {
            'mean_norm': norm.mean().item(),
            'std_norm': norm.std().item(),
            'mean_value': embeddings.mean().item(),
            'std_value': embeddings.std().item(),
        }

        # Cosine similarity matrix (measure of representation diversity)
        similarity_matrix = normalized @ normalized.T
        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=embeddings.device)
        off_diagonal_sim = similarity_matrix[mask]

        stats['mean_pairwise_sim'] = off_diagonal_sim.mean().item()
        stats['std_pairwise_sim'] = off_diagonal_sim.std().item()

        # Effective rank (measure of dimension usage)
        _, s, _ = torch.svd(normalized)
        s_normalized = s / s.sum()
        entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()
        stats['effective_rank'] = effective_rank

    return stats


def visualize_embeddings_2d(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    method: str = "pca"
) -> np.ndarray:
    """
    Reduce embeddings to 2D for visualization.

    Args:
        embeddings: Embeddings to visualize (N, D)
        labels: Optional labels for coloring (N,)
        method: Dimensionality reduction method ('pca' or 'tsne')

    Returns:
        2D coordinates (N, 2)
    """
    embeddings_np = embeddings.cpu().numpy()

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        coords_2d = reducer.fit_transform(embeddings_np)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings_np)

    else:
        raise ValueError(f"Unknown method: {method}")

    return coords_2d


def log_metrics(
    epoch: int,
    metrics: Dict[str, float],
    phase: str = "train",
    step: Optional[int] = None
):
    """
    Log metrics in a formatted way.

    Args:
        epoch: Current epoch
        metrics: Dictionary of metrics
        phase: Training phase ('train', 'val', etc.)
        step: Optional step within epoch
    """
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    if step is not None:
        print(f"[{phase.upper()}] Epoch {epoch} | Step {step} | {metrics_str}")
    else:
        print(f"[{phase.upper()}] Epoch {epoch} | {metrics_str}")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
