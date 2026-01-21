"""
Data Loaders for V-JEPA
Supports synthetic data for testing and real video/image data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Callable
import numpy as np


class SyntheticVideoDataset(Dataset):
    """
    Synthetic video dataset for testing V-JEPA.

    Generates random tensors simulating video frames with optional
    temporal structure (e.g., smooth transitions).
    """

    def __init__(
        self,
        num_samples: int = 1000,
        img_size: int = 224,
        channels: int = 3,
        temporal_length: int = 10,
        context_length: int = 4,
        target_offset: int = 1,
        add_temporal_structure: bool = True,
        noise_level: float = 0.1
    ):
        """
        Args:
            num_samples: Number of video sequences
            img_size: Height/width of frames (assumes square)
            channels: Number of channels (3 for RGB)
            temporal_length: Total number of frames per sequence
            context_length: Number of context frames to sample
            target_offset: Temporal offset between context and target
            add_temporal_structure: Add smooth transitions between frames
            noise_level: Noise level for temporal structure
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.channels = channels
        self.temporal_length = temporal_length
        self.context_length = context_length
        self.target_offset = target_offset
        self.add_temporal_structure = add_temporal_structure
        self.noise_level = noise_level

        # Pre-generate base sequences for efficiency
        if add_temporal_structure:
            self._generate_structured_data()

    def _generate_structured_data(self):
        """Generate video sequences with temporal structure."""
        # Generate smooth trajectories in a latent space
        latent_dim = 32
        self.latent_trajectories = []

        for _ in range(self.num_samples):
            # Random walk in latent space
            trajectory = torch.randn(self.temporal_length, latent_dim)
            # Smooth with cumulative sum to create temporal coherence
            trajectory = torch.cumsum(trajectory * 0.3, dim=0)
            self.latent_trajectories.append(trajectory)

    def _generate_frame_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Generate a frame from latent representation.

        In practice, this simulates a simple generative process.
        """
        # Simple projection to image space with spatial structure
        frame = torch.randn(self.channels, self.img_size, self.img_size)

        # Add some structure based on latent code
        for i in range(min(4, len(latent))):
            # Create simple patterns influenced by latent dimensions
            scale = latent[i].item() * 0.5
            frame += scale * torch.randn(self.channels, self.img_size, self.img_size)

        # Add noise
        frame += torch.randn_like(frame) * self.noise_level

        return frame

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Sample index

        Returns:
            (context_frames, target_frame) tuple
                context_frames: (context_length, C, H, W)
                target_frame: (C, H, W)
        """
        if self.add_temporal_structure and hasattr(self, 'latent_trajectories'):
            # Use pre-generated temporal structure
            trajectory = self.latent_trajectories[idx]

            # Sample context frames
            context_start = np.random.randint(0, self.temporal_length - self.context_length - self.target_offset)
            context_indices = range(context_start, context_start + self.context_length)

            context_frames = torch.stack([
                self._generate_frame_from_latent(trajectory[i])
                for i in context_indices
            ])

            # Sample target frame (offset from last context frame)
            target_idx = context_start + self.context_length + self.target_offset - 1
            target_frame = self._generate_frame_from_latent(trajectory[target_idx])

        else:
            # Fully random (no temporal structure)
            context_frames = torch.randn(
                self.context_length, self.channels, self.img_size, self.img_size
            )
            target_frame = torch.randn(self.channels, self.img_size, self.img_size)

        # Normalize to [0, 1] range
        context_frames = torch.clamp((context_frames + 3) / 6, 0, 1)
        target_frame = torch.clamp((target_frame + 3) / 6, 0, 1)

        # For simplicity, return single context frame (last one)
        # Can be extended to use all context frames
        context_frame = context_frames[-1]

        return context_frame, target_frame


class ImagePairDataset(Dataset):
    """
    Dataset that generates pairs of augmented views from single images.

    Useful for learning from static images by treating different
    augmentations as temporal sequence.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        img_size: int = 224,
        channels: int = 3,
        augmentation_strength: float = 0.3
    ):
        """
        Args:
            num_samples: Number of base images
            img_size: Image size
            channels: Number of channels
            augmentation_strength: Strength of augmentation differences
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.channels = channels
        self.augmentation_strength = augmentation_strength

        # Generate base images
        self.base_images = torch.randn(num_samples, channels, img_size, img_size)

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        """Simple augmentation (noise addition)."""
        noise = torch.randn_like(image) * self.augmentation_strength
        augmented = image + noise
        return torch.clamp(augmented, 0, 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Two augmented views of the same image
        """
        base_image = self.base_images[idx]

        # Create two different augmentations
        context_view = self._augment(base_image)
        target_view = self._augment(base_image)

        return context_view, target_view


class VideoFrameDataset(Dataset):
    """
    Dataset for real video frames.

    Placeholder for loading actual video data. Can be extended with
    video loading libraries like decord, torchvision, or OpenCV.
    """

    def __init__(
        self,
        video_paths: list,
        img_size: int = 224,
        temporal_stride: int = 4,
        context_length: int = 1,
        target_offset: int = 1,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            video_paths: List of paths to video files
            img_size: Target image size
            temporal_stride: Stride for frame sampling
            context_length: Number of context frames
            target_offset: Offset to target frame
            transform: Optional transform to apply
        """
        self.video_paths = video_paths
        self.img_size = img_size
        self.temporal_stride = temporal_stride
        self.context_length = context_length
        self.target_offset = target_offset
        self.transform = transform

        # TODO: Implement video loading with decord/torchvision
        raise NotImplementedError(
            "VideoFrameDataset requires video loading implementation. "
            "Consider using decord, torchvision.io, or OpenCV."
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return video frame pairs."""
        # TODO: Implement frame loading
        raise NotImplementedError


def build_dataloader(
    dataset_type: str = "synthetic",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Factory function to build dataloaders.

    Args:
        dataset_type: 'synthetic', 'image_pair', or 'video'
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        **dataset_kwargs: Arguments passed to dataset constructor

    Returns:
        DataLoader instance
    """
    if dataset_type == "synthetic":
        dataset = SyntheticVideoDataset(**dataset_kwargs)
    elif dataset_type == "image_pair":
        dataset = ImagePairDataset(**dataset_kwargs)
    elif dataset_type == "video":
        dataset = VideoFrameDataset(**dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def create_simple_dataloader(
    num_samples: int = 1000,
    batch_size: int = 32,
    img_size: int = 224,
    channels: int = 3,
    add_structure: bool = True
) -> DataLoader:
    """
    Convenience function to create a simple synthetic dataloader.

    Args:
        num_samples: Number of samples
        batch_size: Batch size
        img_size: Image size
        channels: Number of channels
        add_structure: Whether to add temporal structure

    Returns:
        DataLoader for synthetic data
    """
    return build_dataloader(
        dataset_type="synthetic",
        batch_size=batch_size,
        num_samples=num_samples,
        img_size=img_size,
        channels=channels,
        add_temporal_structure=add_structure,
        shuffle=True,
        num_workers=0  # Use 0 for synthetic data (no I/O bottleneck)
    )
