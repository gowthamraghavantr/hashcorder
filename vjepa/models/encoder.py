"""
Vision Encoders for V-JEPA
Implements both CNN and Vision Transformer variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CNNEncoder(nn.Module):
    """
    Simple CNN encoder for vision embeddings.
    Maps images to fixed-dimensional latent representations.
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dims: Tuple[int, ...] = (64, 128, 256, 512),
        embedding_dim: int = 512,
        input_size: int = 224,
        use_batchnorm: bool = True
    ):
        """
        Args:
            input_channels: Number of input channels (3 for RGB)
            hidden_dims: Channel dimensions for each conv block
            embedding_dim: Final embedding dimension
            input_size: Input image size (assumes square images)
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim

        # Build convolutional blocks
        layers = []
        in_channels = input_channels

        for out_channels in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        # Calculate output spatial size after convolutions
        # Each conv with stride=2 reduces size by half
        num_strides = len(hidden_dims)
        output_size = input_size // (2 ** num_strides)
        flatten_size = hidden_dims[-1] * output_size * output_size

        # Projection head to embedding dimension
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Embedding tensor of shape (B, embedding_dim)
        """
        features = self.conv_blocks(x)
        embeddings = self.projector(features)
        return embeddings


class PatchEmbed(nn.Module):
    """
    Image to patch embedding using convolution.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            Patch embeddings (B, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block with self-attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) where N is sequence length

        Returns:
            Transformed features (B, N, D)
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for V-JEPA.
    Converts images to patch sequences and applies transformer blocks.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        embedding_dim: Optional[int] = None
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Transformer embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            embedding_dim: Final output embedding dimension (if different from embed_dim)
        """
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Optional projection to different embedding dimension
        self.embedding_dim = embedding_dim or embed_dim
        if embedding_dim and embedding_dim != embed_dim:
            self.projection = nn.Sequential(
                nn.Linear(embed_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        else:
            self.projection = nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)

        Returns:
            Class token embeddings (B, embedding_dim)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract class token
        cls_token_final = x[:, 0]  # (B, embed_dim)

        # Project to final embedding dimension
        embeddings = self.projection(cls_token_final)

        return embeddings


def build_encoder(
    encoder_type: str = "cnn",
    **kwargs
) -> nn.Module:
    """
    Factory function to build encoders.

    Args:
        encoder_type: 'cnn' or 'vit'
        **kwargs: Arguments passed to encoder constructor

    Returns:
        Encoder module
    """
    if encoder_type.lower() == "cnn":
        return CNNEncoder(**kwargs)
    elif encoder_type.lower() == "vit":
        return ViTEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
