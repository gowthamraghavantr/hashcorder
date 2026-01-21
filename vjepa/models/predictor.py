"""
Predictor Network for V-JEPA
Maps context embeddings to predicted target embeddings
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MLPPredictor(nn.Module):
    """
    MLP-based predictor for V-JEPA.

    Takes context embedding and predicts target embedding in latent space.
    Uses layer normalization for stable training.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: Tuple[int, ...] = (2048, 2048),
        dropout: float = 0.0,
        use_layer_norm: bool = True
    ):
        """
        Args:
            embedding_dim: Dimension of input/output embeddings
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            use_layer_norm: Whether to use LayerNorm (vs BatchNorm)
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        layers = []
        in_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            in_dim = hidden_dim

        # Final projection back to embedding space
        layers.append(nn.Linear(in_dim, embedding_dim))

        # Optional final normalization for stable cosine similarity
        if use_layer_norm:
            layers.append(nn.LayerNorm(embedding_dim))

        self.predictor = nn.Sequential(*layers)

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_embedding: Context representation (B, embedding_dim)

        Returns:
            Predicted target embedding (B, embedding_dim)
        """
        return self.predictor(context_embedding)


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor for temporal sequence prediction.

    Useful for multi-step prediction or when context involves
    multiple frames/patches.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim)
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(
        self,
        context_embedding: torch.Tensor,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Args:
            context_embedding: Context representation
                - If 2D: (B, embedding_dim) - single embedding per sample
                - If 3D: (B, T, embedding_dim) - sequence of embeddings
            return_sequence: If True, return full sequence; else return last token

        Returns:
            Predicted target embedding (B, embedding_dim) or (B, T, embedding_dim)
        """
        # Ensure 3D input
        if context_embedding.dim() == 2:
            context_embedding = context_embedding.unsqueeze(1)  # (B, 1, D)

        # Apply transformer
        transformed = self.transformer(context_embedding)  # (B, T, D)

        # Project output
        output = self.output_proj(transformed)

        if return_sequence:
            return output
        else:
            # Return last token as prediction
            return output[:, -1, :]  # (B, D)


class MultiStepPredictor(nn.Module):
    """
    Predictor for multiple future timesteps.

    Useful for predicting sequences of future embeddings from
    a single or multiple context embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_steps: int = 1,
        hidden_dims: Tuple[int, ...] = (2048, 2048),
        share_weights: bool = False,
        dropout: float = 0.0
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_steps: Number of future steps to predict
            hidden_dims: Hidden layer dimensions for each predictor
            share_weights: Whether to share predictor weights across steps
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_steps = num_steps
        self.share_weights = share_weights

        if share_weights:
            # Single shared predictor
            self.predictors = nn.ModuleList([
                MLPPredictor(embedding_dim, hidden_dims, dropout)
            ])
        else:
            # Separate predictor for each step
            self.predictors = nn.ModuleList([
                MLPPredictor(embedding_dim, hidden_dims, dropout)
                for _ in range(num_steps)
            ])

    def forward(
        self,
        context_embedding: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            context_embedding: Context representation (B, embedding_dim)
            num_steps: Override number of prediction steps

        Returns:
            Predicted embeddings (B, num_steps, embedding_dim)
        """
        num_steps = num_steps or self.num_steps
        predictions = []

        current_input = context_embedding

        for step in range(num_steps):
            # Select predictor (shared or step-specific)
            predictor_idx = 0 if self.share_weights else step
            predictor = self.predictors[predictor_idx]

            # Predict next embedding
            predicted = predictor(current_input)
            predictions.append(predicted)

            # For autoregressive prediction, use prediction as next input
            # (in practice, V-JEPA typically predicts independently from context)
            # Uncomment below for autoregressive mode:
            # current_input = predicted.detach()

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # (B, num_steps, D)

        return predictions


def build_predictor(
    predictor_type: str = "mlp",
    **kwargs
) -> nn.Module:
    """
    Factory function to build predictors.

    Args:
        predictor_type: 'mlp', 'transformer', or 'multistep'
        **kwargs: Arguments passed to predictor constructor

    Returns:
        Predictor module
    """
    if predictor_type.lower() == "mlp":
        return MLPPredictor(**kwargs)
    elif predictor_type.lower() == "transformer":
        return TransformerPredictor(**kwargs)
    elif predictor_type.lower() == "multistep":
        return MultiStepPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
