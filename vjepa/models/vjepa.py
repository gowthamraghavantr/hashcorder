"""
V-JEPA: Vision Joint-Embedding Predictive Architecture

Core model implementation with stop-gradient and cosine similarity loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy


class VJEPA(nn.Module):
    """
    Vision Joint-Embedding Predictive Architecture.

    Key principles:
    1. Shared encoder for context and target
    2. Stop-gradient on target embeddings
    3. Predictor maps context → predicted target
    4. Training via cosine similarity in latent space
    5. No pixel reconstruction
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        embedding_dim: int,
        momentum: float = 0.996,
        use_momentum_encoder: bool = False
    ):
        """
        Args:
            encoder: Vision encoder (shared for context and target)
            predictor: Predictor network (context → target embedding)
            embedding_dim: Dimension of latent embeddings
            momentum: Momentum coefficient for target encoder updates (if used)
            use_momentum_encoder: Whether to use momentum encoder for target
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.momentum = momentum
        self.use_momentum_encoder = use_momentum_encoder

        # Context encoder (trainable)
        self.context_encoder = encoder

        # Target encoder
        if use_momentum_encoder:
            # Momentum encoder (exponential moving average)
            self.target_encoder = copy.deepcopy(encoder)
            # Freeze momentum encoder
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            # Shared encoder with stop-gradient
            self.target_encoder = self.context_encoder

        # Predictor network
        self.predictor = predictor

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """
        Update momentum encoder parameters using exponential moving average.

        θ_target = m * θ_target + (1 - m) * θ_context
        """
        if not self.use_momentum_encoder:
            return

        for param_context, param_target in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_target.data.mul_(self.momentum).add_(
                param_context.data, alpha=1 - self.momentum
            )

    def encode_context(self, context_frames: torch.Tensor) -> torch.Tensor:
        """
        Encode context frames.

        Args:
            context_frames: Input frames (B, C, H, W) or (B, T, C, H, W)

        Returns:
            Context embeddings (B, D) or (B, T, D)
        """
        # Handle batched frames
        if context_frames.dim() == 5:
            B, T, C, H, W = context_frames.shape
            context_frames = context_frames.view(B * T, C, H, W)
            embeddings = self.context_encoder(context_frames)
            embeddings = embeddings.view(B, T, -1)
        else:
            embeddings = self.context_encoder(context_frames)

        return embeddings

    def encode_target(self, target_frames: torch.Tensor) -> torch.Tensor:
        """
        Encode target frames with stop-gradient.

        Args:
            target_frames: Target frames (B, C, H, W) or (B, T, C, H, W)

        Returns:
            Target embeddings (B, D) or (B, T, D) - detached from computation graph
        """
        # Handle batched frames
        if target_frames.dim() == 5:
            B, T, C, H, W = target_frames.shape
            target_frames = target_frames.view(B * T, C, H, W)

            with torch.no_grad():
                embeddings = self.target_encoder(target_frames)

            embeddings = embeddings.view(B, T, -1)
        else:
            with torch.no_grad():
                embeddings = self.target_encoder(target_frames)

        # Stop gradient (crucial for V-JEPA)
        return embeddings.detach()

    def predict(self, context_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict target embeddings from context.

        Args:
            context_embeddings: Context embeddings (B, D) or (B, T, D)

        Returns:
            Predicted target embeddings (B, D) or (B, T, D)
        """
        return self.predictor(context_embeddings)

    def forward(
        self,
        context_frames: torch.Tensor,
        target_frames: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode context and target, predict, compute loss.

        Args:
            context_frames: Context input (B, C, H, W)
            target_frames: Target input (B, C, H, W)
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary containing:
                - loss: V-JEPA loss (1 - cosine_similarity)
                - cosine_sim: Cosine similarity between predicted and target
                - [optional] context_emb, target_emb, predicted_emb
        """
        # Encode context (trainable)
        context_emb = self.encode_context(context_frames)

        # Encode target (stop-gradient)
        target_emb = self.encode_target(target_frames)

        # Predict target from context
        predicted_emb = self.predict(context_emb)

        # Compute cosine similarity
        # Normalize embeddings for stable similarity computation
        predicted_norm = F.normalize(predicted_emb, dim=-1)
        target_norm = F.normalize(target_emb, dim=-1)

        # Cosine similarity (higher is better)
        cosine_sim = (predicted_norm * target_norm).sum(dim=-1).mean()

        # V-JEPA loss: maximize cosine similarity
        # Equivalent to minimizing (1 - cosine_sim) or negative cosine_sim
        loss = 1.0 - cosine_sim

        # Update momentum encoder if used
        if self.training and self.use_momentum_encoder:
            self._update_momentum_encoder()

        # Prepare output
        output = {
            'loss': loss,
            'cosine_sim': cosine_sim.detach()
        }

        if return_embeddings:
            output.update({
                'context_emb': context_emb.detach(),
                'target_emb': target_emb,
                'predicted_emb': predicted_emb.detach()
            })

        return output

    def compute_embedding_distance(
        self,
        predicted_emb: torch.Tensor,
        target_emb: torch.Tensor,
        metric: str = 'cosine'
    ) -> torch.Tensor:
        """
        Compute distance between predicted and target embeddings.

        Args:
            predicted_emb: Predicted embeddings (B, D)
            target_emb: Target embeddings (B, D)
            metric: Distance metric ('cosine', 'l2', or 'l1')

        Returns:
            Distance tensor (B,)
        """
        if metric == 'cosine':
            # Cosine similarity (higher = more similar)
            pred_norm = F.normalize(predicted_emb, dim=-1)
            target_norm = F.normalize(target_emb, dim=-1)
            return (pred_norm * target_norm).sum(dim=-1)

        elif metric == 'l2':
            # Euclidean distance (lower = more similar)
            return torch.norm(predicted_emb - target_emb, p=2, dim=-1)

        elif metric == 'l1':
            # Manhattan distance (lower = more similar)
            return torch.norm(predicted_emb - target_emb, p=1, dim=-1)

        else:
            raise ValueError(f"Unknown metric: {metric}")


class VJEPALoss(nn.Module):
    """
    V-JEPA loss function with optional regularization.
    """

    def __init__(
        self,
        loss_type: str = 'cosine',
        temperature: float = 1.0,
        variance_weight: float = 0.0,
        covariance_weight: float = 0.0
    ):
        """
        Args:
            loss_type: 'cosine' or 'mse'
            temperature: Temperature scaling for cosine similarity
            variance_weight: Weight for variance regularization (VICReg-style)
            covariance_weight: Weight for covariance regularization (VICReg-style)
        """
        super().__init__()

        self.loss_type = loss_type
        self.temperature = temperature
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predicted: Predicted embeddings (B, D)
            target: Target embeddings (B, D)

        Returns:
            Total loss and dictionary of loss components
        """
        losses = {}

        # Main prediction loss
        if self.loss_type == 'cosine':
            pred_norm = F.normalize(predicted, dim=-1)
            target_norm = F.normalize(target, dim=-1)
            cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean()
            prediction_loss = (1.0 - cosine_sim) / self.temperature
            losses['cosine_sim'] = cosine_sim.detach()
        else:  # MSE
            prediction_loss = F.mse_loss(predicted, target)

        losses['prediction'] = prediction_loss

        total_loss = prediction_loss

        # Optional variance regularization (prevent collapse)
        if self.variance_weight > 0:
            std_pred = torch.sqrt(predicted.var(dim=0) + 1e-4)
            std_target = torch.sqrt(target.var(dim=0) + 1e-4)
            variance_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))
            losses['variance'] = variance_loss
            total_loss = total_loss + self.variance_weight * variance_loss

        # Optional covariance regularization (decorrelate dimensions)
        if self.covariance_weight > 0:
            B, D = predicted.shape
            pred_centered = predicted - predicted.mean(dim=0)
            cov_pred = (pred_centered.T @ pred_centered) / (B - 1)
            covariance_loss = (cov_pred.pow(2).sum() - cov_pred.diag().pow(2).sum()) / D
            losses['covariance'] = covariance_loss
            total_loss = total_loss + self.covariance_weight * covariance_loss

        losses['total'] = total_loss

        return total_loss, losses
