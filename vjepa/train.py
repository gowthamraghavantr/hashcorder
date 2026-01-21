"""
Training script for V-JEPA

Implements the complete training loop with logging and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from typing import Optional

from models import build_encoder, build_predictor, VJEPA, VJEPALoss
from data import create_simple_dataloader
from config import ExperimentConfig, get_cnn_config, get_vit_config, get_debug_config
from utils import (
    set_seed,
    get_device,
    count_parameters,
    MetricsTracker,
    save_checkpoint,
    load_checkpoint,
    save_config,
    log_metrics,
    format_time,
    compute_embedding_statistics
)


class VJEPATrainer:
    """
    Trainer class for V-JEPA experiments.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = get_device(config.training.device)

        # Set random seed
        set_seed(config.seed)

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)

        # Build loss function
        self.criterion = VJEPALoss(
            loss_type=config.model.loss_type,
            temperature=config.model.temperature,
            variance_weight=config.model.variance_weight,
            covariance_weight=config.model.covariance_weight
        )

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Build scheduler
        self.scheduler = self._build_scheduler()

        # Build dataloaders
        self.train_loader = self._build_dataloader()

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')

        # Create checkpoint directory
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = Path(config.training.checkpoint_dir) / "config.json"
        save_config(config, str(config_path))

        print("\n" + "=" * 80)
        print("V-JEPA Training Configuration")
        print("=" * 80)
        print(f"Experiment: {config.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Encoder: {config.model.encoder.encoder_type}")
        print(f"Predictor: {config.model.predictor.predictor_type}")
        print(f"Embedding dim: {config.model.encoder.embedding_dim}")
        print(f"Batch size: {config.data.batch_size}")
        print(f"Learning rate: {config.training.learning_rate}")
        print(f"Max epochs: {config.training.max_epochs}")
        print("=" * 80 + "\n")

    def _build_model(self) -> VJEPA:
        """Build V-JEPA model from config."""
        # Build encoder
        encoder_cfg = self.config.model.encoder
        if encoder_cfg.encoder_type == "cnn":
            encoder_kwargs = {
                'input_channels': encoder_cfg.input_channels,
                'hidden_dims': encoder_cfg.cnn_hidden_dims,
                'embedding_dim': encoder_cfg.embedding_dim,
                'input_size': encoder_cfg.input_size,
                'use_batchnorm': encoder_cfg.cnn_use_batchnorm
            }
        else:  # vit
            encoder_kwargs = {
                'img_size': encoder_cfg.input_size,
                'patch_size': encoder_cfg.vit_patch_size,
                'in_channels': encoder_cfg.input_channels,
                'embed_dim': encoder_cfg.vit_embed_dim,
                'depth': encoder_cfg.vit_depth,
                'num_heads': encoder_cfg.vit_num_heads,
                'mlp_ratio': encoder_cfg.vit_mlp_ratio,
                'dropout': encoder_cfg.vit_dropout,
                'embedding_dim': encoder_cfg.embedding_dim
            }

        encoder = build_encoder(
            encoder_type=encoder_cfg.encoder_type,
            **encoder_kwargs
        )

        # Build predictor
        pred_cfg = self.config.model.predictor
        if pred_cfg.predictor_type == "mlp":
            predictor_kwargs = {
                'embedding_dim': pred_cfg.embedding_dim,
                'hidden_dims': pred_cfg.mlp_hidden_dims,
                'dropout': pred_cfg.mlp_dropout,
                'use_layer_norm': pred_cfg.mlp_use_layer_norm
            }
        elif pred_cfg.predictor_type == "transformer":
            predictor_kwargs = {
                'embedding_dim': pred_cfg.embedding_dim,
                'num_heads': pred_cfg.transformer_num_heads,
                'num_layers': pred_cfg.transformer_num_layers,
                'mlp_ratio': pred_cfg.transformer_mlp_ratio,
                'dropout': pred_cfg.transformer_dropout
            }
        else:  # multistep
            predictor_kwargs = {
                'embedding_dim': pred_cfg.embedding_dim,
                'num_steps': pred_cfg.multistep_num_steps,
                'hidden_dims': pred_cfg.mlp_hidden_dims,
                'share_weights': pred_cfg.multistep_share_weights,
                'dropout': pred_cfg.mlp_dropout
            }

        predictor = build_predictor(
            predictor_type=pred_cfg.predictor_type,
            **predictor_kwargs
        )

        # Build V-JEPA model
        model = VJEPA(
            encoder=encoder,
            predictor=predictor,
            embedding_dim=pred_cfg.embedding_dim,
            momentum=self.config.model.momentum,
            use_momentum_encoder=self.config.model.use_momentum_encoder
        )

        return model

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config."""
        cfg = self.config.training

        if cfg.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg.learning_rate,
                momentum=0.9,
                weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        return optimizer

    def _build_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler from config."""
        cfg = self.config.training

        if cfg.scheduler is None:
            return None

        if cfg.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.max_epochs - cfg.warmup_epochs,
                eta_min=cfg.learning_rate * 0.01
            )
        elif cfg.scheduler.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.max_epochs // 3,
                gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

        return scheduler

    def _build_dataloader(self) -> DataLoader:
        """Build dataloader from config."""
        cfg = self.config.data

        if cfg.dataset_type == "synthetic":
            dataloader = create_simple_dataloader(
                num_samples=cfg.num_samples,
                batch_size=cfg.batch_size,
                img_size=cfg.img_size,
                channels=cfg.channels,
                add_structure=cfg.add_temporal_structure
            )
        else:
            # For other dataset types, use the factory function
            from data import build_dataloader
            dataloader = build_dataloader(
                dataset_type=cfg.dataset_type,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                shuffle=True,
                img_size=cfg.img_size,
                channels=cfg.channels,
                num_samples=cfg.num_samples
            )

        return dataloader

    def train_epoch(self, epoch: int):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
        """
        self.model.train()
        self.metrics_tracker.reset()

        epoch_start_time = time.time()

        for batch_idx, (context_frames, target_frames) in enumerate(self.train_loader):
            # Move to device
            context_frames = context_frames.to(self.device)
            target_frames = target_frames.to(self.device)

            # Forward pass
            output = self.model(context_frames, target_frames, return_embeddings=True)

            # Compute loss
            loss, loss_components = self.criterion(
                output['predicted_emb'],
                output['target_emb']
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            metrics = {
                'loss': loss.item(),
                'cosine_sim': output['cosine_sim'].item(),
            }
            metrics.update({k: v.item() for k, v in loss_components.items()})

            self.metrics_tracker.update(metrics)

            # Log progress
            if (batch_idx + 1) % self.config.training.log_interval == 0:
                log_metrics(
                    epoch=epoch,
                    metrics=self.metrics_tracker.get_averages(),
                    phase="train",
                    step=batch_idx + 1
                )

        epoch_time = time.time() - epoch_start_time

        # Log epoch summary
        print(f"\nEpoch {epoch} completed in {format_time(epoch_time)}")
        print(f"Average metrics: {self.metrics_tracker}")

        # Compute embedding statistics periodically
        if epoch % 10 == 0:
            with torch.no_grad():
                sample_context, sample_target = next(iter(self.train_loader))
                sample_context = sample_context.to(self.device)
                output = self.model(sample_context, sample_target.to(self.device), return_embeddings=True)

                emb_stats = compute_embedding_statistics(output['context_emb'])
                print(f"Embedding statistics:")
                for k, v in emb_stats.items():
                    print(f"  {k}: {v:.4f}")

        return self.metrics_tracker.get_averages()

    def train(self):
        """
        Main training loop.
        """
        print("\nStarting training...\n")

        total_start_time = time.time()

        for epoch in range(self.start_epoch, self.config.training.max_epochs):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{self.config.training.max_epochs}")
            print(f"{'=' * 80}")

            # Train one epoch
            metrics = self.train_epoch(epoch + 1)

            # Update learning rate
            if self.scheduler is not None:
                if epoch >= self.config.training.warmup_epochs:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_interval == 0:
                checkpoint_path = Path(self.config.training.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                is_best = metrics['loss'] < self.best_loss
                if is_best:
                    self.best_loss = metrics['loss']

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    metrics=metrics,
                    checkpoint_path=str(checkpoint_path),
                    is_best=is_best
                )

        total_time = time.time() - total_start_time
        print(f"\n{'=' * 80}")
        print(f"Training completed in {format_time(total_time)}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"{'=' * 80}\n")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        info = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.device
        )
        self.start_epoch = info['epoch']
        self.best_loss = info['metrics'].get('loss', float('inf'))


def main():
    """
    Main entry point for training.
    """
    # Select configuration
    # Options: get_debug_config(), get_cnn_config(), get_vit_config()
    config = get_cnn_config()

    # Create trainer
    trainer = VJEPATrainer(config)

    # Resume from checkpoint if specified
    if config.training.resume_from is not None:
        trainer.resume_from_checkpoint(config.training.resume_from)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
