"""
Evaluation and experiment harness for V-JEPA

Provides tools for evaluating trained models and running experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import VJEPA
from data import create_simple_dataloader
from utils import (
    get_device,
    load_checkpoint,
    compute_embedding_statistics,
    visualize_embeddings_2d,
    AverageMeter
)


class VJEPAEvaluator:
    """
    Evaluator for V-JEPA models.

    Provides methods for:
    - Computing embedding quality metrics
    - Visualizing embeddings
    - Analyzing prediction accuracy
    - Temporal consistency evaluation
    """

    def __init__(
        self,
        model: VJEPA,
        device: torch.device,
        dataloader: DataLoader
    ):
        """
        Args:
            model: Trained V-JEPA model
            device: Torch device
            dataloader: Data loader for evaluation
        """
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.model.eval()

    @torch.no_grad()
    def compute_prediction_accuracy(self) -> Dict[str, float]:
        """
        Compute prediction accuracy metrics.

        Returns:
            Dictionary of metrics
        """
        cosine_sim_meter = AverageMeter("cosine_sim")
        l2_dist_meter = AverageMeter("l2_dist")

        for context_frames, target_frames in self.dataloader:
            context_frames = context_frames.to(self.device)
            target_frames = target_frames.to(self.device)

            # Forward pass
            output = self.model(context_frames, target_frames, return_embeddings=True)

            # Compute distances
            cosine_sim = output['cosine_sim'].item()
            l2_dist = torch.norm(
                output['predicted_emb'] - output['target_emb'],
                p=2,
                dim=-1
            ).mean().item()

            cosine_sim_meter.update(cosine_sim, context_frames.size(0))
            l2_dist_meter.update(l2_dist, context_frames.size(0))

        return {
            'cosine_similarity': cosine_sim_meter.avg,
            'l2_distance': l2_dist_meter.avg
        }

    @torch.no_grad()
    def extract_embeddings(
        self,
        max_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from dataset.

        Args:
            max_samples: Maximum number of samples to process

        Returns:
            Tuple of (context_embeddings, target_embeddings, predicted_embeddings)
        """
        context_embs = []
        target_embs = []
        predicted_embs = []

        num_processed = 0

        for context_frames, target_frames in self.dataloader:
            context_frames = context_frames.to(self.device)
            target_frames = target_frames.to(self.device)

            # Encode
            output = self.model(context_frames, target_frames, return_embeddings=True)

            context_embs.append(output['context_emb'].cpu())
            target_embs.append(output['target_emb'].cpu())
            predicted_embs.append(output['predicted_emb'].cpu())

            num_processed += context_frames.size(0)

            if max_samples is not None and num_processed >= max_samples:
                break

        # Concatenate
        context_embs = torch.cat(context_embs, dim=0)
        target_embs = torch.cat(target_embs, dim=0)
        predicted_embs = torch.cat(predicted_embs, dim=0)

        return context_embs, target_embs, predicted_embs

    def analyze_embedding_quality(self) -> Dict[str, float]:
        """
        Analyze quality of learned embeddings.

        Returns:
            Dictionary of quality metrics
        """
        # Extract embeddings
        context_embs, target_embs, predicted_embs = self.extract_embeddings(max_samples=1000)

        # Compute statistics for each type
        context_stats = compute_embedding_statistics(context_embs)
        target_stats = compute_embedding_statistics(target_embs)
        predicted_stats = compute_embedding_statistics(predicted_embs)

        # Combine into single report
        report = {}
        for prefix, stats in [
            ('context', context_stats),
            ('target', target_stats),
            ('predicted', predicted_stats)
        ]:
            for key, value in stats.items():
                report[f'{prefix}_{key}'] = value

        return report

    def visualize_embeddings(
        self,
        save_path: Optional[str] = None,
        method: str = "pca"
    ):
        """
        Visualize embeddings in 2D.

        Args:
            save_path: Optional path to save figure
            method: Dimensionality reduction method ('pca' or 'tsne')
        """
        # Extract embeddings
        context_embs, target_embs, predicted_embs = self.extract_embeddings(max_samples=500)

        # Reduce to 2D
        context_2d = visualize_embeddings_2d(context_embs, method=method)
        target_2d = visualize_embeddings_2d(target_embs, method=method)
        predicted_2d = visualize_embeddings_2d(predicted_embs, method=method)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].scatter(context_2d[:, 0], context_2d[:, 1], alpha=0.5, s=10)
        axes[0].set_title("Context Embeddings")
        axes[0].set_xlabel(f"{method.upper()} 1")
        axes[0].set_ylabel(f"{method.upper()} 2")

        axes[1].scatter(target_2d[:, 0], target_2d[:, 1], alpha=0.5, s=10, c='orange')
        axes[1].set_title("Target Embeddings")
        axes[1].set_xlabel(f"{method.upper()} 1")
        axes[1].set_ylabel(f"{method.upper()} 2")

        axes[2].scatter(predicted_2d[:, 0], predicted_2d[:, 1], alpha=0.5, s=10, c='green')
        axes[2].set_title("Predicted Embeddings")
        axes[2].set_xlabel(f"{method.upper()} 1")
        axes[2].set_ylabel(f"{method.upper()} 2")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()

    def compute_prediction_errors(self) -> Dict[str, np.ndarray]:
        """
        Compute per-sample prediction errors.

        Returns:
            Dictionary with error arrays
        """
        cosine_sims = []
        l2_dists = []

        with torch.no_grad():
            for context_frames, target_frames in self.dataloader:
                context_frames = context_frames.to(self.device)
                target_frames = target_frames.to(self.device)

                output = self.model(context_frames, target_frames, return_embeddings=True)

                # Compute per-sample metrics
                pred_norm = torch.nn.functional.normalize(output['predicted_emb'], dim=-1)
                target_norm = torch.nn.functional.normalize(output['target_emb'], dim=-1)
                cos_sim = (pred_norm * target_norm).sum(dim=-1)

                l2_dist = torch.norm(
                    output['predicted_emb'] - output['target_emb'],
                    p=2,
                    dim=-1
                )

                cosine_sims.append(cos_sim.cpu().numpy())
                l2_dists.append(l2_dist.cpu().numpy())

        return {
            'cosine_similarities': np.concatenate(cosine_sims),
            'l2_distances': np.concatenate(l2_dists)
        }


def evaluate_model(
    checkpoint_path: str,
    model: VJEPA,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model: Model instance (architecture)
        dataloader: Data loader
        device: Torch device

    Returns:
        Dictionary of evaluation metrics
    """
    # Load checkpoint
    load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    model.eval()

    # Create evaluator
    evaluator = VJEPAEvaluator(model, device, dataloader)

    print("\n" + "=" * 80)
    print("Evaluating V-JEPA Model")
    print("=" * 80)

    # Compute prediction accuracy
    print("\nPrediction Accuracy:")
    pred_metrics = evaluator.compute_prediction_accuracy()
    for key, value in pred_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Analyze embedding quality
    print("\nEmbedding Quality Analysis:")
    quality_metrics = evaluator.analyze_embedding_quality()
    for key, value in quality_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Combine all metrics
    all_metrics = {**pred_metrics, **quality_metrics}

    print("\n" + "=" * 80 + "\n")

    return all_metrics


def run_experiment(
    config_name: str = "cnn",
    num_epochs: int = 10,
    visualize: bool = True
):
    """
    Run a complete experiment: train and evaluate.

    Args:
        config_name: Configuration preset ('cnn', 'vit', or 'debug')
        num_epochs: Number of training epochs
        visualize: Whether to visualize embeddings
    """
    from config import get_cnn_config, get_vit_config, get_debug_config
    from train import VJEPATrainer

    # Select config
    if config_name == "cnn":
        config = get_cnn_config()
    elif config_name == "vit":
        config = get_vit_config()
    elif config_name == "debug":
        config = get_debug_config()
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Override epochs
    config.training.max_epochs = num_epochs

    print(f"\n{'=' * 80}")
    print(f"Running Experiment: {config.experiment_name}")
    print(f"{'=' * 80}\n")

    # Train model
    trainer = VJEPATrainer(config)
    trainer.train()

    # Evaluate
    device = get_device(config.training.device)
    eval_dataloader = create_simple_dataloader(
        num_samples=500,
        batch_size=32,
        img_size=config.data.img_size,
        channels=config.data.channels
    )

    best_checkpoint = Path(config.training.checkpoint_dir) / "best_model.pt"
    if best_checkpoint.exists():
        metrics = evaluate_model(
            str(best_checkpoint),
            trainer.model,
            eval_dataloader,
            device
        )

        # Visualize if requested
        if visualize:
            evaluator = VJEPAEvaluator(trainer.model, device, eval_dataloader)
            viz_path = Path(config.training.checkpoint_dir) / "embeddings_visualization.png"
            evaluator.visualize_embeddings(save_path=str(viz_path))

        return metrics
    else:
        print("No checkpoint found for evaluation")
        return {}


if __name__ == "__main__":
    # Run a quick experiment
    run_experiment(config_name="debug", num_epochs=5, visualize=True)
