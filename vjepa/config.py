"""
Configuration for V-JEPA experiments
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class EncoderConfig:
    """Configuration for vision encoder."""

    # Encoder type: 'cnn' or 'vit'
    encoder_type: str = "cnn"

    # Common parameters
    input_channels: int = 3
    embedding_dim: int = 512
    input_size: int = 224

    # CNN-specific parameters
    cnn_hidden_dims: Tuple[int, ...] = (64, 128, 256, 512)
    cnn_use_batchnorm: bool = True

    # ViT-specific parameters
    vit_patch_size: int = 16
    vit_embed_dim: int = 768
    vit_depth: int = 12
    vit_num_heads: int = 12
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.0


@dataclass
class PredictorConfig:
    """Configuration for predictor network."""

    # Predictor type: 'mlp', 'transformer', or 'multistep'
    predictor_type: str = "mlp"

    # Common parameters
    embedding_dim: int = 512

    # MLP-specific parameters
    mlp_hidden_dims: Tuple[int, ...] = (2048, 2048)
    mlp_dropout: float = 0.1
    mlp_use_layer_norm: bool = True

    # Transformer-specific parameters
    transformer_num_heads: int = 8
    transformer_num_layers: int = 4
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.1

    # Multi-step specific parameters
    multistep_num_steps: int = 1
    multistep_share_weights: bool = False


@dataclass
class VJEPAConfig:
    """Configuration for V-JEPA model."""

    # Model architecture
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    # Training parameters
    use_momentum_encoder: bool = False
    momentum: float = 0.996

    # Loss parameters
    loss_type: str = "cosine"  # 'cosine' or 'mse'
    temperature: float = 1.0
    variance_weight: float = 0.0  # VICReg-style regularization
    covariance_weight: float = 0.0


@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Dataset type: 'synthetic', 'image_pair', or 'video'
    dataset_type: str = "synthetic"

    # Common parameters
    batch_size: int = 64
    num_workers: int = 4
    img_size: int = 224
    channels: int = 3

    # Synthetic dataset parameters
    num_samples: int = 10000
    temporal_length: int = 10
    context_length: int = 4
    target_offset: int = 1
    add_temporal_structure: bool = True
    noise_level: float = 0.1

    # Image pair parameters
    augmentation_strength: float = 0.3

    # Video dataset parameters
    video_paths: Optional[list] = None
    temporal_stride: int = 4


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # 'adam', 'adamw', or 'sgd'
    scheduler: Optional[str] = "cosine"  # 'cosine', 'step', or None
    warmup_epochs: int = 10
    max_epochs: int = 100

    # Training loop
    log_interval: int = 10  # Log every N batches
    eval_interval: int = 1  # Evaluate every N epochs
    save_interval: int = 10  # Save checkpoint every N epochs

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None

    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'
    mixed_precision: bool = False

    # Distributed training (optional)
    distributed: bool = False
    world_size: int = 1
    rank: int = 0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Experiment metadata
    experiment_name: str = "vjepa_experiment"
    seed: int = 42

    # Component configs
    model: VJEPAConfig = field(default_factory=VJEPAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Ensure consistency between encoder and predictor dimensions."""
        # Sync embedding dimensions
        if self.model.encoder.encoder_type == "vit":
            self.model.encoder.embedding_dim = self.model.encoder.vit_embed_dim

        self.model.predictor.embedding_dim = self.model.encoder.embedding_dim

        # Sync data config with encoder
        self.data.img_size = self.model.encoder.input_size
        self.data.channels = self.model.encoder.input_channels


# Preset configurations

def get_cnn_config() -> ExperimentConfig:
    """Small CNN-based configuration for quick experiments."""
    config = ExperimentConfig(
        experiment_name="vjepa_cnn_small"
    )
    config.model.encoder.encoder_type = "cnn"
    config.model.encoder.embedding_dim = 512
    config.model.predictor.predictor_type = "mlp"
    config.data.batch_size = 64
    config.training.max_epochs = 50
    return config


def get_vit_config() -> ExperimentConfig:
    """ViT-based configuration for larger-scale experiments."""
    config = ExperimentConfig(
        experiment_name="vjepa_vit_base"
    )
    config.model.encoder.encoder_type = "vit"
    config.model.encoder.vit_patch_size = 16
    config.model.encoder.vit_embed_dim = 768
    config.model.encoder.vit_depth = 12
    config.model.encoder.vit_num_heads = 12
    config.model.encoder.embedding_dim = 768
    config.model.predictor.predictor_type = "mlp"
    config.model.predictor.embedding_dim = 768
    config.data.batch_size = 32
    config.training.max_epochs = 100
    config.training.learning_rate = 1e-4
    return config


def get_debug_config() -> ExperimentConfig:
    """Minimal configuration for debugging."""
    config = ExperimentConfig(
        experiment_name="vjepa_debug"
    )
    config.model.encoder.encoder_type = "cnn"
    config.model.encoder.cnn_hidden_dims = (32, 64)
    config.model.encoder.embedding_dim = 128
    config.model.predictor.mlp_hidden_dims = (256,)
    config.model.predictor.embedding_dim = 128
    config.data.batch_size = 8
    config.data.num_samples = 100
    config.training.max_epochs = 5
    config.training.log_interval = 1
    return config
