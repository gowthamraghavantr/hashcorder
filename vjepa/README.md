# V-JEPA: Vision Joint-Embedding Predictive Architecture

A **minimal, research-grade PyTorch implementation** of Meta's V-JEPA for self-supervised vision representation learning.

## Core Principles

V-JEPA learns visual representations by **predicting in latent space** rather than pixel space:

1. **Joint-Embedding**: Context and target frames share the same embedding space
2. **Predictive**: Predictor forecasts target embeddings from context embeddings
3. **Stop-Gradient**: Target encoder uses stop-gradient to prevent collapse
4. **No Reconstruction**: Learns representations without pixel-level reconstruction

## Architecture Overview

```
Context Frame          Target Frame
     ↓                      ↓
  Encoder   ←(shared)→   Encoder (stop-grad)
     ↓                      ↓
Context Emb            Target Emb
     ↓                      ↓
 Predictor  -------→   [Cosine Loss]
     ↓                      ↑
Predicted Emb  -----------→
```

### Key Components

#### 1. Vision Encoder
- **CNN Encoder**: Convolutional architecture with progressive downsampling
- **ViT Encoder**: Vision Transformer with patch embeddings and self-attention
- Outputs fixed-dimensional embeddings (default: 512D)

#### 2. Predictor Network
- **MLP Predictor**: Multi-layer perceptron with LayerNorm
- **Transformer Predictor**: For temporal sequence modeling
- **Multi-step Predictor**: Forecasts multiple future timesteps
- Maps context embeddings to predicted target embeddings

#### 3. Training Mechanism
- **Stop-Gradient**: Target embeddings are detached from computation graph
- **Cosine Similarity Loss**: `loss = 1 - cosine_similarity(predicted, target)`
- **Optional Regularization**: VICReg-style variance and covariance terms

## Installation

```bash
# Clone repository (if not already done)
cd vjepa

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```python
from vjepa import get_cnn_config
from vjepa.train import VJEPATrainer

# Load configuration
config = get_cnn_config()  # or get_vit_config(), get_debug_config()

# Create trainer
trainer = VJEPATrainer(config)

# Train
trainer.train()
```

### Command Line

```bash
# Train with default CNN configuration
python -m vjepa.train

# Run quick experiment with evaluation
python -m vjepa.experiments.evaluate
```

## Configuration

Three preset configurations are available:

### 1. Debug Config (Fast Iteration)
```python
from vjepa.config import get_debug_config
config = get_debug_config()
```
- Small CNN (32→64 channels)
- 128D embeddings
- 100 samples, 5 epochs
- Perfect for testing

### 2. CNN Config (Balanced)
```python
from vjepa.config import get_cnn_config
config = get_cnn_config()
```
- CNN encoder (64→128→256→512 channels)
- 512D embeddings
- 10K samples, 50 epochs
- Good for medium-scale experiments

### 3. ViT Config (Large Scale)
```python
from vjepa.config import get_vit_config
config = get_vit_config()
```
- Vision Transformer (12 layers, 12 heads)
- 768D embeddings
- 10K samples, 100 epochs
- Research-grade setup

### Custom Configuration

```python
from vjepa.config import ExperimentConfig, EncoderConfig, PredictorConfig

# Create custom config
config = ExperimentConfig(
    experiment_name="my_experiment"
)

# Customize encoder
config.model.encoder.encoder_type = "cnn"
config.model.encoder.embedding_dim = 256

# Customize predictor
config.model.predictor.predictor_type = "mlp"
config.model.predictor.mlp_hidden_dims = (1024, 1024)

# Customize training
config.training.learning_rate = 3e-4
config.training.max_epochs = 200
```

## Architecture Details

### Encoder Options

#### CNN Encoder (`vjepa/models/encoder.py:CNNEncoder`)
```python
CNNEncoder(
    input_channels=3,
    hidden_dims=(64, 128, 256, 512),
    embedding_dim=512,
    input_size=224,
    use_batchnorm=True
)
```

#### ViT Encoder (`vjepa/models/encoder.py:ViTEncoder`)
```python
ViTEncoder(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.0
)
```

### Predictor Options

#### MLP Predictor (`vjepa/models/predictor.py:MLPPredictor`)
```python
MLPPredictor(
    embedding_dim=512,
    hidden_dims=(2048, 2048),
    dropout=0.1,
    use_layer_norm=True
)
```

#### Transformer Predictor (`vjepa/models/predictor.py:TransformerPredictor`)
```python
TransformerPredictor(
    embedding_dim=512,
    num_heads=8,
    num_layers=4,
    mlp_ratio=4.0,
    dropout=0.1
)
```

### Core V-JEPA Model (`vjepa/models/vjepa.py:VJEPA`)

```python
from vjepa.models import VJEPA, build_encoder, build_predictor

# Build components
encoder = build_encoder(encoder_type="cnn", embedding_dim=512)
predictor = build_predictor(predictor_type="mlp", embedding_dim=512)

# Create V-JEPA model
model = VJEPA(
    encoder=encoder,
    predictor=predictor,
    embedding_dim=512,
    use_momentum_encoder=False  # Set True for EMA target encoder
)

# Forward pass
output = model(context_frames, target_frames)
print(output['loss'])  # V-JEPA loss
print(output['cosine_sim'])  # Cosine similarity
```

## Data Loading

### Synthetic Data (Default)

```python
from vjepa.data import create_simple_dataloader

# Create synthetic video data with temporal structure
dataloader = create_simple_dataloader(
    num_samples=1000,
    batch_size=32,
    img_size=224,
    channels=3,
    add_structure=True  # Smooth temporal transitions
)

for context, target in dataloader:
    # context: (B, 3, 224, 224)
    # target: (B, 3, 224, 224)
    pass
```

### Custom Data

Extend `VideoFrameDataset` for real video data:

```python
from vjepa.data.loader import VideoFrameDataset
import torch
from torch.utils.data import DataLoader

class MyVideoDataset(VideoFrameDataset):
    def __getitem__(self, idx):
        # Load video frames using decord/torchvision
        video_path = self.video_paths[idx]

        # TODO: Implement video loading
        frames = load_video(video_path)  # Your implementation

        # Sample context and target
        context_frame = frames[t]
        target_frame = frames[t + offset]

        return context_frame, target_frame

# Use with DataLoader
dataset = MyVideoDataset(video_paths=["video1.mp4", "video2.mp4"])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Evaluation

### Embedding Quality Analysis

```python
from vjepa.experiments import VJEPAEvaluator

evaluator = VJEPAEvaluator(model, device, dataloader)

# Prediction accuracy
metrics = evaluator.compute_prediction_accuracy()
print(metrics)  # {'cosine_similarity': 0.85, 'l2_distance': 1.23}

# Embedding quality
quality = evaluator.analyze_embedding_quality()
print(quality)  # Stats on embedding norms, diversity, effective rank

# Visualize embeddings in 2D
evaluator.visualize_embeddings(save_path="embeddings.png", method="pca")
```

### Load and Evaluate Checkpoint

```python
from vjepa.experiments import evaluate_model

metrics = evaluate_model(
    checkpoint_path="checkpoints/best_model.pt",
    model=model,
    dataloader=eval_dataloader,
    device=device
)
```

## Key Implementation Details

### Stop-Gradient Mechanism

The target encoder **must** use stop-gradient to prevent representation collapse:

```python
# In vjepa/models/vjepa.py:VJEPA.encode_target()
def encode_target(self, target_frames):
    with torch.no_grad():
        embeddings = self.target_encoder(target_frames)
    return embeddings.detach()  # Crucial: stop gradient
```

### Loss Function

```python
# Cosine similarity loss (higher is better)
predicted_norm = F.normalize(predicted_emb, dim=-1)
target_norm = F.normalize(target_emb, dim=-1)
cosine_sim = (predicted_norm * target_norm).sum(dim=-1).mean()

# Minimize negative similarity
loss = 1.0 - cosine_sim
```

### Momentum Encoder (Optional)

Enable exponential moving average (EMA) for target encoder:

```python
config.model.use_momentum_encoder = True
config.model.momentum = 0.996  # EMA coefficient

# Target encoder updated as: θ_target = m·θ_target + (1-m)·θ_context
```

## Extensions and Future Work

### 1. Temporal Masking
Mask random patches in context and predict full target:

```python
# Add to encoder forward pass
def forward_with_mask(self, x, mask):
    # Apply mask to patch embeddings
    x = self.patch_embed(x)
    x = x * mask.unsqueeze(-1)  # Mask out patches
    # ... rest of forward pass
```

### 2. Multi-Step Prediction

Already implemented via `MultiStepPredictor`:

```python
from vjepa.models import MultiStepPredictor

predictor = MultiStepPredictor(
    embedding_dim=512,
    num_steps=5,  # Predict 5 future timesteps
    share_weights=False  # Separate predictor per step
)

# Predict multiple futures
predictions = predictor(context_emb)  # (B, 5, 512)
```

### 3. Video Dataset Integration

Replace synthetic data with real videos:

```python
# Use decord for efficient video loading
import decord
from decord import VideoReader

class RealVideoDataset(VideoFrameDataset):
    def __getitem__(self, idx):
        vr = VideoReader(self.video_paths[idx])
        frames = vr.get_batch(indices)  # Sample frame indices
        # Process and return context/target
```

### 4. Downstream Task Evaluation

Freeze encoder and train linear probe on classification:

```python
# Freeze encoder
for param in model.context_encoder.parameters():
    param.requires_grad = False

# Add linear classifier
classifier = nn.Linear(512, num_classes)

# Train classifier only
for data, labels in dataloader:
    features = model.encode_context(data)
    logits = classifier(features)
    loss = F.cross_entropy(logits, labels)
```

### 5. Agent-Based Orchestration

Use V-JEPA within an LLM-driven agent controller:

```python
class VJEPAAgent:
    """Agent wrapper for V-JEPA model."""

    def __init__(self, model):
        self.model = model
        self.embedding_history = []

    def process_frame(self, frame):
        """Process single frame and update history."""
        emb = self.model.encode_context(frame)
        self.embedding_history.append(emb)
        return emb

    def predict_future(self, steps=1):
        """Predict future embeddings."""
        context = self.embedding_history[-1]
        predictions = self.model.predict(context)
        return predictions

    def evaluate_prediction_quality(self):
        """Return metrics for LLM decision making."""
        if len(self.embedding_history) < 2:
            return {"status": "insufficient_data"}

        # Compute recent prediction accuracy
        recent_context = self.embedding_history[-2]
        recent_target = self.embedding_history[-1]
        predicted = self.model.predict(recent_context)

        similarity = cosine_similarity(predicted, recent_target)

        return {
            "status": "ready",
            "prediction_quality": similarity.item(),
            "embedding_diversity": compute_diversity(self.embedding_history),
            "recommendation": "good" if similarity > 0.8 else "needs_improvement"
        }

# Use with LLM controller
agent = VJEPAAgent(model)

for frame in video_stream:
    embedding = agent.process_frame(frame)
    quality_report = agent.evaluate_prediction_quality()

    # LLM decides next action based on quality report
    llm_decision = llm_controller(quality_report)
```

## Project Structure

```
vjepa/
├── __init__.py              # Package initialization
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── encoder.py          # CNN and ViT encoders
│   ├── predictor.py        # Predictor networks
│   └── vjepa.py            # Core V-JEPA model
├── data/                    # Data loading
│   ├── __init__.py
│   └── loader.py           # Dataset and DataLoader utilities
├── experiments/             # Evaluation and experiments
│   ├── __init__.py
│   └── evaluate.py         # Evaluation harness
├── config.py               # Configuration dataclasses
├── utils.py                # Utility functions
├── train.py                # Training script
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Performance Tips

1. **Use Mixed Precision**: Enable for faster training on modern GPUs
   ```python
   config.training.mixed_precision = True
   ```

2. **Adjust Batch Size**: Larger batches improve embedding statistics
   ```python
   config.data.batch_size = 128  # If GPU memory allows
   ```

3. **Warmup Learning Rate**: Prevents early training instability
   ```python
   config.training.warmup_epochs = 10
   ```

4. **Monitor Embedding Diversity**: Low diversity indicates collapse
   ```python
   # Check effective rank in training logs
   # Should be > 0.5 * embedding_dim for good representations
   ```

## Troubleshooting

### Representation Collapse

**Symptoms**: Cosine similarity stuck at 1.0, all embeddings identical

**Solutions**:
- Increase variance regularization: `config.model.variance_weight = 0.1`
- Reduce learning rate: `config.training.learning_rate = 1e-5`
- Add covariance regularization: `config.model.covariance_weight = 0.01`

### Poor Prediction Accuracy

**Symptoms**: Cosine similarity < 0.5

**Solutions**:
- Increase model capacity: larger encoder or predictor
- Add temporal structure to data: `add_temporal_structure=True`
- Train longer: `config.training.max_epochs = 200`

### Out of Memory

**Solutions**:
- Reduce batch size: `config.data.batch_size = 16`
- Use smaller encoder: `get_debug_config()`
- Enable gradient checkpointing (requires manual implementation)

## Citation

Based on Meta's V-JEPA architecture. For the original paper:

```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and LeCun, Yann},
  journal={arXiv preprint arXiv:2404.08471},
  year={2024}
}
```

## License

MIT License - See parent repository for details.

## Contributing

This is a minimal research implementation. Extensions welcome:
- Real video dataset loaders
- Distributed training support
- Advanced masking strategies
- Downstream task evaluation pipelines

---

**Contact**: For questions about this implementation, open an issue in the parent repository.
