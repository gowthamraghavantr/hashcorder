# V-JEPA Architecture Deep Dive

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Architecture Components](#architecture-components)
3. [Training Dynamics](#training-dynamics)
4. [Implementation Details](#implementation-details)
5. [Design Decisions](#design-decisions)

---

## Core Concepts

### Joint-Embedding Predictive Architecture (JEPA)

V-JEPA is based on Yann LeCun's JEPA framework for self-supervised learning:

```
Traditional SSL: x → Encoder → Reconstruct x (pixel space)
JEPA: x → Encoder → z, predict z' (latent space)
```

**Key insight**: Predicting in latent space forces the model to learn semantic features rather than low-level pixel patterns.

### Stop-Gradient: The Critical Mechanism

```python
# Context path (trainable)
context_emb = encoder(context_frames)
predicted_emb = predictor(context_emb)

# Target path (stop-gradient)
with torch.no_grad():
    target_emb = encoder(target_frames).detach()

# Loss in latent space
loss = 1 - cosine_similarity(predicted_emb, target_emb)
```

**Why stop-gradient?**
- Prevents trivial solution: encoder outputs constant embeddings
- Forces predictor to learn meaningful transformations
- Creates asymmetry between context and target paths

### Comparison with Other Methods

| Method | Space | Stop-Grad | Predictor | Architecture |
|--------|-------|-----------|-----------|--------------|
| **V-JEPA** | Latent | ✓ | MLP/Transformer | Shared encoder |
| **SimCLR** | Latent | ✗ | None | Contrastive |
| **MAE** | Pixel | N/A | Decoder | Masked autoencoder |
| **BYOL** | Latent | ✓ | MLP | Momentum encoder |
| **SimSiam** | Latent | ✓ | MLP | Shared encoder |

---

## Architecture Components

### 1. Vision Encoder

#### CNN Encoder Design

```
Input: (B, 3, 224, 224)
  ↓
Conv Block 1: (3 → 64, stride=2)    → (B, 64, 112, 112)
  ↓ BatchNorm + ReLU
Conv Block 2: (64 → 128, stride=2)  → (B, 128, 56, 56)
  ↓ BatchNorm + ReLU
Conv Block 3: (128 → 256, stride=2) → (B, 256, 28, 28)
  ↓ BatchNorm + ReLU
Conv Block 4: (256 → 512, stride=2) → (B, 512, 14, 14)
  ↓ BatchNorm + ReLU
Flatten & Project                   → (B, 512)
  ↓ Linear(512*14*14 → 1024) + LayerNorm + ReLU
  ↓ Linear(1024 → 512) + LayerNorm
Output: (B, 512)
```

**Design choices**:
- Progressive downsampling (224→112→56→28→14)
- LayerNorm in projection head for stable training
- Projection MLP creates separation from representation space

#### ViT Encoder Design

```
Input: (B, 3, 224, 224)
  ↓
Patch Embedding: (16×16 patches)    → (B, 196, 768)
  ↓
Add CLS Token                       → (B, 197, 768)
  ↓ Add Positional Embedding
Transformer Blocks (×12)
  ↓ MultiHeadAttention + MLP + Residual
  ↓ LayerNorm before each sublayer
Extract CLS Token                   → (B, 768)
  ↓
Optional Projection                 → (B, embedding_dim)
Output: (B, 768)
```

**Design choices**:
- Standard ViT-Base architecture (12 layers, 12 heads)
- Pre-normalization (LayerNorm before attention)
- CLS token aggregates patch information
- Optional projection aligns dimension with predictor

### 2. Predictor Network

#### MLP Predictor

```
Input: (B, D)
  ↓
Linear(D → 2048) + LayerNorm + GELU + Dropout
  ↓
Linear(2048 → 2048) + LayerNorm + GELU + Dropout
  ↓
Linear(2048 → D) + LayerNorm
  ↓
Output: (B, D)
```

**Design rationale**:
- Hidden dim = 4×embedding_dim (common in transformers)
- LayerNorm over BatchNorm for batch size independence
- GELU activation for smooth gradients
- Final LayerNorm ensures normalized output for cosine similarity

#### Transformer Predictor

```
Input: (B, T, D)  # T = temporal context length
  ↓
Transformer Encoder (×4 layers)
  ↓ Self-attention captures temporal dependencies
  ↓ MLP with GELU
  ↓ Pre-norm architecture
Output Projection
  ↓
Output: (B, D) or (B, T, D)
```

**Use case**: When context consists of multiple frames/patches

### 3. Training Loss

#### Cosine Similarity Loss

```python
def vjepa_loss(predicted, target):
    # L2 normalize
    pred_norm = F.normalize(predicted, dim=-1)
    target_norm = F.normalize(target, dim=-1)

    # Cosine similarity
    sim = (pred_norm * target_norm).sum(dim=-1).mean()

    # Minimize negative similarity
    return 1.0 - sim
```

**Properties**:
- Scale invariant (only direction matters)
- Range: [0, 2] (0 = identical, 2 = opposite)
- Smooth gradients across entire range

#### Optional Regularization (VICReg-style)

```python
# Variance regularization (prevent collapse)
std = torch.sqrt(embeddings.var(dim=0) + 1e-4)
variance_loss = torch.mean(F.relu(1 - std))

# Covariance regularization (decorrelate dimensions)
emb_centered = embeddings - embeddings.mean(dim=0)
cov = (emb_centered.T @ emb_centered) / (B - 1)
off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
covariance_loss = off_diag / embedding_dim
```

---

## Training Dynamics

### Initialization Phase (Epochs 1-10)

**Characteristics**:
- High loss (≈1.0)
- Low cosine similarity (≈0.0)
- Encoder learns basic features
- Predictor aligns with target distribution

**What's happening**:
1. Encoder maps random inputs to diverse embeddings
2. Predictor learns identity mapping initially
3. Gradients flow only through context path
4. Target embeddings provide stable learning signal

### Learning Phase (Epochs 10-50)

**Characteristics**:
- Decreasing loss (1.0 → 0.3)
- Increasing cosine similarity (0.0 → 0.7)
- Encoder captures semantic features
- Predictor learns temporal transformations

**What's happening**:
1. Encoder discovers visual patterns (edges, textures, objects)
2. Predictor learns to forecast temporal changes
3. Representations become more structured
4. Embedding space organizes semantically

### Convergence Phase (Epochs 50+)

**Characteristics**:
- Stable loss (≈0.2-0.3)
- High cosine similarity (≈0.7-0.8)
- Refined representations
- Diminishing returns

**What's happening**:
1. Fine-tuning of feature representations
2. Predictor captures subtle temporal dynamics
3. Embeddings reach optimal trade-off between diversity and predictability

### Common Pitfalls

#### 1. Representation Collapse

**Symptoms**:
- Cosine similarity = 1.0 immediately
- All embeddings identical
- Zero gradients

**Causes**:
- Missing stop-gradient
- Too high learning rate
- Insufficient variance regularization

**Solutions**:
```python
# Ensure stop-gradient
target_emb = encoder(target).detach()

# Add variance loss
config.model.variance_weight = 0.1

# Reduce learning rate
config.training.learning_rate = 1e-5
```

#### 2. Prediction Failure

**Symptoms**:
- Loss stuck at 1.0
- Random cosine similarity (≈0.0)
- No learning progress

**Causes**:
- Insufficient model capacity
- No temporal structure in data
- Poor initialization

**Solutions**:
```python
# Increase capacity
config.model.encoder.embedding_dim = 768
config.model.predictor.hidden_dims = (3072, 3072)

# Add temporal structure
config.data.add_temporal_structure = True

# Warmup learning rate
config.training.warmup_epochs = 20
```

---

## Implementation Details

### Memory Efficiency

```python
# Context path: Requires gradients
context_emb = encoder(context)  # Stores activations
predicted = predictor(context_emb)  # Stores activations

# Target path: No gradients (memory efficient)
with torch.no_grad():
    target_emb = encoder(target)  # No activation storage

# Loss computation
loss = criterion(predicted, target_emb.detach())
```

**Memory savings**: ~50% compared to symmetric architecture

### Computational Complexity

Per training step:
```
CNN Encoder:
  - FLOPs: ~1B (for 224×224 input)
  - Memory: ~500MB (batch=32)

ViT Encoder:
  - FLOPs: ~5B (12 layers, 768D)
  - Memory: ~2GB (batch=32)

MLP Predictor:
  - FLOPs: ~100M (2048D hidden)
  - Memory: ~100MB

Total (CNN): ~1.1B FLOPs, ~600MB memory
Total (ViT): ~5.1B FLOPs, ~2.1GB memory
```

### Parallelization Strategy

```python
# Data parallelism (multi-GPU)
model = nn.DataParallel(model)

# Batch encoding
# Both context and target can be encoded in parallel
# (they use the same encoder but different gradients)

# Memory-efficient forward
# Process large batches by accumulating gradients:
for mini_batch in split_batch(batch, accumulation_steps):
    loss = model(mini_batch) / accumulation_steps
    loss.backward()
optimizer.step()
```

---

## Design Decisions

### Why Shared Encoder?

**Alternative**: Separate encoders for context and target

**Our choice**: Shared encoder with stop-gradient

**Rationale**:
1. Parameter efficiency (half the parameters)
2. Enforces semantic alignment
3. Simpler architecture
4. Better generalization

### Why Cosine Similarity?

**Alternatives**: L2 distance, InfoNCE loss, triplet loss

**Our choice**: Cosine similarity

**Rationale**:
1. Scale invariant (robust to norm changes)
2. Smooth gradients
3. Interpretable (range [0, 1])
4. No hyperparameters (unlike temperature in InfoNCE)

### Why MLP Predictor?

**Alternatives**: Transformer, GRU, None (direct comparison)

**Our choice**: MLP with LayerNorm

**Rationale**:
1. Sufficient capacity for single-step prediction
2. Fast and parameter-efficient
3. No temporal dependencies needed for i.i.d. frames
4. Can be upgraded to Transformer for sequences

### Why No Momentum Encoder?

**Alternative**: Exponential moving average (like BYOL)

**Our choice**: Simple stop-gradient (default)

**Rationale**:
1. Simpler implementation
2. Faster training (no EMA overhead)
3. More direct gradients
4. Works well for video (temporal structure provides stability)

**When to use momentum**: Static images without temporal structure

---

## Performance Characteristics

### Expected Metrics

| Metric | Debug Config | CNN Config | ViT Config |
|--------|-------------|------------|------------|
| **Training Time** | 2 min | 30 min | 2 hours |
| **Final Loss** | 0.4-0.5 | 0.2-0.3 | 0.1-0.2 |
| **Cosine Sim** | 0.5-0.6 | 0.7-0.8 | 0.8-0.9 |
| **Effective Rank** | 50-64 | 250-300 | 400-500 |
| **GPU Memory** | 1 GB | 4 GB | 12 GB |

### Scaling Laws

**Embedding dimension**:
- 128D: Fast, good for small datasets
- 512D: Balanced, general purpose
- 768D: High capacity, large datasets
- 1024D+: Research-scale

**Model depth**:
- CNN: 4-6 conv blocks sufficient
- ViT: 12 layers (Base), 24 layers (Large)

**Predictor capacity**:
- Hidden dim = 4× embedding dim (standard)
- 2 layers sufficient for single-step prediction
- 4+ layers for complex temporal dynamics

---

## References

**V-JEPA Original Work**:
- Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video" (2024)

**Related Methods**:
- SimSiam: Chen & He, "Exploring Simple Siamese Representation Learning" (2020)
- BYOL: Grill et al., "Bootstrap Your Own Latent" (2020)
- VICReg: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization" (2021)

**JEPA Framework**:
- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)

---

## Implementation Quality Checklist

- [x] Stop-gradient on target encoder
- [x] Cosine similarity loss
- [x] LayerNorm in predictor
- [x] Normalized embeddings before loss
- [x] Proper initialization (truncated normal)
- [x] Gradient clipping (optional, not implemented)
- [x] Mixed precision support (configurable)
- [x] Checkpointing and resumption
- [x] Comprehensive metrics logging
- [x] Embedding quality analysis

---

**Last Updated**: 2026-01-21
