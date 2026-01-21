"""
Quick demonstration of V-JEPA usage

Run this script to verify the installation and see V-JEPA in action.
"""

import torch
from models import build_encoder, build_predictor, VJEPA
from data import create_simple_dataloader
from config import get_debug_config
from utils import set_seed, get_device, count_parameters


def demo_model_architecture():
    """Demonstrate model creation and forward pass."""
    print("\n" + "=" * 80)
    print("V-JEPA Architecture Demo")
    print("=" * 80 + "\n")

    # Set seed for reproducibility
    set_seed(42)
    device = get_device("cuda")

    # Build encoder
    print("Building CNN Encoder...")
    encoder = build_encoder(
        encoder_type="cnn",
        input_channels=3,
        embedding_dim=256,
        input_size=224,
        hidden_dims=(64, 128, 256)
    )
    print(f"  Encoder parameters: {count_parameters(encoder):,}")

    # Build predictor
    print("\nBuilding MLP Predictor...")
    predictor = build_predictor(
        predictor_type="mlp",
        embedding_dim=256,
        hidden_dims=(1024, 1024),
        dropout=0.1
    )
    print(f"  Predictor parameters: {count_parameters(predictor):,}")

    # Build V-JEPA model
    print("\nBuilding V-JEPA Model...")
    model = VJEPA(
        encoder=encoder,
        predictor=predictor,
        embedding_dim=256,
        use_momentum_encoder=False
    )
    model.to(device)
    print(f"  Total parameters: {count_parameters(model):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    context_frames = torch.randn(batch_size, 3, 224, 224).to(device)
    target_frames = torch.randn(batch_size, 3, 224, 224).to(device)

    output = model(context_frames, target_frames, return_embeddings=True)

    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Cosine similarity: {output['cosine_sim'].item():.4f}")
    print(f"  Context embedding shape: {output['context_emb'].shape}")
    print(f"  Target embedding shape: {output['target_emb'].shape}")
    print(f"  Predicted embedding shape: {output['predicted_emb'].shape}")

    print("\n✓ Architecture demo completed successfully!\n")


def demo_data_loading():
    """Demonstrate data loading."""
    print("\n" + "=" * 80)
    print("V-JEPA Data Loading Demo")
    print("=" * 80 + "\n")

    print("Creating synthetic dataloader...")
    dataloader = create_simple_dataloader(
        num_samples=100,
        batch_size=8,
        img_size=224,
        channels=3,
        add_structure=True
    )

    print(f"  Dataset size: {len(dataloader.dataset)}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Number of batches: {len(dataloader)}")

    # Load one batch
    print("\nLoading sample batch...")
    context, target = next(iter(dataloader))

    print(f"  Context shape: {context.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Context range: [{context.min():.3f}, {context.max():.3f}]")
    print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")

    print("\n✓ Data loading demo completed successfully!\n")


def demo_training_step():
    """Demonstrate a single training step."""
    print("\n" + "=" * 80)
    print("V-JEPA Training Step Demo")
    print("=" * 80 + "\n")

    set_seed(42)
    device = get_device("cuda")

    # Build small model
    print("Building model...")
    encoder = build_encoder("cnn", embedding_dim=128, hidden_dims=(32, 64))
    predictor = build_predictor("mlp", embedding_dim=128, hidden_dims=(256,))
    model = VJEPA(encoder, predictor, embedding_dim=128)
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Get sample data
    print("Loading data...")
    dataloader = create_simple_dataloader(num_samples=32, batch_size=8, img_size=224)
    context, target = next(iter(dataloader))
    context, target = context.to(device), target.to(device)

    # Training step
    print("\nPerforming training step...")
    model.train()

    # Forward pass
    output = model(context, target)
    loss = output['loss']

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Cosine similarity: {output['cosine_sim'].item():.4f}")

    # Second forward pass (should show improvement)
    with torch.no_grad():
        output2 = model(context, target)
        print(f"  Loss after update: {output2['loss'].item():.4f}")
        print(f"  Cosine similarity after update: {output2['cosine_sim'].item():.4f}")

    print("\n✓ Training step demo completed successfully!\n")


def demo_config_system():
    """Demonstrate configuration system."""
    print("\n" + "=" * 80)
    print("V-JEPA Configuration Demo")
    print("=" * 80 + "\n")

    from config import get_cnn_config, get_vit_config, get_debug_config

    configs = [
        ("Debug", get_debug_config()),
        ("CNN", get_cnn_config()),
        ("ViT", get_vit_config())
    ]

    for name, config in configs:
        print(f"{name} Configuration:")
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Encoder: {config.model.encoder.encoder_type}")
        print(f"  Embedding dim: {config.model.encoder.embedding_dim}")
        print(f"  Predictor: {config.model.predictor.predictor_type}")
        print(f"  Batch size: {config.data.batch_size}")
        print(f"  Max epochs: {config.training.max_epochs}")
        print(f"  Learning rate: {config.training.learning_rate}")
        print()

    print("✓ Configuration demo completed successfully!\n")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("V-JEPA Complete Demo")
    print("=" * 80)

    try:
        demo_model_architecture()
        demo_data_loading()
        demo_training_step()
        demo_config_system()

        print("=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run training: python -m vjepa.train")
        print("  2. Run experiments: python -m vjepa.experiments.evaluate")
        print("  3. See README.md for detailed usage")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
