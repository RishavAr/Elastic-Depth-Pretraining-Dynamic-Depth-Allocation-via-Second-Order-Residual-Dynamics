"""
Run Elastic-Depth Pretraining (EDP) transformer training.

Usage:
    python -m experiments.run_edp --dataset tinystories --epochs 10
"""
import os
import sys
import argparse
import json
import torch
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import ExperimentConfig, ModelConfig, EDPConfig, TrainingConfig, DataConfig
from models import create_edp_model
from data import create_dataloaders
from training import train_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EDP transformer")
    
    # Data
    parser.add_argument('--dataset', type=str, default='tinystories', choices=['tinystories', 'wikitext103'])
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    
    # Model (base)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # EDP-specific
    parser.add_argument('--early_layers', type=int, default=3)
    parser.add_argument('--late_layers', type=int, default=3)
    parser.add_argument('--use_first_order', action='store_true', help='Use first-order signal instead of second-order')
    parser.add_argument('--fixed_threshold', action='store_true', help='Use fixed threshold instead of learned')
    parser.add_argument('--initial_threshold', type=float, default=0.1)
    parser.add_argument('--no_flop_encoding', action='store_true', help='Use static step encoding')
    parser.add_argument('--no_warp_forward', action='store_true')
    
    # Loss weights
    parser.add_argument('--lambda_sparsity', type=float, default=0.1)
    parser.add_argument('--lambda_budget', type=float, default=0.05)
    parser.add_argument('--target_ratio', type=float, default=0.6)
    parser.add_argument('--use_entropy_aware', action='store_true')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    
    # Logging
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--experiment_name', type=str, default='edp')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_amp', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create config
    config = ExperimentConfig(
        model=ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            max_seq_len=args.seq_len,
        ),
        edp=EDPConfig(
            early_layers=args.early_layers,
            late_layers=args.late_layers,
            use_second_order_signal=not args.use_first_order,
            learned_threshold=not args.fixed_threshold,
            initial_threshold=args.initial_threshold,
            use_flop_aware_encoding=not args.no_flop_encoding,
            use_warp_forward=not args.no_warp_forward,
            lambda_sparsity=args.lambda_sparsity,
            lambda_budget=args.lambda_budget,
            target_compute_ratio=args.target_ratio,
            use_entropy_aware_sparsity=args.use_entropy_aware,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            device=args.device,
            mixed_precision=not args.no_amp,
            seed=args.seed,
        ),
        data=DataConfig(
            dataset=args.dataset,
            seq_len=args.seq_len,
        ),
        experiment_name=f"{args.experiment_name}_{args.dataset}",
        output_dir=args.output_dir,
    )
    
    print("=" * 60)
    print("ELASTIC-DEPTH PRETRAINING (EDP)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}")
    print(f"EDP: early={args.early_layers}, late={args.late_layers}, middle={args.n_layers - args.early_layers - args.late_layers}")
    print(f"Signal: {'Second-order' if not args.use_first_order else 'First-order'}")
    print(f"Threshold: {'Learned' if not args.fixed_threshold else 'Fixed'}")
    print(f"Step encoding: {'FLOP-aware' if not args.no_flop_encoding else 'Static'}")
    print(f"Loss: λ_sparse={args.lambda_sparsity}, λ_budget={args.lambda_budget}, target={args.target_ratio}")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 60)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Update vocab size
    config.model.vocab_size = tokenizer.vocab_size
    
    # Create model
    print("\nCreating EDP model...")
    model = create_edp_model(config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Print routing info
    print(f"\nRouting configuration:")
    print(f"  - Always-on layers: 0-{args.early_layers-1} and {args.n_layers - args.late_layers}-{args.n_layers-1}")
    print(f"  - Gated layers: {args.early_layers}-{args.n_layers - args.late_layers - 1}")
    
    # Train
    print("\nStarting training...")
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        is_edp=True,
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final validation loss: {results['final_val_loss']:.4f}")
    
    if 'flops_stats' in results:
        flops_stats = results['flops_stats']
        print(f"Average FLOPs ratio: {flops_stats['avg_flops_ratio']:.4f}")
        
    if 'edp_metrics' in results:
        edp_metrics = results['edp_metrics']
        print(f"Average gate ratio: {edp_metrics['avg_gate_ratio']:.4f}")
        
    print(f"Results saved to: {os.path.join(config.output_dir, config.experiment_name)}")
    
    return results


if __name__ == "__main__":
    main()
