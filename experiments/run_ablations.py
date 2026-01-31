"""
Run all ablation studies for EDP paper.

Required ablations:
1. First-order vs second-order signal
2. Static vs FLOP-aware step encoding
3. No warp-forward vs warp-forward
4. No sparsity loss vs with sparsity loss
5. Fixed vs learned threshold
6. Middle-only vs full-depth routing
7. Post-training skip vs pretraining skip

Usage:
    python -m experiments.run_ablations --dataset tinystories --epochs 5
"""
import os
import sys
import argparse
import json
import torch
import random
import numpy as np
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import ExperimentConfig, ModelConfig, EDPConfig, TrainingConfig, DataConfig, get_ablation_config, ABLATION_NAMES
from models import create_edp_model, create_baseline_model
from data import create_dataloaders
from training import train_model
from evaluation import EDPEvaluator


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Define ablation configurations
ABLATIONS = {
    # Signal ablations
    "second_order_signal": {
        "description": "EDP with second-order (acceleration) signal (default)",
        "edp_changes": {"use_second_order_signal": True},
    },
    "first_order_signal": {
        "description": "EDP with first-order (velocity) signal",
        "edp_changes": {"use_second_order_signal": False},
    },
    
    # Step encoding ablations
    "flop_aware_encoding": {
        "description": "FLOP-aware step encoding (default)",
        "edp_changes": {"use_flop_aware_encoding": True},
    },
    "static_step_encoding": {
        "description": "Static (layer-based) step encoding",
        "edp_changes": {"use_flop_aware_encoding": False},
    },
    
    # Warp-forward ablations
    "with_warp_forward": {
        "description": "With warp-forward mechanism (default)",
        "edp_changes": {"use_warp_forward": True},
    },
    "no_warp_forward": {
        "description": "Without warp-forward (naive skip)",
        "edp_changes": {"use_warp_forward": False},
    },
    
    # Sparsity loss ablations
    "with_sparsity_loss": {
        "description": "With sparsity loss (default)",
        "edp_changes": {"lambda_sparsity": 0.1},
    },
    "no_sparsity_loss": {
        "description": "Without sparsity loss",
        "edp_changes": {"lambda_sparsity": 0.0},
    },
    
    # Threshold ablations
    "learned_threshold": {
        "description": "Learned per-layer thresholds (default)",
        "edp_changes": {"learned_threshold": True},
    },
    "fixed_threshold": {
        "description": "Fixed thresholds",
        "edp_changes": {"learned_threshold": False},
    },
    
    # Routing scope ablations
    "bowl_shaped_routing": {
        "description": "Bowl-shaped routing (early/late always on, middle gated) (default)",
        "edp_changes": {"early_layers": 3, "late_layers": 3},
    },
    "full_depth_routing": {
        "description": "Full-depth routing (all layers gatable)",
        "edp_changes": {"early_layers": 0, "late_layers": 0},
    },
    
    # Budget ablations
    "target_ratio_0.6": {
        "description": "Target compute ratio 0.6 (default)",
        "edp_changes": {"target_compute_ratio": 0.6},
    },
    "target_ratio_0.4": {
        "description": "Target compute ratio 0.4",
        "edp_changes": {"target_compute_ratio": 0.4},
    },
    "target_ratio_0.8": {
        "description": "Target compute ratio 0.8",
        "edp_changes": {"target_compute_ratio": 0.8},
    },
    
    # Budget loss ablations
    "with_budget_loss": {
        "description": "With budget loss (default)",
        "edp_changes": {"lambda_budget": 0.05},
    },
    "no_budget_loss": {
        "description": "Without budget loss",
        "edp_changes": {"lambda_budget": 0.0},
    },
}


def create_ablation_config(base_config: ExperimentConfig, ablation_name: str) -> ExperimentConfig:
    """Create config for a specific ablation."""
    config = ExperimentConfig(
        model=ModelConfig(**vars(base_config.model)),
        edp=EDPConfig(**vars(base_config.edp)),
        training=TrainingConfig(**{k: v for k, v in vars(base_config.training).items()}),
        data=DataConfig(**vars(base_config.data)),
        experiment_name=f"ablation_{ablation_name}",
        output_dir=base_config.output_dir,
    )
    
    # Apply ablation changes
    if ablation_name in ABLATIONS:
        for key, value in ABLATIONS[ablation_name]["edp_changes"].items():
            setattr(config.edp, key, value)
            
    return config


def run_single_ablation(
    ablation_name: str,
    base_config: ExperimentConfig,
    train_loader,
    val_loader,
    tokenizer,
) -> Dict:
    """Run a single ablation experiment."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_name}")
    print(f"{'='*60}")
    
    if ablation_name in ABLATIONS:
        print(f"Description: {ABLATIONS[ablation_name]['description']}")
        
    # Create ablation config
    config = create_ablation_config(base_config, ablation_name)
    config.model.vocab_size = tokenizer.vocab_size
    
    # Create model
    model = create_edp_model(config)
    
    # Train
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        is_edp=True,
    )
    
    # Add ablation metadata
    results['ablation_name'] = ablation_name
    results['config'] = config.to_dict()
    
    return results


def run_post_training_ablation(
    dense_checkpoint: str,
    base_config: ExperimentConfig,
    val_loader,
    tokenizer,
) -> Dict:
    """
    Special ablation: Apply routing AFTER dense pretraining.
    
    This proves the central claim: representations must be trained to be skipped.
    """
    print(f"\n{'='*60}")
    print("ABLATION: Post-training skip (vs pretraining skip)")
    print("='*60")
    
    # Load dense model
    base_config.model.vocab_size = tokenizer.vocab_size
    dense_model = create_baseline_model(base_config)
    
    checkpoint = torch.load(dense_checkpoint)
    dense_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create EDP model and copy weights
    edp_model = create_edp_model(base_config)
    
    # Copy matching weights
    dense_state = dense_model.state_dict()
    edp_state = edp_model.state_dict()
    
    for name, param in dense_state.items():
        if name in edp_state and edp_state[name].shape == param.shape:
            edp_state[name] = param
            
    edp_model.load_state_dict(edp_state)
    
    # Evaluate without further training
    device = torch.device(base_config.training.device if torch.cuda.is_available() else 'cpu')
    evaluator = EDPEvaluator(edp_model, device=str(device))
    
    results = evaluator.evaluate(val_loader, return_detailed=True)
    results['ablation_name'] = 'post_training_skip'
    results['note'] = 'Routing applied after dense pretraining, no EDP training'
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDP ablation studies")
    
    # Data
    parser.add_argument('--dataset', type=str, default='tinystories', choices=['tinystories', 'wikitext103'])
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    
    # Model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=12)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    
    # Ablation selection
    parser.add_argument('--ablations', type=str, nargs='+', default=None,
                       help='Specific ablations to run. If not specified, runs all.')
    parser.add_argument('--skip_baseline', action='store_true')
    
    # Logging
    parser.add_argument('--output_dir', type=str, default='./outputs/ablations')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create base config
    base_config = ExperimentConfig(
        model=ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_seq_len=args.seq_len,
        ),
        edp=EDPConfig(),  # Default EDP settings
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
        ),
        data=DataConfig(
            dataset=args.dataset,
            seq_len=args.seq_len,
        ),
        output_dir=args.output_dir,
    )
    
    print("=" * 60)
    print("EDP ABLATION STUDIES")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: d={args.d_model}, L={args.n_layers}, H={args.n_heads}")
    print(f"Training: epochs={args.epochs}")
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
    
    # Select ablations to run
    if args.ablations:
        ablations_to_run = args.ablations
    else:
        # Default: run key ablations
        ablations_to_run = [
            "second_order_signal",
            "first_order_signal",
            "flop_aware_encoding",
            "static_step_encoding",
            "with_warp_forward",
            "no_warp_forward",
            "with_sparsity_loss",
            "no_sparsity_loss",
            "learned_threshold",
            "fixed_threshold",
            "bowl_shaped_routing",
            "full_depth_routing",
        ]
        
    print(f"\nAblations to run: {len(ablations_to_run)}")
    for name in ablations_to_run:
        if name in ABLATIONS:
            print(f"  - {name}: {ABLATIONS[name]['description']}")
        else:
            print(f"  - {name}")
    
    # Store all results
    all_results = {}
    
    # Run ablations
    for ablation_name in ablations_to_run:
        try:
            results = run_single_ablation(
                ablation_name=ablation_name,
                base_config=base_config,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
            )
            all_results[ablation_name] = results
            
            # Save intermediate results
            results_path = os.path.join(args.output_dir, f"{ablation_name}_results.json")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump({
                    'ablation': ablation_name,
                    'best_val_loss': results['best_val_loss'],
                    'final_val_loss': results['final_val_loss'],
                    'flops_ratio': results.get('flops_stats', {}).get('avg_flops_ratio', None),
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error running ablation {ablation_name}: {e}")
            all_results[ablation_name] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    print(f"{'Ablation':<30} {'Val Loss':<12} {'FLOPs Ratio':<12}")
    print("-" * 54)
    
    for name, results in all_results.items():
        if 'error' in results:
            print(f"{name:<30} ERROR")
        else:
            val_loss = results.get('best_val_loss', 'N/A')
            flops = results.get('flops_stats', {}).get('avg_flops_ratio', 'N/A')
            
            val_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
            flops_str = f"{flops:.4f}" if isinstance(flops, float) else str(flops)
            
            print(f"{name:<30} {val_str:<12} {flops_str:<12}")
    
    # Save complete results
    summary_path = os.path.join(args.output_dir, 'ablation_summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': base_config.to_dict(),
        'results': {
            name: {
                'best_val_loss': r.get('best_val_loss'),
                'final_val_loss': r.get('final_val_loss'),
                'flops_ratio': r.get('flops_stats', {}).get('avg_flops_ratio'),
            } if 'error' not in r else {'error': r['error']}
            for name, r in all_results.items()
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nResults saved to: {summary_path}")
    
    return all_results


if __name__ == "__main__":
    main()
