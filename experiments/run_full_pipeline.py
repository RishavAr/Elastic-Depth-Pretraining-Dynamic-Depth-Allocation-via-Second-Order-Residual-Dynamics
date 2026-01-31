"""
Complete End-to-End Experiment Pipeline for EDP Paper.

This script runs the complete experimental protocol:
1. Train dense baseline on TinyStories and WikiText-103
2. Train EDP model on both datasets
3. Run all ablation studies on TinyStories
4. Run key ablations on WikiText-103
5. Generate all paper figures
6. Compile results summary

Usage:
    python -m experiments.run_full_pipeline --quick_test  # Fast test run
    python -m experiments.run_full_pipeline               # Full run
"""
import os
import sys
import argparse
import json
import torch
import random
import numpy as np
from datetime import datetime
from typing import Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import ExperimentConfig, ModelConfig, EDPConfig, TrainingConfig, DataConfig
from models import create_edp_model, create_baseline_model
from data import create_dataloaders
from training import train_model
from evaluation import EDPEvaluator, generate_all_figures


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_baseline_experiment(dataset: str, config: ExperimentConfig, output_dir: str) -> Dict:
    """Train and evaluate dense baseline."""
    print(f"\n{'='*60}")
    print(f"BASELINE: {dataset}")
    print(f"{'='*60}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        dataset_name=dataset,
        batch_size=config.training.batch_size,
        seq_len=config.data.seq_len,
        max_train_samples=config.data.tinystories_subset,
    )
    
    config.model.vocab_size = tokenizer.vocab_size
    config.experiment_name = f"baseline_{dataset}"
    config.output_dir = output_dir
    
    # Create and train model
    model = create_baseline_model(config)
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        is_edp=False,
    )
    
    # Evaluate on test set
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    evaluator = EDPEvaluator(model, device=str(device))
    test_results = evaluator.evaluate(test_loader)
    results['test_results'] = test_results
    
    return results


def run_edp_experiment(dataset: str, config: ExperimentConfig, output_dir: str) -> Dict:
    """Train and evaluate EDP model."""
    print(f"\n{'='*60}")
    print(f"EDP: {dataset}")
    print(f"{'='*60}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        dataset_name=dataset,
        batch_size=config.training.batch_size,
        seq_len=config.data.seq_len,
        max_train_samples=config.data.tinystories_subset,
    )
    
    config.model.vocab_size = tokenizer.vocab_size
    config.experiment_name = f"edp_{dataset}"
    config.output_dir = output_dir
    
    # Create and train model
    model = create_edp_model(config)
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        is_edp=True,
    )
    
    # Evaluate on test set
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    evaluator = EDPEvaluator(model, device=str(device))
    test_results = evaluator.evaluate(test_loader, return_detailed=True)
    results['test_results'] = test_results
    
    # Routing analysis
    routing_analysis = evaluator.analyze_routing(test_loader, n_batches=10)
    results['routing_analysis'] = routing_analysis
    
    return results


def run_ablations(dataset: str, config: ExperimentConfig, output_dir: str) -> Dict:
    """Run ablation studies."""
    print(f"\n{'='*60}")
    print(f"ABLATIONS: {dataset}")
    print(f"{'='*60}")
    
    # Import ablation runner
    from experiments.run_ablations import run_single_ablation, ABLATIONS
    
    # Create dataloaders
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        dataset_name=dataset,
        batch_size=config.training.batch_size,
        seq_len=config.data.seq_len,
        max_train_samples=config.data.tinystories_subset,
    )
    
    config.output_dir = os.path.join(output_dir, f"ablations_{dataset}")
    
    # Key ablations to run
    key_ablations = [
        "first_order_signal",      # vs second-order
        "static_step_encoding",    # vs FLOP-aware
        "no_warp_forward",         # vs with warp-forward
        "no_sparsity_loss",        # vs with sparsity
        "fixed_threshold",         # vs learned
        "full_depth_routing",      # vs bowl-shaped
    ]
    
    results = {}
    for ablation in key_ablations:
        try:
            ablation_results = run_single_ablation(
                ablation_name=ablation,
                base_config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
            )
            results[ablation] = ablation_results
        except Exception as e:
            print(f"Error in ablation {ablation}: {e}")
            results[ablation] = {'error': str(e)}
            
    return results


def compile_results(all_results: Dict, output_dir: str) -> Dict:
    """Compile all results into summary."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiments': {},
    }
    
    # Main results
    for exp_name in ['baseline_tinystories', 'baseline_wikitext', 
                     'edp_tinystories', 'edp_wikitext']:
        if exp_name in all_results:
            r = all_results[exp_name]
            summary['experiments'][exp_name] = {
                'best_val_loss': r.get('best_val_loss'),
                'test_loss': r.get('test_results', {}).get('loss'),
                'test_perplexity': r.get('test_results', {}).get('perplexity'),
            }
            
            # EDP-specific metrics
            if 'edp' in exp_name:
                summary['experiments'][exp_name].update({
                    'avg_flops_ratio': r.get('test_results', {}).get('avg_flops_ratio'),
                    'avg_gate_ratio': r.get('test_results', {}).get('avg_gate_ratio'),
                })
    
    # Ablation results
    if 'ablations_tinystories' in all_results:
        summary['ablations'] = {}
        for abl_name, abl_results in all_results['ablations_tinystories'].items():
            if 'error' not in abl_results:
                summary['ablations'][abl_name] = {
                    'val_loss': abl_results.get('best_val_loss'),
                    'flops_ratio': abl_results.get('flops_stats', {}).get('avg_flops_ratio'),
                }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete EDP experiment pipeline")
    
    # Mode
    parser.add_argument('--quick_test', action='store_true', 
                       help='Quick test run with reduced data/epochs')
    
    # Data
    parser.add_argument('--seq_len', type=int, default=256)
    
    # Model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=12)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    
    # What to run
    parser.add_argument('--skip_baseline', action='store_true')
    parser.add_argument('--skip_edp', action='store_true')
    parser.add_argument('--skip_ablations', action='store_true')
    parser.add_argument('--skip_wikitext', action='store_true')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/full_pipeline')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Quick test mode
    if args.quick_test:
        args.epochs = 2
        max_samples = 1000
        ablation_epochs = 1
        print("\n*** QUICK TEST MODE - Reduced data and epochs ***\n")
    else:
        max_samples = None
        ablation_epochs = 5
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("ELASTIC-DEPTH PRETRAINING: COMPLETE EXPERIMENT PIPELINE")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"Model: d={args.d_model}, L={args.n_layers}, H={args.n_heads}")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}")
    print("=" * 70)
    
    # Base config
    base_config = ExperimentConfig(
        model=ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_seq_len=args.seq_len,
        ),
        edp=EDPConfig(),
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
        ),
        data=DataConfig(seq_len=args.seq_len),
    )
    
    if args.quick_test:
        base_config.data.tinystories_subset = max_samples
    
    all_results = {}
    
    # =========================================
    # PHASE 1: Baseline Training
    # =========================================
    if not args.skip_baseline:
        print("\n" + "=" * 70)
        print("PHASE 1: BASELINE TRAINING")
        print("=" * 70)
        
        # TinyStories
        baseline_ts = run_baseline_experiment(
            'tinystories', base_config, args.output_dir
        )
        all_results['baseline_tinystories'] = baseline_ts
        
        # WikiText-103
        if not args.skip_wikitext:
            baseline_wt = run_baseline_experiment(
                'wikitext103', base_config, args.output_dir
            )
            all_results['baseline_wikitext'] = baseline_wt
    
    # =========================================
    # PHASE 2: EDP Training
    # =========================================
    if not args.skip_edp:
        print("\n" + "=" * 70)
        print("PHASE 2: EDP TRAINING")
        print("=" * 70)
        
        # TinyStories
        edp_ts = run_edp_experiment(
            'tinystories', base_config, args.output_dir
        )
        all_results['edp_tinystories'] = edp_ts
        
        # WikiText-103
        if not args.skip_wikitext:
            edp_wt = run_edp_experiment(
                'wikitext103', base_config, args.output_dir
            )
            all_results['edp_wikitext'] = edp_wt
    
    # =========================================
    # PHASE 3: Ablation Studies
    # =========================================
    if not args.skip_ablations:
        print("\n" + "=" * 70)
        print("PHASE 3: ABLATION STUDIES")
        print("=" * 70)
        
        # Ablations on TinyStories
        ablation_config = ExperimentConfig(
            model=ModelConfig(**vars(base_config.model)),
            edp=EDPConfig(**vars(base_config.edp)),
            training=TrainingConfig(
                batch_size=args.batch_size,
                epochs=ablation_epochs,
                learning_rate=args.lr,
                device=args.device,
            ),
            data=DataConfig(seq_len=args.seq_len),
        )
        if args.quick_test:
            ablation_config.data.tinystories_subset = max_samples
            
        ablations_ts = run_ablations(
            'tinystories', ablation_config, args.output_dir
        )
        all_results['ablations_tinystories'] = ablations_ts
    
    # =========================================
    # PHASE 4: Generate Figures
    # =========================================
    print("\n" + "=" * 70)
    print("PHASE 4: GENERATING FIGURES")
    print("=" * 70)
    
    figures_dir = os.path.join(args.output_dir, 'figures')
    
    # Prepare results for figures
    figure_data = {}
    if 'edp_tinystories' in all_results:
        edp_results = all_results['edp_tinystories']
        if 'test_results' in edp_results:
            figure_data.update(edp_results['test_results'])
        if 'routing_analysis' in edp_results:
            figure_data['layer_utilization'] = edp_results['routing_analysis'].get('layer_stats', {})
    
    # Save figure data
    figure_data_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(figure_data_path, 'w') as f:
        json.dump(figure_data, f, indent=2, default=str)
    
    generate_all_figures(args.output_dir, figures_dir)
    
    # =========================================
    # PHASE 5: Compile Results
    # =========================================
    print("\n" + "=" * 70)
    print("PHASE 5: COMPILING RESULTS")
    print("=" * 70)
    
    summary = compile_results(all_results, args.output_dir)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE - SUMMARY")
    print("=" * 70)
    
    print("\n--- Main Results ---")
    for exp_name, metrics in summary.get('experiments', {}).items():
        print(f"\n{exp_name}:")
        for key, value in metrics.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    if 'ablations' in summary:
        print("\n--- Ablation Results ---")
        print(f"{'Ablation':<25} {'Val Loss':<12} {'FLOPs':<12}")
        print("-" * 49)
        for abl_name, metrics in summary['ablations'].items():
            loss = metrics.get('val_loss', 'N/A')
            flops = metrics.get('flops_ratio', 'N/A')
            loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
            flops_str = f"{flops:.4f}" if isinstance(flops, float) else str(flops)
            print(f"{abl_name:<25} {loss_str:<12} {flops_str:<12}")
    
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"Figures saved to: {figures_dir}")
    
    return all_results


if __name__ == "__main__":
    main()
