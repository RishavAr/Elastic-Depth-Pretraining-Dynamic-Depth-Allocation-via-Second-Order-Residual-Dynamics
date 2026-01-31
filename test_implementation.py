"""
Quick test to verify EDP implementation structure.

Run: python test_implementation.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")
    
    # Configs
    from configs import ExperimentConfig, ModelConfig, EDPConfig, TrainingConfig, DataConfig
    print("  ✓ configs")
    
    # Utils
    from utils import layer_flops, token_flops, FLOPsTracker, EDPMetrics
    print("  ✓ utils")
    
    # Signals
    from signals import SignalComputer, SecondOrderSignal, FirstOrderSignal
    print("  ✓ signals")
    
    # Routing
    from routing import ElasticRouter, ThresholdRouter, FLOPAwareStepEncoding
    print("  ✓ routing")
    
    # Models
    from models import DenseTransformer, EDPTransformer, create_baseline_model, create_edp_model
    print("  ✓ models")
    
    # Training
    from training import EDPLoss, SparsityLoss, BudgetLoss, Trainer
    print("  ✓ training")
    
    # Evaluation
    from evaluation import EDPEvaluator, generate_all_figures
    print("  ✓ evaluation")
    
    print("\nAll imports successful!")


def test_flops_calculation():
    """Test FLOPs calculation."""
    print("\nTesting FLOPs calculation...")
    
    from utils import layer_flops, model_flops_per_token
    
    d_model = 512
    seq_len = 256
    n_layers = 12
    
    flops_per_layer = layer_flops(d_model, seq_len)
    total_flops = model_flops_per_token(d_model, seq_len, n_layers)
    
    print(f"  d_model={d_model}, seq_len={seq_len}, n_layers={n_layers}")
    print(f"  FLOPs per layer per token: {flops_per_layer:,}")
    print(f"  Total FLOPs per token: {total_flops:,}")
    
    # Expected: 12*d² + 2*seq*d = 12*512² + 2*256*512 = 3,145,728 + 262,144 = 3,407,872
    expected = 12 * d_model**2 + 2 * seq_len * d_model
    assert flops_per_layer == expected, f"FLOPs mismatch: {flops_per_layer} != {expected}"
    print("  ✓ FLOPs calculation correct!")


def test_signal_computation():
    """Test signal computation."""
    print("\nTesting signal computation...")
    
    import torch
    from signals import SignalComputer, SecondOrderSignal
    
    batch_size = 2
    seq_len = 8
    d_model = 64
    
    # Create signal computer
    signal_computer = SignalComputer(use_second_order=True)
    
    # Create mock hidden states
    h1 = torch.randn(batch_size, seq_len, d_model)
    h2 = h1 + 0.1 * torch.randn(batch_size, seq_len, d_model)
    h3 = h2 + 0.05 * torch.randn(batch_size, seq_len, d_model)
    
    # Compute signals
    signal_computer.reset()
    s1 = signal_computer(h1)  # First layer, no previous
    s2 = signal_computer(h2)  # Second-order signal
    s3 = signal_computer(h3)  # Should be decreasing (converging)
    
    print(f"  Signal shapes: {s1.shape}, {s2.shape}, {s3.shape}")
    print(f"  Signal means: {s1.mean():.4f}, {s2.mean():.4f}, {s3.mean():.4f}")
    
    assert s1.shape == (batch_size, seq_len), f"Wrong shape: {s1.shape}"
    print("  ✓ Signal computation working!")


def test_routing():
    """Test elastic routing."""
    print("\nTesting elastic routing...")
    
    import torch
    from routing import ElasticRouter
    
    d_model = 64
    n_layers = 12
    seq_len = 8
    batch_size = 2
    
    router = ElasticRouter(
        d_model=d_model,
        n_layers=n_layers,
        early_layers=3,
        late_layers=3,
        seq_len=seq_len,
    )
    
    # Test routing for each layer
    signal = torch.rand(batch_size, seq_len) * 0.5
    cumulative_flops = torch.zeros(batch_size, seq_len)
    
    for layer_idx in range(n_layers):
        gate, soft_gate, cumulative_flops = router(signal, layer_idx, cumulative_flops)
        is_always_on = router.is_always_on(layer_idx)
        
        if is_always_on:
            assert gate.mean() == 1.0, f"Always-on layer {layer_idx} should have gate=1"
            
        print(f"  Layer {layer_idx}: always_on={is_always_on}, gate_mean={gate.mean():.4f}")
    
    print("  ✓ Elastic routing working!")


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    import torch
    from configs import ExperimentConfig
    from models import create_baseline_model, create_edp_model
    
    config = ExperimentConfig()
    config.model.vocab_size = 1000
    config.model.d_model = 64
    config.model.n_heads = 4
    config.model.n_layers = 6
    config.model.max_seq_len = 32
    
    # Create baseline
    baseline = create_baseline_model(config)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"  Baseline parameters: {baseline_params:,}")
    
    # Create EDP
    config.edp.early_layers = 2
    config.edp.late_layers = 2
    edp = create_edp_model(config)
    edp_params = sum(p.numel() for p in edp.parameters())
    print(f"  EDP parameters: {edp_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Baseline forward
    baseline_out = baseline(input_ids, labels=labels)
    print(f"  Baseline output keys: {list(baseline_out.keys())}")
    print(f"  Baseline loss: {baseline_out['loss'].item():.4f}")
    
    # EDP forward
    edp_out = edp(input_ids, labels=labels, return_routing_info=True)
    print(f"  EDP output keys: {list(edp_out.keys())}")
    print(f"  EDP loss: {edp_out['loss'].item():.4f}")
    print(f"  EDP gates shape: {edp_out['gates'].shape}")
    
    if 'routing_info' in edp_out:
        print(f"  EDP FLOPs ratio: {edp_out['routing_info']['mean_flops_ratio'].item():.4f}")
    
    print("  ✓ Model creation working!")


def test_loss_functions():
    """Test EDP loss functions."""
    print("\nTesting loss functions...")
    
    import torch
    from training import SparsityLoss, BudgetLoss, EDPLoss
    
    batch_size = 2
    seq_len = 8
    n_layers = 6
    
    # Create mock data
    gates = torch.rand(batch_size, seq_len, n_layers)
    lm_loss = torch.tensor(3.5)
    
    # Sparsity loss
    sparsity_loss = SparsityLoss()
    s_loss = sparsity_loss(gates)
    print(f"  Sparsity loss: {s_loss.item():.4f}")
    
    # Budget loss
    budget_loss = BudgetLoss(target_ratio=0.6)
    b_loss = budget_loss(gates)
    print(f"  Budget loss: {b_loss.item():.4f}")
    
    # Full EDP loss
    edp_loss = EDPLoss(lambda_sparsity=0.1, lambda_budget=0.05)
    edp_loss.set_training_steps(1000)
    loss_dict = edp_loss(lm_loss, gates)
    
    print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  LM loss: {loss_dict['lm_loss'].item():.4f}")
    print(f"  Sparsity loss: {loss_dict['sparsity_loss'].item():.4f}")
    print(f"  Budget loss: {loss_dict['budget_loss'].item():.4f}")
    
    print("  ✓ Loss functions working!")


def test_config():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    import tempfile
    from configs import ExperimentConfig, get_ablation_config, ABLATION_NAMES
    
    # Create default config
    config = ExperimentConfig()
    print(f"  Default experiment name: {config.experiment_name}")
    print(f"  Model d_model: {config.model.d_model}")
    print(f"  EDP early_layers: {config.edp.early_layers}")
    
    # Test save/load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config.save(f.name)
        loaded = ExperimentConfig.load(f.name)
        
    assert loaded.model.d_model == config.model.d_model
    print("  ✓ Config save/load working!")
    
    # Test ablation configs
    print(f"  Available ablations: {len(ABLATION_NAMES)}")
    
    print("  ✓ Configuration system working!")


def main():
    print("=" * 60)
    print("ELASTIC-DEPTH PRETRAINING - IMPLEMENTATION TEST")
    print("=" * 60)
    
    try:
        test_imports()
        test_config()
        test_flops_calculation()
        test_signal_computation()
        test_routing()
        test_model_creation()
        test_loss_functions()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe implementation is ready for training.")
        print("Run: python -m experiments.run_full_pipeline --quick_test")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
