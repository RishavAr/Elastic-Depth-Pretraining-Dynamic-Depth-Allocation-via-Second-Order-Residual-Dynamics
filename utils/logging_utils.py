"""
Logging and metrics utilities for EDP experiments.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict

import torch
import numpy as np


def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class MetricsTracker:
    """Track and aggregate metrics during training."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.step_metrics = {}
        
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Add metrics for current step."""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
            
        if step is not None:
            self.step_metrics[step] = metrics.copy()
            
    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric."""
        values = self.metrics[key]
        if last_n is not None:
            values = values[-last_n:]
        return np.mean(values) if values else 0.0
    
    def get_last(self, key: str) -> float:
        """Get last value of a metric."""
        values = self.metrics[key]
        return values[-1] if values else 0.0
    
    def get_all(self, key: str) -> List[float]:
        """Get all values of a metric."""
        return self.metrics[key]
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
        self.step_metrics = {}
        
    def save(self, path: str):
        """Save metrics to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'metrics': dict(self.metrics),
                'step_metrics': {str(k): v for k, v in self.step_metrics.items()}
            }, f, indent=2)
            
    def load(self, path: str):
        """Load metrics from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.metrics = defaultdict(list, data.get('metrics', {}))
        self.step_metrics = {int(k): v for k, v in data.get('step_metrics', {}).items()}


class EDPMetrics:
    """
    Specialized metrics for EDP experiments.
    
    Tracks:
    - Loss components (LM, sparsity, budget)
    - FLOPs usage
    - Gate statistics per layer
    - Acceleration histograms
    - Token entropy vs skipped depth
    """
    
    def __init__(self, n_layers: int, n_middle_layers: int):
        self.n_layers = n_layers
        self.n_middle_layers = n_middle_layers
        
        # Loss tracking
        self.losses = MetricsTracker()
        
        # Gate statistics per layer
        self.gate_stats = {i: MetricsTracker() for i in range(n_middle_layers)}
        
        # Acceleration histograms (binned)
        self.accel_bins = np.linspace(0, 1, 51)  # 50 bins
        self.accel_hist = np.zeros(50)
        self.accel_count = 0
        
        # Depth usage distribution
        self.depth_hist = np.zeros(n_layers + 1)  # 0 to n_layers
        self.depth_count = 0
        
        # Token entropy vs depth
        self.entropy_depth_pairs = []
        
    def update_losses(self, lm_loss: float, sparsity_loss: float, budget_loss: float, total_loss: float, step: int):
        """Update loss metrics."""
        self.losses.update({
            'lm_loss': lm_loss,
            'sparsity_loss': sparsity_loss,
            'budget_loss': budget_loss,
            'total_loss': total_loss,
        }, step)
        
    def update_gates(self, gates: torch.Tensor, layer_idx: int):
        """
        Update gate statistics for a layer.
        
        Args:
            gates: Gate values [batch, seq_len]
            layer_idx: Index of middle layer (0-indexed)
        """
        gate_ratio = gates.float().mean().item()
        self.gate_stats[layer_idx].update({
            'gate_ratio': gate_ratio,
            'gate_std': gates.float().std().item(),
        })
        
    def update_acceleration(self, acceleration: torch.Tensor):
        """
        Update acceleration histogram.
        
        Args:
            acceleration: Acceleration values [batch, seq_len] or flattened
        """
        accel_flat = acceleration.detach().cpu().numpy().flatten()
        
        # Normalize to [0, 1] for binning
        accel_norm = np.clip(accel_flat / (accel_flat.max() + 1e-8), 0, 1)
        
        hist, _ = np.histogram(accel_norm, bins=self.accel_bins)
        self.accel_hist += hist
        self.accel_count += len(accel_flat)
        
    def update_depth_usage(self, depth_per_token: torch.Tensor):
        """
        Update depth usage histogram.
        
        Args:
            depth_per_token: Number of layers used per token [batch, seq_len]
        """
        depths = depth_per_token.detach().cpu().numpy().flatten().astype(int)
        
        for d in depths:
            if 0 <= d <= self.n_layers:
                self.depth_hist[d] += 1
        self.depth_count += len(depths)
        
    def update_entropy_depth(self, entropy: torch.Tensor, depth: torch.Tensor):
        """
        Track relationship between token entropy and depth used.
        
        Args:
            entropy: Token entropy values [batch, seq_len]
            depth: Depth used per token [batch, seq_len]
        """
        ent = entropy.detach().cpu().numpy().flatten()
        dep = depth.detach().cpu().numpy().flatten()
        
        # Sample to avoid memory issues
        if len(ent) > 1000:
            idx = np.random.choice(len(ent), 1000, replace=False)
            ent = ent[idx]
            dep = dep[idx]
            
        self.entropy_depth_pairs.extend(zip(ent, dep))
        
    def get_gate_utilization(self) -> Dict[int, float]:
        """Get average gate utilization per layer."""
        return {
            layer: tracker.get_average('gate_ratio')
            for layer, tracker in self.gate_stats.items()
        }
        
    def get_acceleration_histogram(self) -> np.ndarray:
        """Get normalized acceleration histogram."""
        if self.accel_count > 0:
            return self.accel_hist / self.accel_count
        return self.accel_hist
    
    def get_depth_distribution(self) -> np.ndarray:
        """Get normalized depth usage distribution."""
        if self.depth_count > 0:
            return self.depth_hist / self.depth_count
        return self.depth_hist
    
    def get_entropy_depth_correlation(self) -> float:
        """Get correlation between entropy and depth."""
        if len(self.entropy_depth_pairs) < 10:
            return 0.0
        
        ent, dep = zip(*self.entropy_depth_pairs)
        return np.corrcoef(ent, dep)[0, 1]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            'avg_lm_loss': self.losses.get_average('lm_loss'),
            'avg_sparsity_loss': self.losses.get_average('sparsity_loss'),
            'avg_budget_loss': self.losses.get_average('budget_loss'),
            'avg_total_loss': self.losses.get_average('total_loss'),
            'gate_utilization': self.get_gate_utilization(),
            'avg_gate_ratio': np.mean(list(self.get_gate_utilization().values())),
            'depth_distribution': self.get_depth_distribution().tolist(),
            'entropy_depth_correlation': self.get_entropy_depth_correlation(),
        }
        
    def save(self, path: str):
        """Save all metrics to file."""
        summary = self.get_summary()
        summary['accel_histogram'] = self.get_acceleration_histogram().tolist()
        summary['entropy_depth_pairs'] = self.entropy_depth_pairs[-10000:]  # Keep last 10k
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)


def log_gpu_memory():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        return {
            'gpu_mem_allocated_gb': allocated,
            'gpu_mem_cached_gb': cached,
        }
    return {}


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': total - trainable,
    }
