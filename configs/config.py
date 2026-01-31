"""
Configuration classes for Elastic-Depth Pretraining experiments.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import json


@dataclass
class ModelConfig:
    """Baseline Transformer Configuration (Frozen Spec)"""
    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    ffn_expansion: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256
    vocab_size: int = 50257  # GPT-2 tokenizer
    
    # Pre-LN (more stable for adaptive depth)
    pre_ln: bool = True
    
    @property
    def d_ffn(self) -> int:
        return self.d_model * self.ffn_expansion
    
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class EDPConfig:
    """Elastic-Depth Pretraining Configuration"""
    # Bowl-shaped routing: always compute layers [0, early_layers) and [n_layers - late_layers, n_layers)
    early_layers: int = 3   # Layers 0, 1, 2 always on
    late_layers: int = 3    # Layers 9, 10, 11 always on (for 12-layer model)
    
    # Routing
    use_second_order_signal: bool = True  # Use acceleration, not just delta
    learned_threshold: bool = True        # Learned vs fixed threshold
    initial_threshold: float = 0.1        # Initial tau value
    
    # Warp-forward
    use_warp_forward: bool = True         # Enable warp-forward mechanism
    max_skip_layers: int = 4              # Maximum layers to skip at once
    use_flop_aware_encoding: bool = True  # FLOP-aware vs static step encoding
    
    # Step encoding
    num_step_embeddings: int = 64         # Number of discrete FLOP buckets
    
    # Loss weights
    lambda_sparsity: float = 0.1          # Sparsity loss weight
    lambda_budget: float = 0.05           # Budget loss weight
    target_compute_ratio: float = 0.6     # Target fraction of FLOPs to use
    
    # Warmup schedule
    sparsity_warmup_fraction: float = 0.2  # First 20% steps: no sparsity loss
    
    # Entropy-aware sparsity (optional)
    use_entropy_aware_sparsity: bool = False


@dataclass
class TrainingConfig:
    """Training Configuration"""
    # Basic training
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.95)
    
    # Scheduling
    scheduler: str = "cosine"
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Device
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Data Configuration"""
    dataset: Literal["tinystories", "wikitext103"] = "tinystories"
    seq_len: int = 256
    
    # Dataset-specific
    tinystories_subset: Optional[str] = None  # Use full dataset
    wikitext_version: str = "wikitext-103-v1"
    
    # Preprocessing
    tokenizer: str = "gpt2"
    num_workers: int = 4
    
    # Splits
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"


@dataclass
class ExperimentConfig:
    """Full Experiment Configuration"""
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    edp: EDPConfig = field(default_factory=EDPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment metadata
    experiment_name: str = "edp_default"
    output_dir: str = "./outputs"
    wandb_project: str = "elastic-depth-pretraining"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'model': vars(self.model),
            'edp': vars(self.edp),
            'training': {k: v for k, v in vars(self.training).items()},
            'data': vars(self.data),
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'use_wandb': self.use_wandb,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentConfig':
        """Create from dictionary"""
        return cls(
            model=ModelConfig(**data.get('model', {})),
            edp=EDPConfig(**data.get('edp', {})),
            training=TrainingConfig(**data.get('training', {})),
            data=DataConfig(**data.get('data', {})),
            experiment_name=data.get('experiment_name', 'edp_default'),
            output_dir=data.get('output_dir', './outputs'),
            wandb_project=data.get('wandb_project', 'elastic-depth-pretraining'),
            wandb_entity=data.get('wandb_entity'),
            use_wandb=data.get('use_wandb', True),
        )


# Preset configurations for ablations
def get_ablation_config(ablation_name: str) -> ExperimentConfig:
    """Get configuration for a specific ablation study"""
    config = ExperimentConfig()
    config.experiment_name = f"ablation_{ablation_name}"
    
    if ablation_name == "first_order_signal":
        config.edp.use_second_order_signal = False
        
    elif ablation_name == "static_step_encoding":
        config.edp.use_flop_aware_encoding = False
        
    elif ablation_name == "no_warp_forward":
        config.edp.use_warp_forward = False
        
    elif ablation_name == "no_sparsity_loss":
        config.edp.lambda_sparsity = 0.0
        
    elif ablation_name == "fixed_threshold":
        config.edp.learned_threshold = False
        
    elif ablation_name == "full_depth_routing":
        config.edp.early_layers = 0
        config.edp.late_layers = 0
        
    elif ablation_name == "no_budget_loss":
        config.edp.lambda_budget = 0.0
        
    elif ablation_name == "target_ratio_0.4":
        config.edp.target_compute_ratio = 0.4
        
    elif ablation_name == "target_ratio_0.8":
        config.edp.target_compute_ratio = 0.8
        
    elif ablation_name == "entropy_aware":
        config.edp.use_entropy_aware_sparsity = True
        
    else:
        raise ValueError(f"Unknown ablation: {ablation_name}")
    
    return config


# List of all ablations
ABLATION_NAMES = [
    "first_order_signal",
    "static_step_encoding",
    "no_warp_forward",
    "no_sparsity_loss",
    "fixed_threshold",
    "full_depth_routing",
    "no_budget_loss",
    "target_ratio_0.4",
    "target_ratio_0.8",
    "entropy_aware",
]
