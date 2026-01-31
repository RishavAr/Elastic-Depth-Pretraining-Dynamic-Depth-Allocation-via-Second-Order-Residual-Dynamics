"""
Evaluation and analysis for EDP experiments.

Includes:
- Standard LM metrics (loss, perplexity)
- EDP-specific metrics (FLOPs usage, depth distribution)
- Routing analysis tools
"""
import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

from ..utils.flops import layer_flops, compute_effective_flops


class EDPEvaluator:
    """
    Comprehensive evaluator for EDP models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_amp: bool = True,
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.use_amp = use_amp
        
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        return_detailed: bool = False,
    ) -> Dict:
        """
        Evaluate model on dataloader.
        
        Args:
            dataloader: Evaluation data
            return_detailed: If True, return per-sample statistics
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        # EDP-specific tracking
        all_gate_ratios = []
        all_flops_ratios = []
        all_depth_used = []
        layer_gate_counts = None
        
        # Per-token tracking for analysis
        token_entropies = []
        token_depths = []
        
        is_edp = hasattr(self.model, 'router')
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                if is_edp:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_routing_info=True,
                    )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    
            # Basic metrics
            n_tokens = attention_mask.sum().item()
            total_loss += outputs['loss'].item() * n_tokens
            total_tokens += n_tokens
            
            # EDP metrics
            if is_edp:
                gates = outputs['gates']
                routing_info = outputs['routing_info']
                
                # Gate ratio
                all_gate_ratios.append(gates.float().mean().item())
                
                # FLOPs ratio
                all_flops_ratios.append(routing_info['mean_flops_ratio'].item())
                
                # Depth per token
                all_depth_used.extend(routing_info['depth_used'].cpu().numpy().flatten().tolist())
                
                # Per-layer gate counts
                gate_per_layer = gates.float().mean(dim=(0, 1)).cpu().numpy()
                if layer_gate_counts is None:
                    layer_gate_counts = gate_per_layer
                else:
                    layer_gate_counts += gate_per_layer
                    
                # Entropy-depth correlation
                if return_detailed:
                    entropy = self.model.compute_token_entropy(outputs['logits'])
                    token_entropies.extend(entropy.cpu().numpy().flatten().tolist())
                    token_depths.extend(routing_info['depth_used'].cpu().numpy().flatten().tolist())
                    
        # Aggregate metrics
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        results = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
        }
        
        if is_edp:
            results.update({
                'avg_gate_ratio': np.mean(all_gate_ratios),
                'avg_flops_ratio': np.mean(all_flops_ratios),
                'depth_distribution': {
                    'mean': np.mean(all_depth_used),
                    'std': np.std(all_depth_used),
                    'min': np.min(all_depth_used),
                    'max': np.max(all_depth_used),
                },
                'layer_utilization': (layer_gate_counts / len(dataloader)).tolist(),
            })
            
            if return_detailed and token_entropies:
                # Entropy-depth correlation
                correlation = np.corrcoef(token_entropies, token_depths)[0, 1]
                results['entropy_depth_correlation'] = correlation
                results['token_entropies'] = token_entropies[:10000]  # Sample
                results['token_depths'] = token_depths[:10000]
                
        return results
    
    @torch.no_grad()
    def analyze_routing(
        self,
        dataloader: DataLoader,
        n_batches: int = 10,
    ) -> Dict:
        """
        Detailed routing analysis.
        
        Returns per-layer, per-position routing statistics.
        """
        self.model.eval()
        
        # Collect routing decisions
        all_gates = []
        all_signals = []
        all_positions = []
        
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
                
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_routing_info=True,
                )
                
            all_gates.append(outputs['gates'].cpu())
            all_signals.append(outputs['routing_info']['signals'].cpu())
            
            # Position indices
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            all_positions.append(positions)
            
        # Stack all data
        gates = torch.cat(all_gates, dim=0)  # [total_samples, seq_len, n_layers]
        signals = torch.cat(all_signals, dim=0)  # [total_samples, seq_len, n_layers]
        positions = torch.cat(all_positions, dim=0)  # [total_samples, seq_len]
        
        n_layers = gates.shape[-1]
        seq_len = gates.shape[1]
        
        # Per-layer statistics
        layer_stats = {}
        for layer_idx in range(n_layers):
            layer_gates = gates[..., layer_idx]
            layer_signals = signals[..., layer_idx]
            
            layer_stats[f'layer_{layer_idx}'] = {
                'gate_ratio': layer_gates.float().mean().item(),
                'gate_std': layer_gates.float().std().item(),
                'signal_mean': layer_signals.mean().item(),
                'signal_std': layer_signals.std().item(),
            }
            
        # Per-position statistics
        position_stats = {}
        for pos in range(min(seq_len, 256)):  # First 256 positions
            pos_gates = gates[:, pos, :]
            
            position_stats[f'pos_{pos}'] = {
                'avg_depth': pos_gates.sum(dim=-1).float().mean().item(),
                'gate_ratio': pos_gates.float().mean().item(),
            }
            
        return {
            'layer_stats': layer_stats,
            'position_stats': position_stats,
            'n_samples': gates.shape[0],
        }
    
    @torch.no_grad()
    def compute_flops_vs_loss(
        self,
        dataloader: DataLoader,
        target_ratios: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ) -> Dict:
        """
        Compute loss at different FLOPs budgets.
        
        Used for scaling law plots.
        """
        results = {}
        
        for target in target_ratios:
            # Temporarily adjust budget target
            if hasattr(self.model, 'router'):
                # This would require modifying inference-time routing
                # For now, we measure natural distribution
                pass
                
        # For actual implementation, this would require:
        # 1. Inference-time threshold adjustment
        # 2. Or training multiple models at different budgets
        
        # Return natural distribution for now
        eval_results = self.evaluate(dataloader)
        
        return {
            'natural_flops_ratio': eval_results.get('avg_flops_ratio', 1.0),
            'natural_loss': eval_results['loss'],
            'natural_perplexity': eval_results['perplexity'],
        }


def compare_models(
    baseline_path: str,
    edp_path: str,
    test_loader: DataLoader,
    config,
) -> Dict:
    """
    Compare baseline and EDP models.
    """
    from ..models import create_baseline_model, create_edp_model
    
    # Load baseline
    baseline = create_baseline_model(config)
    baseline_ckpt = torch.load(baseline_path)
    baseline.load_state_dict(baseline_ckpt['model_state_dict'])
    
    baseline_evaluator = EDPEvaluator(baseline)
    baseline_results = baseline_evaluator.evaluate(test_loader)
    
    # Load EDP
    edp = create_edp_model(config)
    edp_ckpt = torch.load(edp_path)
    edp.load_state_dict(edp_ckpt['model_state_dict'])
    
    edp_evaluator = EDPEvaluator(edp)
    edp_results = edp_evaluator.evaluate(test_loader, return_detailed=True)
    
    # Comparison
    comparison = {
        'baseline': baseline_results,
        'edp': edp_results,
        'loss_diff': edp_results['loss'] - baseline_results['loss'],
        'perplexity_diff': edp_results['perplexity'] - baseline_results['perplexity'],
        'flops_savings': 1.0 - edp_results.get('avg_flops_ratio', 1.0),
    }
    
    return comparison
