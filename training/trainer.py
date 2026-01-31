"""
Training loop for EDP and baseline models.

Handles:
- Mixed precision training
- Gradient clipping
- Learning rate scheduling
- Logging and checkpointing
- Routing statistics tracking
"""
import os
import time
import json
from typing import Dict, Optional, Tuple
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import EDPLoss, compute_perplexity
from ..utils import FLOPsTracker, EDPMetrics, setup_logging, count_parameters


class Trainer:
    """
    Trainer for EDP and baseline transformer models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        is_edp: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.is_edp = is_edp
        
        # Device
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.training.betas,
        )
        
        # Calculate total steps
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.steps_per_epoch * config.training.epochs
        
        # Learning rate scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.training.warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - config.training.warmup_steps,
            eta_min=1e-6,
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.training.warmup_steps],
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.mixed_precision else None
        self.use_amp = config.training.mixed_precision
        
        # EDP-specific loss
        if is_edp:
            self.edp_loss = EDPLoss(
                lambda_sparsity=config.edp.lambda_sparsity,
                lambda_budget=config.edp.lambda_budget,
                target_compute_ratio=config.edp.target_compute_ratio,
                sparsity_warmup_fraction=config.edp.sparsity_warmup_fraction,
                use_entropy_aware=config.edp.use_entropy_aware_sparsity,
                vocab_size=config.model.vocab_size,
            )
            self.edp_loss.set_training_steps(self.total_steps)
            
            # FLOPs tracker
            self.flops_tracker = FLOPsTracker(
                config.model.d_model,
                config.model.max_seq_len,
                config.model.n_layers,
            )
            
            # EDP metrics
            n_middle_layers = config.model.n_layers - config.edp.early_layers - config.edp.late_layers
            self.edp_metrics = EDPMetrics(config.model.n_layers, n_middle_layers)
        
        # Output directory
        self.output_dir = os.path.join(config.output_dir, config.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Logging
        self.logger = setup_logging(self.output_dir, config.experiment_name)
        self.logger.info(f"Model parameters: {count_parameters(model)}")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Total steps: {self.total_steps}")
        
        # Save config
        config.save(os.path.join(self.output_dir, 'config.json'))
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'lm_loss': 0.0,
        }
        if self.is_edp:
            epoch_losses.update({
                'sparsity_loss': 0.0,
                'budget_loss': 0.0,
                'avg_gate_ratio': 0.0,
            })
            
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                if self.is_edp:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_routing_info=True,
                    )
                    
                    # Compute EDP loss
                    loss_dict = self.edp_loss(
                        lm_loss=outputs['loss'],
                        gates=outputs['gates'],
                        logits=outputs['logits'],
                        attention_mask=attention_mask,
                    )
                    loss = loss_dict['total_loss']
                    
                    # Update EDP loss step
                    self.edp_loss.step()
                    
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs['loss']
                    loss_dict = {'total_loss': loss, 'lm_loss': loss}
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
                
            self.scheduler.step()
            
            # Update metrics
            epoch_losses['total_loss'] += loss.item()
            epoch_losses['lm_loss'] += loss_dict['lm_loss'].item()
            
            if self.is_edp:
                epoch_losses['sparsity_loss'] += loss_dict['sparsity_loss'].item()
                epoch_losses['budget_loss'] += loss_dict['budget_loss'].item()
                
                # Gate ratio
                gate_ratio = outputs['gates'].float().mean().item()
                epoch_losses['avg_gate_ratio'] += gate_ratio
                
                # Update FLOPs tracker
                middle_gates = outputs['gates'][..., self.config.edp.early_layers:self.config.model.n_layers - self.config.edp.late_layers]
                self.flops_tracker.update(middle_gates, self.config.edp.early_layers, self.config.edp.late_layers)
                
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
            
            # Logging
            self.global_step += 1
            if self.global_step % self.config.training.log_interval == 0:
                self._log_step(loss_dict, outputs if self.is_edp else None)
                
            # Evaluation
            if self.global_step % self.config.training.eval_interval == 0:
                val_loss = self.evaluate()
                self.val_losses.append((self.global_step, val_loss))
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best')
                    
                self.model.train()
                
            # Save checkpoint
            if self.global_step % self.config.training.save_interval == 0:
                self.save_checkpoint(f'step_{self.global_step}')
                
        # Average epoch losses
        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
            
        return epoch_losses
    
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_lm_loss = 0.0
        total_tokens = 0
        
        gate_ratios = []
        flops_ratios = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                with autocast(enabled=self.use_amp):
                    if self.is_edp:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_routing_info=True,
                        )
                        
                        gate_ratios.append(outputs['gates'].float().mean().item())
                        flops_ratios.append(outputs['routing_info']['mean_flops_ratio'].item())
                        
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        
                n_tokens = attention_mask.sum().item()
                total_loss += outputs['loss'].item() * n_tokens
                total_lm_loss += outputs['loss'].item() * n_tokens
                total_tokens += n_tokens
                
        avg_loss = total_loss / total_tokens
        avg_lm_loss = total_lm_loss / total_tokens
        perplexity = compute_perplexity(torch.tensor(avg_lm_loss)).item()
        
        self.logger.info(f"Validation - Loss: {avg_loss:.4f}, LM Loss: {avg_lm_loss:.4f}, PPL: {perplexity:.2f}")
        
        if self.is_edp and gate_ratios:
            avg_gate_ratio = sum(gate_ratios) / len(gate_ratios)
            avg_flops_ratio = sum(flops_ratios) / len(flops_ratios)
            self.logger.info(f"Validation - Gate Ratio: {avg_gate_ratio:.4f}, FLOPs Ratio: {avg_flops_ratio:.4f}")
            
        return avg_loss
    
    def train(self) -> Dict:
        """Full training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.training.epochs):
            self.logger.info(f"\n=== Epoch {epoch + 1}/{self.config.training.epochs} ===")
            
            # Train
            train_metrics = self.train_epoch(epoch + 1)
            self.train_losses.append((epoch, train_metrics))
            
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['total_loss']:.4f}")
            if self.is_edp:
                self.logger.info(f"Epoch {epoch + 1} - Gate Ratio: {train_metrics['avg_gate_ratio']:.4f}")
            
            # Evaluate
            val_loss = self.evaluate()
            self.val_losses.append((epoch, val_loss))
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}')
            
        # Final evaluation
        final_val_loss = self.evaluate()
        
        # Save final model
        self.save_checkpoint('final')
        
        # Save training history
        self._save_history()
        
        results = {
            'final_val_loss': final_val_loss,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if self.is_edp:
            results['flops_stats'] = self.flops_tracker.get_stats()
            results['edp_metrics'] = self.edp_metrics.get_summary()
            
        return results
    
    def _log_step(self, loss_dict: Dict, outputs: Optional[Dict] = None):
        """Log training step."""
        log_str = f"Step {self.global_step}: "
        log_str += f"loss={loss_dict['total_loss'].item():.4f}, "
        log_str += f"lm_loss={loss_dict['lm_loss'].item():.4f}"
        
        if self.is_edp and 'sparsity_loss' in loss_dict:
            log_str += f", sparse={loss_dict['sparsity_loss'].item():.4f}"
            log_str += f", budget={loss_dict['budget_loss'].item():.4f}"
            
        if outputs is not None and 'gates' in outputs:
            gate_ratio = outputs['gates'].float().mean().item()
            log_str += f", gate_ratio={gate_ratio:.4f}"
            
        self.logger.info(log_str)
        
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        path = os.path.join(self.output_dir, f'{name}.pt')
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.logger.info(f"Loaded checkpoint from {path}")
        
    def _save_history(self):
        """Save training history."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.is_edp:
            history['flops_stats'] = self.flops_tracker.get_stats()
            
        path = os.path.join(self.output_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=2, default=str)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    is_edp: bool = True,
) -> Dict:
    """
    Convenience function to train a model.
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        is_edp=is_edp,
    )
    
    return trainer.train()
