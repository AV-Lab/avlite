#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 

@author: Murdism
"""
import torch 
from torch import Tensor
import torch.nn as nn
import math
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from typing import Tuple, Dict, Optional, List, Union,Any
from dataclasses import dataclass, asdict
from c70_extension.prediction_utils import TrajectoryHandler,calculate_ade,calculate_fde,visualize_occupancy_grid_tkinter
from tqdm import tqdm
import os
import logging
import sys
from pathlib import Path
import time

@dataclass
class ModelConfig:
    """Configuration for the AttentionGMM model."""
    # Model architecture parameters
    mode: Optional[str] = 'train' # predict -> set to predict to load trained model
    past_trajectory: int = 10
    future_trajectory: int = 30
    device: Optional[torch.device] = None
    normalize: bool = True
    saving_checkpoint_path: Optional[str] = None  # Allow user-defined checkpoint
    win_size: int = 3
    lambda_value: Optional[float] = 0.0
    diagonal_sigma: Optional[bool] = True # coveriance matrix is diagonal
    mean: torch.tensor = torch.tensor([0.0, 0.0, 0.0, 0.0])
    std: torch.tensor = torch.tensor([1.0, 1.0, 1.0, 1.0])
    in_features: int = 2
    out_features: int = 2
    num_heads: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    embedding_size: int = 64
    dropout: float = 0.3
    batch_first: bool = True
    actn: str = "gelu"
    
    

    # GMM parameters
    n_gaussians: int = 6
    n_hidden: int = 32

    # Optimizer parameters
    lr_mul: float = 0.04
    n_warmup_steps: int = 2200 #2000 #3000 #3500 #4000
    optimizer_betas: Tuple[float, float] = (0.9, 0.98)
    optimizer_eps: float = 1e-9

    # Early stopping parameters
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001

    # logging:
    log_save_path = 'results/metrics/training_metrics'

    #eval_metrics
    best_of_k = 5 
    

    def __post_init__(self):
        """Post-init processing."""
        # if self.checkpoint_file is None:
        #     self.checkpoint_file = f'GMM_transformer_P_{self.past_trajectory}_F_{self.future_trajectory}_W_x.pth'
        if self.lr_mul <= 0:
            raise ValueError("Learning rate multiplier must be positive")
        if self.n_warmup_steps < 0:
            raise ValueError("Warmup steps must be non-negative")

    def get_device(self) -> torch.device:
        """Return the device for computation."""
        return self.device if self.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def display_config(self, verbose: bool = False) -> None:
        """
        Pretty print the model configuration using logging.
        
        Args:
            verbose (bool): If True, logs additional information and formatting
        """
        logger = logging.getLogger('AttentionGMM')
        
        if verbose:
            logger.info("\n" + "="*50)
            logger.info("AttentionGMM Model Configuration")
            logger.info("="*50)
            
            logger.info("\nModel Architecture:")
            logger.info("-"*20)
            logger.info(f"Input Features:      {self.in_features}")
            logger.info(f"Output Features:     {self.out_features}")
            logger.info(f"Number of Heads:     {self.num_heads}")
            logger.info(f"Encoder Layers:      {self.num_encoder_layers}")
            logger.info(f"Decoder Layers:      {self.num_decoder_layers}")
            logger.info(f"Embedding Size:      {self.embedding_size}")
            logger.info(f"Dropout Rate:        {self.dropout}")
            logger.info(f"Batch First:         {self.batch_first}")
            logger.info(f"Activation Function: {self.actn}")

            logger.info(f"Dataset info:")
            logger.info("-"*20)
            logger.info(f"sliding window: {self.win_size}")
            logger.info(f"past trajectory:   {self.past_trajectory}")
            logger.info(f"future trajectory: {self.future_trajectory}")
            
            logger.info("\nGMM Settings:")
            logger.info("-"*20)
            logger.info(f"Number of Gaussians: {self.n_gaussians}")
            logger.info(f"Coveraiance Configuration: diagonal matrix: {self.diagonal_sigma}")
            logger.info(f"Hidden Size:         {self.n_hidden}")
            
            logger.info("\nOptimizer Settings:")
            logger.info("-"*20)
            logger.info(f"Learning Rate Multiplier: {self.lr_mul}")
            logger.info(f"Warmup Steps:            {self.n_warmup_steps}")
            logger.info(f"Optimizer Betas:         {self.optimizer_betas}")
            logger.info(f"Optimizer Epsilon:       {self.optimizer_eps}")
            
            logger.info("\nEarly Stopping Settings:")
            logger.info("-"*20)
            logger.info(f"Patience:               {self.early_stopping_patience}")
            logger.info(f"Delta:                  {self.early_stopping_delta}")

            logger.info("\Loss info:")
            logger.info("-"*20)
            logger.info(f" Entropy loss weight (lambda_value) : {self.lambda_value}")
            
            
            logger.info("\nDevice Configuration:")
            logger.info("-"*20)
            logger.info(f"Device: {self.get_device()}")
            logger.info("\n" + "="*50)
        else:
            # Simple log of key parameters
            logger.info(
                f"AttentionGMM Config: in_features={self.in_features}, "
                f"out_features={self.out_features}, num_heads={self.num_heads}, "
                f"embedding_size={self.embedding_size}, dropout={self.dropout}, "
                f"n_gaussians={self.n_gaussians}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)   

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        #print(self.n_warmup_steps)
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
   
class AttentionGMM(nn.Module):
    """
    Attention-based Encoder-Decoder Transformer Model for time series forecasting.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        num_heads (int): Number of attention heads
        num_encoder_layers (int): Number of transformer encoder layers
        num_decoder_layers (int): Number of transformer decoder layers
        embedding_size (int): Size of the embedding dimension
        dropout (float): Dropout rate for encoder and decoder
        max_length (int): Maximum sequence length
        batch_first (bool): If True, batch dimension is first
        actn (str): Activation function to use
        device (torch.device): Device to use for computation
    """
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        **kwargs
    ):
        super().__init__()
        # Create config object first
        self.config = config or ModelConfig(**kwargs)
        
        self._validate_config()
        self._init_device()
        self._init_model_params()
        self._init_layers()
        self._init_optimizer_params()
        
        self.tracker = MetricTracker()
        
        if self.mode == 'predict':
            self.load_model(self.checkpoint_file)
            # Intialize TrajectoryHandler to track history of detected objects
            self.traj_handler = TrajectoryHandler(device='cuda')

        
        
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.embedding_size % self.config.num_heads != 0:
            raise ValueError("Embedding size must be divisible by number of heads")
        if self.config.num_heads < 1:
            raise ValueError("Number of heads must be positive")
        if self.config.n_gaussians < 1:
            raise ValueError("Number of Gaussians must be positive")
        if self.config.n_hidden < 1:
            raise ValueError("Hidden size must be positive")
        if not 0 <= self.config.dropout <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if self.config.past_trajectory < 1 or self.config.future_trajectory < 1:
            raise ValueError("Trajectory lengths must be positive")
        
    def _init_device(self):
        """Initialize device configuration."""
        self.device = self.config.get_device()
        self.mean = self.config.mean.to(self.device)
        self.std = self.config.std.to(self.device)
        self.mode = self.config.mode
        self.diagonal_covariance = self.config.diagonal_sigma
        
    def _init_model_params(self):
        """Initialize model parameters."""
        self.num_heads = self.config.num_heads
        self.max_len = max(self.config.past_trajectory, self.config.future_trajectory)
        self.past_trajectory = self.config.past_trajectory
        self.future_trajectory = self.config.future_trajectory
        self.num_gaussians = self.config.n_gaussians
        self.hidden = self.config.n_hidden
        
        self.normalized = self.config.normalize
        self.d_model = self.config.embedding_size
        self.input_features = self.config.in_features
        self.output_features = self.config.out_features
        self.dim_feedforward = 4 * self.d_model #Set feedforward dimensions (4x larger than d_model as per original paper)

        # Define dropout rates
        self.dropout_encoder = self.config.dropout
        self.dropout_decoder = self.config.dropout

        # Logging path
        self.log_save_path = self.config.log_save_path
        self.checkpoint_file = self.config.saving_checkpoint_path
        self.best_of_k = self.config.best_of_k

        # loss 
        self.lambda_value = self.config.lambda_value
    
    def _init_optimizer_params(self):
        # Store optimizer parameters
        self.lr_mul = self.config.lr_mul 
        self.n_warmup_steps = self.config.n_warmup_steps
        self.optimizer_betas = self.config.optimizer_betas
        self.optimizer_eps = self.config.optimizer_eps

        #Initialize early stopping parameters
        self.early_stop_counter = 0
        self.early_stopping_patience = self.config.early_stopping_patience
        self.early_stopping_delta = self.config.early_stopping_delta
        self.best_metrics = {
            'ade': float('inf'),
            'fde': float('inf'),
            'best_ade': float('inf'),
            'best_fde': float('inf')
        }

    def _init_layers(self):
        """Initialize model layers."""
        # Embeddings
        self.encoder_input_layer = Linear_Embeddings(self.config.in_features, self.d_model)
        self.decoder_input_layer = Linear_Embeddings(self.config.out_features, self.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.max_len,
            batch_first=self.config.batch_first
        )
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Decoder
        self.decoder = self._build_decoder()
        
        # GMM layers
        self._init_gmm_layers()
    
    def _build_encoder(self):
        """Build encoder layers."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_encoder,
            batch_first=self.config.batch_first,
            activation=self.config.actn
        )
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config.num_encoder_layers
        )
    
    def _build_decoder(self):
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = self.d_model,
            nhead = self.num_heads,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout_decoder,
            batch_first = self.config.batch_first,
            activation = self.config.actn 
            )
        return nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers = self.config.num_decoder_layers
        )

    def _create_gmm_embedding(self):
        """Create a simple GMM embedding network."""
        return nn.Sequential(
            nn.Linear(self.d_model, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, int(self.hidden * 0.75)),
            nn.ELU(),
            nn.Linear(int(self.hidden * 0.75), self.hidden // 2),
            nn.ELU()
        )
    
    def _init_gmm_layers(self):
        """Initialize Gaussian Mixture Model layers."""
        # Create embedding networks
        self.embedding_sigma = self._create_gmm_embedding()
        self.embedding_mue = self._create_gmm_embedding()
        self.embedding_pi = self._create_gmm_embedding()
        
        
        # Create output heads
        self.pis_head = nn.Linear(self.hidden // 2, self.num_gaussians)
        self.sigma_head = nn.Linear(self.hidden // 2, self.num_gaussians * 2)
        if not self.diagonal_covariance:
            self.embedding_rho = self._create_gmm_embedding()
            self.rho_head = nn.Linear(self.hidden // 2, self.num_gaussians )
        self.mu_head = nn.Linear(self.hidden // 2, self.num_gaussians * 2)
    
    def _init_weights(self):
        """Initialize the model weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            src (torch.Tensor): Source sequence
            tgt (torch.Tensor): Target sequence
            src_mask (torch.Tensor, optional): Mask for source sequence
            tgt_mask (torch.Tensor, optional): Mask for target sequence
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Add input validation
        if src.dim() != 3 or tgt.dim() != 3:
            raise ValueError("Expected 3D tensors for src and tgt")
        
        # Move inputs to device
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self.device)
        
        # Encoder forward pass
        encoder_embed = self.encoder_input_layer(src)
        encoder_embed = self.positional_encoding(encoder_embed)
        encoder_output = self.encoder(src=encoder_embed)
        
        # Decoder forward pass
        decoder_embed = self.decoder_input_layer(tgt)
        decoder_embed = self.positional_encoding(decoder_embed)
        decoder_output = self.decoder(
            tgt=decoder_embed,
            memory=encoder_output,
            tgt_mask=tgt_mask
            # memory_mask=src_mask
        )
        

        # Compute embeddings
        sigma_embedded = self.embedding_sigma(decoder_output)
        mue_embedded = self.embedding_mue(decoder_output)
        pi_embedded = self.embedding_pi(decoder_output)  # <-- Apply embedding to pi
        
        
        if not self.diagonal_covariance:
            rho_embedded = self.embedding_rho(decoder_output)
            # Apply tanh to constrain rho between -1 and 1
            rho = torch.tanh(self.rho_head(rho_embedded))
            rho = torch.clamp(rho, min=-0.99, max=0.99)
        else:
            rho = None
        # Mixture weights (apply softmax)
        pi = torch.softmax(self.pis_head(pi_embedded), dim=-1)
        
        # Compute Sigmas with softplus to ensure positivity
        sigma = nn.functional.softplus(self.sigma_head(sigma_embedded))
        sigma_x, sigma_y = sigma.chunk(2, dim=-1)
        
        # Compute Means
        mu = self.mu_head(mue_embedded)
        mu_x, mu_y = mu.chunk(2, dim=-1)
        

        return pi, sigma_x,sigma_y, mu_x ,mu_y, rho 
    
    def train(
        self,
        train_dl: DataLoader,
        eval_dl: DataLoader = None,
        epochs: int = 50,
        verbose: bool = True,
        save_path: str = 'results',
        save_model: bool = True,
        save_frequency: int = 20,
    ) -> Tuple[nn.Module, Dict]:
        """
        Train the model with metrics tracking and visualization.
        """
        # Setup logger
        logger = logging.getLogger('AttentionGMM')
        if not logger.handlers:
            logger = self.setup_logger(save_path=self.log_save_path)
        
        self.to(self.device)
        self._init_weights()
        
        #  if verbose print config:
        self.config.display_config(verbose)

        # Setup optimizer with model's configuration
        optimizer = self.configure_optimizer(
            lr_mul=self.lr_mul,
            n_warmup_steps=self.n_warmup_steps,
            optimizer_betas=self.optimizer_betas,
            optimizer_eps=self.optimizer_eps
        )

        # set metrics tracker:
        self.tracker.train_available = True

        # Set up directory structure
        models_dir = os.path.join(save_path, 'pretrained_models')
        metrics_dir = os.path.join(save_path, 'metrics')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        # get mean and standard deviation from training dataset
        self.mean= train_dl.dataset.mean.to(self.device)
        self.std = train_dl.dataset.std.to(self.device)
       
        for epoch in range(epochs):
            super().train()  # Set train mode again for safety

            # Training loop with progress bar
            load_train = tqdm(train_dl, desc=f"Epoch: {epoch+1}/{epochs}") if verbose else train_dl

            for id_b, batch in enumerate(load_train):
                # Prepare input data
                obs_tensor, target_tensor = batch
                batch_size, enc_seq_len, feat_dim = obs_tensor.shape
                dec_seq_len = target_tensor.shape[1]
                
                # Move to device and normalize
                obs_tensor = obs_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)

                input_train = (obs_tensor[:,1:,2:4] - self.mean[2:])/self.std[2:]
                updated_enq_length = input_train.shape[1]
                target = ((target_tensor[:, :, 2:4] - self.mean[2:]) / self.std[2:]).clone()

                tgt = torch.zeros((target.shape[0], dec_seq_len, 2), dtype=torch.float32, device=self.device)


                # Generate masks
                tgt_mask = self._generate_square_mask(
                    dim_trg=dec_seq_len,
                    dim_src=updated_enq_length,
                    mask_type="tgt"
                ).to(self.device)
                

                # Forward pass
                optimizer.zero_grad()

    
                pi, sigma_x,sigma_y, mu_x ,mu_y, rho = self(input_train,tgt,tgt_mask = tgt_mask)
                mus = torch.cat((mu_x.unsqueeze(-1),mu_y.unsqueeze(-1)),-1)
                sigmas = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)

                
                # Calculate loss
                train_loss = self._mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,target,self.num_gaussians,rho)
                
                # Backward pass
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                # print(f"Gradient Norm: {total_norm:.4f}")
                optimizer.step_and_update_lr()

                with torch.no_grad(): # to avoid data leakage during sampling
                    
                    highest_prob_pred, best_of_n_pred = self._sample_gmm_predictions(pi, mus)
                    # batch_trajs,batch_weights,best_trajs,best_weights = self._run_cluster(mus,pi)
                    # print(batch_trajs.shape,batch_weights.shape)
                    # print(batch_trajs[0][0],batch_weights[0])
                    
                    obs_last_pos = obs_tensor[:, -1:, 0:2]

                    # using heighest probability values
                    mad, fad = self.calculate_metrics(
                        highest_prob_pred.detach(), target.detach(), obs_last_pos)
                    
                    # Best of n_predictions error
                    mad_best_n, fad_best_n = self.calculate_metrics(
                        best_of_n_pred.detach(), target.detach(), obs_last_pos)
                   
                    # Update metrics using tracker
                    batch_metrics = {
                        'loss': train_loss.item(),
                        'ade': mad,
                        'fde': fad,
                        'best_ade': mad_best_n,
                        'best_fde': fad_best_n
                    }
                    self.tracker.update(batch_metrics, obs_tensor.shape[0], phase='train')      
                    #Update progress bar
                    if verbose:
                        train_avgs = self.tracker.get_averages('train')
                        load_train.set_postfix({
                            'Loss': f"{train_avgs['loss']:.4f}",
                            'ADE': f"{train_avgs['ade']:.4f}",
                            'FDE': f"{train_avgs['fde']:.4f}",
                            'Best_ADE': f"{train_avgs['best_ade']:.4f}",
                            'Best_FDE': f"{train_avgs['best_fde']:.4f}"
                        })
                    
            # At end of epoch
            self.tracker.compute_epoch_metrics(phase='train')
            # Test evaluation
            if eval_dl is not None:
                self.evaluate(eval_dl,from_train=True)

            # Print epoch metrics
            self.tracker.print_epoch_metrics(epoch, epochs, verbose)

            # Check early stopping conditions
            phase = 'test' if eval_dl else 'train'
            current_metrics = {
                'loss':self.tracker.history[f'{phase}_loss'][-1],
                'ade': self.tracker.history[f'{phase}_ade'][-1],
                'fde': self.tracker.history[f'{phase}_fde'][-1],
                'best_ade': self.tracker.history[f'{phase}_best_ade'][-1],
                'best_fde': self.tracker.history[f'{phase}_best_fde'][-1]
            }
            
            should_stop,found_better = self.check_early_stopping(current_metrics, verbose,stop_metric='ade')
            
            if should_stop:
                logger.info(f"Stoped training -> Early stoping! ade has not improved in {self.config.early_stopping_patience} epochs!.")
                break
            # Save model if save_frequency reached 
            if found_better:
                self._save_checkpoint(optimizer, epoch,save_model,1,save_path,better_metric=True)
            
            else:
                self._save_checkpoint(optimizer, epoch,save_model,save_frequency,save_path)
            # Break if early stopping triggered
            # if should_stop:
            #     logger.info("Early stopping triggered. Ending training.")
            #     break

            # Reset metrics for next epoch
            self.tracker.reset('train')
            self.tracker.reset('test')

        # Plot training history if verbose
        if verbose:
            self.plot_metrics(
                self.tracker.history['train_loss'],
                self.tracker.history['test_loss'],
                self.tracker.history['train_ade'],
                self.tracker.history['test_ade'],
                self.tracker.history['train_fde'],
                self.tracker.history['test_fde'],
                self.tracker.history['train_best_ade'],
                self.tracker.history['test_best_ade'],
                self.tracker.history['train_best_fde'],
                self.tracker.history['test_best_fde'],
                enc_seq_len,
                dec_seq_len
            )
            logger.info(f"Training plots saved to {metrics_dir}")

        return self, self.tracker.history

    def _save_checkpoint(self,optimizer, epoch=10,save_model=True,save_frequency=10,save_path="/results",better_metric=False):
        # Set up directory structure
        models_dir = os.path.join(save_path, 'pretrained_models')
        os.makedirs(models_dir, exist_ok=True)
        logger = logging.getLogger('AttentionGMM')

        if save_model and (epoch + 1) % save_frequency == 0 or better_metric :
            model_state = {
                'model_state_dict': self.state_dict(),  # Save directly
                'optimizer_state_dict': optimizer._optimizer.state_dict(),
                'training_history': self.tracker.history,
                'best_metrics': self.tracker.best_metrics,
                'train_mean': self.mean,
                'train_std': self.std,
                'num_gaussians': self.num_gaussians,
                'model_config': {
                    # Only save what you actually use for loading
                    'in_features': self.input_features,
                    'out_features': self.output_features,
                    'num_heads': self.num_heads,
                    'num_encoder_layers': self.config.num_encoder_layers,
                    'num_decoder_layers': self.config.num_decoder_layers,
                    'embedding_size': self.d_model,
                    'dropout': self.dropout_encoder
                }
            }
            # torch.save(model_state, os.path.join(models_dir, checkpoint_name))
            # Save the model
            # checkpoint_name = f'GMM_transformer_P_{self.past_trajectory}_F_{self.future_trajectory}_Warm_{self.n_warmup_steps}_W_{self.config.win_size}.pth'
            if better_metric:
                checkpoint_name = f'GMM_transformer_P_{self.past_trajectory}_F_{self.future_trajectory}_Warm_{self.n_warmup_steps}_W_{self.config.win_size}_epoch{epoch+1}_lambda_{self.lambda_value}_best_ade.pth'
            else:
                checkpoint_name = f'GMM_transformer_P_{self.past_trajectory}_F_{self.future_trajectory}_Warm_{self.n_warmup_steps}_W_{self.config.win_size}_lambda_{self.lambda_value}_epoch_{epoch+1}.pth'
            os.makedirs(save_path, exist_ok=True)
            torch.save(model_state, os.path.join(models_dir, f"{checkpoint_name}"))
            logger.info(f"Saved checkpoint to: {save_path}")
    def load_model(self, ckpt_path: str):
        """
        Load a complete model with all necessary state.
        
        Args:
            ckpt_path (str): Path to checkpoint file
        """
        if ckpt_path is None:
            raise ValueError("Checkpoint path cannot be None")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.load_state_dict(state_dict=checkpoint['model_state_dict'])
            self.to(self.device)  # Move the entire model to device
            
            # Load and move tensors to device
            self.mean = checkpoint['train_mean'].to(self.device)
            self.std = checkpoint['train_std'].to(self.device)
            self.num_gaussians = checkpoint['num_gaussians']
            
            if 'model_config' in checkpoint:
                # Update any config parameters if needed
                for key, value in checkpoint['model_config'].items():
                    setattr(self.config, key, value)
            
            return self
                
        except KeyError as e:
            raise KeyError(f"Checkpoint missing required key: {e}")
        except Exception as e:
            raise Exception(f"Error loading checkpoint: {e}")
        
    def configure_optimizer(
        self,
        lr_mul: Optional[float] = None,
        n_warmup_steps: Optional[int] = None,
        optimizer_betas: Optional[Tuple[float, float]] = None,
        optimizer_eps: Optional[float] = None
    ) -> ScheduledOptim:
        """
        Configure the scheduled optimizer with optional parameter overrides.
        
        Args:
            lr_mul (float, optional): Learning rate multiplier
            n_warmup_steps (int, optional): Number of warmup steps
            optimizer_betas (tuple, optional): Beta parameters for Adam
            optimizer_eps (float, optional): Epsilon parameter for Adam
            
        Returns:
            ScheduledOptim: Configured optimizer with scheduling
        """
        # Use provided values or fall back to initialization values
        lr_mul = lr_mul if lr_mul is not None else self.lr_mul
        n_warmup_steps = n_warmup_steps if n_warmup_steps is not None else self.n_warmup_steps
        optimizer_betas = optimizer_betas if optimizer_betas is not None else self.optimizer_betas
        optimizer_eps = optimizer_eps if optimizer_eps is not None else self.optimizer_eps

        # Create base optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            betas=optimizer_betas,
            eps=optimizer_eps
        )

        # Wrap with scheduler
        return ScheduledOptim(
            optimizer=optimizer,
            lr_mul=lr_mul,
            d_model=self.d_model,
            n_warmup_steps=n_warmup_steps
        )
    
    def _sample_gmm_predictions(self, pi, mue):
        """
        Returns both highest probability and best-of-N predictions
        
        Args:
            pi (torch.Tensor): Mixture weights (batch_size, seq_len, n_mixtures)
            sigma (torch.Tensor): Standard deviations
            mue (torch.Tensor): Means (batch_size, seq_len, n_mixtures, 2)
            gt_normalized: Normalized ground truth for best-of-N selection
            
        Returns:
            tuple: (highest_prob_pred)
        """
        # 1. Get highest probability predictions
        max_indices = torch.argmax(pi, dim=2).unsqueeze(-1)
        highest_prob_pred = torch.gather(mue, dim=2, 
                                    index=max_indices.unsqueeze(dim=-1).repeat(1, 1, 1, 2))
        highest_prob_pred = highest_prob_pred.squeeze(dim=2)
        
        return highest_prob_pred
    
    def _bivariate(self,pi,sigma_x,sigma_y, mu_x , mu_y,target_points,rho):
        """
        Calculate bivariate Gaussian probability density.

        Args:
            pi (torch.Tensor): Mixture weights (batch_size, seq_len, n_mixtures)
            sigma_x (torch.Tensor): Standard deviations in x direction (batch_size, seq_len, n_mixtures)
            sigma_y (torch.Tensor): Standard deviations in y direction (batch_size, seq_len, n_mixtures)
            mu_x (torch.Tensor): Means in x direction (batch_size, seq_len, n_mixtures)
            mu_y (torch.Tensor): Means in y direction (batch_size, seq_len, n_mixtures)
            target_points (torch.Tensor): Target coordinates (batch_size, seq_len, 2) or (batch_size, 2)
            rho (torch.Tensor, optional): Correlation coefficients (batch_size, seq_len, n_mixtures)

        Returns:
            torch.Tensor: Log probability densities (batch_size, seq_len, n_mixtures)
        """
        # Validate input values
        if torch.isnan(sigma_x).any() or torch.isnan(sigma_y).any():
            raise ValueError("NaN values detected in sigma computation")    
        if torch.any(pi <= 0):
            raise ValueError("Mixture weights must be positive")

        # Validate shapes match
        expected_shape = pi.shape
        for tensor, name in [(sigma_x, 'sigma_x'), (sigma_y, 'sigma_y'), 
                            (mu_x, 'mu_x'), (mu_y, 'mu_y')]:
            if tensor.shape != expected_shape:
                raise ValueError(f"Shape mismatch: {name} has shape {tensor.shape}, "
                            f"expected {expected_shape}")

        # Extract x, y coordinates and add necessary dimensions
        if target_points.ndim == 3:
            x = target_points[:,:,0].unsqueeze(-1)
            y = target_points[:,:,1].unsqueeze(-1)
        elif target_points.ndim == 2:
            x = target_points[:,0].unsqueeze(1)
            y = target_points[:,1].unsqueeze(1)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {target_points.ndim}D")

        # Calculate squared normalized distances
        norm_x = torch.square((x.expand_as(mu_x) - mu_x) * torch.reciprocal(sigma_x))
        norm_y = torch.square((y.expand_as(mu_y) - mu_y) * torch.reciprocal(sigma_y))

        # Calculate log probabilities
        log_pi = torch.log(pi)
        if rho is None:
            # Diagonal case - original calculation
            exponent = -0.5 * (norm_x + norm_y)
            log_normalization = -torch.log(2.0 * np.pi * sigma_x * sigma_y)
        else:
            # Correlated case - new calculation using norm_x and norm_y
            # We need the non-squared terms for the cross-product term
            norm_xy = ((x.expand_as(mu_x) - mu_x) * torch.reciprocal(sigma_x)) * \
                    ((y.expand_as(mu_y) - mu_y) * torch.reciprocal(sigma_y))
            
            exponent = -0.5 / (1.0 - rho**2) * (norm_x - 2*rho*norm_xy + norm_y)
            log_normalization = -torch.log(2.0 * np.pi * sigma_x * sigma_y * torch.sqrt(1.0 - rho**2))
        
        return log_pi + log_normalization.expand_as(log_pi) + exponent
    def _mdn_loss_fn(self,pi, sigma_x,sigma_y, mu_x , mu_y,targets,n_mixtures,rho=None):
        """
        Calculate the Mixture Density Network loss using LogSumExp trick for numerical stability.
        
        Args:
            pi (torch.Tensor): Mixture weights
            sigma_x (torch.Tensor): Standard deviations in x direction
            sigma_y (torch.Tensor): Standard deviations in y direction
            mu_x (torch.Tensor): Means in x direction
            mu_y (torch.Tensor): Means in y direction
            targets (torch.Tensor): Target points
            n_mixtures (int): Number of Gaussian mixtures
        
        Returns:
            torch.Tensor: Mean negative log likelihood loss
        """

        logger = logging.getLogger('AttentionGMM')
        # Calculate log probabilities for each mixture component
        log_probs = self._bivariate(pi, sigma_x, sigma_y, mu_x, mu_y, targets,rho)

        # Apply LogSumExp trick for numerical stability
        max_log_probs = torch.max(log_probs, dim=2, keepdim=True)[0]
        max_log_probs_repeated = max_log_probs.repeat(1, 1, n_mixtures)


        # Calculate stable log sum exp
        exp_term = torch.exp(log_probs - max_log_probs_repeated)
        epsilon = 0.00001 #torch.finfo(torch.float32).eps  # Use machine epsilon
        sum_exp = torch.sum(exp_term, dim=-1) + epsilon

        # Final loss calculation
        neg_log_likelihood = -(max_log_probs[:,:,0] + torch.log(sum_exp))
        
        # Check for numerical instability
        if torch.isnan(neg_log_likelihood).any():
            raise ValueError("NaN values detected in loss computation")
        
        # calculate entropy
        entropy = -torch.sum(pi * torch.log(pi + 1e-8), dim=-1)
        entropy_loss = -entropy.mean()  # minimize - entropy = maximize entropy
        mdn_loss = torch.mean(neg_log_likelihood)
        
        # logger.info(f"entropy_loss:{entropy_loss}")
        # logger.info(f"mdn loss:{mdn_loss}")
        
        # logger.info(f"pi [0]:{pi[0]}")

        total_loss = mdn_loss +  self.lambda_value* entropy_loss
        # logger.info(f"total_loss:{total_loss}")

        return total_loss
    def check_early_stopping(self, current_metrics: dict, verbose: bool = True, stop_metric='ade') -> Tuple[bool, dict]:
        """
        Check if training should stop based on the specified metric.
        
        Args:
            current_metrics (dict): Dictionary containing current metric values
            verbose (bool): Whether to print early stopping information
            metric (str): The specific metric to monitor for early stopping
            
        Returns:
            Tuple[bool, dict]: (should_stop, best_metrics)
        """
        should_stop = True
        logger = logging.getLogger('AttentionGMM') 

        found_better = False
        # Only check the specified metric
        if stop_metric in current_metrics and stop_metric in self.best_metrics:
            current_value = current_metrics[stop_metric]
            
            # Check if the current value is better than the best value
            # print(self.best_metrics[stop_metric] , type(self.best_metrics[stop_metric]),self.best_metrics[stop_metric])
            if current_value < self.best_metrics[stop_metric]:
                self.best_metrics[stop_metric] = current_value
                should_stop = False
                found_better = True
                logger.info(f"\nImprovement in {stop_metric}! \n{self.early_stop_counter} epochs without improvements.")
        
        # Update counter based on improvement
        if should_stop:
            self.early_stop_counter += 1
            if verbose and self.early_stop_counter > 0:
                logger.info(f"\nNo improvement in {stop_metric} for {self.early_stop_counter} epochs.")
                logger.info(f"Best {stop_metric.upper()}: {self.best_metrics[stop_metric]:.4f}")
                
        else:
            self.early_stop_counter = 0
        
        # Check if we should stop training
        should_stop = self.early_stop_counter >= self.config.early_stopping_patience
        
        # Log early stopping information if triggered
        if should_stop and verbose:
            logger.info(f"\nEarly stopping triggered after {self.early_stop_counter} epochs without improvement in {stop_metric}")
            logger.info(f"Best {stop_metric.upper()}: {self.best_metrics[stop_metric]:.4f}")
        
        return should_stop,found_better

    def denormalize_to_absolute(self,pred: torch.Tensor, obs_last_pos: torch.Tensor,target: torch.Tensor=None, mue_values: torch.Tensor = None):
        """
       Denormalize and change to absolute positions
        Args:
            pred: predicted velocities [batch, seq_len, 2]
            target: target velocities [batch, seq_len, 2]
            obs_last_pos: last observed position [batch, 1, 2]
            mean: mean values for denormalization
            std: standard deviation values for denormalization
            device: computation device
        """
        
        if self.normalized:
            pred = pred * self.std[..., 2:] + self.mean[..., 2:]
            target =  target * self.std[..., 2:] + self.mean[..., 2:] if target is not None else None
        

        # Expand obs_last_pos if needed
        if pred.ndim == 4:
            # pred.shape = (y, x, 30, 2)
            y, x, _, _ = pred.shape
            # obs_last_pos_expanded = obs_last_pos.unsqueeze(1).expand(-1, x, -1).reshape(-1, obs_last_pos.shape[-1])
            obs_last_pos_expanded = obs_last_pos.repeat_interleave(x, dim=0)  # (128 * x, 1, 2)


        else:
            obs_last_pos_expanded = obs_last_pos


        # Always flatten batch dims if needed
        pred_flat   = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-2], target.shape[-1]) if target is not None else None

        if isinstance(pred_flat, torch.Tensor):
            pred_flat = pred_flat.cpu().numpy()

        # print(f"obs_last_pos_expanded: {obs_last_pos_expanded.shape}")

        # Now everything matches shape
        pred_pos = pred_flat.cumsum(1) + obs_last_pos_expanded.cpu().numpy()
        target_pos = target_flat.cpu().numpy().cumsum(1) + obs_last_pos.cpu().numpy() if target is not None else None

        # Reshape back
        pred_pos = pred_pos.reshape(*pred.shape[:-2], *pred.shape[-2:])
        target_pos = target_pos.reshape(*target.shape[:-2], *target.shape[-2:]) if target is not None else None

        if mue_values is not None:
            # Current shape: [128, 30, 6, 2]
            # Need shape:    [128, 6, 30, 2]
            mus_permuted = mue_values.permute(0, 2, 1, 3)
            mue_flat = mus_permuted.reshape(-1, mus_permuted.shape[-2], mus_permuted.shape[-1])
            if isinstance(mue_flat, torch.Tensor):
                mue_flat = mue_flat.cpu().numpy()
            y, x, _, _ = mus_permuted.shape
            # obs_last_pos_expanded = obs_last_pos.unsqueeze(1).expand(-1, x, -1).reshape(-1, obs_last_pos.shape[-1])
            obs_last_pos_expanded = obs_last_pos.repeat_interleave(x, dim=0)  # (128 * x, 1, 2)
            mue_pos = mue_flat.cumsum(1) + obs_last_pos_expanded.cpu().numpy()
            mue_pos = mue_pos.reshape(*mus_permuted.shape[:-2], *mus_permuted.shape[-2:])
            return pred_pos,target_pos,mue_pos

        
        return pred_pos,target_pos,np.array([])
    def calculate_metrics(self,pred: torch.Tensor, target: torch.Tensor, obs_last_pos: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate ADE and FDE for predictions
        Args:
            pred: predicted velocities [batch, seq_len, 2]
            target: target velocities [batch, seq_len, 2]
            obs_last_pos: last observed position [batch, 1, 2]
            mean: mean values for denormalization
            std: standard deviation values for denormalization
            device: computation device
        """
        
        # Convert velocities to absolute positions through cumsum
        pred_pos,target_pos = self.denormalize_to_absolute(pred, obs_last_pos, target)
        
        # Calculate metrics
        ade = calculate_ade(pred_pos, target_pos.tolist())
        fde = calculate_fde(pred_pos, target_pos.tolist())
        
        return ade, fde
    def calculate_all_metrics(self, pred: torch.Tensor, target: torch.Tensor, obs_last_pos: torch.Tensor,

                        batch_trajs: Optional[torch.Tensor] = None, 

                        batch_weights: Optional[torch.Tensor] = None) -> Tuple[float, float, float, float, float, float]:

        """

        Calculate ADE and FDE for original prediction and both weighted & unweighted best from k trajectories

        Args:

            pred: predicted velocities [batch, seq_len, 2]

            target: target velocities [batch, seq_len, 2]

            obs_last_pos: last observed position [batch, 1, 2]

            batch_trajs: optional k trajectories [batch, num_k, seq_len, 2]

            batch_weights: optional weights for k trajectories [batch, num_k]

        Returns:

            ade: ADE for original prediction

            fde: FDE for original prediction

            weighted_best_ade: Best weighted ADE from k trajectories

            weighted_best_fde: Best weighted FDE from k trajectories

            best_k_ade: Best ADE from k trajectories (unweighted)

            best_k_fde: Best FDE from k trajectories (unweighted)

        """

        if self.normalized:

            # Denormalize all velocities

            pred = pred * self.std[2:] + self.mean[2:]

            target = target * self.std[2:] + self.mean[2:]

            if batch_trajs is not None:
                std = self.std[2:].cpu().numpy()
                mean = self.mean[2:].cpu().numpy()
                batch_trajs = batch_trajs * std + mean

        
        # Convert to absolute positions
        pred_pos = pred.cpu().numpy().cumsum(1) + obs_last_pos.cpu().numpy()
        target_pos = target.cpu().numpy().cumsum(1) + obs_last_pos.cpu().numpy()

        

        # Calculate original metrics
        ade = calculate_ade(pred_pos, target_pos.tolist())
        fde = calculate_fde(pred_pos, target_pos.tolist())

        

        if batch_trajs is not None and batch_weights is not None:

            # Convert k trajectories to absolute positions [batch, num_k, seq_len, 2]
            k_trajs_pos = batch_trajs.cumsum(2) + obs_last_pos.cpu().numpy()[:, None]
            weights = batch_weights

            

            # Reshape to [batch*num_k, seq_len, 2]
            batch_size, num_k = k_trajs_pos.shape[:2]
            k_trajs_pos_flat = k_trajs_pos.reshape(batch_size * num_k, -1, 2)


            # Repeat target to match k trajectories [batch*num_k, seq_len, 2]
            target_pos_repeat = np.repeat(target_pos, num_k, axis=0)
            

            # Calculate per-sample metrics for all trajectories
            # (Using per_sample=True returns an array of errors, one per trajectory)
            k_ades = calculate_ade(k_trajs_pos_flat, target_pos_repeat.tolist(), per_sample=True)
            k_fdes = calculate_fde(k_trajs_pos_flat, target_pos_repeat.tolist(), per_sample=True)
    
            

            # Reshape back to per-batch metrics
            k_ades = k_ades.reshape(batch_size, num_k)
            k_fdes = k_fdes.reshape(batch_size, num_k)

            

            # Get weighted best (min per batch)
            weighted_best_ade_per_sample = (k_ades * weights).min(axis=1)
            weighted_best_fde_per_sample = (k_fdes * weights).min(axis=1)

            
            # Get unweighted best (min per batch)
            best_k_ade_per_sample = k_ades.min(axis=1)
            best_k_fde_per_sample = k_fdes.min(axis=1)
            
            # Now average these best metrics over the batch to obtain a single scalar
            weighted_best_ade = weighted_best_ade_per_sample.mean()
            weighted_best_fde = weighted_best_fde_per_sample.mean()
            best_k_ade = best_k_ade_per_sample.mean()
            best_k_fde = best_k_fde_per_sample.mean()


            return ade, fde, best_k_ade, best_k_fde,weighted_best_ade, weighted_best_fde

        

        return ade, fde, ade, fde, ade, fde
    def evaluate(self, test_loader=None, ckpt_path=None, from_train=False):
        """
        Evaluate the model on test data
        Args:
            test_loader: DataLoader for test data
            ckpt_path: Path to checkpoint file
            from_train: Boolean indicating if called during training
        """
        if test_loader is None:
            raise ValueError("test_loader cannot be None")
            
        # Store initial training mode
        training = self.training
        
        try:
            if not from_train:
                # Setup logger
               
                logger = logging.getLogger('AttentionGMM')
                if not logger.handlers:
                    logger = self.setup_logger(save_path=self.log_save_path,eval=True)    
                print("logger initialized") 
                logger.info(f"Loading Model")  
                self.load_model(self.checkpoint_file)
                logger.info(f"Model Loaded!")                                              
                        
           
            # Set evaluation mode :  Need to use nn.Module's train method for mode setting
            super().train(False)  
            self.tracker.test_available = True

            # logger.info(f"Starting evaluation on {len(test_loader)} batches")
            num_evaluated = 0
            load_test = tqdm(test_loader)

            preds,all_preds = [],[]
            weights,variances = [],[]
            obs,targets = [],[]
            means_mue,variance_sigma,mixture_weight = [],[],[]

            
            with torch.no_grad():
                for batch in load_test:
                    obs_tensor_eval, target_tensor_eval = batch
                    
                    # dimension check
                    assert obs_tensor_eval.shape[-1] == 4, "Expected input with 4 features (pos_x, pos_y, vel_x, vel_y)"
                    
                    obs_tensor_eval = obs_tensor_eval.to(self.device)
                    target_tensor_eval = target_tensor_eval.to(self.device)
                    dec_seq_len = target_tensor_eval.shape[1]

                    input_eval = (obs_tensor_eval[:,1:,2:4] - self.mean[2:])/self.std[2:]
                    updated_enq_length = input_eval.shape[1]
                    target_eval = (target_tensor_eval[:,:,2:4] - self.mean[2:])/self.std[2:]

                    tgt_eval = torch.zeros((target_eval.shape[0], dec_seq_len, 2), dtype=torch.float32, device=self.device)

                    tgt_mask = self._generate_square_mask(
                        dim_trg=dec_seq_len,
                        dim_src=updated_enq_length,
                        mask_type="tgt"
                    ).to(self.device)

                    pi_eval, sigma_x_eval,sigma_y_eval, mu_x_eval , mu_y_eval,rho = self(input_eval,tgt_eval,tgt_mask = tgt_mask)
                    mus_eval = torch.cat((mu_x_eval.unsqueeze(-1),mu_y_eval.unsqueeze(-1)),-1)
                    sigmas_eval = torch.cat((sigma_x_eval.unsqueeze(-1),sigma_y_eval.unsqueeze(-1)),-1)


                    # highest_prob_pred and best of n prediction -> best_of_n_pred is used to calculate the best of n from the GMM 
                    highest_prob_pred = self._sample_gmm_predictions(pi_eval, mus_eval)
                    batch_trajs,batch_weights,batch_variances,best_trajs,best_weights = self._process_trajectory_clusters(mus_eval,pi_eval,sigmas_eval,prediction_length=dec_seq_len) 
                    
                    

                    
                    # Calculate metrics
                    eval_loss = self._mdn_loss_fn(pi_eval, sigma_x_eval,sigma_y_eval, mu_x_eval , mu_y_eval,target_eval,self.num_gaussians,rho)
                    eval_obs_last_pos = obs_tensor_eval[:, -1:, 0:2]


                    pred_pos,target_pos,_ = self.denormalize_to_absolute(highest_prob_pred,eval_obs_last_pos,target_eval)
                    batch_pos,batch_target,denormalized_mue = self.denormalize_to_absolute(batch_trajs,eval_obs_last_pos,target_eval,mus_eval)

                    
                    


                    # Generate occupancy grid predictions
                    pre_permuted_mue = denormalized_mue.transpose(0, 2, 1, 3)
                  
                    occupancy_grid, grid_bounds, grid_points = self.occupancy_grid_prediction(pre_permuted_mue[0:1], sigmas_eval.cpu()[0:1], pi_eval.cpu()[0:1], grid_steps=100, padding_factor=4.0)


                    # self.diagnose_sigma_vs_mue_issue(pre_permuted_mue, sigmas_eval.cpu(), grid_bounds, grid_points)
                        
    

                    observed_traj = obs_tensor_eval[:, :,0:2]
                
                    preds.append(pred_pos)
                    targets.append(target_pos)
                    obs.append(observed_traj.cpu())
                    means_mue.append(denormalized_mue)
                    # Current shape: [128, 30, 6, 2]
                    # Need shape:    [128, 6, 30, 2]
                    sigmas_permuted = sigmas_eval.permute(0, 2, 1, 3)
                    # Need shape:    [128, 6, 30]
                    pie_permuted = pi_eval.permute(0, 2, 1)
                    # print(f"shape of batch_trajs{batch_trajs.shape} , mus_eval shape before :{mus_eval.shape}  mus_eval shape after {denormalized_mue.shape} sigmas_permuted shape :{sigmas_permuted.shape},pi_eval.shape : {pi_eval.shape}")

                    variance_sigma.append(sigmas_permuted.cpu())
                    mixture_weight.append(pie_permuted.cpu())

                    all_preds.append(batch_pos)
                    weights.append(batch_weights)
                    variances.append(batch_variances)

                    eval_ade, eval_fde, best_k_ade, best_k_fde,_,_ = self.calculate_all_metrics(highest_prob_pred, target_eval,eval_obs_last_pos,batch_trajs,batch_weights)
                    
                    batch_metrics = {
                                'loss': eval_loss.item(),
                                'ade': eval_ade,
                                'fde': eval_fde,
                                'best_ade': best_k_ade,
                                'best_fde': best_k_fde
                            }
                    num_evaluated += obs_tensor_eval.shape[0]
                    self.tracker.update(batch_metrics, obs_tensor_eval.shape[0], phase='test')
                    
                    # if hasattr(torch.cuda, 'empty_cache'):
                    #     torch.cuda.empty_cache()

            # logger.info(f"Completed evaluation of {num_evaluated} samples")
            self.tracker.compute_epoch_metrics(phase='test')
            # Print epoch metrics
            if not from_train:
                self.tracker.print_epoch_metrics(epoch=1, epochs=1, verbose=True)
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise        
        finally:
            # Restore original training mode
            super().train(training)
        
        if len(preds)>0:
            combined_pred = np.concatenate(preds, axis=0)
            combined_variances = np.concatenate(variances,axis=0)
            combined_targets = np.concatenate(targets,axis=0)
            combined_weights = np.concatenate(weights, axis=0)
            combined_all_preds =  np.concatenate(all_preds, axis=0)
            combined_obs = np.concatenate(obs, axis=0)
            combined_means_mue = np.concatenate(means_mue, axis=0)
            combined_variance_sigma = np.concatenate(variance_sigma, axis=0)
            combined_mixture_weights = np.concatenate(mixture_weight,axis=0)

        return combined_pred,combined_targets,combined_obs,combined_all_preds,combined_weights,combined_variances,combined_means_mue,combined_variance_sigma,combined_mixture_weights

    def predict(self, detection_msg, output_mode='grid',ego_location=(0,0),grid_steps=100, padding_factor=3.0):
        """
        Evaluate the model on live detections and generate predictions.

        Args:
            detection_msg: vision_msgs/Detection3DArray
            output_mode (str): Determines what type of output to return
                - 'single': Returns only the highest probability trajectory
                - 'multi': Returns multiple trajectories with their probabilities
                - 'gmm': Returns GMM parameters along with trajectories
                - 'grid': Returns occupancy grid predictions

        Returns:
            Based on output_mode:
            - 'single': highest_prob_pred, observed_traj
            - 'multi':  predicted trajectories, weights, observed_traj
            - 'gmm': μ , σ, π, observed_traj
            - 'grid': occupancy_grid, grid_bounds
        """
        if detection_msg is None:
            raise Warning("No detection_msg provided!")
        torch.cuda.synchronize()
        t0 = time.time()
        self.traj_handler.update(detection_msg)
        
        torch.cuda.synchronize()
        t1 = time.time()

        # print(f"Time taken to update trajectory handler: {t1 - t0:.4f} seconds")
        try:
            super().train(False)
            with torch.no_grad():
                for batch in self.traj_handler.data_iter(batch_size=32):
                    obs_tensor = batch

                    # Sanity check
                    if obs_tensor.shape[-1] != 4:
                        raise ValueError("Input must have 4 features (pos_x, pos_y, vel_x, vel_y)")

                    obs_tensor = obs_tensor.to(self.device)
                  
                    dec_seq_len = self.config.future_trajectory
                    input_norm = (obs_tensor[:, 1:, 2:] - self.mean[2:]) / self.std[2:]
        
                    tgt_zeros = torch.zeros((obs_tensor.shape[0], dec_seq_len, 2),
                                            dtype=torch.float32, device=self.device)

                    tgt_mask = self._generate_square_mask(
                        dim_trg=dec_seq_len,
                        dim_src=input_norm.shape[1],
                        mask_type="tgt"
                    ).to(self.device)
                    t2 = time.time()
                    pie, sigma_x, sigma_y, mu_x, mu_y,rho = self(input_norm, tgt_zeros, tgt_mask=tgt_mask)

                    t3 = time.time()
                
                    mus = torch.stack((mu_x, mu_y), dim=-1)
                    sigmas = torch.stack((sigma_x, sigma_y), dim=-1)

                    observed_traj = obs_tensor[:, :, 0:2]
                    pred_obs_last_pos = obs_tensor[:, -1:, 0:2]
                    

                    highest_prob_pred = self._sample_gmm_predictions(pie, mus)

                    # Return different outputs based on output_mode
                    if output_mode == 'single':
                        denormalized_pred,_,_ =  self.denormalize_to_absolute(highest_prob_pred,pred_obs_last_pos)
                        t3 = time.time()
                        print(f"Time taken to denormalize: {t3 - t2:.4f} seconds")
                        return denormalized_pred, observed_traj
                    
                    elif output_mode == 'multi':
                        trajs, weights, _, _, _ = self._process_trajectory_clusters(
                            mus, pie, sigmas, prediction_length=dec_seq_len)
                      
                        denormalized_pred,_,_ =  self.denormalize_to_absolute(trajs,pred_obs_last_pos)
                        
                        return denormalized_pred, weights, observed_traj
                    
                    elif output_mode == 'gmm':

                        _,_,denormalized_mue = self.denormalize_to_absolute(highest_prob_pred,pred_obs_last_pos,None,mus)
                    
                        # Current shape: [128, 30, 6, 2],  Need shape:    [128, 6, 30, 2]
                        sigmas_permuted = sigmas.permute(0, 2, 1, 3)
                        # Need shape:    [128, 6, 30]
                        pie_permuted = pie.permute(0, 2, 1)

                        if rho is not None:
                            rho_permuted = rho.permute(0, 2, 1)

                        return denormalized_mue, sigmas_permuted, pie_permuted, rho_permuted, observed_traj

                    elif output_mode == 'grid':
                        
                        _,_,denormalized_mue = self.denormalize_to_absolute(highest_prob_pred,pred_obs_last_pos,None,mus)
                    
                        # Current shape: sigma[128, 30, 6, 2],  Need shape:    [128, 6, 30, 2] Pie shape  [128, 30,6]
                        # Current shape: denormalized_mue [128, 6, 30, 2] ,  Need shape:    [128, 30, 6, 2]
                        permuted_mue = denormalized_mue.transpose(0, 2, 1, 3)

                        # Generate occupancy grid predictions
                        occupancy_grid, grid_bounds, grid_points = self.occupancy_grid_prediction(
                            permuted_mue, sigmas, pie, ego_location, grid_steps, padding_factor)
                        
                        # visualize_occupancy_grid_tkinter(
                        #     None,occupancy_grid, grid_bounds
                        # )

                        # torch.cuda.synchronize()
                        # t4 = time.time()
                        
                        # print(f'Time taken to update trajectory handler: {t1 - t0:.4f} seconds')
                        # print(f'Time taken for model forward pass: {t3 - t2:.4f} seconds')
                        # print(f"Time taken to generate occupancy grid: {t4 - t3:.4f} seconds")
                        # print(f"Total time taken for prediction: {t4 - t0:.4f} seconds")
                        # print(f"Occupancy grid shape: {occupancy_grid.shape}, Grid bounds: {grid_bounds}")

                        return occupancy_grid, grid_bounds
                    
         
                    else:
                        raise ValueError(f"Invalid output_mode: {output_mode}. Must be 'single', 'multi', or 'gmm'")
                    


        except Exception as e:
            print(f"[predict] Error during inference: {e}")
            raise
    
     
    def occupancy_grid_prediction(self, mue, sigma, pi, ego_location, grid_steps=100, padding_factor=2.0):
        """
        GPU-optimized occupancy grid prediction expecting tensors.
        
        Args:
            mue: Mean vectors, shape (num_objects, num_timesteps, num_components, 2) - 
            sigma: Diagonal variances, shape (num_objects, num_timesteps, num_components, 2) 
            pi: Mixture weights, shape (num_objects, num_timesteps, num_components)
            grid_steps: Number of grid cells per dimension
            padding_factor: How many standard deviations to extend beyond means
        
        Returns:
            occupancy_grid: Shape (num_timesteps, grid_steps, grid_steps)
            grid_bounds: Dictionary with grid coordinate bounds
            grid_points: All grid coordinates, shape (grid_steps^2, 2)
        """
        
        # Convert NumPy arrays to PyTorch tensors and move to GPU immediately
        if isinstance(mue, np.ndarray):
            mue = torch.from_numpy(mue).float().to(self.device)
        else:
            mue = mue.to(self.device)
            
        if isinstance(sigma, np.ndarray):
            sigma = torch.from_numpy(sigma).float().to(self.device)
        else:
            sigma = sigma.to(self.device)
            
        if isinstance(pi, np.ndarray):
            pi = torch.from_numpy(pi).float().to(self.device)
        else:
            pi = pi.to(self.device)

        # Extract dimensions from pre-transposed tensors
        num_objects, num_timesteps, num_components = mue.shape[:3]
        
        ego_tensor = torch.tensor(ego_location, dtype=torch.float32, device=self.device)
        
        # 1. COMPUTE BOUNDING BOX - ALL ON GPU
        all_means = mue.reshape(-1, 2)  # Flatten all dimensions except last
        all_sigmas = sigma.reshape(-1, 2)
        
        std_devs = torch.sqrt(all_sigmas)
        padding_tensor = torch.tensor(padding_factor, dtype=torch.float32, device=self.device)
        
        min_bounds = (all_means - padding_tensor * std_devs).min(dim=0)[0]
        max_bounds = (all_means + padding_tensor * std_devs).max(dim=0)[0]
        
        # Ensure ego location is also inside the bounds
        min_bounds = torch.min(min_bounds, ego_tensor)
        max_bounds = torch.max(max_bounds, ego_tensor)
        
        min_x, min_y = min_bounds.tolist()
        max_x, max_y = max_bounds.tolist()
        
        # 2. CREATE SPATIAL GRID - ON GPU
        grid_x = torch.linspace(min_x, max_x, steps=grid_steps, device=self.device)
        grid_y = torch.linspace(min_y, max_y, steps=grid_steps, device=self.device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid_points = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        
        # 3. Reshape to combine objects and timesteps for vectorized processing
        mue_reshaped = mue.reshape(num_objects * num_timesteps, num_components, 2)
        sigma_reshaped = sigma.reshape(num_objects * num_timesteps, num_components, 2)
        pi_reshaped = pi.reshape(num_objects * num_timesteps, num_components)
        
        # Get combined occupancy for all object-timestep combinations - ON GPU
        all_occupancies = self.evaluate_all_gmms_vectorized_gpu(
            grid_points, mue_reshaped, sigma_reshaped, pi_reshaped, num_objects, num_timesteps
        )
        
        # Reshape back to separate objects and timesteps
        all_occupancies = all_occupancies.reshape(num_objects, num_timesteps, -1)
        
        # Combine objects for each timestep using vectorized operations - ON GPU
        combined_occupancies = self.combine_objects_vectorized_gpu(all_occupancies)
        
        # Reshape to grid format
        occupancy_grid = combined_occupancies.reshape(num_timesteps, grid_steps, grid_steps)
        
        grid_bounds = {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'resolution': (max_x - min_x) / (grid_steps - 1)
        }
        
        # move to CPU/NumPy 
        occupancy_grid_np = occupancy_grid.detach().cpu().numpy()
        grid_points_np = grid_points.detach().cpu().numpy()
        
        return occupancy_grid_np, grid_bounds, grid_points_np


    def evaluate_all_gmms_vectorized_gpu(self, grid_points, mue_all, sigma_all, pi_all, num_objects, num_timesteps):
        """
        GPU-optimized vectorized evaluation of multiple GMMs with per-object normalization.
        
        Args:
            grid_points: Grid coordinates, shape (n_points, 2) - ON GPU
            mue_all: Means, shape (num_gmms, num_components, 2) - ON GPU
            sigma_all: Variances, shape (num_gmms, num_components, 2) - ON GPU
            pi_all: Weights, shape (num_gmms, num_components) - ON GPU
            num_objects: Number of objects 
            num_timesteps: Number of timesteps
        
        Returns:
            occupancies: Shape (num_gmms, n_points) - ON GPU
        """
        num_gmms, num_components, _ = mue_all.shape
        n_points = grid_points.shape[0]
        
        # Initialize output tensor on GPU
        all_densities = torch.zeros(num_gmms, n_points, device=self.device)
        
        # Process all GMMs and components simultaneously
        for comp_idx in range(num_components):
            means = mue_all[:, comp_idx, :]  # (num_gmms, 2)
            variances = sigma_all[:, comp_idx, :]  # (num_gmms, 2)
            weights = pi_all[:, comp_idx]  # (num_gmms,)
            
            # Skip components with negligible weights
            valid_mask = weights >= 1e-6
            if not valid_mask.any():
                continue
                
            valid_means = means[valid_mask]
            valid_variances = variances[valid_mask]
            valid_weights = weights[valid_mask]
            
            # GPU-vectorized PDF computation
            densities = self.diagonal_gaussian_pdf_batch_gpu(
                grid_points, valid_means, valid_variances
            )
            
            # Weight the densities
            weighted_densities = valid_weights.unsqueeze(1) * densities
            
            # Add to the corresponding GMMs
            valid_indices = torch.where(valid_mask)[0]
            all_densities[valid_indices] += weighted_densities
        
        # UPDATED: Convert densities to occupancies with PER-OBJECT normalization
        return self.density_to_occupancy_per_object_gpu(all_densities, num_objects, num_timesteps)


    def density_to_occupancy_per_object_gpu(self, densities, num_objects, num_timesteps, method='normalized'):
        """
        GPU-optimized per-object normalization of densities to occupancies.
        
        Args:
            densities: Shape (num_gmms, n_points) where num_gmms = num_objects * num_timesteps
            num_objects: Number of objects
            num_timesteps: Number of timesteps
            method: Normalization method ('normalized', 'sigmoid_per_object', 'linear_per_object')
        
        Returns:
            occupancies: Shape (num_gmms, n_points) - normalized per object
        """
        num_gmms, n_points = densities.shape
        
        # Reshape to separate objects and timesteps: (num_objects, num_timesteps, n_points)
        densities_reshaped = densities.reshape(num_objects, num_timesteps, n_points)
        
        if method == 'normalized':
            # Find max density per object across all timesteps and points
            # Shape: (num_objects, 1, 1) for broadcasting
            max_densities_per_object = densities_reshaped.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            
            # Normalize each object by its own maximum density
            occupancies_reshaped = torch.where(
                max_densities_per_object > 1e-10,  # Avoid division by zero
                densities_reshaped / max_densities_per_object,
                torch.zeros_like(densities_reshaped)
            )
            
        elif method == 'sigmoid':
            # Scale factor adapted per object based on its density range
            std_per_object = densities_reshaped.std(dim=(1,2), keepdim=True)
            scale_factor = 1.0 / (std_per_object + 1e-6)  # Adaptive scaling
            occupancies_reshaped = torch.sigmoid(scale_factor * densities_reshaped)
            
        elif method == 'linear':
            # Linear scaling per object to [0,1] range
            min_per_object = densities_reshaped.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
            max_per_object = densities_reshaped.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            range_per_object = max_per_object - min_per_object
            
            occupancies_reshaped = torch.where(
                range_per_object > 1e-10,
                (densities_reshaped - min_per_object) / range_per_object,
                torch.zeros_like(densities_reshaped)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reshape back to (num_gmms, n_points)
        occupancies = occupancies_reshaped.reshape(num_gmms, n_points)
        
        return occupancies


    def diagonal_gaussian_pdf_batch_gpu(self, points, means, variances):
        """
        GPU-optimized batch computation of diagonal Gaussian PDFs.
        
        Args:
            points: Shape (n_points, 2) - ON GPU
            means: Shape (num_gaussians, 2) - ON GPU
            variances: Shape (num_gaussians, 2) - ON GPU
        
        Returns:
            pdf_values: Shape (num_gaussians, n_points) - ON GPU
        """
        # Add regularization
        reg_variances = variances + 1e-6
        
        # Compute differences using GPU broadcasting
        # (n_points, 2) -> (1, n_points, 2) and (num_gaussians, 2) -> (num_gaussians, 1, 2)
        diff = points.unsqueeze(0) - means.unsqueeze(1)  # (num_gaussians, n_points, 2)
        
        # Mahalanobis distance for diagonal case - all on GPU
        mahal_dist = ((diff ** 2) / reg_variances.unsqueeze(1)).sum(dim=2)
        
        # Determinant of diagonal matrix
        det = reg_variances.prod(dim=1, keepdim=True)  # (num_gaussians, 1)
        
        # PDF computation - all GPU operations
        normalization = 1.0 / (2 * torch.pi * torch.sqrt(det))
        pdf = normalization * torch.exp(-0.5 * mahal_dist)
        
        return pdf


    def combine_objects_vectorized_gpu(self, all_occupancies):
        """
        GPU-optimized vectorized combination of object occupancies.
        
        Args:
            all_occupancies: Shape (num_objects, num_timesteps, n_points) - ON GPU
        
        Returns:
            combined_occupancies: Shape (num_timesteps, n_points) - ON GPU
        """
        # Use the independence assumption: P(occupied) = 1 - ∏(1 - P_i)
        # All operations stay on GPU
        prob_not_occupied = 1 - all_occupancies
        prob_not_occupied_any = prob_not_occupied.prod(dim=0)  # Product across objects
        combined_occupancy = 1 - prob_not_occupied_any
        
        return combined_occupancy
           
    def _process_trajectory_clusters(self, means, weights, variances, weight_threshold=0.01, prediction_length=10):
        """
        Process trajectories from input data.

        Parameters:
            means: The input trajectory data.
            weights: The weights for the trajectories.
            variances: The variance values for individual points in the trajectories.
            weight_threshold: Threshold value for selecting valid centroids.
            prediction_length: The length of the prediction.

        Returns:
            all_trajectories (numpy.ndarray): The full list of trajectories with shape (batch_size, trajs, traj_len, 2).
            all_trajectory_weights (numpy.ndarray): The weights for each trajectory.
            best_trajectories (numpy.ndarray): The best trajectories selected based on weights with shape (batch_size, traj_len, 2).
            best_trajectory_weights (numpy.ndarray): The weights for the best trajectories.
            all_trajectory_variances (numpy.ndarray): The variances for individual points in each trajectory.
        """
        batch_weights = weights.detach().cpu().numpy()
        batch_means = means.detach().cpu().numpy()
        batch_variances = variances.detach().cpu().numpy()
        
        all_trajectories = []
        best_trajectories = []
        all_trajectory_weights = []
        best_trajectory_weights = []
        all_trajectory_variances = []

        for sequence_means, sequence_weights, sequence_variances in zip(batch_means, batch_weights, batch_variances):
            root = TreeNode([0, 0], 0, [0, 0], False, 0)
            
            for level, (timestep_means, timestep_weights, timestep_variances) in enumerate(
                zip(sequence_means, sequence_weights, sequence_variances), start=1):
                
                # Filter centroids based on weight threshold
                valid_indices = timestep_weights > weight_threshold
                valid_centroids = timestep_means[valid_indices]
                valid_weights = timestep_weights[valid_indices].reshape(-1, 1)
                valid_variances = timestep_variances[valid_indices]
                
                max_weight_index = np.argmax(valid_weights)
                
                for i, centroid in enumerate(valid_centroids):
                    is_max_weight = (i == max_weight_index)
                    root.add_child(
                        TreeNode(centroid, valid_weights[i], valid_variances[i],is_max_weight, level),
                        level - 1
                    )
                    
                if level == prediction_length:
                    # Get trajectories and their weights
                    possible_trajectories, possible_weights, possible_variances = root.get_exact_k_trajectories(
                        self.best_of_k, prediction_length)
                    
                    # Normalize weights
                    normalized_weights = possible_weights / np.sum(possible_weights)
                    
                    all_trajectories.append(possible_trajectories)
                    all_trajectory_variances.append(possible_variances)
                    all_trajectory_weights.append(normalized_weights)
                    
                    # Select best trajectory based on highest weight
                    best_idx = np.argmax(normalized_weights)
                    best_trajectories.append(np.array(possible_trajectories[best_idx]))
                    best_trajectory_weights.append(np.array(normalized_weights[best_idx]))
                    
        return np.array(all_trajectories), np.array(all_trajectory_weights), np.array(all_trajectory_variances), best_trajectories, best_trajectory_weights
    def _generate_square_mask(
        self,
        dim_trg: int,
        dim_src: int,
        mask_type: str = "tgt"
    ) -> torch.Tensor:
        """
        Generate a square mask for transformer attention mechanisms.
        
        Args:
            dim_trg (int): Target sequence length.
            dim_src (int): Source sequence length.
            mask_type (str): Type of mask to generate. Can be "src", "tgt", or "memory".
        
        Returns:
            torch.Tensor: A mask tensor with `-inf` values to block specific positions.
        """
        # Initialize a square matrix filled with -inf (default to a fully masked state)
        mask = torch.ones(dim_trg, dim_trg) * float('-inf')

        if mask_type == "src":
            # Source mask (self-attention in the encoder)
            # Creates an upper triangular matrix with -inf above the diagonal
            mask = torch.triu(mask, diagonal=1)

        elif mask_type == "tgt":
            # Target mask (self-attention in the decoder)
            # Prevents the decoder from attending to future tokens
            mask = torch.triu(mask, diagonal=1)

        elif mask_type == "memory":
            # Memory mask (cross-attention between encoder and decoder)
            # Controls which encoder outputs the decoder can attend to
            mask = torch.ones(dim_trg, dim_src) * float('-inf')
            mask = torch.triu(mask, diagonal=1)  # Prevents attending to future positions

        return mask
    
    def setup_logger(self,name: str = 'AttentionGMM', save_path: str = None, level=logging.INFO,eval=False):
        """Set up logger configuration.
        
        Args:
            name (str): Logger name
            save_path (str): Directory to save log file
            level: Logging level
            
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter('%(message)s')
        
        # Stream handler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(simple_formatter)
        logger.addHandler(stream_handler)
        
        # File handler if save_path is provided
        if save_path:
            if eval: 
                log_path = Path(save_path) / f'transformerGMM_eval_metrics_model_{self.past_trajectory}_{self.future_trajectory}_W_{self.config.win_size}.log'
            else:
                 log_path = Path(save_path) / f'transformerGMM_training_metrics_model_{self.past_trajectory}_{self.future_trajectory}_training_{self.n_warmup_steps}_W_{self.config.win_size}_lr_mul_{self.lr_mul}.log'
            
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path),mode='w')
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def plot_metrics(self,
        train_losses: List[float],
        test_losses: List[float],
        train_ades: List[float],
        test_ades: List[float],
        train_fdes: List[float],
        test_fdes: List[float],
        train_best_ades: List[float],
        test_best_ades: List[float],
        train_best_fdes: List[float],
        test_best_fdes: List[float],
        enc_seq_len: int,
        dec_seq_len: int,
    ) -> None:
        """Plot training metrics including best-of-N predictions.
        
        Args:
            train_losses: Training loss values
            test_losses: Test loss values
            train_ades: Training ADE values
            test_ades: Test ADE values
            train_fdes: Training FDE values
            test_fdes: Test FDE values
            train_best_ades: Training Best-of-N ADE values
            test_best_ades: Test Best-of-N ADE values
            train_best_fdes: Training Best-of-N FDE values
            test_best_fdes: Test Best-of-N FDE values
            enc_seq_len: Encoder sequence length
            dec_seq_len: Decoder sequence length
            save_path: Path to save the plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(test_losses, label='Test Loss', color='orange')
        ax1.set_title('Loss', pad=20)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ADE plot
        ax2.plot(train_ades, label='Train ADE', color='blue')
        ax2.plot(test_ades, label='Test ADE', color='orange')
        ax2.plot(train_best_ades, label='Train Best ADE', color='blue', linestyle='--')
        ax2.plot(test_best_ades, label='Test Best ADE', color='orange', linestyle='--')
        ax2.set_title('Average Displacement Error (ADE)', pad=20)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('ADE Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # FDE plot
        ax3.plot(train_fdes, label='Train FDE', color='blue')
        ax3.plot(test_fdes, label='Test FDE', color='orange')
        ax3.plot(train_best_fdes, label='Train Best FDE', color='blue', linestyle='--')
        ax3.plot(test_best_fdes, label='Test Best FDE', color='orange', linestyle='--')
        ax3.set_title('Final Displacement Error (FDE)', pad=20)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('FDE Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        os.makedirs(self.log_save_path, exist_ok=True)
        save_file = os.path.join(self.log_save_path, f'training_metrics_model_{enc_seq_len}_{dec_seq_len}_W_{self.config.win_size}.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()

class Linear_Embeddings(nn.Module):
    def __init__(self, input_features,d_model):
        super(Linear_Embeddings, self).__init__()
        self.lut = nn.Linear(input_features, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000, batch_first: bool=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if batch_first: 
            pe = torch.zeros(1,max_len, d_model)
            pe[0,:, 0::2] = torch.sin(position * div_term)
            pe[0,:, 1::2] = torch.cos(position * div_term)
        else: 
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] 
            x: Tensor, shape [batch_size, seq_len, embedding_dim]batch first
        """
        #print("pe[:,:x.size(1),:] shape: ",self.pe.shape)
        x = x + self.pe[:,:x.size(1),:] if self.batch_first else x + self.pe[:x.size(0)]

        return self.dropout(x)

class MetricTracker:
    def __init__(self):
        self.train_available = False
        self.test_available = False

        # Separate running metrics for train and test
        self.running_metrics = {
            'train': self._init_metric_dict(),
            'test': self._init_metric_dict()
        }
        
        self.history = {
            'train_loss': [], 'test_loss': [],
            'train_ade': [], 'test_ade': [],
            'train_fde': [], 'test_fde': [],
            'train_best_ade': [], 'test_best_ade': [],
            'train_best_fde': [], 'test_best_fde': []
        }

        self.best_metrics = {
            'ade': float('inf'),
            'fde': float('inf'),
            'best_ade': float('inf'),
            'best_fde': float('inf')
        }

    def _init_metric_dict(self):
        """Helper to initialize metrics dictionary."""
        return {key: {'value': 0, 'count': 0} for key in ['loss', 'ade', 'fde', 'best_ade', 'best_fde']}
    
    def update(self, metrics_dict, batch_size, phase='train'):
        """Update running metrics with batch results"""
        for key, value in metrics_dict.items():
            self.running_metrics[phase][key]['value'] += value * batch_size
            self.running_metrics[phase][key]['count'] += batch_size

    def get_averages(self, phase='train'):
        """Compute averages for specified phase."""
        if phase not in self.running_metrics:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train' or 'test'.")

        return {
            key: (metric['value'] / metric['count'] if metric['count'] > 0 else 0)
            for key, metric in self.running_metrics[phase].items()
        }

    def compute_epoch_metrics(self, phase='train'):
        """Compute and store metrics for completed epoch."""
        epoch_metrics = self.get_averages(phase)
        
        # Store epoch averages in history
        self.history[f'{phase}_loss'].append(epoch_metrics['loss'])
        self.history[f'{phase}_ade'].append(epoch_metrics['ade'])
        self.history[f'{phase}_fde'].append(epoch_metrics['fde'])
        self.history[f'{phase}_best_ade'].append(epoch_metrics['best_ade'])
        self.history[f'{phase}_best_fde'].append(epoch_metrics['best_fde'])

        # Reset running metrics for next epoch
        self.running_metrics[phase] = self._init_metric_dict()
        
        return epoch_metrics

    def get_current_epoch_metrics(self, phase='train'):
        """Get most recent epoch metrics."""
        if not self.history[f'{phase}_loss']:  # if history is empty
            return None
            
        return {
            'loss': self.history[f'{phase}_loss'][-1],
            'ade': self.history[f'{phase}_ade'][-1],
            'fde': self.history[f'{phase}_fde'][-1],
            'best_ade': self.history[f'{phase}_best_ade'][-1],
            'best_fde': self.history[f'{phase}_best_fde'][-1]
        }

    def get_previous_epoch_metrics(self, phase='train'):
        """Get previous epoch metrics."""
        if len(self.history[f'{phase}_loss']) < 2:  # need at least 2 epochs
            return None
            
        return {
            'loss': self.history[f'{phase}_loss'][-2],
            'ade': self.history[f'{phase}_ade'][-2],
            'fde': self.history[f'{phase}_fde'][-2],
            'best_ade': self.history[f'{phase}_best_ade'][-2],
            'best_fde': self.history[f'{phase}_best_fde'][-2]
        }
    def print_epoch_metrics(self, epoch, epochs, verbose=True):
        """Print epoch metrics including best-of-N results in a side-by-side format."""
        if not verbose:
            return

        logger = logging.getLogger('AttentionGMM')
        
        # Get current metrics from history
        train_metrics = self.get_current_epoch_metrics('train')
        test_metrics = self.get_current_epoch_metrics('test') if self.test_available else None

        # Get previous metrics for improvements
        train_prev = self.get_previous_epoch_metrics('train')
        test_prev = self.get_previous_epoch_metrics('test') if self.test_available else None

        # Header
        logger.info(f"\nEpoch [{epoch+1}/{epochs}]")
        logger.info("-" * 100)
        logger.info(f"{'Metric':12} {'Training':35} {'Validation':35}")
        logger.info("-" * 100)

        # Print metrics side by side
        for metric, name in [('loss', 'Loss'), ('ade', 'ADE'), ('fde', 'FDE'),
                            ('best_ade', 'Best ADE'), ('best_fde', 'Best FDE')]:
            train_str = "N/A"
            val_str = "N/A"

            if train_metrics:
                train_val = train_metrics[metric]
                train_str = f"{train_val:.4f}"
                if train_prev:
                    train_imp = train_prev[metric] - train_val
                    arrow = "↓" if train_imp > 0 else "↑"
                    train_str += f" ({arrow} {abs(train_imp):.4f})"
                    # train_str += f" (↓ {train_imp:.4f})"

            if test_metrics:
                val_val = test_metrics[metric]
                val_str = f"{val_val:.4f}"
                if test_prev:
                    val_imp = test_prev[metric] - val_val
                    arrow = "↓" if val_imp > 0 else "↑"
                    val_str += f" ({arrow} {abs(val_imp):.4f})" #f" (↓ {val_imp:.4f})"

            logger.info(f"{name:12} {train_str:35} {val_str:35}")

        logger.info("-" * 100)
    def reset(self, phase='train'):
        """Reset running metrics for specified phase."""
        self.running_metrics[phase] = self._init_metric_dict()



class TreeNode:
    def __init__(self, value: List[float], weight: float,sigma: List[float], maximum: bool, level: int = 0):
        self.value = value
        self.weight = weight
        self.sigma = sigma
        self.children = []
        self.level = level
        self.is_max = maximum
        self.max_connected = False


    def add_child(self, child, prev_level, multi_connection=False):
        """
        Add a child node to the tree structure.
        
        Parameters:
            child: The node to be added
            prev_level: Target level to connect to
            multi_connection: If True and child is max, it will be connected to both 
                            the closest node and any max probability nodes
        """
        # Direct connection if at the target level
        if self.level == prev_level:
            self.children.append(child)
            return

        # Initialize tracking variables
        min_distance = math.inf
        closest_node = None
        max_connected = False  # Flag to track if connected to a max node
        stack = [self]

        # Depth-first search through the tree
        while stack:
            node = stack.pop()
            if node.level == prev_level:
                # Find and track closest node by distance
                distance = math.dist(node.value, child.value)
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node

                # Connect max nodes (special priority connection)
                if node.is_max and child.is_max:
                    node.children.append(child)
                    max_connected = True
                    node.max_connected = True
            
            # Add children to stack to continue traversal
            elif node.children:
                stack.extend(node.children)

        # Connect to closest node if:
        # 1. Not already connected to a max node, OR
        # 2. Multi-connection is enabled and closest node isn't already max-connected
        if not max_connected or (multi_connection and not closest_node.max_connected):
            closest_node.children.append(child)

    def get_exact_k_trajectories(self, k: int = 5, depth: int = 10):
        """
        Get exactly k trajectories based on weights, repeating top trajectories if needed
        Args:
            k: number of trajectories to return
            depth: trajectory depth required for valid branches
        Returns:
            trajectories: np.array of shape [k, depth, 2]
            weights: np.array of shape [k]
        """
        branches, weights, sigmas = self.get_all_branches()
        
        # Filter for full branches
        # Filter valid branches (those with at least `depth` waypoints)
        valid_indices = [i for i, branch in enumerate(branches) if len(branch) >= depth]
        branches = np.array([branches[i][:depth] for i in valid_indices])
        weights  =  np.array([weights[i] for i in valid_indices]).flatten()
        sigmas = np.array([sigmas[i][:depth] for i in valid_indices])
        
        # Sort all by weights in descending order
        sorted_indices = np.argsort(-weights)
        sorted_branches = branches[sorted_indices]
        sorted_weights = weights[sorted_indices]
        sorted_sigmas = sigmas[sorted_indices]

        # Repeat trajectories if we have fewer than k
        if len(weights) < k:
            repeat_times = k // len(weights) + 1
            sorted_branches = np.tile(sorted_branches, (repeat_times, 1, 1))
            sorted_sigmas = np.tile(sorted_sigmas, (repeat_times, 1, 1))
            sorted_weights = np.tile(sorted_weights, repeat_times)

        # Return exactly k trajectories
        return sorted_branches[:k], sorted_weights[:k],sorted_sigmas[:k]
    
    def get_trajectories(self,depth = 10,full=True):
        branches,weights = self.get_all_branches()
        full_branches = [np.array(branch) for branch in branches if len(branch) >= depth]
        full_weights = [weights[i] for i in range(len(weights)) if len(branches[i]) >= depth]
        return full_branches,full_weights
        
    def get_all_branches(self):
        """Recursively finds all branches of the tree."""
        if not self.children:
            return [[self.value]], [self.weight], [[self.sigma]]
        branches = []
        weights = []
        sigmas = []
        for child in self.children:
                child_branches, child_weights,child_sigmas = child.get_all_branches()
                for branch, weight,sigma in zip(child_branches, child_weights,child_sigmas):
                    if self.level == 0:
                        branches.append(branch) 
                        weights.append(weight) 
                        sigmas.append(sigma)
                    else:
                        branches.append([self.value.tolist()] + branch)
                        sigmas.append([self.sigma.tolist()] + sigma)
                        weights.append(self.weight + weight)
        return branches,weights,sigmas  
    

