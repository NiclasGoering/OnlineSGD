import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class DeepNN(nn.Module):
    """
    Modified DeepNN class that can store activations for load factor calculation.
    Uses HardTanh activation instead of ReLU.
    """
    def __init__(
        self,
        d: int,             # input dimension
        hidden_size: int,   # model width n
        depth: int,
        mode: str = 'standard',
        alignment: bool = False,  # Parameter to control alignment assumption
        base_width: int = 1024,   # Base width for LR scaling
        embed_lr_scale: float = 1.0,
        hidden_lr_scale: float = 1.0,
        readout_lr_scale: float = 1.0,
        gamma: float = 1.0,
        store_activations: bool = False  # Flag to enable activation storage
    ):
        super().__init__()
        
        self.mode = mode
        self.depth = depth
        self.hidden_size = hidden_size
        self.input_dim = d
        self.base_width = base_width
        self.alignment = alignment
    
        # Whether to store activations during forward pass (for load factor calculation)
        self.store_activations = store_activations
        self.hidden_activations = None
        
        # Per-layer LR scales multipliers (user-configurable)
        self.embed_lr_scale = embed_lr_scale
        self.hidden_lr_scale = hidden_lr_scale
        self.readout_lr_scale = readout_lr_scale

        # Factor for forward pass
        self.gamma = gamma

        # We'll store the "relative" LR scale for each Linear in this list
        self.layer_lrs: List[float] = []
        
        # Store all layers, parameter multipliers, and configurations
        self.layers = nn.ModuleList()
        self.param_multipliers = []
        
        prev_dim = d
        
        # Build embedding + hidden layers
        for layer_idx in range(depth):
            # Create linear layer with bias=False
            linear = nn.Linear(prev_dim, hidden_size, bias=False)
            is_embedding = (layer_idx == 0)
            
            # Step 1: Configure initialization variance and parameter multiplier
            if mode.startswith('standard'):
                # Standard parameterization
                if is_embedding:
                    # Embedding layer - initialization variance ~ 1
                    init_std = 1.0 / np.sqrt(prev_dim)  #normally  init_std = 1.0
                    param_multiplier = 1.0  # param multiplier = 1
                else:
                    # Hidden layers - initialization variance ~ 1/n
                    init_std = 1.0 / np.sqrt(hidden_size)
                    param_multiplier = 1.0  # param multiplier = 1
                
            elif mode.startswith('ntk'):
                # NTK parameterization
                if is_embedding:
                    # Embedding layer - initialization variance ~ 1
                    init_std = 1.0
                    param_multiplier = 1.0  # param multiplier = 1
                else:
                    # Hidden layers - initialization variance ~ 1
                    init_std = 1.0
                    param_multiplier = 1.0 / np.sqrt(hidden_size)  # param multiplier = 1/√n
            
            elif mode.startswith('mup'):
                # muP parameterization
                if is_embedding:
                    # Embedding layer - initialization variance ~ 1/n
                    init_std = 1.0 / np.sqrt(hidden_size)
                    param_multiplier = np.sqrt(hidden_size)  # param multiplier = √n
                else:
                    # Hidden layers - initialization variance ~ 1/n
                    init_std = 1.0 / np.sqrt(hidden_size)
                    param_multiplier = 1.0  # param multiplier = 1
                    
            elif mode.startswith('mfp'):
                # Mean Field parameterization
                if is_embedding:
                    # Embedding layer - initialization variance ~ 1
                    init_std = 1.0
                    param_multiplier = 1.0  # param multiplier = 1
                else:
                    # Hidden layers - initialization variance ~ 1
                    init_std = 1.0
                    param_multiplier = 1.0 / np.sqrt(hidden_size)  # param multiplier = 1/√n
            
            # Apply initialization with correct variance
            with torch.no_grad():
                linear.weight.data = torch.randn(hidden_size, prev_dim) * init_std
            
            # Step 2: Determine learning rate scaling
            # If it's a *_lr mode, set learning rate scale to 1.0 (no scaling)
            if mode.endswith('_lr'):
                lr_scale = 1.0
            else:
                # Standard parameterization
                if mode.startswith('standard'):
                    if is_embedding:
                        # Embedding layer: SGD scaling with √n
                        lr_scale = self.embed_lr_scale * (hidden_size / self.base_width) ** 0.5  # LR ~ √n
                    else:
                        # Hidden layer scaling depends on alignment
                        if alignment:
                            # Full alignment: Scale by (base_width/width)^0.5
                            lr_scale = self.hidden_lr_scale * (self.base_width / hidden_size) ** 0.5  # LR ~ 1/√n
                        else:
                            # No alignment: No scaling with width
                            lr_scale = self.hidden_lr_scale  # LR ~ 1
                
                # NTK parameterization
                elif mode.startswith('ntk'):
                    if is_embedding:
                        # Embedding layer: SGD scaling with √n
                        lr_scale = self.embed_lr_scale * (hidden_size / self.base_width) ** 0.5  # LR ~ √n
                    else:
                        # Hidden layer scaling depends on alignment
                        if alignment:
                            # Full alignment: Scale by (width/base_width)^0.5
                            lr_scale = self.hidden_lr_scale * (hidden_size / self.base_width) ** 0.5  # LR ~ √n
                        else:
                            # No alignment: Scale by width/base_width
                            lr_scale = self.hidden_lr_scale * (hidden_size / self.base_width)  # LR ~ n
                
                # muP parameterization
                elif mode.startswith('mup'):
                    if is_embedding:
                        # Embedding layer: No scaling with width
                        lr_scale = self.embed_lr_scale  # LR ~ 1
                    else:
                        # Hidden layer scaling depends on alignment
                        if alignment:
                            # Full alignment: No scaling with width
                            lr_scale = self.hidden_lr_scale  # LR ~ 1
                        else:
                            # No alignment: Scale by (width/base_width)^0.5
                            lr_scale = self.hidden_lr_scale * (hidden_size / self.base_width) ** 0.5  # LR ~ √n
                
                # Mean Field parameterization
                elif mode.startswith('mfp'):
                    if is_embedding:
                        # Embedding layer: Scale with width
                        lr_scale = self.embed_lr_scale * (hidden_size / self.base_width)  # LR ~ n
                    else:
                        # Hidden layer scaling depends on alignment
                        if alignment:
                            # Full alignment: Scale by width/base_width
                            lr_scale = self.hidden_lr_scale * (hidden_size / self.base_width)  # LR ~ n
                        else:
                            # No alignment: Scale by (width/base_width)^1.5
                            lr_scale = self.hidden_lr_scale * (hidden_size / self.base_width) ** 1.5  # LR ~ n^1.5
            
            self.layer_lrs.append(lr_scale)
            self.param_multipliers.append(param_multiplier)
            self.layers.append(linear)
            # Replace ReLU with HardTanh
            self.layers.append(nn.ELU())
            prev_dim = hidden_size
        
        # Build readout layer with bias=False^1
        final_layer = nn.Linear(prev_dim, 1, bias=False)
        
        # Configure readout layer based on mode
        if mode.startswith('standard'):
            # Standard readout - initialization variance ~ 1/n
            init_std = 1.0 / np.sqrt(hidden_size)
            param_multiplier = 1.0  # param multiplier = 1
            
        elif mode.startswith('ntk'):
            # NTK readout - initialization variance ~ 1
            init_std = 1.0
            param_multiplier = 1.0 / np.sqrt(hidden_size)  # param multiplier = 1/√n
            
        elif mode.startswith('mup'):
            # muP readout - initialization variance ~ 1/n
            init_std = 1.0 / np.sqrt(hidden_size)
            param_multiplier = 1.0 / np.sqrt(hidden_size)  # param multiplier = 1/√n
            
        elif mode.startswith('mfp'):
            # Mean Field readout - initialization variance ~ 1
            init_std = 1.0
            param_multiplier = 1.0 / hidden_size  # param multiplier = 1/n
        
        # Apply initialization with correct variance
        with torch.no_grad():
            final_layer.weight.data = torch.randn(1, hidden_size) * init_std
        
        # Determine learning rate scaling for readout layer
        if mode.endswith('_lr'):
            lr_scale = 1.0
        else:
            # Standard parameterization
            if mode.startswith('standard'):
                if alignment:
                    # Full alignment: Scale by (base_width/width)
                    lr_scale = self.readout_lr_scale * (self.base_width / hidden_size)  # LR ~ 1/n
                else:
                    # No alignment: Scale by (base_width/width)^0.5
                    lr_scale = self.readout_lr_scale * (self.base_width / hidden_size) ** 0.5  # LR ~ 1/√n
            
            # NTK parameterization
            elif mode.startswith('ntk'):
                if alignment:
                    # Full alignment: No scaling with width
                    lr_scale = self.readout_lr_scale  # LR ~ 1
                else:
                    # No alignment: Scale by (width/base_width)^0.5
                    lr_scale = self.readout_lr_scale * (hidden_size / self.base_width) ** 0.5  # LR ~ √n
            
            # muP parameterization
            elif mode.startswith('mup'):
                if alignment:
                    # Full alignment: No scaling with width
                    lr_scale = self.readout_lr_scale  # LR ~ 1
                else:
                    # No alignment: No scaling with width
                    lr_scale = self.readout_lr_scale  # LR ~ 1
                    
            # Mean Field parameterization
            elif mode.startswith('mfp'):
                if alignment:
                    # Full alignment: Scale by width/base_width
                    lr_scale = self.readout_lr_scale * (hidden_size / self.base_width)  # LR ~ n
                else:
                    # No alignment: Scale by width/base_width
                    lr_scale = self.readout_lr_scale * (hidden_size / self.base_width)  # LR ~ n
        
        self.layer_lrs.append(lr_scale)
        self.param_multipliers.append(param_multiplier)
        self.layers.append(final_layer)
        
        # Convert param_multipliers to a tensor - it will be moved to the correct device in forward()
        self.register_buffer('param_multipliers_tensor', torch.tensor(self.param_multipliers, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with parameter multipliers applied correctly.
        Option to store activations for load factor calculation.
        """
        # Ensure param_multipliers is on the same device as input x
        param_multipliers = self.param_multipliers_tensor.to(x.device)
        
        # Initialize activations storage if needed
        if self.store_activations:
            self.hidden_activations = []
        
        linear_idx = 0
        # Store the original input if we're collecting activations
        if self.store_activations:
            self.hidden_activations.append(x)
            
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # Calculate weight multiplication
                weight_output = F.linear(x, layer.weight)
                
                # Apply parameter multiplier to the result of weight multiplication
                weight_output = weight_output * param_multipliers[linear_idx]
                
                # Add bias (which isn't shown in the paper's formulation)
                if layer.bias is not None:
                    x = weight_output + layer.bias
                else:
                    x = weight_output
                    
                linear_idx += 1
            else:
                # For non-linear layers (now HardTanh), just apply normally
                x = layer(x)
            
            # Store activations after each layer if needed
            if self.store_activations:
                self.hidden_activations.append(x)
        
        # Return result divided by gamma
        return x.squeeze() / self.gamma

    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        """
        Returns the per-layer effective LR = base_lr * (relative scale in self.layer_lrs).
        """
        return [base_lr * lr for lr in self.layer_lrs]
        
    def enable_activation_storage(self):
        """Enable storing activations during forward pass"""
        self.store_activations = True
        
    def disable_activation_storage(self):
        """Disable storing activations during forward pass"""
        self.store_activations = False
        self.hidden_activations = None