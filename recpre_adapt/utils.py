from typing import Literal
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from recpre.raven_modeling_minimal import RavenForCausalLM, CausalSelfAttention, SandwichBlock


def update_huggingface_implementation(model):
    """This function selectively updates function implementations in the huggingface model."""
    import types
    model.iterate_forward = types.MethodType(RavenForCausalLM.iterate_forward, model)
    for name, module in model.named_modules():
        if module.__class__.__name__ == "CausalSelfAttention":
            module.forward = types.MethodType(CausalSelfAttention.forward, module)
        elif module.__class__.__name__ == "SandwichBlock":
            module.forward = types.MethodType(SandwichBlock.forward, module)


def generate_causal_mask(seq_len: int, device=None):
    """
    Generate a causal attention mask compatible with transformer models.
    Returns a mask where 1 means don't attend and 0 means attend.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.to(device) if device is not None else mask
    return mask


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, decay_type: Literal["cosine", "linear", "step", "none"] = "cosine"):
    """
    Create a learning rate scheduler with warmup and decay.
    
    Args:
        optimizer: The optimizer to modify learning rates for
        warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        decay_type: Type of decay after warmup ("cosine", "linear", or "step")
    
    Returns:
        A learning rate scheduler
    """
    def lr_lambda(epoch):
        # Linear warmup phase
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        
        # Decay phase
        if decay_type == "cosine":
            # Cosine decay from 1 to 0 over remaining epochs
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        elif decay_type == "linear":
            # Linear decay from 1 to 0.1 over remaining epochs
            decay_ratio = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 1.0 - 0.9 * min(1, decay_ratio)
        elif decay_type == "step":
            # Step decay by 0.1 every 1/3 of remaining epochs
            decay_step = (total_epochs - warmup_epochs) / 3
            return 0.1 ** (int((epoch - warmup_epochs) / decay_step))
        else:
            # No decay
            return 1.0
            
    return LambdaLR(optimizer, lr_lambda)
