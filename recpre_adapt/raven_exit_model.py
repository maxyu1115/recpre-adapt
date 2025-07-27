import abc
import math
from typing import Callable, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers.cache_utils import Cache
from transformers import GenerationConfig


from recpre.raven_modeling_minimal import CausalLMOutputRecurrentLatents, HuginnDynamicCache, RavenConfig, RavenForCausalLM, RavenGenerateDecoderOnlyOutput, SandwichBlock, precompute_freqs_cis

from recpre_adapt.autoencoder import Autoencoder
from recpre_adapt.utils import generate_causal_mask

class LatentDiffExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, prev_latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass

class LatentDiffEmbeddingExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, input_embeddings: torch.Tensor, prev_latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass

class LatentRecurrentExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def iterate_forward(self, latent: torch.Tensor, recurrent_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def initialize_recurrent(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(latent)
        # return latent

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        recurrent_input = self.initialize_recurrent(latents[:, :, 0, :])
        exit_probs = torch.zeros(latents.shape[0], latents.shape[1], 1, 2, device=latents.device)
        for i in range(latents.shape[2] - 1):
            exit_prob, recurrent_input = self.iterate_forward(latents[:, :, i, :], recurrent_input)
            exit_probs = torch.cat([exit_probs, exit_prob.unsqueeze(2)], dim=2)
        return exit_probs

class LatentTransformerExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, latents: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

class TransformerExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        pass

# RavenLatentTransformerExitModel + input_embeddings
class LTEExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, input_embeddings: torch.Tensor, latents: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class FeatureRecurrentExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def iterate_forward(self, input_embeddings: torch.Tensor, feature_vecs: torch.Tensor, recurrent_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def forward(self, input_embeddings: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        pass

class ProbsRecurrentExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def iterate_forward(self, probs: torch.Tensor, recurrent_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def initialize_recurrent(self, probs: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, probs: list[torch.Tensor]) -> torch.Tensor:
        # probs = [F.softmax(logit, dim=-1) for logit in logits]
        recurrent_input = self.initialize_recurrent(probs[0])
        exit_probs = torch.zeros(probs[0].shape[0], probs[0].shape[1], 1, 2, device=probs[0].device)
        for i in range(len(probs) - 1):
            # print(f"probs[{i}].shape: {probs[i].shape}, recurrent_input.shape: {recurrent_input.shape}")
            exit_prob, recurrent_input = self.iterate_forward(probs[i], recurrent_input)
            exit_probs = torch.cat([exit_probs, exit_prob.unsqueeze(2)], dim=2)
        return exit_probs


class GatedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, output_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2
        if output_dim is None:
            output_dim = input_dim
        self.fc = nn.Linear(input_dim, hidden_dim * 2)
        self.nonlin = nn.SiLU()
        self.proj = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        x_fc_1, x_fc_2 = self.fc(x).chunk(2, dim=-1)
        x = self.nonlin(x_fc_1) * x_fc_2
        return self.proj(x)


# class RavenExitModel(nn.Module):
#     def __init__(self, config: RavenConfig):
#         super().__init__()
#         self.ln = nn.Linear(config.n_embd * 2, config.n_embd)
#         self.norm1 = nn.LayerNorm(config.n_embd)
#         self.attn = nn.MultiheadAttention(config.n_embd, config.n_heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(config.n_embd)
#         self.mlp = GatedMLP(config)
#         self.norm3 = nn.LayerNorm(config.n_embd)
#         self.out = nn.Linear(config.n_embd, 2)

#         # Initialize weights
#         nn.init.xavier_uniform_(self.ln.weight)
#         nn.init.zeros_(self.ln.bias)
#         nn.init.xavier_uniform_(self.out.weight)
#         nn.init.zeros_(self.out.bias)
#         nn.init.xavier_normal_(self.attn.in_proj_weight)
#         nn.init.zeros_(self.attn.in_proj_bias)
#         nn.init.xavier_uniform_(self.mlp.fc.weight)
#         nn.init.zeros_(self.mlp.fc.bias)
#         nn.init.xavier_uniform_(self.mlp.proj.weight)
#         nn.init.zeros_(self.mlp.proj.bias)

#     def forward(self, embeddings: torch.Tensor, prev_x: torch.Tensor, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
#         x = self.ln(torch.cat([x.to(self.ln.weight.dtype), prev_x.to(self.ln.weight.dtype)], dim=-1))
#         x = self.norm1(x)
#         attn_output, attn_weights = self.attn(x, embeddings, embeddings, is_causal=(attn_mask is not None), attn_mask=attn_mask)
#         x = self.norm2(x + attn_output)
#         x = self.norm3(x + self.mlp(x))
#         x = self.out(x)
#         return F.softmax(x.float(), dim=-1)


class RavenExitModel(LTEExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()
        self.ln = nn.Linear(config.n_embd * 2, config.n_embd)
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp = GatedMLP(config.n_embd)
        self.norm3 = nn.LayerNorm(config.n_embd)
        self.out = nn.Linear(config.n_embd, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.xavier_normal_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)

    def forward(self, input_embeds: torch.Tensor, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        assert len(input_embeds.shape) == 2
        assert len(x.shape) == 3
        
        x = x.to(self.ln.weight.dtype)
        input_embeds = input_embeds.to(self.ln.weight.dtype)
        input_embeds_expanded = input_embeds.unsqueeze(1).expand(-1, x.shape[1], -1)
        # concatenate x and input_embeds
        x = self.ln(torch.cat([x, input_embeds_expanded], dim=-1))
        x = self.norm1(x)
        attn_output, attn_weights = self.attn(x, x, x, is_causal=(attn_mask is not None), attn_mask=attn_mask)
        x = self.norm2(x + attn_output)
        x = self.norm3(x + self.mlp(x))
        x = self.out(x)
        return F.softmax(x.float(), dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = GatedMLP(n_embd)

        # Initialize weights
        nn.init.xavier_normal_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_output, attn_weights = self.attn(x, x, x, is_causal=(attn_mask is not None), attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))
        return x

class RavenLatentTransformerExitModel(LatentTransformerExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp = GatedMLP(config.n_embd)
        self.out = nn.Linear(config.n_embd, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.xavier_normal_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)

    def forward(self, latents: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        assert len(latents.shape) == 3
        x = latents.to(self.out.weight.dtype)
        attn_output, attn_weights = self.attn(x, x, x, is_causal=(attn_mask is not None), attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))
        x = self.out(x)
        return F.softmax(x.float(), dim=-1)

class RavenLatentTransformerExitModel2(LatentTransformerExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp = GatedMLP(config.n_embd)
        self.out = nn.Linear(config.n_embd, 2)
        
        # Add positional embeddings
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        # Initialize weights
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.xavier_normal_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        # Initialize positional embeddings
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, latents: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        assert len(latents.shape) == 3
        batch_size, seq_len, n_embd = latents.shape
        
        x = latents.to(self.out.weight.dtype)
        
        # Add positional embeddings
        pos_indices = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos_embeddings = self.pos_emb(pos_indices)  # [seq_len, n_embd]
        x = x + pos_embeddings.unsqueeze(0)  # Broadcast to [batch_size, seq_len, n_embd]
        
        attn_output, attn_weights = self.attn(x, x, x, is_causal=(attn_mask is not None), attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))
        x = self.out(x)
        return F.softmax(x.float(), dim=-1)


class RavenAutoencoderLTExitModel(LatentTransformerExitModel):
    def __init__(self, autoencoder: Autoencoder):
        super().__init__()
        self.autoencoder = autoencoder
        self.latent_dim = autoencoder.hidden_dim
        
        self.transformer_block1 = TransformerBlock(self.latent_dim, self.latent_dim // 128)
        self.transformer_block2 = TransformerBlock(self.latent_dim, self.latent_dim // 128)
        self.out = nn.Linear(self.latent_dim, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, latents: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        assert len(latents.shape) == 3
        with torch.no_grad():
            encoded_latents = self.autoencoder.encode(latents)
        x = self.transformer_block1(encoded_latents, attn_mask)
        x = self.transformer_block2(x, attn_mask)
        x = self.out(x)
        return F.softmax(x.float(), dim=-1)


class RavenLatentRecurrentExitModel(LatentRecurrentExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()        
        self.mlp = GatedMLP(config.n_embd * 2, output_dim=config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd * 2)
        self.out = nn.Linear(config.n_embd, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def iterate_forward(self, latent: torch.Tensor, recurrent_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([latent, recurrent_input], dim=-1)
        new_recurrent = recurrent_input + self.mlp(self.norm(x))
        logits = self.out(new_recurrent)
        return F.softmax(logits.float(), dim=-1), new_recurrent


class RavenLatentRecurrentExitModel2(LatentRecurrentExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()
        self.ln = nn.Linear(config.n_embd * 2, config.n_embd)
        self.mlp = GatedMLP(config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd)
        self.out = nn.Linear(config.n_embd, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def iterate_forward(self, latent: torch.Tensor, recurrent_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.ln(torch.cat([latent, recurrent_input], dim=-1))
        new_recurrent = self.norm(x + self.mlp(x))
        logits = self.out(new_recurrent)
        return F.softmax(logits.float(), dim=-1), new_recurrent


class RavenAutoencoderRecurrentExitModel(LatentRecurrentExitModel):
    def __init__(self, autoencoder: Autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

        self.recurrent_dim = autoencoder.hidden_dim
        # self.recurrent_dim = autoencoder.hidden_dim // 3
        self.mlp = GatedMLP(autoencoder.hidden_dim + self.recurrent_dim, output_dim=self.recurrent_dim)
        self.norm = nn.LayerNorm(autoencoder.hidden_dim + self.recurrent_dim)
        self.out = nn.Linear(self.recurrent_dim, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def iterate_forward(self, encoded_latent: torch.Tensor, recurrent_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([encoded_latent, recurrent_input], dim=-1)
        new_recurrent = recurrent_input + self.mlp(self.norm(x))
        logits = self.out(new_recurrent)
        return F.softmax(logits.float(), dim=-1), new_recurrent

    def initialize_recurrent(self, encoded_latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(encoded_latent.shape[0], encoded_latent.shape[1], self.recurrent_dim, device=encoded_latent.device, dtype=encoded_latent.dtype)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded_latents = self.autoencoder.encode(latents)

        recurrent_input = self.initialize_recurrent(encoded_latents[:, :, 0, :])
        exit_probs = torch.zeros(latents.shape[0], latents.shape[1], 1, 2, device=latents.device)
        for i in range(latents.shape[2] - 1):
            exit_prob, recurrent_input = self.iterate_forward(encoded_latents[:, :, i, :], recurrent_input)
            exit_probs = torch.cat([exit_probs, exit_prob.unsqueeze(2)], dim=2)
        return exit_probs

class RavenFinalProbExitModel(nn.Module):
    def __init__(self, config: RavenConfig):
        super().__init__()
        self.ln = nn.Linear(config.vocab_size * 2, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd)
        self.out = nn.Linear(config.n_embd, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, prev_probs: torch.Tensor, x: torch.Tensor):
        x = self.ln(torch.cat([x.to(self.ln.weight.dtype), prev_probs.to(self.ln.weight.dtype)], dim=-1))
        x = F.silu(x)
        x = self.norm(x)
        x = self.out(x)
        return F.softmax(x.float(), dim=-1)


class RavenLatentExitModel(LatentDiffExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()
        self.ln = nn.Linear(config.n_embd * 2, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd)
        self.mlp = GatedMLP(config.n_embd)
        self.out = nn.Linear(config.n_embd, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # torch.nn.init.trunc_normal_(self.ln.weight, mean=0.0, std=1.0)
        # torch.nn.init.trunc_normal_(self.out.weight, mean=0.0, std=1.0)

    def forward(self, prev_latents: torch.Tensor, x: torch.Tensor):
        x = self.ln(torch.cat([x.to(self.ln.weight.dtype), prev_latents.to(self.ln.weight.dtype)], dim=-1))
        x = self.norm(x + self.mlp(x))
        x = self.out(x)
        return F.softmax(x.float(), dim=-1)

class RavenLatentEmbeddingExitModel(LatentDiffEmbeddingExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()
        self.ln = nn.Linear(config.n_embd * 2, config.n_embd)
        self.nonlin = nn.SiLU()
        self.ln2 = nn.Linear(config.n_embd * 2, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd)
        self.mlp = GatedMLP(config.n_embd)
        self.out = nn.Linear(config.n_embd, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.xavier_uniform_(self.ln2.weight)
        nn.init.zeros_(self.ln2.bias)
        # torch.nn.init.trunc_normal_(self.ln.weight, mean=0.0, std=1.0)
        # torch.nn.init.trunc_normal_(self.out.weight, mean=0.0, std=1.0)

    def forward(self, input_embeddings: torch.Tensor, prev_latents: torch.Tensor, x: torch.Tensor):
        x = self.ln(torch.cat([x.to(self.ln.weight.dtype), prev_latents.to(self.ln.weight.dtype)], dim=-1))
        x = self.nonlin(x)
        x = self.ln2(torch.cat([x, input_embeddings.to(self.ln.weight.dtype)], dim=-1))
        x = self.norm(x + self.mlp(x))
        x = self.out(x)
        return F.softmax(x.float(), dim=-1)


class RavenBasicLatentExitModel(LatentDiffExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()
        self.ln = nn.Linear(config.n_embd * 2, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd)
        self.out = nn.Linear(config.n_embd, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # torch.nn.init.trunc_normal_(self.ln.weight, mean=0.0, std=1.0)
        # torch.nn.init.trunc_normal_(self.out.weight, mean=0.0, std=1.0)

    def forward(self, prev_x: torch.Tensor, x: torch.Tensor):
        x = self.ln(torch.cat([x.to(self.ln.weight.dtype), prev_x.to(self.ln.weight.dtype)], dim=-1))
        x = F.silu(x)
        x = self.norm(x)
        x = self.out(x)
        return F.softmax(x.float(), dim=-1)


class RavenFeatureRecurrentExitModel(FeatureRecurrentExitModel):
    def __init__(self, config: RavenConfig, feature_count: int):
        super().__init__()
        self.feature_count = feature_count
        self.mlp = GatedMLP((config.n_embd + self.feature_count) * 2, output_dim=config.n_embd + self.feature_count)
        self.norm = nn.LayerNorm((config.n_embd + self.feature_count) * 2)
        self.out = nn.Linear(config.n_embd + self.feature_count, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def iterate_forward(self, input_embeddings: torch.Tensor, feature_vecs: torch.Tensor, recurrent_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([input_embeddings, feature_vecs, recurrent_input], dim=-1)
        new_recurrent = recurrent_input + self.mlp(self.norm(x))
        logits = self.out(new_recurrent)
        return F.softmax(logits.float(), dim=-1), new_recurrent

    def forward(self, input_embeddings: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # input_embeddings: [batch_size, seq_len, emb_dim]
        # features: [batch_size, seq_len, num_loops, feature_dim]
        assert len(input_embeddings.shape) == 3
        assert len(features.shape) == 4
        input_embeddings = input_embeddings.to(self.out.weight.dtype)

        recurrent_input = torch.zeros(input_embeddings.shape[0], input_embeddings.shape[1], input_embeddings.shape[2] + self.feature_count, device=input_embeddings.device, dtype=self.out.weight.dtype)
        exit_probs = torch.zeros(input_embeddings.shape[0], input_embeddings.shape[1], 1, 2, device=input_embeddings.device, dtype=self.out.weight.dtype)
        for i in range(features.shape[2]):
            exit_prob, recurrent_input = self.iterate_forward(input_embeddings, features[:, :, i, :], recurrent_input)
            exit_probs = torch.cat([exit_probs, exit_prob.unsqueeze(2)], dim=2)
        return exit_probs


# class RavenProbsRecurrentExitModel(ProbsRecurrentExitModel):
#     def __init__(self, config: RavenConfig):
#         super().__init__()
#         self.recurrent_dim = config.n_embd
#         self.ln = nn.Linear(config.vocab_size + self.recurrent_dim, self.recurrent_dim)
#         self.mlp = GatedMLP(self.recurrent_dim)
#         self.norm = nn.LayerNorm(self.recurrent_dim)
#         self.out = nn.Linear(self.recurrent_dim, 2)

#         # Initialize weights
#         nn.init.xavier_uniform_(self.ln.weight)
#         nn.init.zeros_(self.ln.bias)
#         nn.init.xavier_uniform_(self.out.weight)
#         nn.init.zeros_(self.out.bias)

#     def initialize_recurrent(self, probs: torch.Tensor) -> torch.Tensor:
#         return torch.zeros(probs.shape[0], probs.shape[1], self.recurrent_dim, device=probs.device, dtype=probs.dtype)

#     def iterate_forward(self, probs: torch.Tensor, recurrent_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = self.ln(torch.cat([probs, recurrent_input], dim=-1))
#         new_recurrent = self.norm(x + self.mlp(x))
#         logits = self.out(new_recurrent)
#         return F.softmax(logits.float(), dim=-1), new_recurrent


def make_block_sum_matrix(m: int, n: int,
                          dtype: torch.dtype = torch.float32,
                          device=None) -> torch.Tensor:
    """
    Returns a tensor P of shape (n, m) such that, for any x in R^m,
        y = P @ x
    is in R^n and
        y[i] = sum_{j: floor(i*m/n) <= j < floor((i+1)*m/n)} x[j].
    Blocks may differ in size by at most one if m/n is not integer.
    """
    # Allocate zero matrix
    P = torch.zeros((n, m), dtype=dtype, device=device)
    # Compute block boundaries
    for i in range(n):
        start = int((i    ) * m / n)
        end   = int((i + 1) * m / n)
        P[i, start:end] = 1.0
    return P


def make_modulo_sum_matrix(m: int,
                           n: int,
                           dtype: torch.dtype = torch.float32,
                           device=None) -> torch.Tensor:
    """
    Returns P of shape (n, m) such that for any x in R^m,
        y = P @ x
    is in R^n and
        y[i] = sum_{j: j % n == i} x[j].
    """
    # 1) allocate
    P = torch.zeros((n, m), dtype=dtype, device=device)
    # 2) compute for each column j which row it belongs to
    cols = torch.arange(m, device=device)
    rows = cols % n
    # 3) set those positions to 1
    P[rows, cols] = 1.0
    return P

class RavenProbsRecurrentExitModel(ProbsRecurrentExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()
        self.recurrent_dim = 4096
        # self.ln = nn.Linear(config.vocab_size, self.recurrent_dim)
        self.mlp = GatedMLP(self.recurrent_dim * 2, output_dim=self.recurrent_dim)
        self.norm = nn.LayerNorm(self.recurrent_dim)
        self.out = nn.Linear(self.recurrent_dim, 2)

        # Initialize weights
        # nn.init.xavier_uniform_(self.ln.weight)
        # self.ln.weight = torch.nn.Parameter(make_block_sum_matrix(config.vocab_size, self.recurrent_dim, dtype=self.ln.weight.dtype, device=self.ln.weight.device))
        # self.ln.weight = torch.nn.Parameter(make_modulo_sum_matrix(config.vocab_size, self.recurrent_dim, dtype=self.ln.weight.dtype, device=self.ln.weight.device))
        # nn.init.zeros_(self.ln.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def initialize_recurrent(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(probs.shape[0], probs.shape[1], self.recurrent_dim, device=probs.device, dtype=probs.dtype)

    def iterate_forward(self, probs: torch.Tensor, recurrent_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x = self.ln(probs)
        x = probs
        x = self.norm(x + self.mlp(torch.cat([recurrent_input, x], dim=-1)))
        logits = self.out(x)
        return F.softmax(logits.float(), dim=-1), x


class RavenTransformerExitModel(TransformerExitModel):
    def __init__(self, config: RavenConfig):
        super().__init__()
        # self.seq_tf = TransformerBlock(config.n_embd, config.n_heads)
        self.seq_tf = SandwichBlock(config, 0)
        self.steps_tf = TransformerBlock(config.n_embd, config.n_heads)
        self.out = GatedMLP(config.n_embd * 2, output_dim=2)
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(config), persistent=True)

    def _precompute_freqs_cis(self, config: RavenConfig):
        # can actually be a buffer now, and remains in fp32! (at least in the settings I tested)
        freqs_cis = precompute_freqs_cis(
            config.n_embd // config.num_attention_heads, config.block_size, config.rope_base, 1
        )
        return freqs_cis

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [batch_size, seq_len, n_steps, n_embd]
        # latent_seq: [batch_size * n_steps, seq_len, n_embd]
        x_seq = latents.permute(0, 2, 1, 3).contiguous().reshape(-1, latents.shape[1], latents.shape[3])
        # seq_mask = generate_causal_mask(latents.shape[1], latents.device)
        x_seq, _ = self.seq_tf(x_seq, self.freqs_cis[:, : x_seq.shape[1]], 0)
        x_seq = x_seq.reshape(latents.shape[0], latents.shape[2], latents.shape[1], latents.shape[3]).permute(0, 2, 1, 3).contiguous()
        # latent_steps: [batch_size * seq_len, n_steps, n_embd]
        steps_mask = generate_causal_mask(latents.shape[2], latents.device)
        x_steps = self.steps_tf(latents.reshape(-1, latents.shape[2], latents.shape[3]), steps_mask)
        x_steps = x_steps.reshape(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])

        x = self.out(torch.cat([x_seq, x_steps], dim=-1))
        return F.softmax(x.float(), dim=-1)
