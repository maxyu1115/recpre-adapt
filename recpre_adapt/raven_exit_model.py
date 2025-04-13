import abc
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers.cache_utils import Cache
from transformers import GenerationConfig


from recpre.raven_modeling_minimal import HuginnDynamicCache, RavenConfig, RavenForCausalLM, RavenGenerateDecoderOnlyOutput


class LatentDiffExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, prev_latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass

class LatentDiffEmbeddingExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, input_embeddings: torch.Tensor, prev_latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass

class LatentTransformerExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, latents: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

# RavenLatentTransformerExitModel + input_embeddings
class LTEExitModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, input_embeddings: torch.Tensor, latents: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


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


class RavenAdaptiveModel(RavenForCausalLM):
    def __init__(self, base_config: RavenConfig, save_latents: bool = False):
        super().__init__(base_config)
        self.exit_model: Union[LatentDiffExitModel, LTEExitModel, LatentTransformerExitModel] = RavenExitModel(base_config)
        self.save_latents = save_latents
        self.latents = None

    @staticmethod
    def from_models(model: RavenForCausalLM, exit_model: Union[LatentDiffExitModel, LTEExitModel, LatentTransformerExitModel]):
        config = model.config
        new_model = RavenAdaptiveModel(config)
        new_model.transformer = model.transformer
        new_model.emb_scale = model.emb_scale
        new_model.lm_head = model.lm_head

        new_model.exit_model = exit_model
        return new_model

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def iterate_forward(
        self,
        input_embeds,
        input_states,
        freqs_cis,
        block_idx,
        mask,
        past_key_values: Optional[Cache] = None,
        num_steps: Optional[torch.Tensor] = None,
        attn_maps: dict = {},
        return_attn: bool = False,
    ):
        x = xk = self.initialize_state(input_embeds) if input_states is None else input_states.clone()
        if num_steps is None:
            raise NotImplementedError("Doesn't support num_steps is None")
        elif hasattr(num_steps, "__len__") and len(num_steps) > 1:
            num_steps_no_grad, num_steps_with_grad = num_steps
        else:
            num_steps_no_grad, num_steps_with_grad = num_steps, torch.tensor(0) if not x.is_meta else 0

        if self.save_latents:
            latents = [x.clone().detach()]

        for step in range(num_steps_no_grad + num_steps_with_grad):
            xk = x
            x, block_idx, attn_maps = self.core_block_forward(
                xk, input_embeds, freqs_cis, mask, past_key_values, block_idx, attn_maps, return_attn
            )
            if self.save_latents:
                latents.append(x.clone().detach())
            # self.exit_model(x, xk)

        if self.save_latents:
            self.latents = latents

        return self.transformer.ln_f(x), num_steps_no_grad, num_steps_with_grad, xk.detach(), block_idx, attn_maps


    @torch.no_grad()
    def generate_minimal(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        num_steps: int,
    ) -> Union[torch.Tensor, RavenGenerateDecoderOnlyOutput]:
        """Minimal single-sequence generation. Template for more complicated generate tasks"""
        gen_length = max_length - input_ids.shape[1]

        # Generate tokens
        for _ in range(gen_length):
            # Forward pass
            outputs = self.forward(input_ids, num_steps=torch.tensor((num_steps,)))
            next_token_logits = outputs.logits[:, -1, :] # type: ignore

            # Sample or select next token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)  # type: ignore

        return input_ids

    @torch.no_grad()
    def generate_with_adaptive_compute(
        self,
        input_ids: torch.LongTensor,
        generation_config: Optional[GenerationConfig] = None,  # type: ignore
        tokenizer=None,
        streamer=None,
        continuous_compute=False,  # warm-start state / continuous CoT
        latent_dampening=False,
        criterion="entropy-diff",
        exit_threshold: Union[str, float, int] = "auto",
        cache_kwargs: dict = {},
        **model_kwargs,
    ) -> Union[torch.Tensor, RavenGenerateDecoderOnlyOutput]:
        if criterion != "auto" and criterion != "adaptive":
            return super().generate_with_adaptive_compute(
                input_ids, generation_config, tokenizer, streamer, continuous_compute, latent_dampening, criterion, exit_threshold, cache_kwargs, **model_kwargs
            )
        
        if continuous_compute or latent_dampening:
            raise NotImplementedError("Doesn't support continuous compute or latent dampening")
        
        if exit_threshold != "auto":
            raise NotImplementedError("Doesn't support exit threshold")
        
        if generation_config is None:
            generation_config: GenerationConfig = self.generation_config  # type: ignore
        model_kwargs["past_key_values"] = HuginnDynamicCache(**cache_kwargs)
        model_kwargs["use_cache"] = True
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        stop_tokens = self._get_stops(generation_config, tokenizer).to(input_ids.device)
        batch_size = input_ids.shape[0]
        compute_steps = []
        seq_steps = [0] * batch_size
        initial_seq_len = input_ids.shape[1]
        avg_compute_steps = None

        # Track which sequences have finished
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Generate tokens
        for step in range(generation_config.max_length - initial_seq_len):
            # Adaptive compute forward
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            aux_inputs = {
                k: model_inputs[k] for k in ["cache_position", "past_key_values", "attention_mask"] if k in model_inputs
            }
            embedded_inputs, block_idx, _ = self.embed_inputs(model_inputs["input_ids"], **aux_inputs)
            current_latents = self.initialize_state(embedded_inputs, deterministic=not generation_config.do_sample)

            # Initialize criterion tracking for each sequence in batch
            exit_values_per_seq = [[] for _ in range(batch_size)]
            compute_steps_per_seq = [0] * batch_size
            exit_reached = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

            all_latents = current_latents.clone().unsqueeze(2)
            next_token_logits = None

            # Iterate through compute steps
            for compute_step in range(model_inputs["num_steps"]):
                prev_latents = current_latents.clone()
                current_latents, block_idx, _ = self.iterate_one_step(
                    embedded_inputs, current_latents, block_idx=block_idx, **aux_inputs
                )
                all_latents = torch.cat([all_latents, current_latents.unsqueeze(2)], dim=2)

                if step > 0 and compute_step > 3:  # do not exit in prefill:
                    # Check exit condition for each sequence in batch
                    if isinstance(self.exit_model, LatentDiffExitModel):
                        exit_policy = self.exit_model.forward(all_latents[:, :, -2, :], all_latents[:, :, -1, :])
                    elif isinstance(self.exit_model, LTEExitModel):
                        exit_policy = self.exit_model(embedded_inputs[:, -1, :], all_latents.flatten(start_dim=0, end_dim=1))
                        exit_policy = exit_policy.unflatten(dim=0, sizes=(batch_size, -1))
                        exit_policy = exit_policy[:, :, -1, :]
                    elif isinstance(self.exit_model, LatentTransformerExitModel):
                        exit_policy = self.exit_model.forward(all_latents.flatten(start_dim=0, end_dim=1))
                        exit_policy = exit_policy.unflatten(dim=0, sizes=(batch_size, -1))
                        exit_policy = exit_policy[:, :, -1, :]

                    if generation_config.do_sample:
                        exit_actions = torch.distributions.Categorical(exit_policy).sample()
                        new_exits = exit_actions[:, -1] == 0
                    else:
                        new_exits = exit_policy[:, :, 0] > 0.5
                    new_exits = new_exits & ~exit_reached & ~finished_sequences

                    if new_exits.any():
                        exit_reached = exit_reached | new_exits
                        outputs = self.predict_from_latents(current_latents, **aux_inputs)
                        logits: torch.Tensor = outputs.logits  # type: ignore
                        if next_token_logits is None:
                            next_token_logits = logits[:, -1, :].clone()
                        else:
                            next_token_logits = torch.where(
                                new_exits.unsqueeze(1).expand_as(logits[:, -1, :]), logits[:, -1, :], next_token_logits
                            )
                        for i in range(batch_size):
                            if new_exits[i].item():
                                compute_steps_per_seq[i] = compute_step + 1

                    # If all sequences have exited, break early
                    if (exit_reached | finished_sequences).all():
                        break
            # This else is if the for loop finished without breaking
            else:
                outputs = self.predict_from_latents(current_latents, **aux_inputs)

                # For sequences that didn't exit early, use the final logits
                if next_token_logits is None:
                    # If no sequence exited early
                    next_token_logits = outputs.logits[:, -1, :]  # type: ignore
                    for i in range(batch_size):
                        compute_steps_per_seq[i] = model_inputs["num_steps"]
                else:
                    # Only update logits for sequences that didn't exit early
                    non_exit_mask = ~exit_reached & ~finished_sequences
                    next_token_logits = torch.where(
                        non_exit_mask.unsqueeze(1).expand_as(next_token_logits),
                        outputs.logits[:, -1, :],  # type: ignore
                        next_token_logits,
                    )

                    # Record compute steps for non-exited sequences
                    for i in range(batch_size):
                        if non_exit_mask[i].item():
                            compute_steps_per_seq[i] = model_inputs["num_steps"]

            # Record compute steps for this token generation
            compute_steps.append((compute_steps_per_seq, exit_values_per_seq))

            # Sample or select next token based on generation config
            if generation_config.do_sample:
                next_token = self._sample_next_token(
                    next_token_logits,
                    generation_config.temperature,
                    generation_config.top_k,
                    generation_config.top_p,
                    generation_config.min_p,
                )
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # type: ignore

            # print(next_token)

            input_ids = torch.cat([input_ids, next_token], dim=-1)  # type: ignore

            if streamer:
                streamer.put(next_token.cpu())

            # Update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)

            # Check for finished sequences
            for i in range(batch_size):
                if not finished_sequences[i] and stop_tokens is not None and next_token[i, 0] in stop_tokens:
                    finished_sequences[i] = True
                    seq_steps[i] = step + 1

            # Break if all sequences are finished
            if finished_sequences.all():
                break

        if streamer:
            streamer.end()

        # print([step[0][0] for step in compute_steps])
        seq_lens = [seq_steps[i] if seq_steps[i] > 0 else generation_config.max_length - initial_seq_len for i in range(batch_size)]
        avg_compute_steps = [sum([step[0][i] for step in compute_steps]) / seq_lens[i] for i in range(batch_size)]

        if generation_config.return_dict_in_generate:
            return RavenGenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=compute_steps,  # type: ignore
                logits=None,
                attentions=None,
                hidden_states=None,
                past_key_values=model_kwargs.get("past_key_values"),
                avg_compute_steps=avg_compute_steps,
            )
        return input_ids
