from typing import Callable, Optional, Union, Tuple
import torch
import torch.nn.functional as F

from recpre.raven_modeling_minimal import RavenForCausalLM, RavenConfig
from recpre.raven_modeling_minimal import CausalLMOutputRecurrentLatents, GenerationConfig, Cache, HuginnDynamicCache
from recpre.raven_modeling_minimal import RavenGenerateDecoderOnlyOutput

from recpre_adapt.raven_exit_model import LatentDiffExitModel, LTEExitModel, LatentTransformerExitModel, LatentRecurrentExitModel


class PerLoopExitEvaluator:
    # returns None when we should continue, otherwise the exits
    def evaluate_exit(self, latents: torch.Tensor, embedded_inputs: torch.Tensor, compute_step: int, do_sample: bool) -> Optional[torch.Tensor]:
        raise NotImplementedError("Not implemented")

    def init(self, aux_inputs: dict):
        self.aux_inputs = aux_inputs

class PredictiveExitEvaluator:
    def calculate_exit(self) -> int:
        raise NotImplementedError("Not implemented")


class ExitModelEvaluator(PerLoopExitEvaluator):
    def __init__(self, exit_model: Union[None, LatentDiffExitModel, LTEExitModel, LatentTransformerExitModel], min_loops: int = 4, loops_per_exit_eval: int = 1):
        self.exit_model = exit_model
        self.loops_per_exit_eval = loops_per_exit_eval
        self.min_loops = min_loops

    def evaluate_exit(self, latents: torch.Tensor, embedded_inputs: torch.Tensor, compute_step: int, do_sample: bool) -> Optional[torch.Tensor]:
        if compute_step % self.loops_per_exit_eval != 0:
            return None
        if compute_step < self.min_loops:
            return None

        batch_size = latents.shape[0]

        latents = latents[:, :, 0::self.loops_per_exit_eval, :]
        # Check exit condition for each sequence in batch
        if isinstance(self.exit_model, LatentDiffExitModel):
            exit_policy = self.exit_model.forward(latents[:, :, -2, :], latents[:, :, -1, :])
        elif isinstance(self.exit_model, LTEExitModel):
            exit_policy = self.exit_model(embedded_inputs[:, -1, :], latents.flatten(start_dim=0, end_dim=1))
            exit_policy = exit_policy.unflatten(dim=0, sizes=(batch_size, -1))
            exit_policy = exit_policy[:, :, -1, :]
        elif isinstance(self.exit_model, LatentTransformerExitModel):
            exit_policy = self.exit_model.forward(latents.flatten(start_dim=0, end_dim=1))
            exit_policy = exit_policy.unflatten(dim=0, sizes=(batch_size, -1))
            exit_policy = exit_policy[:, :, -1, :]
        elif isinstance(self.exit_model, LatentRecurrentExitModel):
            exit_policy = self.exit_model.forward(latents)

        if do_sample:
            exit_actions = torch.distributions.Categorical(exit_policy).sample()
            new_exits = exit_actions[:, -1] == 0
        else:
            new_exits = exit_policy[:, -1, 0] > 0.5
        return new_exits


class HeuristicExitEvaluator(PerLoopExitEvaluator):
    def __init__(self, model: RavenForCausalLM, exit_threshold: float):
        self.exit_threshold = exit_threshold
        self.model = model

    def evaluate_exit(self, latents: torch.Tensor, embedded_inputs: torch.Tensor, compute_step: int, do_sample: bool) -> Optional[torch.Tensor]:
        raise NotImplementedError("Not implemented")


class EntropyDiffExitEvaluator(HeuristicExitEvaluator):
    def __init__(self, model: RavenForCausalLM, exit_threshold: float = 1e-3):
        super().__init__(model, exit_threshold)

    def init(self, aux_inputs: dict):
        super().init(aux_inputs)
        self.prev_entropy = None

    def evaluate_exit(self, latents: torch.Tensor, embedded_inputs: torch.Tensor, compute_step: int, do_sample: bool) -> Optional[torch.Tensor]:
        if self.prev_entropy is None:
            batch_size = latents.shape[0]
            self.prev_entropy = torch.ones(batch_size, device=latents.device) * 100.0

        outputs = self.model.predict_from_latents(latents[:, :, -1, :], **self.aux_inputs)
        logits: torch.Tensor = outputs.logits  # type: ignore
        probs = F.softmax(logits[:, -1, :], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        exit_values = (entropy - self.prev_entropy).abs()
        self.prev_entropy = entropy
        return exit_values < self.exit_threshold


class LatentDiffExitEvaluator(HeuristicExitEvaluator):
    def __init__(self, model: RavenForCausalLM, exit_threshold: float = 0.03):
        super().__init__(model, exit_threshold)

    def evaluate_exit(self, latents: torch.Tensor, embedded_inputs: torch.Tensor, compute_step: int, do_sample: bool) -> Optional[torch.Tensor]:
        exit_values = (latents[:, :, -1, :] - latents[:, :, -2, :]).norm(dim=-1) / latents[:, :, -1, :].norm(dim=-1)
        return exit_values < self.exit_threshold

class KLExitEvaluator(HeuristicExitEvaluator):
    def __init__(self, model: RavenForCausalLM, exit_threshold: float = 5e-4):
        super().__init__(model, exit_threshold)
        self.V = model.config.padded_vocab_size

    def init(self, aux_inputs: dict):
        super().init(aux_inputs)
        self.prev_log_probs = None

    def calc_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(logits[:, -1, :], dim=-1)

    def evaluate_exit(self, latents: torch.Tensor, embedded_inputs: torch.Tensor, compute_step: int, do_sample: bool) -> Optional[torch.Tensor]:
        if self.prev_log_probs is None:
            batch_size = latents.shape[0]
            self.prev_log_probs = ((1 / self.V) * torch.ones(batch_size, self.V, device=latents.device)).log()

        outputs = self.model.predict_from_latents(latents[:, :, -1, :], **self.aux_inputs)
        logits: torch.Tensor = outputs.logits  # type: ignore
        log_probs = self.calc_log_probs(logits)

        exit_values = F.kl_div(log_probs, self.prev_log_probs, reduction="none", log_target=True).sum(dim=-1)
        self.prev_log_probs = log_probs
        return exit_values < self.exit_threshold

class MinKLExitEvaluator(KLExitEvaluator):
    def __init__(self, model: RavenForCausalLM, exit_threshold: float = 1e-6):
        super().__init__(model, exit_threshold)

    def calc_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits[:, -1, :], dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True)[0]
        probs_mask = probs < (0.1 * max_probs)
        masked_probs = probs
        masked_probs[probs_mask] = 1 / self.V
        probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        return probs.log()

class ArgmaxStabilityExitEvaluator(HeuristicExitEvaluator):
    def __init__(self, model: RavenForCausalLM, exit_threshold: float = 5):
        super().__init__(model, exit_threshold)
        self.prev_argmax: Optional[torch.Tensor] = None
        self.stable_for_n_steps: Optional[torch.Tensor] = None

    def init(self, aux_inputs: dict):
        super().init(aux_inputs)
        self.prev_argmax = None
        self.stable_for_n_steps = None

    def evaluate_exit(self, latents: torch.Tensor, embedded_inputs: torch.Tensor, compute_step: int, do_sample: bool) -> Optional[torch.Tensor]:
        if self.prev_argmax is None or self.stable_for_n_steps is None:
            batch_size = latents.shape[0]
            self.prev_argmax = torch.ones(batch_size, dtype=torch.long, device=latents.device) * -1
            self.stable_for_n_steps = torch.zeros(batch_size, dtype=torch.long, device=latents.device)

        outputs = self.model.predict_from_latents(latents[:, :, -1, :], **self.aux_inputs)
        logits: torch.Tensor = outputs.logits  # type: ignore
        current_argmax = logits[:, -1, :].argmax(dim=-1)
        stable_for_n_steps = torch.where(
            current_argmax == self.prev_argmax, self.stable_for_n_steps + 1, torch.zeros_like(self.stable_for_n_steps)
        )
        exit_values = stable_for_n_steps
        self.prev_argmax = current_argmax
        self.stable_for_n_steps = stable_for_n_steps
        return exit_values < self.exit_threshold


class NumStepsGenerator(PredictiveExitEvaluator):
    def __init__(self, num_steps_generator: Callable[[int], int]):
        self.num_steps_generator: Callable[[int], int] = num_steps_generator
        self.counter = 0

    def calculate_exit(self) -> int:
        steps = self.num_steps_generator(self.counter)
        self.counter += 1
        return steps


class RavenAdaptiveModel(RavenForCausalLM):
    def __init__(self, base_config: RavenConfig, save_latents: bool = False):
        super().__init__(base_config)
        self.save_latents = save_latents
        self.latents = None
        self.exit_evaluator: Optional[PerLoopExitEvaluator | PredictiveExitEvaluator] = None

    @staticmethod
    def from_models(model: RavenForCausalLM, exit_evaluator: Optional[PerLoopExitEvaluator | PredictiveExitEvaluator], **kwargs):
        config = model.config
        new_model = RavenAdaptiveModel(config, **kwargs)
        new_model.transformer = model.transformer
        new_model.emb_scale = model.emb_scale
        new_model.lm_head = model.lm_head

        new_model.exit_evaluator = exit_evaluator
        return new_model

    @torch.no_grad()
    def prefill_with_varied_exit_steps(self, input_ids: torch.Tensor, deterministic: bool = True) -> Tuple[torch.Tensor, HuginnDynamicCache, float]:
        # currently the cache doesn't support batching with adaptive compute
        assert(input_ids.shape[0] == 1)

        past_key_values = HuginnDynamicCache()
        attention_mask = None
        output: torch.Tensor | None = None
        total_compute_steps = 0
        for pos in range(input_ids.shape[1]):
            freqs_cis = self.freqs_cis[:, pos]
            if isinstance(self.exit_evaluator, PredictiveExitEvaluator):
                num_steps = self.exit_evaluator.calculate_exit()
            else:
                num_steps = self.config.mean_recurrence

            input_embeds = self.transformer.wte(input_ids[:, pos]).unsqueeze(1)
            if self.emb_scale != 1:
                input_embeds = input_embeds * self.emb_scale  # type: ignore
            # Non-recurrent prelude
            for block_idx, block in enumerate(self.transformer.prelude):
                input_embeds, _ = block(
                    input_embeds, freqs_cis, block_idx, attention_mask, past_key_values
                )

            current_latents = self.initialize_state(input_embeds, deterministic=deterministic)
            if isinstance(self.exit_evaluator, PerLoopExitEvaluator):
                all_latents = current_latents.clone().unsqueeze(2)
                aux_inputs = {
                    "cache_position": pos,
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask,
                }
                self.exit_evaluator.init(aux_inputs=aux_inputs)

            # Main recurrence
            for compute_step in range(num_steps):
                current_latents, block_idx, _ = self.iterate_one_step(
                    input_embeds, current_latents, block_idx=block_idx, attention_mask=attention_mask, past_key_values=past_key_values
                )
                if isinstance(self.exit_evaluator, PerLoopExitEvaluator):
                    all_latents = torch.cat([all_latents, current_latents.unsqueeze(2)], dim=2)
                    new_exits = self.exit_evaluator.evaluate_exit(all_latents, input_embeds, compute_step, not deterministic)
                    if new_exits is not None and new_exits.any():
                        break
            total_compute_steps += compute_step + 1

            x = self.transformer.ln_f(current_latents)

            # Coda layers
            for block_idx, block in enumerate(self.transformer.coda, start=1):
                x, attn_map = block(x, freqs_cis, -block_idx, attention_mask, past_key_values)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x).float()
            if output is None:
                output = logits
            else:
                output = torch.cat([output, logits], dim=1)
        return output, past_key_values, total_compute_steps / input_ids.shape[1] # type: ignore


    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        input_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: Optional[torch.Tensor | int] = None,
        past_key_values: Optional[Cache] = None,
        output_details: dict = {
            "return_logits": True,
            "return_latents": True,
            "return_attention": False,
            "return_head": False,
            "return_stats": False,
        },
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputRecurrentLatents:
        assert input_embeds is None
        assert input_states is None
        assert position_ids is None
        assert cache_position is None
        assert past_key_values is None
        assert use_cache is False
        assert num_steps is None

        logits, past_key_values, avg_compute_steps = self.prefill_with_varied_exit_steps(input_ids)

        # Prediction head, assuming labels really are labels and not equal to input_ids
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
            log_ppl = loss.clone().detach()
        else:
            loss, log_ppl = torch.as_tensor(0.0), torch.as_tensor(0.0)

        return CausalLMOutputRecurrentLatents(
            loss=loss,
            log_ppl=log_ppl,
            logits=logits if output_details["return_logits"] else None,
            past_key_values=past_key_values,
            hidden_states=None,
            latent_states=None,
            attention_maps=None,
            stats={"avg_compute_steps": avg_compute_steps},
        )

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def iterate_forward(
        self,
        input_embeds,
        input_states,
        freqs_cis,
        block_idx,
        mask,
        past_key_values: Optional[Cache] = None,
        num_steps: Optional[torch.Tensor | int] = None,
        attn_maps: dict = {},
        return_attn: bool = False,
    ):
        x = xk = self.initialize_state(input_embeds, deterministic=True) if input_states is None else input_states.clone()
        if num_steps is None:
            raise NotImplementedError("Doesn't support num_steps is None")
        elif hasattr(num_steps, "__len__") and len(num_steps) > 1: # type: ignore
            num_steps_no_grad, num_steps_with_grad = num_steps # type: ignore
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

        if self.save_latents:
            self.latents = latents

        return self.transformer.ln_f(x), num_steps_no_grad, num_steps_with_grad, xk.detach(), block_idx, attn_maps


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
        if continuous_compute or latent_dampening:
            raise NotImplementedError("Doesn't support continuous compute or latent dampening")
        
        if exit_threshold != "auto":
            raise NotImplementedError("Doesn't support exit threshold")
        
        if generation_config is None:
            generation_config: GenerationConfig = self.generation_config  # type: ignore
        model_kwargs["past_key_values"] = HuginnDynamicCache(**cache_kwargs)
        model_kwargs["use_cache"] = True
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        num_steps_series = model_kwargs.get("num_steps_series", None)
        if num_steps_series is not None:
            self.num_steps_series = num_steps_series
        stop_tokens = self._get_stops(generation_config, tokenizer).to(input_ids.device)
        batch_size = input_ids.shape[0]
        compute_steps = []
        seq_steps = [0] * batch_size
        initial_seq_len = input_ids.shape[1]
        avg_compute_steps = None

        # Track which sequences have finished
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        exit_evaluator = self.exit_evaluator

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
            override_num_steps = exit_evaluator.calculate_exit() if isinstance(exit_evaluator, PredictiveExitEvaluator) else None
            num_steps = override_num_steps if override_num_steps is not None else model_inputs["num_steps"]
            if isinstance(exit_evaluator, PerLoopExitEvaluator):
                exit_evaluator.init(aux_inputs=aux_inputs)

            # Iterate through compute steps
            for compute_step in range(num_steps):
                prev_latents = current_latents.clone()
                current_latents, block_idx, _ = self.iterate_one_step(
                    embedded_inputs, current_latents, block_idx=block_idx, **aux_inputs
                )
                all_latents = torch.cat([all_latents, current_latents.unsqueeze(2)], dim=2)

                if override_num_steps is not None:
                    # if we're overriding the number of steps, we don't need to check the exit condition
                    continue
                if step > 0:  # do not exit in prefill:
                    continue
                if exit_evaluator is None:
                    continue

                assert isinstance(exit_evaluator, PerLoopExitEvaluator)
                new_exits = exit_evaluator.evaluate_exit(all_latents, embedded_inputs, compute_step, generation_config.do_sample)
                if new_exits is None:
                    continue
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
                        compute_steps_per_seq[i] = num_steps
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
                            compute_steps_per_seq[i] = num_steps

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
