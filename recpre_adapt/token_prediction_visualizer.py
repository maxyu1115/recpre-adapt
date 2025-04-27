from dataclasses import dataclass
from enum import Enum
import os
import sys
import json
from typing import Callable, Literal, Optional
import torch
import gradio as gr
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
from html import escape as html_escape

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenForCausalLM
from recpre_adapt.train import update_huggingface_implementation, generate_causal_mask
from recpre_adapt.raven_exit_model import *

EPSILON = 1e-10


VERBOSE = False
def _print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Calculate KL divergence: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
    # Adding small epsilon to avoid log(0)
    return torch.sum(p * torch.log((p + EPSILON) / (q + EPSILON)), dim=-1)

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def top_3_exact_match(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    top_3_p = torch.topk(p, 3, dim=-1).indices
    top_3_q = torch.topk(q, 3, dim=-1).indices
    return 3 - (torch.sum(top_3_p == top_3_q, dim=-1))

def top_5_exact_match(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    top_5_p = torch.topk(p, 5, dim=-1).indices
    top_5_q = torch.topk(q, 5, dim=-1).indices
    return 5 - (torch.sum(top_5_p == top_5_q, dim=-1))

def top_10_exact_match(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    top_10_p = torch.topk(p, 10, dim=-1).indices
    top_10_q = torch.topk(q, 10, dim=-1).indices
    return 10 - (torch.sum(top_10_p == top_10_q, dim=-1))


SCORE_FUNCTIONS = {
    "KL Divergence": (kl_divergence, 2.0),
    "JS Divergence": (js_divergence, 0.5),
    "Top 3 Exact Match": (top_3_exact_match, 4),
    "Top 5 Exact Match": (top_5_exact_match, 6),
    "Top 10 Exact Match": (top_10_exact_match, 11),
}


def latent_l1_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(p - q), dim=-1)

def latent_l2_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.pow(p - q, 2), dim=-1))

def entropy_diff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(p * torch.log(p + EPSILON) - q * torch.log(q + EPSILON), dim=-1)


@dataclass
class ExitHeuristic:
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    limits: tuple[float, float]
    input_type: Literal["latent", "probs", "incremental-latent", "incremental-probs"]


EXIT_HEURISTIC_FUNCTIONS = {
    "Latent L1 Distance": ExitHeuristic(latent_l1_distance, (0.0, 10000.0), "latent"),
    "Latent L2 Distance": ExitHeuristic(latent_l2_distance, (0.0, 100.0), "latent"),
    "Entropy Diff": ExitHeuristic(entropy_diff, (-1.0, 1.0), "probs"),
    "Incremental Latent L1 Distance": ExitHeuristic(latent_l1_distance, (0.0, 10000.0), "incremental-latent"),
    "Incremental Latent L2 Distance": ExitHeuristic(latent_l2_distance, (0.0, 100.0), "incremental-latent"),
    "Incremental Entropy Diff": ExitHeuristic(entropy_diff, (-1.0, 1.0), "incremental-probs"),
}


class HighlightCompare(Enum):
    REF_VS_TEST = "Ref vs Test"
    REF_VS_EARLY_EXIT = "Ref vs Early Exit"
    TEST_VS_EARLY_EXIT = "Test vs Early Exit"


class TokenPredictionVisualizer:
    def __init__(self, model_name="tomg-group-umd/huginn-0125", exit_model_dir="checkpoints/checkpoints_18/", exit_model_name="exit_model_2999.pt"):
        self.model_name = model_name
        self.exit_model_dir = exit_model_dir
        self.exit_model_name = exit_model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: RavenForCausalLM
        self.exit_model: Union[LatentTransformerExitModel, LTEExitModel, LatentDiffExitModel, LatentDiffEmbeddingExitModel, LatentRecurrentExitModel]
        self.exit_model_min_steps: int
        self.exit_model_first_exit: bool = False
        self.init_model(exit_model_dir, exit_model_name)

        self.selected_score_function: str = "KL Divergence"
        self.score_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = SCORE_FUNCTIONS[self.selected_score_function][0]
        self.score_limit: float = SCORE_FUNCTIONS[self.selected_score_function][1]

        self.selected_exit_heuristic: str = "Latent L2 Distance"
        self.exit_heuristic: ExitHeuristic = EXIT_HEURISTIC_FUNCTIONS["Latent L2 Distance"]

        # cached states
        self.text: str = ""
        self.max_steps: int = 0
        self.tokens: list[int] = []
        self.token_strings: list[str] = []
        self.spans: list[tuple[int, int]] = []
        self.top_k: int = 10
        self.latents: list[torch.Tensor] = []
        self.input_embeds: torch.Tensor
        self.logits: list[torch.Tensor] = []
        self.probs: list[torch.Tensor] = []
        self.exit_policy: Optional[torch.Tensor] = None
        self.expected_scores: Optional[torch.Tensor] = None
        self.expected_probs: Optional[torch.Tensor] = None
        self.expected_steps: Optional[torch.Tensor] = None
        self.current_test_steps: int = 0

        self.highlight_top_k: int = 1
        self.highlight_compare: HighlightCompare = HighlightCompare.REF_VS_EARLY_EXIT

        # Flags to track state changes
        self._base_model_output_stale: bool = True
        self._exit_policy_stale: bool = True
        self._expected_values_stale: bool = True

    def init_model(self, exit_model_dir: str, exit_model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        update_huggingface_implementation(self.model)
        self.model.to(self.device, dtype=torch.bfloat16) # type: ignore
        self.model.save_latents = True
        self.model.eval()

        with open(os.path.join(exit_model_dir, "training_params.json"), "r") as f:
            training_params = json.load(f)

        if training_params.get("model") == "RavenLatentExitModel":
            self.exit_model = RavenLatentExitModel(self.model.config)
        elif training_params.get("model") == "RavenLatentEmbeddingExitModel":
            self.exit_model = RavenLatentEmbeddingExitModel(self.model.config)
        elif training_params.get("model") == "RavenLatentTransformerExitModel":
            self.exit_model = RavenLatentTransformerExitModel(self.model.config)
        elif training_params.get("model") == "RavenLatentRecurrentExitModel":
            self.exit_model = RavenLatentRecurrentExitModel(self.model.config)
        else:
            self.exit_model = RavenExitModel(self.model.config)
        self.exit_model_min_steps = training_params.get("min_loops", 4)
        self.exit_model.load_state_dict(torch.load(os.path.join(exit_model_dir, exit_model_name)))
        self.exit_model.to(self.device, dtype=torch.bfloat16) # type: ignore
        self.exit_model.eval()

    def tokenize_text(self, text: str):
        """Tokenize text and return token strings and their indices."""
        self.tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.token_strings = self.tokenizer.convert_ids_to_tokens(self.tokens) # type: ignore
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        self.spans = encoding.offset_mapping

    def get_token_index(self, char_index: int) -> int:
        if char_index < 0:
            return -1
        for i, (start, end) in enumerate(self.spans):
            if start <= char_index < end:
                return i
        return -1

    def _run_base_model(self):
        """Runs the base model forward pass if text or max_steps changed."""
        if not self._base_model_output_stale:
            return

        _print("Running base model forward pass...") # For debugging
        self.tokenize_text(self.text)
        self.latents = []
        self.logits = []
        self.probs = []
        if not self.tokens: # Handle empty text case
            self._base_model_output_stale = False
            self._exit_policy_stale = True # Need to potentially clear/recalculate exit policy
            self._expected_values_stale = True # Need to potentially clear/recalculate expected values
            return

        input_ids = torch.tensor([self.tokens]).to(self.device)

        with torch.no_grad():
            # Run the model with the specified number of steps
            self.model.forward(input_ids, attention_mask=None, num_steps=torch.tensor((self.max_steps,)))
            self.latents = self.model.latents
            self.input_embeds = self.model.input_embeds
            for i in range(self.max_steps + 1):
                logits = self.model.predict_from_latents(self.latents[i]).logits
                assert logits is not None
                self.logits.append(logits)
                probs = torch.softmax(logits, dim=-1)
                self.probs.append(probs)

        self._base_model_output_stale = False
        self._exit_policy_stale = True
        self._expected_values_stale = True
        _print("Base model forward pass complete.")

    def _run_exit_model(self):
        """Runs the exit model forward pass if base model output is new."""
        if not self._exit_policy_stale or self._base_model_output_stale:
            # If base model is stale, it needs to run first
            # If exit policy is not stale, no need to run
            return

        _print("Running exit model forward pass...") # For debugging
        if not self.latents: # Handle case where base model didn't run (e.g., empty text)
            self.exit_policy = None
            self._exit_policy_stale = False
            self._expected_values_stale = True # Expected values depend on exit policy
            return

        input_ids_shape = (self.logits[0].shape[0], self.logits[0].shape[1]) # (batch_size, seq_len)

        with torch.no_grad():
            # Ensure latents list has enough elements
            if len(self.latents) < self.max_steps + 1:
                print(f"Warning: Not enough latents ({len(self.latents)}) for max_steps ({self.max_steps}).")
                self.exit_policy = None # Cannot compute

            elif isinstance(self.exit_model, LatentDiffExitModel):
                self.exit_policy = torch.zeros((input_ids_shape[0], input_ids_shape[1], self.max_steps, 2), device=self.device, dtype=self.logits[0].dtype)
                for i in range(self.exit_model_min_steps, self.max_steps):
                    # Check if indices are valid before accessing latents
                    if i - 1 < len(self.latents) and i < len(self.latents):
                        policy = self.exit_model.forward(self.latents[i - 1], self.latents[i])
                        self.exit_policy[:, :, i, :] = policy
                    else:
                        print(f"Warning: Index out of bounds for latents at step {i} in LatentDiffExitModel.")
                        # Handle error case, maybe fill with default policy or break
                        break # Stop calculating policy if latents are missing

            elif isinstance(self.exit_model, LTEExitModel) or isinstance(self.exit_model, LatentTransformerExitModel):
                # Stack latents up to max_steps (latents has max_steps + 1 elements)
                latents_stack = torch.stack(self.latents[:self.max_steps], dim=2) # Shape: [batch, seq, steps, hidden]
                latents_flat = latents_stack.flatten(start_dim=0, end_dim=1) # Shape: [batch*seq, steps, hidden]
                attn_mask = generate_causal_mask(self.max_steps, self.device)

                if isinstance(self.exit_model, LTEExitModel):
                    input_embeds_flat = self.input_embeds.flatten(start_dim=0, end_dim=1) # Shape: [batch*seq, hidden]
                    policy = self.exit_model.forward(input_embeds_flat, latents_flat, attn_mask=attn_mask) # Shape: [batch*seq, steps, 2]
                elif isinstance(self.exit_model, LatentTransformerExitModel):
                    policy = self.exit_model.forward(latents_flat, attn_mask=attn_mask) # Shape: [batch*seq, steps, 2]

                self.exit_policy = policy.unflatten(dim=0, sizes=input_ids_shape) # Shape: [batch, seq, steps, 2]
            elif isinstance(self.exit_model, LatentRecurrentExitModel):
                latents_stack = torch.stack(self.latents[:self.max_steps], dim=2) # Shape: [batch, seq, steps, hidden]
                policy = self.exit_model.forward(latents_stack) # Shape: [batch, seq, steps, 2]
                self.exit_policy = policy
            else:
                print(f"Warning: Unsupported exit model type: {type(self.exit_model)}")
                self.exit_policy = None # Mark as not computed

        self._exit_policy_stale = False
        self._expected_values_stale = True # New policy means expected values need recalculation
        _print("Exit model forward pass complete.")


    def _calculate_and_cache_expected_values(self):
        """Calculates expected scores, steps, and probabilities based on current state."""
        if not self._expected_values_stale:
            return

        _print("Calculating expected values...") # For debugging
        # Ensure base model and exit model outputs are ready
        if self._base_model_output_stale or not self.probs or self.exit_policy is None:
            _print("Skipping expected value calculation: Base model output or exit policy not ready.")
            self.expected_scores = None
            self.expected_steps = None
            self.expected_probs = None
            # Don't mark as not stale, as calculation didn't complete successfully
            return

        expected_score, expected_steps, expected_probs = self.calculate_exit_model_results()
        self.expected_steps = expected_steps.clone() if expected_steps is not None else None
        self.expected_scores = expected_score.clone() if expected_score is not None else None
        self.expected_probs = expected_probs.clone() if expected_probs is not None else None
        self._expected_values_stale = False
        _print("Expected values calculation complete.")

    def process_text(self, text: str, max_steps: int):
        """
        Processes the input text, running model forward passes and calculating
        expected values only if necessary based on state changes.
        """
        # --- Check for state changes requiring re-computation ---
        if text != self.text:
            self.text = text
            self._base_model_output_stale = True

        if max_steps != self.max_steps:
            # Ensure max_steps is valid before updating
            if max_steps < 1:
                _print(f"Warning: Invalid max_steps ({max_steps}). Using 1 instead.")
                max_steps = 1
            self.max_steps = max_steps
            self._base_model_output_stale = True

        # --- Run computations if inputs are stale ---
        self._run_base_model()
        self._run_exit_model()
        self._calculate_and_cache_expected_values()

    def get_predictions(self, probs: Optional[torch.Tensor], k: Optional[int] = None):
        # Handle cases where probs might be None
        if probs is None:
            return [], np.array([])
        with torch.no_grad():
            if k is None:
                k = self.top_k
            # Get the top k predictions
            top_probs, top_indices = torch.topk(probs, k)
            top_probs = top_probs.cpu().numpy()

            top_tokens = [idx.item() for idx in top_indices]

            # Add the "other" category
            return top_tokens, top_probs

    def set_score_function(self, function_name: str):
        """Set the score function and mark expected values as stale."""
        if function_name in SCORE_FUNCTIONS and function_name != self.selected_score_function:
            _print(f"Setting score function to: {function_name}")
            self.selected_score_function = function_name
            self.score_function = SCORE_FUNCTIONS[function_name][0]
            self.score_limit = SCORE_FUNCTIONS[function_name][1]
            self._expected_values_stale = True # Expected values depend on the score function

    def set_exit_heuristic(self, heuristic_name: str):
        """Set the exit criteria function to use."""
        if heuristic_name in EXIT_HEURISTIC_FUNCTIONS and heuristic_name != self.selected_exit_heuristic:
            _print(f"Setting exit heuristic to: {heuristic_name}")
            self.selected_exit_heuristic = heuristic_name
            self.exit_heuristic = EXIT_HEURISTIC_FUNCTIONS[heuristic_name]
            # This only affects the score progression plot, not core calculations

    def set_exit_criteria_mode(self, first_exit_mode: bool):
        """Sets the exit criteria calculation mode (expected vs. first exit)."""
        if first_exit_mode != self.exit_model_first_exit:
            _print(f"Setting first exit mode to: {first_exit_mode}")
            self.exit_model_first_exit = first_exit_mode
            self._expected_values_stale = True # Expected values depend on this mode

    def set_top_k(self, top_k: int):
        """Sets the top_k value for prediction visualization."""
        if top_k != self.top_k:
            _print(f"Setting top_k to: {top_k}")
            self.top_k = top_k
            # This only affects the prediction plots, not core calculations

    def calculate_scores(self, test_steps: int) -> Optional[torch.Tensor]:
        """Calculate divergence scores between reference and test distributions for all tokens."""
        # Ensure base model outputs are ready and test_steps is valid
        if self._base_model_output_stale or not self.logits or test_steps < 0 or test_steps >= len(self.logits):
            _print(f"Skipping score calculation: Base model output not ready or invalid test_steps ({test_steps}).")
            return None
        # Ensure reference step (max_steps) is valid
        if self.max_steps >= len(self.logits):
            _print(f"Skipping score calculation: max_steps ({self.max_steps}) out of bounds for logits ({len(self.logits)}).")
            return None

        with torch.no_grad():
            # Get full distributions from both models (not just top-k)
            ref_logits = self.logits[self.max_steps][0, :, :]
            test_logits = self.logits[test_steps][0, :, :]

            # Convert to probabilities
            ref_probs = torch.softmax(ref_logits, dim=-1)
            test_probs = torch.softmax(test_logits, dim=-1)

            return self.score_function(ref_probs, test_probs)

    def calculate_top_k_matches(self, probs: Optional[torch.Tensor], ref_probs: Optional[torch.Tensor], top_k: int) -> int:
        # Handle None inputs
        if probs is None or ref_probs is None:
            return 0
        top_tokens, top_probs = self.get_predictions(probs, top_k)
        top_ref_tokens, top_ref_probs = self.get_predictions(ref_probs, top_k)

        count = 0
        for i in range(top_k):
            count += top_tokens[i] == top_ref_tokens[i]
        return count

    def calculate_exit_model_results(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Calculate divergence scores between reference and test distributions for all tokens."""
        # Check prerequisites
        if not self.probs or self.exit_policy is None or self.max_steps >= len(self.probs):
            print("Cannot calculate exit model results: Missing probabilities or exit policy, or max_steps out of bounds.")
            return None, None, None

        with torch.no_grad():
            ref_logits = self.logits[self.max_steps][0, :, :]
            ref_probs = torch.softmax(ref_logits, dim=-1)
            # Calculate score for step max_steps vs itself (should be near zero)
            ref_score = self.score_function(ref_probs, ref_probs) # Shape: [seq_len]

            # Initialize expected values
            seq_len = ref_probs.shape[0]
            expected_score = torch.zeros_like(ref_score)
            expected_steps = torch.zeros_like(ref_score)
            expected_probs = torch.zeros_like(ref_probs)

            if not self.exit_model_first_exit:
                # --- Expected Value Calculation ---
                expected_score = ref_score.clone()
                expected_steps = torch.ones_like(ref_score, dtype=torch.float32) * self.max_steps
                expected_probs = ref_probs.clone()

                # Iterate backwards from max_steps - 1 down to min_steps
                for i in range(self.max_steps - 1, self.exit_model_min_steps - 1, -1):
                    # Ensure index i is valid for logits/probs
                    if i >= len(self.logits):
                        print(f"Warning: Index {i} out of bounds for logits during expected value calculation.")
                        continue # Skip this step

                    test_logits = self.logits[i][0, :, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    score = self.score_function(ref_probs, test_probs) # Score of step i vs final step

                    # Policy for exiting at step i (index i in policy corresponds to decision *after* step i)
                    # Policy shape: [seq_len, steps, 2]
                    policy_exit = self.exit_policy[0, :, i, 0] # Probability of exiting *at* step i
                    policy_continue = self.exit_policy[0, :, i, 1] # Probability of continuing *past* step i

                    expected_score = policy_exit * score + policy_continue * expected_score
                    expected_steps = policy_exit * i + policy_continue * expected_steps
                    expected_probs = policy_exit.unsqueeze(-1) * test_probs + policy_continue.unsqueeze(-1) * expected_probs

            else:
                # --- First >50% Exit Calculation ---
                exit_reached = torch.zeros(seq_len, device=self.device, dtype=torch.bool)
                # Iterate forwards from min_steps up to max_steps - 1
                for i in range(self.exit_model_min_steps, self.max_steps):
                     # Ensure index i is valid for logits/probs
                    if i >= len(self.logits):
                        print(f"Warning: Index {i} out of bounds for logits during first exit calculation.")
                        continue # Skip this step

                    test_logits = self.logits[i][0, :, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    score = self.score_function(ref_probs, test_probs)

                    # Policy for exiting at step i
                    policy_exit = self.exit_policy[0, :, i, 0] # Prob of exiting AT step i
                    is_first_exit = (policy_exit > 0.5) & ~exit_reached

                    # Accumulate values for tokens exiting at this step
                    expected_steps += torch.where(is_first_exit, float(i), 0.0)
                    expected_probs += torch.where(is_first_exit.unsqueeze(-1), test_probs, torch.zeros_like(expected_probs))
                    expected_score += torch.where(is_first_exit, score, 0.0)

                    exit_reached |= is_first_exit # Update exit_reached mask

                # Handle tokens that never exited
                remaining = ~exit_reached
                expected_steps += torch.where(remaining, float(self.max_steps), 0.0)
                expected_probs += torch.where(remaining.unsqueeze(-1), ref_probs, torch.zeros_like(expected_probs))
                expected_score += torch.where(remaining, ref_score, 0.0) # Score for non-exiting tokens is the ref_score

            return expected_score, expected_steps, expected_probs

    def visualize_text_scores(self, test_steps: int, char_index: int) -> str:
        """
        Create HTML with score values displayed above each token, with highlighting for selected token.
        If selected_token_idx is < 0, no highlighting is done.
        """
        # Ensure calculations are up-to-date
        # self._calculate_and_cache_expected_values() # Should be called by the main update loop

        # Check if data is ready
        if not self.spans or not self.text:
            return "<div>Please enter text and process.</div>"
        if self.expected_scores is None or self.expected_steps is None or self.expected_probs is None:
            return "<div>Expected values not calculated. Ensure model ran successfully.</div>"
        if self._base_model_output_stale or test_steps >= len(self.logits):
            return f"<div>Base model output stale or test_steps ({test_steps}) invalid.</div>"


        scores = self.calculate_scores(test_steps)
        if scores is None:
            return "<div>Could not calculate scores for visualization.</div>"

        selected_token_idx = self.get_token_index(char_index)
        html = ""
        ref_probs = torch.softmax(self.logits[self.max_steps][0, :, :], dim=-1)
        test_probs = torch.softmax(self.logits[test_steps][0, :, :], dim=-1)

        # Ensure expected values have the correct shape
        if scores.shape[0] != self.expected_scores.shape[0] or \
           scores.shape[0] != self.expected_steps.shape[0]:
            _print(f"Shape mismatch between scores ({scores.shape[0]}) and expected values ({self.expected_scores.shape[0]}, {self.expected_steps.shape[0]}, {self.expected_probs.shape[0]})")
            return "<div>Error: Shape mismatch in calculated values.</div>"


        for i, (start, end) in enumerate(self.spans):
            token_text = self.text[start:end]
            score = scores[i].item()
            expected_score = self.expected_scores[i].item()
            expected_step = self.expected_steps[i].item()

            # Create a color gradient based on KL value (higher values = more red)
            # Use score_limit for normalization instead of max_score for consistency
            # max_score = scores.max().item()
            normalized_score = min(max(score / (self.score_limit + EPSILON), 0.0), 1.0) # Normalize to [0,1]
            r_score = min(255, int(255 * normalized_score))
            g_score = min(255, int(255 * (1 - normalized_score)))

            normalized_expected_score = min(max(expected_score / (self.score_limit + EPSILON), 0.0), 1.0)
            r_expected_score = min(255, int(255 * normalized_expected_score))
            g_expected_score = min(255, int(255 * (1 - normalized_expected_score)))

            # Color for expected steps: Blue if early exit (< test_steps), Red if later exit (>= test_steps)
            # Normalize within the relevant range
            min_step = self.exit_model_min_steps
            max_step = self.max_steps
            current_test = max(min_step, test_steps) # Ensure test_steps is at least min_step for normalization

            if expected_step < current_test: # Early exit (Blue gradient)
                # Normalize between min_step and current_test
                norm_range = current_test - min_step
                normalized_expected_step = 1.0 - min(max((expected_step - min_step) / (norm_range + EPSILON), 0.0), 1.0) if norm_range > 0 else 0.0
                r_expected_step = 0
                g_expected_step = min(255, int(255 * (1 - normalized_expected_step))) # Less blue = earlier exit
                b_expected_step = min(255, int(255 * normalized_expected_step)) # More blue = later exit (closer to test_steps)
            else: # Later exit or exact match (Red gradient)
                # Normalize between current_test and max_step
                norm_range = max_step - current_test
                normalized_expected_step = min(max((expected_step - current_test) / (norm_range + EPSILON), 0.0), 1.0) if norm_range > 0 else 0.0
                r_expected_step = min(255, int(255 * normalized_expected_step)) # More red = later exit
                g_expected_step = min(255, int(255 * (1 - normalized_expected_step))) # Less red = closer to test_steps
                b_expected_step = 0

            # Check if top-1 prediction matches
            if self.highlight_compare == HighlightCompare.REF_VS_TEST:
                top_k_matches = self.calculate_top_k_matches(test_probs[i, :], ref_probs[i, :], self.highlight_top_k)
            elif self.highlight_compare == HighlightCompare.REF_VS_EARLY_EXIT:
                top_k_matches = self.calculate_top_k_matches(self.expected_probs[i, :], ref_probs[i, :], self.highlight_top_k)
            elif self.highlight_compare == HighlightCompare.TEST_VS_EARLY_EXIT:
                top_k_matches = self.calculate_top_k_matches(self.expected_probs[i, :], test_probs[i, :], self.highlight_top_k)

            # Add yellow background highlight for selected token
            # Add red background if the top-1 prediction from early exit differs from reference
            if top_k_matches < self.highlight_top_k:
                top_k_matches_ratio = top_k_matches / self.highlight_top_k
                if i == selected_token_idx:
                    # orange when both selected and marked
                    bg_style = f"background-color: rgba(255, 165, 0, {0.5 * (1 - top_k_matches_ratio) : .2f});"
                else:
                    bg_style = f"background-color: rgba(255, 0, 0, {0.5 * (1 - top_k_matches_ratio) : .2f});" # Light red background
            elif i == selected_token_idx:
                bg_style = "background-color: yellow;"
            else:
                bg_style = ""

            # Create HTML with score value above each token
            html += f"""
            <div style="display:inline-block; text-align:center; margin:0 2px; vertical-align: top;">
              <div style="font-size:0.8em; color:rgb({r_score},{g_score},0); font-weight:bold;">{score:.3f}</div>
              <div style="font-size:0.8em; font-weight:bold;">
                <span style="color:rgb({r_expected_score},{g_expected_score},0);">{expected_score:.3f}</span>
                <span style="color:rgb({r_expected_step},{g_expected_step},{b_expected_step});">({expected_step:.1f})</span>
              </div>
              <span style="{bg_style} white-space: pre-wrap;">{html_escape(token_text)}</span>
            </div>
            """
        return f'<div style="line-height:1.8;">{html}</div>'

    def _plot_predictions(self, ax, tokens, probs, title, color):
        other = 1.0 - probs.sum()
        # Add labels for top-k tokens
        labels = [f"'{self.tokenizer.decode([token])}'[{token}] ({prob * 100:.1f}%)" for token, prob in zip(tokens, probs)]
        # Add the "other" category
        labels.append(f"Other ({other * 100:.1f}%)")
        # Combine top-k probabilities with "other" probability
        all_probs = np.append(probs, other)

        ax.barh(range(len(labels)), all_probs, color=[color]*len(probs) + ['lightgray'])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        ax.set_xlabel("Probability")
        ax.invert_yaxis()

    def get_average_expected_values(self) -> tuple[Optional[float], Optional[float]]:
        """Calculates the average expected steps and score across all tokens."""
        if self.expected_steps is None or self.expected_scores is None:
            return None, None
        if self.expected_steps.numel() == 0 or self.expected_scores.numel() == 0:
            return 0.0, 0.0 # Handle empty tensor case

        avg_steps = torch.mean(self.expected_steps.float()).item()
        avg_score = torch.mean(self.expected_scores.float()).item()
        return avg_steps, avg_score

    def calculate_token_score_by_steps(self, token_idx) -> list[float]:
        # Add checks for readiness
        if self._base_model_output_stale or not self.logits or token_idx < 0 or token_idx >= self.logits[0].shape[1]:
            return [0.0] * self.max_steps # Return default list if not ready

        scores = []
        with torch.no_grad():
            # Get reference distribution from maximum steps
            ref_logits = self.logits[self.max_steps][0, token_idx, :]
            ref_probs = torch.softmax(ref_logits, dim=-1)

            # Calculate score for each step
            for step in range(1, self.max_steps + 1):
                test_logits = self.logits[step][0, token_idx, :]
                test_probs = torch.softmax(test_logits, dim=-1)

                score = self.score_function(ref_probs, test_probs)
                scores.append(score.item())
        return scores

    def calculate_token_exit_criteria_by_steps(self, token_idx) -> list[float]:
         # Add checks for readiness
        if self._base_model_output_stale or not self.latents or not self.logits or token_idx < 0 or token_idx >= self.logits[0].shape[1]:
            # Return default list based on heuristic limits if not ready
            default_val = (self.exit_heuristic.limits[0] + self.exit_heuristic.limits[1]) / 2.0
            return [default_val] * self.max_steps

        values = []
        with torch.no_grad():
            if self.exit_heuristic.input_type == "latent":
                # Get reference latent from maximum steps
                ref_latent = self.latents[self.max_steps][0, token_idx, :]

                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    test_latent = self.latents[step][0, token_idx, :]
                    value = self.exit_heuristic.function(ref_latent, test_latent)
                    values.append(value.item())
            elif self.exit_heuristic.input_type == "probs":
                # Get reference probabilities from maximum steps
                ref_logits = self.logits[self.max_steps][0, token_idx, :]
                ref_probs = torch.softmax(ref_logits, dim=-1)

                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    test_logits = self.logits[step][0, token_idx, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    value = self.exit_heuristic.function(ref_probs, test_probs)
                    values.append(value.item())
            elif self.exit_heuristic.input_type == "incremental-latent":
                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    prev_latent = self.latents[step - 1][0, token_idx, :]
                    test_latent = self.latents[step][0, token_idx, :]
                    value = self.exit_heuristic.function(prev_latent, test_latent)
                    values.append(value.item())
            elif self.exit_heuristic.input_type == "incremental-probs":
                # Calculate exit criteria for each step
                for step in range(1, self.max_steps + 1):
                    prev_logits = self.logits[step - 1][0, token_idx, :]
                    prev_probs = torch.softmax(prev_logits, dim=-1)
                    test_logits = self.logits[step][0, token_idx, :]
                    test_probs = torch.softmax(test_logits, dim=-1)
                    value = self.exit_heuristic.function(prev_probs, test_probs)
                    values.append(value.item())
        return values

    def plot_token_score_progression(self, token_idx):
        """Create a plot showing score progression by step count for the selected token."""
        # Add checks for readiness
        if self._base_model_output_stale or token_idx < 0 or not self.token_strings or token_idx >= len(self.token_strings):
            # Return an empty figure or a message
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Data not ready or token index invalid.", ha='center', va='center')
            plt.close(fig)
            return fig

        scores = self.calculate_token_score_by_steps(token_idx)
        exit_criteria_values = self.calculate_token_exit_criteria_by_steps(token_idx)
        steps = list(range(1, self.max_steps + 1))

        fig, ax1 = plt.subplots(figsize=(10, 7))

        # Plot score on primary y-axis
        color1 = 'purple'
        ax1.set_xlabel('Number of Steps')
        ax1.set_ylabel(f'Score ({self.selected_score_function})', color=color1)
        ax1.plot(steps, scores, marker='o', linestyle='-', color=color1, label=self.selected_score_function)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, self.score_limit)

        # Create secondary y-axis for exit criteria
        ax2 = ax1.twinx()
        color2 = 'green'
        ax2.set_ylabel(f'Exit Criteria ({self.selected_exit_heuristic})', color=color2)
        ax2.plot(steps, exit_criteria_values, marker='s', linestyle='--', color=color2, label=self.selected_exit_heuristic)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(self.exit_heuristic.limits[0], self.exit_heuristic.limits[1])
        
        ax1.set_title(f'Progression for Token: "{self.token_strings[token_idx]}"')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Add a vertical line at the test_steps position
        if hasattr(self, 'current_test_steps'):
            ax1.axvline(x=self.current_test_steps, color='red', linestyle='--', 
                       label=f'Current test steps: {self.current_test_steps}')

        # Create a single legend for both curves
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        result_fig = fig
        plt.close(fig)
        return result_fig

    def visualize_selected_token(self, char_index: int, test_steps: int):
        """Visualize token predictions for a selected character position."""
        # Ensure calculations are up-to-date
        # self._calculate_and_cache_expected_values() # Should be called by the main update loop

        # Find which token corresponds to the clicked position
        token_index = self.get_token_index(char_index)

        # Default return values
        default_fig, default_info, default_prog_plot = None, "Select a token.", None

        if token_index == -1:
            return default_fig, default_info, default_prog_plot

        # Check if data is ready
        if self._base_model_output_stale or not self.probs or not self.token_strings or token_index >= len(self.token_strings):
            return default_fig, f"Data not ready for token {token_index}.", default_prog_plot
        if self.expected_probs is None or self.expected_steps is None:
            return default_fig, f"Expected values not ready for token {token_index}.", default_prog_plot
        if test_steps >= len(self.probs) or self.max_steps >= len(self.probs):
            return default_fig, f"Invalid steps ({test_steps}/{self.max_steps}) for token {token_index}.", default_prog_plot
        if token_index >= self.expected_probs.shape[0] or token_index >= self.expected_steps.shape[0]:
            return default_fig, f"Token index {token_index} out of bounds for expected values.", default_prog_plot


        # Store current test steps for the KL progression plot
        self.current_test_steps = test_steps

        selected_token = self.token_strings[token_index]
        # Clean the token for display by removing special characters like Ġ
        display_token = selected_token.replace('Ġ', ' ')

        # Get predictions for both step counts
        test_tokens, test_probs_np = self.get_predictions(self.probs[test_steps][0, token_index, :])
        ref_tokens, ref_probs_np = self.get_predictions(self.probs[self.max_steps][0, token_index, :])
        exit_tokens, exit_probs_np = self.get_predictions(self.expected_probs[token_index, :])

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Reference steps plot
        self._plot_predictions(ax1, ref_tokens, ref_probs_np, f"Predictions with {self.max_steps} steps", "skyblue")

        # Test steps plot
        self._plot_predictions(ax2, test_tokens, test_probs_np, f"Predictions with {test_steps} steps", "salmon")

        # Early exit plot
        expected_step_val = self.expected_steps[token_index].item()
        self._plot_predictions(ax3, exit_tokens, exit_probs_np, f"Early exit predictions ({expected_step_val:.1f} steps)", "mediumseagreen") # Changed color slightly

        plt.tight_layout()

        # Generate score progression plot
        score_progression_plot = self.plot_token_score_progression(token_index)

        # Generate token information
        token_info = f"Selected token: '{display_token}' (Index: {token_index})"

        result_fig = fig
        plt.close(fig) # Close the figure to prevent double display in some environments
        return result_fig, token_info, score_progression_plot


def create_interface(visualizer: TokenPredictionVisualizer):
    """Create the Gradio interface."""
    # Default values
    default_text = """Question: Stella wanted to buy a new dress for the upcoming dance.  At the store she found out that the dress she wanted was $50.  
The store was offering 30% off of everything in the store.  What was the final cost of the dress?
Answer: The dress was $50 and 30% off so 50*.30 = $<<50*.30=15>>15 discount price
The dress cost $50 minus $15 (30% off discount) so 50-15 = $<<50-15=35>>35
#### 35"""
    default_ref_steps = 32
    default_test_steps = 8
    default_top_k = 10
    default_score_func = visualizer.selected_score_function
    default_exit_heuristic = visualizer.selected_exit_heuristic
    default_exit_criteria = "Expected Policy"
    default_highlight_compare = visualizer.highlight_compare
    default_highlight_top_k = visualizer.highlight_top_k

    # Initialize visualizer state (run initial processing)
    _print("Initial processing...")
    visualizer.set_top_k(default_top_k)
    visualizer.set_score_function(default_score_func)
    visualizer.set_exit_heuristic(default_exit_heuristic)
    visualizer.set_exit_criteria_mode(default_exit_criteria == "First >50% Exit")
    visualizer.process_text(default_text, default_ref_steps)
    initial_score_viz = visualizer.visualize_text_scores(default_test_steps, -1)
    _print("Initial processing complete.")


    with gr.Blocks() as interface:
        gr.Markdown("# Token Prediction Visualizer")
        gr.Markdown(f"## Visualizing {visualizer.exit_model_dir}/{visualizer.exit_model_name}")
        gr.Markdown("Click on any token in the text to see the model's predictions with different recurrence steps.")

        # Store the state of the selected character index
        last_click_position = gr.State(-1)

        with gr.Row():
            with gr.Column(scale=1):
                top_k = gr.Number(minimum=1, maximum=100, value=default_top_k, step=1, label="Top K", interactive=True)
                score_function_dropdown = gr.Dropdown(
                    choices=list(SCORE_FUNCTIONS.keys()),
                    value=default_score_func,
                    label="Score Function",
                    interactive=True
                )
                exit_heuristic_dropdown = gr.Dropdown(
                    choices=list(EXIT_HEURISTIC_FUNCTIONS.keys()),
                    value=default_exit_heuristic,
                    label="Exit Heuristic (for plot)", # Clarify it's for the plot
                    interactive=True
                )

            with gr.Column(scale=1):
                reference_steps = gr.Number(minimum=1, maximum=64, value=default_ref_steps, step=1, label="Reference Steps", interactive=True)
                test_steps = gr.Slider(minimum=1, maximum=default_ref_steps, value=default_test_steps, step=1, label="Test Steps", interactive=True)

            with gr.Column(scale=1):
                highlight_top_k = gr.Number(value=default_highlight_top_k, label="Highlight Top K", interactive=True)
                highlight_compare_dropdown = gr.Dropdown(
                    choices=[hc.value for hc in HighlightCompare],
                    value=default_highlight_compare.value,
                    label="Highlight Comparison",
                    interactive=True
                )
                exit_criteria_dropdown = gr.Dropdown(
                    choices=["Expected Policy", "First >50% Exit"],
                    value=default_exit_criteria,
                    label="Exit Criteria (for expected values)", # Clarify what it affects
                    interactive=True
                )

            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=5,
                    value=default_text
                )

        # UI Outputs
        with gr.Row():
            with gr.Column(scale=1):
                token_info = gr.Textbox(label="Token Information", value="Select a token.")
                average_info = gr.Textbox(label="Average Expected Values", value="Processing...", interactive=False) # New component
            with gr.Column(scale=5):
                score_visualization = gr.HTML(label="Score Visualization", value=initial_score_viz)
        with gr.Row():
            with gr.Column(scale=4):
                prediction_plot = gr.Plot(label="Prediction Distributions")
            with gr.Column(scale=2):
                score_progression_plot = gr.Plot(label="Score & Heuristic by Step Count") # Updated label

        # --- Central Update Function ---
        def update_ui(
            text: str,
            ref_steps_val: float, # Gradio Number gives float
            test_steps_val: float, # Gradio Number gives float
            score_func: str,
            exit_heuristic: str,
            exit_criteria: str,
            top_k_val: float, # Gradio Number gives float
            char_index: int,
            highlight_top_k: int,
            highlight_compare: str,
        ):
            _print(f"\n--- UI Update Triggered ---")
            _print(f"Inputs: ref={ref_steps_val}, test={test_steps_val}, score={score_func}, heuristic={exit_heuristic}, criteria={exit_criteria}, top_k={top_k_val}, char_idx={char_index}")
            # --- 1. Update Visualizer State ---
            ref_steps = int(ref_steps_val)
            test_steps = int(test_steps_val)
            top_k_int = int(top_k_val)

            visualizer.set_score_function(score_func)
            visualizer.set_exit_heuristic(exit_heuristic)
            visualizer.set_exit_criteria_mode(exit_criteria == "First >50% Exit")
            visualizer.set_top_k(top_k_int)

            visualizer.highlight_compare = HighlightCompare(highlight_compare)
            visualizer.highlight_top_k = highlight_top_k

            # --- 2. Process Text (runs model if needed) ---
            # Ensure test_steps doesn't exceed ref_steps after potential ref_steps change
            test_steps = min(test_steps, ref_steps)
            visualizer.process_text(text, ref_steps)

            # --- 3. Generate Visualizations ---
            # Update score visualization HTML
            score_viz_html = visualizer.visualize_text_scores(test_steps, char_index)

            # Update token-specific plots and info
            if char_index != -1:
                pred_plot, tk_info, score_prog_plot = visualizer.visualize_selected_token(char_index, test_steps)
            else:
                # Return empty/default values if no token is selected
                pred_plot, tk_info, score_prog_plot = None, "Select a token.", None

            # --- 3.5 Calculate and Format Average Values ---
            avg_steps, avg_score = visualizer.get_average_expected_values()
            if avg_steps is not None and avg_score is not None:
                avg_info_text = f"Avg. Expected Steps: {avg_steps:.2f}\nAvg. Expected Score ({visualizer.selected_score_function}): {avg_score:.3f}"
            else:
                avg_info_text = "Could not calculate average values."


            _print(f"--- UI Update Complete ---")
            # --- 4. Return Updates for UI Components ---
            # Use gr.update() for components whose values might not change
            # Return test_steps as well in case it was capped
            return score_viz_html, pred_plot, tk_info, avg_info_text, score_prog_plot, gr.update(value=test_steps) # Added avg_info_text

        # --- Event Wiring ---

        # List of all inputs required by update_ui
        inputs = [
            text_input,
            reference_steps,
            test_steps,
            score_function_dropdown,
            exit_heuristic_dropdown,
            exit_criteria_dropdown,
            top_k,
            last_click_position, # The state holding the char index
            highlight_top_k,
            highlight_compare_dropdown,
        ]

        # List of all outputs updated by update_ui
        outputs = [
            score_visualization,
            prediction_plot,
            token_info,
            average_info, # Added new output
            score_progression_plot,
            test_steps # Update test_steps value if it was capped
        ]

        # Helper function to store click position
        def store_click_position(evt: gr.SelectData):
            _print(f"Click detected at index: {evt.index[0]}")
            return evt.index[0]

        # Helper function to reset click position (e.g., when text changes)
        def reset_click_position():
            _print("Resetting click position.")
            return -1

        # Update Test Steps Max Value when Reference Steps changes
        def update_test_steps_max(ref_steps_val):
            ref_steps = int(ref_steps_val)
            # Keep current value if valid, otherwise cap it
            current_test_val = int(test_steps.value) if test_steps.value is not None else 1
            new_test_val = min(current_test_val, ref_steps)
            _print(f"Updating test_steps max to {ref_steps}, value to {new_test_val}")
            return gr.update(maximum=ref_steps, value=new_test_val)

        reference_steps.change(
            fn=update_test_steps_max,
            inputs=[reference_steps],
            outputs=[test_steps]
        ).then( # Chain the main UI update after adjusting the slider
            fn=update_ui,
            inputs=inputs,
            outputs=outputs
        )

        # Connect text selection: store index first, then update UI
        text_input.select(
            fn=store_click_position,
            inputs=[], # No direct inputs, uses evt
            outputs=[last_click_position]
        ).then(
            fn=update_ui,
            inputs=inputs,
            outputs=outputs
        )

        # Connect text change: reset index first, then update UI
        text_input.change(
            fn=reset_click_position,
            inputs=[],
            outputs=[last_click_position]
        ).then(
            fn=update_ui,
            inputs=inputs,
            outputs=outputs
        )

        # Connect other controls directly to the main update function
        # (test_steps was already connected via reference_steps.change)
        # We still need a direct connection for when only test_steps changes
        test_steps.change(fn=update_ui, inputs=inputs, outputs=outputs)
        score_function_dropdown.change(fn=update_ui, inputs=inputs, outputs=outputs)
        exit_heuristic_dropdown.change(fn=update_ui, inputs=inputs, outputs=outputs)
        exit_criteria_dropdown.change(fn=update_ui, inputs=inputs, outputs=outputs)
        top_k.change(fn=update_ui, inputs=inputs, outputs=outputs)

        highlight_top_k.change(fn=update_ui, inputs=inputs, outputs=outputs)
        highlight_compare_dropdown.change(fn=update_ui, inputs=inputs, outputs=outputs)


    return interface

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Token Prediction Visualizer")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio sharing."
    )
    parser.add_argument(
        "--exit-model-dir",
        type=str,
        default="checkpoints/checkpoints_18/",
        help="Path to the exit model checkpoint directory."
    )
    parser.add_argument(
        "--exit-model-name",
        type=str,
        default="exit_model_2999.pt",
        help="Name of the exit model checkpoint."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="tomg-group-umd/huginn-0125",
        help="Name or path of the base model."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )


    # Parse arguments
    args = parser.parse_args()

    VERBOSE = args.verbose

    # Create the visualizer instance with parsed arguments
    visualizer = TokenPredictionVisualizer(
        model_name=args.model_name,
        exit_model_dir=args.exit_model_dir,
        exit_model_name=args.exit_model_name,
    )

    # Create and launch the interface
    interface = create_interface(visualizer)
    interface.launch(share=args.share)
